"""
Diagnostic reproducer for the Ausas one-step solver — Task 11 of the
diesel-THD ladder.

Goal
----
Decide whether a single application of `ausas_unsteady_one_step_gpu` to a
*fixed* (H_prev -> H_curr, P_prev, theta_prev) state already fails on the
finer grids (160/80, 240/60, 240/80) — independent of the dieselt
mechanical Picard loop. Output is a CSV that can be diffed against the
existing 160/60 baseline.

The script intentionally does NOT simulate the full transient. Each row in
the CSV is exactly one call to the solver with a *frozen* state.

Cases (indexed by `case`)
-------------------------
  smooth_e070     — smooth journal, eps_prev=eps_curr=0.70
  smooth_e085     — smooth journal, eps_prev=0.84, eps_curr=0.85
  smooth_e091     — smooth journal, eps_prev=0.90, eps_curr=0.91
  textured_e085   — herringbone-pattern texture, eps=(0.84, 0.85)
  textured_e091   — herringbone-pattern texture, eps=(0.90, 0.91)

The textured H mimics the diesel `g4_same_depth_safe` preset:
  depth_nondim   = 0.125
  beta_deg       = 30
  N_branch       = 10 per side
  branch_width_rad = 0.148

It is built directly here (no dependency on the article-dump-truck repo)
to keep this script self-contained.

Grids
-----
  (N_phi, N_z) in {(160, 60), (160, 80), (160, 100),
                   (240, 60), (240, 80), (320, 60)}

Solver parameters mirror the diesel runner:
  alpha          = 1.0
  omega_p        = omega_theta = 1.0
  ausas_tol      = 1e-4
  ausas_max_inner = 5000
  periodic_phi   = True, periodic_z = False
  dt             = omega * dt_s for n=2100 rpm and Δφ ≈ 1°
  check_every    = 25

Output
------
  diagnose_ausas_one_step_grid.csv  with columns:
    case, scheme, N_phi, N_z, converged, n_inner,
    residual_linf, residual_rms, residual_l2_abs,
    pmax_nd, theta_min, theta_max,
    nonfinite_count, wall_time_s

Usage
-----
  python scripts/diagnose_ausas_one_step_grid.py
  python scripts/diagnose_ausas_one_step_grid.py --schemes rb jacobi
  python scripts/diagnose_ausas_one_step_grid.py --output /tmp/run.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Solver import — must be on the path. We assume the script is run from the
# repository root or that gpu_reynolds is installed in editable mode.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Solver import is deferred to keep ``--help`` usable on hosts without
# CuPy / a CUDA device.
def _import_solver():
    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        ausas_unsteady_one_step_gpu,
    )
    return ausas_unsteady_one_step_gpu


# Module-level placeholder so callers can `ausas_unsteady_one_step_gpu(...)`
# after ``run_one`` / ``run_shape_lite_one`` resolve it lazily.
ausas_unsteady_one_step_gpu = None


# ---------------------------------------------------------------------------
# Geometry — non-dimensional gap h(phi, z) on a (N_z, N_phi) grid.
# Convention matches reynolds_solver/cavitation/ausas/solver_dynamic_gpu.py:
# the leading axis is Z, the trailing axis is phi. phi-axis is periodic
# (one ghost column on each side); z-axis is Dirichlet (one ghost row on
# each side).
# ---------------------------------------------------------------------------
def _phi_z_grids(N_phi: int, N_z: int):
    """Return (phi, z, d_phi, d_Z) including the +/- ghost rows.

    The interior carries N_phi-2 cells in phi and N_z-2 cells in Z. d_phi
    is one full period / (N_phi - 2). d_Z is L/R / (N_z - 2) for an L/D
    aspect ratio of 1 (the dieseluses ~0.5; we just need a consistent
    alpha factor — the actual value doesn't change the qualitative
    convergence behaviour we are diagnosing).
    """
    d_phi = 2.0 * math.pi / (N_phi - 2)
    # arrange phi so that interior j = 1..N_phi-2 maps to phi in [0, 2*pi)
    phi = (np.arange(N_phi) - 1) * d_phi  # ghosts at -d_phi and 2*pi
    L_over_D = 1.0
    d_Z = L_over_D / (N_z - 2)
    z = (np.arange(N_z) - 0.5) * d_Z      # cell-centres in [0, L/D]
    return phi, z, d_phi, d_Z


def smooth_gap(N_phi: int, N_z: int, eps: float):
    """h_nd(phi, z) = 1 + eps * cos(phi). Shape (N_z, N_phi)."""
    phi, _z, _, _ = _phi_z_grids(N_phi, N_z)
    h_phi = 1.0 + eps * np.cos(phi)
    return np.broadcast_to(h_phi[None, :], (N_z, N_phi)).copy()


def herringbone_relief(
    N_phi: int,
    N_z: int,
    depth_nondim: float = 0.125,
    beta_deg: float = 30.0,
    N_branch: int = 10,
    branch_width_rad: float = 0.148,
):
    """
    Builds a herringbone-pattern relief mimicking g4_same_depth_safe.

    The pattern is a 2D field of N_branch oblique grooves per side
    spanning the bearing length. Each branch is a Gaussian-tube in a
    rotated (phi', z') frame with full-width branch_width_rad. Two
    mirrored sets meet at z = L/2.

    Returns relief (>= 0) of shape (N_z, N_phi). Subtract from the smooth
    gap to get the textured H.
    """
    phi, z, _, _ = _phi_z_grids(N_phi, N_z)
    z_norm = z / max(z[-2] - z[1], 1e-12)  # ~[0, 1] interior

    beta = math.radians(beta_deg)
    sigma = branch_width_rad / 2.355      # Gaussian FWHM -> sigma

    relief = np.zeros((N_z, N_phi), dtype=np.float64)

    # Lower half: oblique grooves slope (+); upper half: mirrored (-).
    for half in (0, 1):
        sign = 1.0 if half == 0 else -1.0
        z_off = 0.0 if half == 0 else 0.5
        for k in range(N_branch):
            phi_center = (k + 0.5) * (2.0 * math.pi / N_branch)
            for iz in range(N_z):
                if half == 0 and z_norm[iz] > 0.5:
                    continue
                if half == 1 and z_norm[iz] < 0.5:
                    continue
                z_local = (z_norm[iz] - z_off) * 0.5     # [0, 0.5]
                # rotated phi distance to branch axis
                phi_axis = phi_center + sign * z_local * math.tan(beta)
                d = ((phi - phi_axis + math.pi) % (2.0 * math.pi)) - math.pi
                relief[iz, :] += depth_nondim * np.exp(
                    -0.5 * (d / sigma) ** 2
                )
    # Cap relief at depth_nondim to avoid unphysical superposition.
    np.minimum(relief, depth_nondim, out=relief)
    return relief


def textured_gap(N_phi: int, N_z: int, eps: float):
    h = smooth_gap(N_phi, N_z, eps)
    return h - herringbone_relief(N_phi, N_z)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
CASES = [
    # (name, gap_builder, eps_prev, eps_curr)
    ("smooth_e070",   smooth_gap,   0.70, 0.70),
    ("smooth_e085",   smooth_gap,   0.84, 0.85),
    ("smooth_e091",   smooth_gap,   0.90, 0.91),
    ("textured_e085", textured_gap, 0.84, 0.85),
    ("textured_e091", textured_gap, 0.90, 0.91),
]

GRIDS = [
    (160,  60),
    (160,  80),
    (160, 100),
    (240,  60),
    (240,  80),
    (320,  60),
]


# ---------------------------------------------------------------------------
# Shape-lite mode (Task 31)
#
# A coarse screen for shape-dependent kernel/ghost/BC bugs on synthetic H.
# Passing --shape-lite is a NECESSARY-but-not-SUFFICIENT check: a clean
# shape-lite run does NOT prove that real diesel transient dumps will
# pass — it only filters the most obvious shape-dependent failures.
# Real dumps must still go through scripts/replay_ausas_one_step_dump.py.
# ---------------------------------------------------------------------------
SHAPE_LITE_N_PHI = [159, 160, 161, 239, 240, 241, 319, 320, 321]
SHAPE_LITE_N_Z   = [59,  60,  61,  79,  80,  81,  99,  100, 101]

SHAPE_LITE_CASES = {
    # name -> (case_detail, gap_builder, eps_prev, eps_curr)
    "smooth_e06":  ("smooth",       smooth_gap,   0.55, 0.60),
    "smooth_e085": ("smooth",       smooth_gap,   0.84, 0.85),
    "g4_e06":      ("g4_synthetic", textured_gap, 0.55, 0.60),
    "g4_e085":     ("g4_synthetic", textured_gap, 0.84, 0.85),
}

SHAPE_LITE_FIELDS = [
    "N_phi", "N_z", "padded_width", "scheme", "case", "case_detail",
    "converged", "failure_kind", "nan_iter", "first_nan_field",
    "first_nan_i", "first_nan_j",
    "first_nan_is_ghost", "first_nan_is_axial_boundary",
    "first_nan_is_phi_seam",
    "residual_linf", "residual_rms", "residual_l2_abs",
    "n_inner",
    "pmax_nd", "theta_min", "theta_max",
    "nonfinite_count", "wall_time_s",
]


def _shape_lite_initial_state(N_phi: int, N_z: int, theta_perturb: float,
                              rng: np.random.Generator):
    """Finite, bounded initial state. P=0, theta in [0, 1]."""
    P_prev = np.zeros((N_z, N_phi), dtype=np.float64)
    theta_prev = np.ones((N_z, N_phi), dtype=np.float64)
    if theta_perturb > 0.0:
        delta = rng.uniform(
            -theta_perturb, theta_perturb, size=theta_prev.shape
        )
        np.clip(theta_prev + delta, 0.0, 1.0, out=theta_prev)
    return P_prev, theta_prev


def _empty_shape_lite_row(*, N_phi, N_z, scheme, case, case_detail,
                          failure_kind):
    return {
        "N_phi": N_phi,
        "N_z": N_z,
        "padded_width": N_phi + 2,
        "scheme": scheme,
        "case": case,
        "case_detail": case_detail,
        "converged": 0,
        "failure_kind": failure_kind,
        "nan_iter": "",
        "first_nan_field": "",
        "first_nan_i": "",
        "first_nan_j": "",
        "first_nan_is_ghost": "",
        "first_nan_is_axial_boundary": "",
        "first_nan_is_phi_seam": "",
        "residual_linf": "",
        "residual_rms": "",
        "residual_l2_abs": "",
        "n_inner": "",
        "pmax_nd": "",
        "theta_min": "",
        "theta_max": "",
        "nonfinite_count": "",
        "wall_time_s": "",
    }


def run_shape_lite_one(
    case_name: str,
    case_detail: str,
    gap_builder,
    eps_prev: float,
    eps_curr: float,
    N_phi: int,
    N_z: int,
    *,
    scheme: str,
    tol: float,
    max_inner: int,
    dt: float,
    debug_checks: bool,
    rng: np.random.Generator,
    theta_perturb: float,
):
    # On the padded one-step grid we want the *physical* phi count to be
    # exactly the user's --n-phi-list value. The solver works in padded
    # coordinates internally, so we feed shape (N_z, N_phi+2).
    N_phi_padded = N_phi + 2

    H_prev = gap_builder(N_phi_padded, N_z, eps_prev)
    H_curr = gap_builder(N_phi_padded, N_z, eps_curr)
    P_prev, theta_prev = _shape_lite_initial_state(
        N_phi_padded, N_z, theta_perturb, rng,
    )

    _phi, _z, d_phi, d_Z = _phi_z_grids(N_phi_padded, N_z)
    R = 1.0
    L = 1.0

    t0 = time.perf_counter()
    out = ausas_unsteady_one_step_gpu(
        H_curr, H_prev, P_prev, theta_prev,
        dt, d_phi, d_Z, R, L,
        alpha=1.0, omega_p=1.0, omega_theta=1.0,
        tol=tol, max_inner=max_inner,
        p_bc_z0=0.0, p_bc_zL=0.0,
        theta_bc_z0=1.0, theta_bc_zL=1.0,
        periodic_phi=True, periodic_z=False,
        check_every=25,
        scheme=scheme,
        residual_norm="linf",
        verbose=False,
        debug_checks=debug_checks,
        debug_check_every=50,
        debug_stop_on_nonfinite=True,
        debug_return_last_finite_state=True,
        debug_return_bad_state=False,
    )
    wall = time.perf_counter() - t0

    P = np.asarray(out["P"])
    theta = np.asarray(out["theta"])
    P_int = P[1:-1, 1:-1] if P.size else P
    theta_int = theta[1:-1, 1:-1] if theta.size else theta
    nonfinite = int(
        np.sum(~np.isfinite(P_int)) + np.sum(~np.isfinite(theta_int))
    )

    idx = out.get("first_nan_index")
    if idx is not None:
        fi, fj = int(idx[0]), int(idx[1])
    else:
        fi = ""
        fj = ""

    return {
        "N_phi": N_phi,
        "N_z": N_z,
        "padded_width": N_phi_padded,
        "scheme": scheme,
        "case": case_name,
        "case_detail": case_detail,
        "converged": int(bool(out.get("converged", False))),
        "failure_kind": out.get("failure_kind") or "",
        "nan_iter": (
            "" if out.get("nan_iter") is None else int(out["nan_iter"])
        ),
        "first_nan_field": out.get("first_nan_field") or "",
        "first_nan_i": fi,
        "first_nan_j": fj,
        "first_nan_is_ghost": int(bool(out.get(
            "first_nan_is_ghost", False
        ))),
        "first_nan_is_axial_boundary": int(bool(out.get(
            "first_nan_is_axial_boundary", False
        ))),
        "first_nan_is_phi_seam": int(bool(out.get(
            "first_nan_is_phi_seam", False
        ))),
        "residual_linf": float(out.get("residual_linf", float("nan"))),
        "residual_rms": float(out.get("residual_rms", float("nan"))),
        "residual_l2_abs": float(out.get(
            "residual_l2_abs", float("nan")
        )),
        "n_inner": int(out.get("n_inner", 0)),
        "pmax_nd": float(np.nanmax(P_int)) if P_int.size else float("nan"),
        "theta_min": float(np.nanmin(theta_int)) if theta_int.size
                     else float("nan"),
        "theta_max": float(np.nanmax(theta_int)) if theta_int.size
                     else float("nan"),
        "nonfinite_count": nonfinite,
        "wall_time_s": wall,
    }


def run_shape_lite(args) -> int:
    """Entry point for --shape-lite. Returns exit code."""
    global ausas_unsteady_one_step_gpu
    if ausas_unsteady_one_step_gpu is None:
        ausas_unsteady_one_step_gpu = _import_solver()
    n_phi_list = (
        [int(x) for x in args.n_phi_list.split(",")]
        if args.n_phi_list else SHAPE_LITE_N_PHI
    )
    n_z_list = (
        [int(x) for x in args.n_z_list.split(",")]
        if args.n_z_list else SHAPE_LITE_N_Z
    )
    case_names = (
        [c.strip() for c in args.cases.split(",")]
        if isinstance(args.cases, str) and args.cases
        else list(SHAPE_LITE_CASES.keys())
    )
    bad = [c for c in case_names if c not in SHAPE_LITE_CASES]
    if bad:
        raise SystemExit(
            f"unknown shape-lite case(s) {bad}. "
            f"Allowed: {sorted(SHAPE_LITE_CASES)}"
        )

    schemes = (
        [s.strip() for s in args.schemes_str.split(",")]
        if args.schemes_str else ["rb", "jacobi"]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    print(
        "shape-lite is a coarse screen only; real transient dumps are"
        " still required to clear the solver."
    )
    for scheme in schemes:
        for case_name in case_names:
            case_detail, builder, eps_prev, eps_curr = SHAPE_LITE_CASES[
                case_name
            ]
            for N_phi in n_phi_list:
                for N_z in n_z_list:
                    tag = (
                        f"[{scheme}] {case_name:<11s} "
                        f"{N_phi:>3d}/{N_z:<3d}"
                    )
                    try:
                        row = run_shape_lite_one(
                            case_name, case_detail, builder,
                            eps_prev, eps_curr,
                            N_phi, N_z,
                            scheme=scheme,
                            tol=args.tol,
                            max_inner=args.max_inner,
                            dt=args.dt_tau,
                            debug_checks=args.debug_checks,
                            rng=rng,
                            theta_perturb=args.random_theta_perturb,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {tag}: EXC {type(exc).__name__}: {exc}")
                        row = _empty_shape_lite_row(
                            N_phi=N_phi, N_z=N_z, scheme=scheme,
                            case=case_name, case_detail=case_detail,
                            failure_kind=f"exception:{type(exc).__name__}",
                        )
                    rows.append(row)
                    print(
                        f"  {tag}: conv={row['converged']} "
                        f"fk={row['failure_kind'] or '-'} "
                        f"ni={row['nan_iter']} "
                        f"linf={row['residual_linf']}"
                    )

    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SHAPE_LITE_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {len(rows)} rows to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_one(
    case_name: str,
    gap_builder,
    eps_prev: float,
    eps_curr: float,
    N_phi: int,
    N_z: int,
    *,
    scheme: str,
    tol: float,
    max_inner: int,
    dt: float,
):
    H_prev = gap_builder(N_phi, N_z, eps_prev)
    H_curr = gap_builder(N_phi, N_z, eps_curr)

    P_prev = np.zeros((N_z, N_phi), dtype=np.float64)
    theta_prev = np.ones((N_z, N_phi), dtype=np.float64)

    _phi, _z, d_phi, d_Z = _phi_z_grids(N_phi, N_z)
    R = 1.0
    L = 1.0  # alpha-factor uses 2*R/L * d_phi/d_Z; concrete value unimportant

    t0 = time.perf_counter()
    out = ausas_unsteady_one_step_gpu(
        H_curr, H_prev, P_prev, theta_prev,
        dt, d_phi, d_Z, R, L,
        alpha=1.0, omega_p=1.0, omega_theta=1.0,
        tol=tol, max_inner=max_inner,
        p_bc_z0=0.0, p_bc_zL=0.0,
        theta_bc_z0=1.0, theta_bc_zL=1.0,
        periodic_phi=True, periodic_z=False,
        check_every=25,
        scheme=scheme,
        residual_norm="linf",
        verbose=False,
    )
    wall = time.perf_counter() - t0

    P = out["P"]
    theta = out["theta"]
    P_int = P[1:-1, 1:-1]
    theta_int = theta[1:-1, 1:-1]
    nonfinite = int(np.sum(~np.isfinite(P_int)) + np.sum(~np.isfinite(theta_int)))

    return {
        "case": case_name,
        "scheme": scheme,
        "N_phi": N_phi,
        "N_z": N_z,
        "converged": int(out["converged"]),
        "n_inner": out["n_inner"],
        "residual_linf": out["residual_linf"],
        "residual_rms": out["residual_rms"],
        "residual_l2_abs": out["residual_l2_abs"],
        "pmax_nd": float(np.max(P_int)) if P_int.size else float("nan"),
        "theta_min": float(np.min(theta_int)) if theta_int.size else float("nan"),
        "theta_max": float(np.max(theta_int)) if theta_int.size else float("nan"),
        "nonfinite_count": nonfinite,
        "wall_time_s": wall,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        default=str(ROOT / "diagnose_ausas_one_step_grid.csv"),
        help="Output CSV path (legacy mode).",
    )
    p.add_argument(
        "--schemes",
        nargs="+",
        default=["rb", "jacobi"],
        choices=["rb", "jacobi"],
        help="Schemes to evaluate (legacy mode). Default: both.",
    )
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--max-inner", type=int, default=5000)
    p.add_argument(
        "--dt-tau",
        type=float,
        default=2.0 * math.pi / 360.0,
        help="dt in tau units. Default: ~1deg of phi at unit omega.",
    )
    p.add_argument(
        "--cases",
        default=None,
        help="Comma-separated subset of case names. "
             "Legacy mode default: all CASES; "
             "--shape-lite default: all four shape-lite cases.",
    )

    # --- shape-lite mode (Task 31) ---------------------------------------
    p.add_argument(
        "--shape-lite", action="store_true",
        help="Coarse screen for shape-dependent kernel/ghost/BC bugs on "
             "synthetic H. Does NOT replace real dump replay.",
    )
    p.add_argument(
        "--n-phi-list", default="",
        help="Comma-separated physical N_phi list (shape-lite). Default: "
             "159,160,161,239,240,241,319,320,321.",
    )
    p.add_argument(
        "--n-z-list", default="",
        help="Comma-separated N_z list (shape-lite). Default: "
             "59,60,61,79,80,81,99,100,101.",
    )
    p.add_argument(
        "--schemes-str", default="",
        help="Comma-separated scheme list for shape-lite mode "
             "(default: rb,jacobi). Distinct from --schemes to keep "
             "legacy mode unaffected.",
    )
    p.add_argument(
        "--debug-checks", action="store_true",
        help="Enable debug_checks=True in shape-lite mode.",
    )
    p.add_argument(
        "--random-theta-perturb", type=float, default=0.0,
        help="Bounded random perturbation amplitude for theta_prev "
             "(shape-lite). Default 0.0 (no perturbation).",
    )
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument(
        "--out", default="shape_sweep_lite.csv",
        help="CSV output path for shape-lite mode.",
    )
    args = p.parse_args()

    if args.shape_lite:
        return run_shape_lite(args)

    global ausas_unsteady_one_step_gpu
    if ausas_unsteady_one_step_gpu is None:
        ausas_unsteady_one_step_gpu = _import_solver()

    cases = CASES
    if args.cases is not None:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        cases = [c for c in CASES if c[0] in keep]
        if not cases:
            raise SystemExit(f"No cases match {args.cases!r}")

    out_path = Path(args.output)
    fieldnames = [
        "case", "scheme", "N_phi", "N_z", "converged", "n_inner",
        "residual_linf", "residual_rms", "residual_l2_abs",
        "pmax_nd", "theta_min", "theta_max",
        "nonfinite_count", "wall_time_s",
    ]

    rows = []
    for scheme in args.schemes:
        for (cname, builder, eps_prev, eps_curr) in cases:
            for (N_phi, N_z) in GRIDS:
                tag = f"[{scheme}] {cname:<14s}  {N_phi:>3d}/{N_z:<3d}"
                try:
                    row = run_one(
                        cname, builder, eps_prev, eps_curr,
                        N_phi, N_z,
                        scheme=scheme,
                        tol=args.tol,
                        max_inner=args.max_inner,
                        dt=args.dt_tau,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  {tag}: FAILED — {type(exc).__name__}: {exc}")
                    row = {
                        "case": cname, "scheme": scheme,
                        "N_phi": N_phi, "N_z": N_z,
                        "converged": 0, "n_inner": -1,
                        "residual_linf": float("nan"),
                        "residual_rms": float("nan"),
                        "residual_l2_abs": float("nan"),
                        "pmax_nd": float("nan"),
                        "theta_min": float("nan"),
                        "theta_max": float("nan"),
                        "nonfinite_count": -1,
                        "wall_time_s": float("nan"),
                    }
                rows.append(row)
                print(
                    f"  {tag}: conv={row['converged']} "
                    f"n_inner={row['n_inner']:>4d} "
                    f"linf={row['residual_linf']:.2e} "
                    f"rms={row['residual_rms']:.2e} "
                    f"l2={row['residual_l2_abs']:.2e} "
                    f"pmax={row['pmax_nd']:.3e} "
                    f"t={row['wall_time_s']:.2f}s"
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
