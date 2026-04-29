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

from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (  # noqa: E402
    ausas_unsteady_one_step_gpu,
)


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
        help="Output CSV path.",
    )
    p.add_argument(
        "--schemes",
        nargs="+",
        default=["rb", "jacobi"],
        choices=["rb", "jacobi"],
        help="Schemes to evaluate. Default: both.",
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
        nargs="+",
        default=None,
        help="Restrict to a subset of case names. Default: all.",
    )
    args = p.parse_args()

    cases = CASES
    if args.cases is not None:
        keep = set(args.cases)
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
