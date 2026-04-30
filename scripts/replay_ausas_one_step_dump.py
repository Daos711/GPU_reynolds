"""
Replay a saved Ausas one-step dump from the diesel transient runner —
Task 30 of the diesel-THD ladder.

Reads a ``.npz`` dump (or a directory of them), reconstructs the inputs
to ``ausas_unsteady_one_step_gpu``, and replays each one across a sweep
of solver parameters (scheme / omega_p / omega_theta / max_inner /
check_every). Output:

* a CSV with one row per (dump, scheme, omega_p, omega_theta,
  max_inner, check_every) combination,
* an optional markdown trace report per dump under ``--trace-nan``,
  produced after a coarse-then-fine refinement of ``nan_iter``.

The script does NOT depend on the article-dump-truck repository — every
field is read from the ``.npz`` directly. Required keys are:

    H_prev, H_curr, P_prev, theta_prev,
    dt_s OR dt_tau,
    d_phi, d_Z,
    periodic_phi, periodic_z,
    solver_kwargs   (optional dict-as-object)

Optional metadata keys (used for the markdown report only): ``step``,
``trial``, ``substep``, ``commit``, ``phi_deg``, ``eps_x``, ``eps_y``,
``config_label``, ``trial_kind``, ``texture_kind``, ``groove_preset``.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CSV_FIELDS = [
    "dump",
    "scheme",
    "omega_p",
    "omega_theta",
    "max_inner",
    "check_every",
    "converged",
    "failure_kind",
    "nan_iter",
    "first_nan_field",
    "first_nan_i",
    "first_nan_j",
    "first_nan_is_ghost",
    "first_nan_is_axial_boundary",
    "first_nan_is_phi_seam",
    "H_at_nan",
    "P_prev_at_nan",
    "theta_prev_at_nan",
    "residual_linf",
    "residual_rms",
    "residual_l2_abs",
    "pmax_nd",
    "theta_min",
    "theta_max",
    "nonfinite_count",
    "wall_time_s",
]


# ---------------------------------------------------------------------------
# Dump I/O
# ---------------------------------------------------------------------------
@dataclass
class DumpPayload:
    path: Path
    H_prev: np.ndarray
    H_curr: np.ndarray
    P_prev: np.ndarray
    theta_prev: np.ndarray
    dt_s: float
    d_phi: float
    d_Z: float
    R: float
    L: float
    periodic_phi: bool
    periodic_z: bool
    solver_kwargs: dict
    metadata: dict


def _coerce_solver_kwargs(raw) -> dict:
    """Allow solver_kwargs to be stored either as a 0-d object array
    (np.savez) or as a JSON string (some dump variants)."""
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray):
        if raw.dtype == object:
            return dict(raw.item()) if raw.size == 1 else {}
        if raw.dtype.kind in ("U", "S"):
            import json
            return json.loads(str(raw))
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, (str, bytes)):
        import json
        return json.loads(raw)
    return {}


def _coerce_scalar(value, default=None):
    if value is None:
        return default
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    if isinstance(value, np.ndarray) and value.size == 0:
        return default
    return value


def load_dump(path: Path) -> DumpPayload:
    """Read a single ``.npz`` dump. Raises ValueError with the missing
    key on failure."""
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"failed to read {path}: {exc}") from exc

    keys = set(z.files)

    def _need(name, alts: Iterable[str] = ()):
        for n in (name, *alts):
            if n in keys:
                return z[n]
        raise ValueError(
            f"{path}: required key {name!r} not found "
            f"(file keys: {sorted(keys)})"
        )

    H_prev = np.asarray(_need("H_prev"), dtype=np.float64)
    H_curr = np.asarray(_need("H_curr"), dtype=np.float64)
    P_prev = np.asarray(_need("P_prev"), dtype=np.float64)
    theta_prev = np.asarray(_need("theta_prev"), dtype=np.float64)

    if "dt_s" in keys:
        dt_s = float(_coerce_scalar(z["dt_s"]))
    elif "dt_tau" in keys:
        dt_s = float(_coerce_scalar(z["dt_tau"]))
    else:
        raise ValueError(
            f"{path}: required key 'dt_s' (or 'dt_tau') not found"
        )

    d_phi = float(_coerce_scalar(_need("d_phi")))
    d_Z = float(_coerce_scalar(_need("d_Z")))

    R = float(_coerce_scalar(z["R"], default=1.0)) if "R" in keys else 1.0
    L = float(_coerce_scalar(z["L"], default=1.0)) if "L" in keys else 1.0

    periodic_phi = bool(
        _coerce_scalar(z["periodic_phi"], default=True)
    ) if "periodic_phi" in keys else True
    periodic_z = bool(
        _coerce_scalar(z["periodic_z"], default=False)
    ) if "periodic_z" in keys else False

    solver_kwargs = _coerce_solver_kwargs(
        z["solver_kwargs"] if "solver_kwargs" in keys else None
    )

    meta_keys = (
        "step", "trial", "substep", "commit", "phi_deg",
        "eps_x", "eps_y", "config_label", "trial_kind",
        "texture_kind", "groove_preset",
    )
    metadata = {}
    for k in meta_keys:
        if k in keys:
            v = _coerce_scalar(z[k])
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            metadata[k] = v

    return DumpPayload(
        path=path,
        H_prev=H_prev,
        H_curr=H_curr,
        P_prev=P_prev,
        theta_prev=theta_prev,
        dt_s=dt_s,
        d_phi=d_phi,
        d_Z=d_Z,
        R=R,
        L=L,
        periodic_phi=periodic_phi,
        periodic_z=periodic_z,
        solver_kwargs=solver_kwargs,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def _parse_omega_list(spec: str, current_value):
    """Parse 'current,1.3,1.45' → [current_value, 1.3, 1.45]. ``current``
    expands to the dump's value or is dropped (with a warning) when the
    dump didn't carry it."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "current":
            if current_value is None:
                print(
                    "  WARNING: 'current' requested but solver_kwargs"
                    " has no matching key — dropping.",
                    file=sys.stderr,
                )
                continue
            out.append(float(current_value))
        else:
            out.append(float(tok))
    return out


def _parse_int_list(spec: str):
    return [int(t.strip()) for t in spec.split(",") if t.strip()]


def _parse_str_list(spec: str):
    return [t.strip() for t in spec.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------
def _safe_index_value(arr: np.ndarray, idx):
    if idx is None:
        return float("nan")
    i, j = idx
    if not (0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]):
        return float("nan")
    return float(arr[i, j])


def replay_one(
    one_step,
    dump: DumpPayload,
    *,
    scheme: str,
    omega_p: float,
    omega_theta: float,
    max_inner: int,
    check_every: int,
    trace_nan: bool,
    debug_check_every: Optional[int] = None,
    debug_check_start_iter: int = 0,
):
    """Run a single replay. Returns a CSV-row dict and the raw result dict.

    ``trace_nan`` toggles ``debug_return_bad_state``. ``debug_check_every``
    overrides the cadence (defaults to ``check_every``).
    """
    if debug_check_every is None:
        debug_check_every = check_every

    t0 = time.perf_counter()
    try:
        result = one_step(
            dump.H_curr, dump.H_prev, dump.P_prev, dump.theta_prev,
            dt=dump.dt_s, d_phi=dump.d_phi, d_Z=dump.d_Z,
            R=dump.R, L=dump.L,
            alpha=float(dump.solver_kwargs.get("alpha", 1.0)),
            omega_p=omega_p,
            omega_theta=omega_theta,
            tol=float(dump.solver_kwargs.get("tol", 1e-4)),
            max_inner=max_inner,
            periodic_phi=dump.periodic_phi,
            periodic_z=dump.periodic_z,
            check_every=check_every,
            scheme=scheme,
            residual_norm=str(dump.solver_kwargs.get(
                "residual_norm", "linf"
            )),
            debug_checks=True,
            debug_check_every=debug_check_every,
            debug_check_start_iter=debug_check_start_iter,
            debug_stop_on_nonfinite=True,
            debug_return_bad_state=trace_nan,
            debug_return_last_finite_state=True,
        )
    except Exception as exc:  # noqa: BLE001
        wall = time.perf_counter() - t0
        return {
            "dump": str(dump.path),
            "scheme": scheme,
            "omega_p": omega_p,
            "omega_theta": omega_theta,
            "max_inner": max_inner,
            "check_every": check_every,
            "converged": 0,
            "failure_kind": f"exception:{type(exc).__name__}",
            "nan_iter": "",
            "first_nan_field": "",
            "first_nan_i": "",
            "first_nan_j": "",
            "first_nan_is_ghost": "",
            "first_nan_is_axial_boundary": "",
            "first_nan_is_phi_seam": "",
            "H_at_nan": "",
            "P_prev_at_nan": "",
            "theta_prev_at_nan": "",
            "residual_linf": "",
            "residual_rms": "",
            "residual_l2_abs": "",
            "pmax_nd": "",
            "theta_min": "",
            "theta_max": "",
            "nonfinite_count": "",
            "wall_time_s": wall,
        }, None
    wall = time.perf_counter() - t0

    P = np.asarray(result["P"])
    theta = np.asarray(result["theta"])
    P_int = P[1:-1, 1:-1] if P.size else P
    theta_int = theta[1:-1, 1:-1] if theta.size else theta
    nonfinite = int(
        np.sum(~np.isfinite(P_int)) + np.sum(~np.isfinite(theta_int))
    )
    pmax_nd = float(np.nanmax(P_int)) if P_int.size else float("nan")
    th_min = float(np.nanmin(theta_int)) if theta_int.size else float("nan")
    th_max = float(np.nanmax(theta_int)) if theta_int.size else float("nan")

    idx = result.get("first_nan_index")
    if idx is not None:
        fi, fj = int(idx[0]), int(idx[1])
        h_at = _safe_index_value(dump.H_curr, idx)
        p_at = _safe_index_value(dump.P_prev, idx)
        t_at = _safe_index_value(dump.theta_prev, idx)
    else:
        fi = ""
        fj = ""
        h_at = ""
        p_at = ""
        t_at = ""

    row = {
        "dump": str(dump.path),
        "scheme": scheme,
        "omega_p": omega_p,
        "omega_theta": omega_theta,
        "max_inner": max_inner,
        "check_every": check_every,
        "converged": int(bool(result.get("converged", False))),
        "failure_kind": result.get("failure_kind") or "",
        "nan_iter": (
            "" if result.get("nan_iter") is None else int(result["nan_iter"])
        ),
        "first_nan_field": result.get("first_nan_field") or "",
        "first_nan_i": fi,
        "first_nan_j": fj,
        "first_nan_is_ghost": int(bool(result.get(
            "first_nan_is_ghost", False
        ))),
        "first_nan_is_axial_boundary": int(bool(result.get(
            "first_nan_is_axial_boundary", False
        ))),
        "first_nan_is_phi_seam": int(bool(result.get(
            "first_nan_is_phi_seam", False
        ))),
        "H_at_nan": h_at,
        "P_prev_at_nan": p_at,
        "theta_prev_at_nan": t_at,
        "residual_linf": float(result.get("residual_linf", float("nan"))),
        "residual_rms": float(result.get("residual_rms", float("nan"))),
        "residual_l2_abs": float(result.get(
            "residual_l2_abs", float("nan")
        )),
        "pmax_nd": pmax_nd,
        "theta_min": th_min,
        "theta_max": th_max,
        "nonfinite_count": nonfinite,
        "wall_time_s": wall,
    }
    return row, result


# ---------------------------------------------------------------------------
# Markdown trace report
# ---------------------------------------------------------------------------
def _interpretation_hint(row_rb: Optional[dict], row_jac: Optional[dict],
                         best: dict) -> str:
    """Pick a one-line root-cause hint based on the best row + RB/Jacobi
    contrast. Returns a multi-line bulleted string."""
    bullets = []

    def _failed(r):
        return r is not None and r.get("failure_kind") not in ("", None)

    rb_fail = _failed(row_rb)
    jac_fail = _failed(row_jac)
    if rb_fail and not jac_fail and row_jac is not None:
        bullets.append(
            "RB fails while Jacobi does not -> suspect RB kernel / RB BC "
            "ordering / in-place colour interaction."
        )
    elif rb_fail and jac_fail and row_rb is not None and row_jac is not None:
        same_loc = (
            row_rb.get("first_nan_i") == row_jac.get("first_nan_i")
            and row_rb.get("first_nan_j") == row_jac.get("first_nan_j")
        )
        if same_loc:
            bullets.append(
                "RB and Jacobi fail at the same cell -> input state, "
                "discretization, complementarity rule, or H-gradient."
            )

    if best.get("first_nan_is_ghost") == 1:
        bullets.append(
            "First nonfinite in the phi ghost column -> BC / ghost-pack / "
            "shape path."
        )
    if best.get("first_nan_is_axial_boundary") == 1:
        bullets.append(
            "First nonfinite on the z-boundary row -> z-BC path."
        )
    if (
        best.get("first_nan_is_phi_seam") == 1
        and best.get("first_nan_is_ghost") != 1
    ):
        bullets.append(
            "First nonfinite in a phi-seam-adjacent interior column "
            "-> seam stencil / periodic projection."
        )
    if best.get("first_nan_field") == "coeff":
        bullets.append(
            "Coefficient buffer non-finite -> H<=0 cell or H-gradient "
            "blowup feeding average-of-cubes."
        )
    if best.get("first_nan_field") == "residual":
        bullets.append(
            "Residual non-finite while state still in bounds at last "
            "snapshot -> NaN propagation between debug checks; tighten "
            "debug_check_every or rerun with --trace-nan."
        )

    # Omega-overshoot heuristic — only reliable if we have multiple omega rows.
    return "\n".join(f"- {b}" for b in bullets) if bullets else "- (none)"


def write_trace_report(
    out_dir: Path,
    dump: DumpPayload,
    rows: list[dict],
    best_row: dict,
):
    """Write ``dump_trace_<stem>.md`` summarizing the failure for one dump."""
    stem = dump.path.stem
    md_path = out_dir / f"dump_trace_{stem}.md"

    rb_rows = [r for r in rows if r["scheme"] == "rb"]
    jac_rows = [r for r in rows if r["scheme"] == "jacobi"]
    rb_first_fail = next(
        (r for r in rb_rows if r.get("failure_kind") not in ("", None)),
        None,
    )
    jac_first = jac_rows[0] if jac_rows else None

    lines = []
    lines.append(f"# Ausas one-step replay trace — `{stem}`\n")
    lines.append("## Dump metadata")
    for k in (
        "step", "trial", "substep", "commit", "phi_deg",
        "eps_x", "eps_y", "config_label", "trial_kind",
        "texture_kind", "groove_preset",
    ):
        if k in dump.metadata:
            lines.append(f"- **{k}**: `{dump.metadata[k]}`")
    lines.append(
        f"- **shape**: H_curr {dump.H_curr.shape}, "
        f"d_phi={dump.d_phi:.6e}, d_Z={dump.d_Z:.6e}, dt_s={dump.dt_s:.6e}"
    )
    lines.append(f"- **path**: `{dump.path}`")
    lines.append("")

    lines.append("## Best reproduction row (first observed failure)")
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    for k in (
        "scheme", "omega_p", "omega_theta", "max_inner", "check_every",
    ):
        lines.append(f"| {k} | `{best_row[k]}` |")
    lines.append("")

    lines.append("## Failure")
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    for k in (
        "failure_kind", "nan_iter", "first_nan_field",
        "first_nan_i", "first_nan_j",
    ):
        lines.append(f"| {k} | `{best_row[k]}` |")
    lines.append("")

    lines.append("## Location flags")
    lines.append(
        f"- first_nan_is_ghost: `{best_row['first_nan_is_ghost']}`"
    )
    lines.append(
        f"- first_nan_is_axial_boundary: "
        f"`{best_row['first_nan_is_axial_boundary']}`"
    )
    lines.append(
        f"- first_nan_is_phi_seam: `{best_row['first_nan_is_phi_seam']}`"
    )
    lines.append("")

    lines.append("## Local values at first_nan_index")
    lines.append(f"- H_curr  : `{best_row['H_at_nan']}`")
    lines.append(f"- P_prev  : `{best_row['P_prev_at_nan']}`")
    lines.append(f"- theta_prev: `{best_row['theta_prev_at_nan']}`")
    lines.append("")

    lines.append("## Interpretation hints")
    lines.append(_interpretation_hint(rb_first_fail, jac_first, best_row))
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {md_path}")
    return md_path


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Replay an Ausas one-step .npz dump."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dump", type=Path, help="Path to a single .npz dump.")
    src.add_argument(
        "--dump-dir", type=Path,
        help="Path to a directory of .npz dumps (recurses with *.npz).",
    )

    p.add_argument(
        "--schemes", default="rb,jacobi",
        help="Comma-separated solver schemes (subset of {rb, jacobi}).",
    )
    p.add_argument(
        "--omega-p-list", default="current,1.3,1.45,1.55,1.65,1.7",
        help="Comma-separated omega_p values. 'current' = read from "
             "solver_kwargs in the dump.",
    )
    p.add_argument(
        "--omega-theta-list", default="current,0.7,0.85,1.0",
    )
    p.add_argument(
        "--max-inner-list", default="2000,5000,10000",
    )
    p.add_argument(
        "--check-every-list", default="50,25,5,1",
    )
    p.add_argument(
        "--trace-nan", action="store_true",
        help="Two-pass coarse->fine refinement of nan_iter and a markdown "
             "trace report per failing dump.",
    )
    p.add_argument(
        "--out", type=Path, default=Path("dump_replay.csv"),
        help="Output CSV path.",
    )
    args = p.parse_args()

    # --- Locate dumps ---
    if args.dump is not None:
        if not args.dump.exists():
            raise SystemExit(f"--dump path does not exist: {args.dump}")
        dump_paths = [args.dump]
    else:
        if not args.dump_dir.exists():
            raise SystemExit(
                f"--dump-dir path does not exist: {args.dump_dir}"
            )
        dump_paths = sorted(args.dump_dir.rglob("*.npz"))
        if not dump_paths:
            raise SystemExit(f"no .npz files under {args.dump_dir}")

    # --- Solver import (fail loudly if cupy/GPU is missing) ---
    try:
        from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
            ausas_unsteady_one_step_gpu,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"failed to import solver: {exc}")

    schemes = _parse_str_list(args.schemes)
    max_inner_list = _parse_int_list(args.max_inner_list)
    check_every_list = _parse_int_list(args.check_every_list)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []

    for dump_path in dump_paths:
        try:
            dump = load_dump(dump_path)
        except ValueError as exc:
            print(f"  SKIP {dump_path.name}: {exc}", file=sys.stderr)
            continue

        sk = dump.solver_kwargs
        omega_p_list = _parse_omega_list(args.omega_p_list, sk.get("omega_p"))
        omega_theta_list = _parse_omega_list(
            args.omega_theta_list, sk.get("omega_theta")
        )

        per_dump_rows = []
        per_dump_first_failure: Optional[dict] = None

        for scheme in schemes:
            for omega_p in omega_p_list:
                for omega_theta in omega_theta_list:
                    for max_inner in max_inner_list:
                        for check_every in check_every_list:
                            row, _ = replay_one(
                                ausas_unsteady_one_step_gpu, dump,
                                scheme=scheme,
                                omega_p=omega_p,
                                omega_theta=omega_theta,
                                max_inner=max_inner,
                                check_every=check_every,
                                trace_nan=args.trace_nan,
                                debug_check_every=check_every,
                            )
                            rows_out.append(row)
                            per_dump_rows.append(row)
                            print(
                                f"  [{dump_path.name}] {scheme} "
                                f"op={omega_p:.3f} ot={omega_theta:.3f} "
                                f"mi={max_inner} ce={check_every} "
                                f"-> conv={row['converged']} "
                                f"fk={row['failure_kind'] or '-'} "
                                f"ni={row['nan_iter'] or '-'}"
                            )
                            if (
                                per_dump_first_failure is None
                                and row.get("failure_kind") not in ("", None)
                            ):
                                per_dump_first_failure = row

        # --- two-pass refinement under --trace-nan ---
        if args.trace_nan and per_dump_first_failure is not None:
            best = per_dump_first_failure
            try:
                nan_iter_coarse = int(best["nan_iter"])
            except (TypeError, ValueError):
                nan_iter_coarse = None
            if nan_iter_coarse is not None and nan_iter_coarse > 0:
                ce_coarse = int(best["check_every"])
                k0 = max(0, nan_iter_coarse - ce_coarse)
                print(
                    f"  trace-nan refine [{dump_path.name}]: window "
                    f"[{k0}, {nan_iter_coarse}] @ check_every=1"
                )
                refined, _ = replay_one(
                    ausas_unsteady_one_step_gpu, dump,
                    scheme=str(best["scheme"]),
                    omega_p=float(best["omega_p"]),
                    omega_theta=float(best["omega_theta"]),
                    max_inner=int(best["max_inner"]),
                    check_every=int(best["check_every"]),
                    trace_nan=True,
                    debug_check_every=1,
                    debug_check_start_iter=k0,
                )
                refined["check_every"] = (
                    f"{best['check_every']}->1"
                )
                rows_out.append(refined)
                if refined.get("failure_kind") not in ("", None):
                    best = refined
            write_trace_report(
                args.out.parent, dump, per_dump_rows, best
            )

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"\nWrote {len(rows_out)} rows to {args.out}")


if __name__ == "__main__":
    main()
