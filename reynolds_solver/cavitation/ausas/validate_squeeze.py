"""
Squeeze-benchmark validation (Phase 4 Step 2.1).

Produces the Ausas-2008-Fig.-2(b) analogue: numerical cavitation
fraction Sigma_num(t) overlaid on the analytic

    Sigma_exact(t) = 1 - sqrt(p0 * h^3 / h')

over the active rupture phase. Also computes summary metrics
(t_rup, t_ref, Sigma_peak, RMS error on [t_rup, t_ref]) and writes
a PNG + JSON into `results/`.

Usage:
    python -m reynolds_solver.cavitation.ausas.validate_squeeze \
        [--N1 450 --dt 6.6e-4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _ensure_results_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(exist_ok=True, parents=True)
    return p


def _sigma_from_theta(theta_snapshot: np.ndarray) -> float:
    """
    Numerical cavitation fraction from a (N_Z, N_phi) theta field. The
    squeeze problem is uniform in x2 so we average over Z rows and then
    count the fraction of x1 interior nodes with theta < 1.
    """
    th_interior = theta_snapshot[1:-1, 1:-1]
    th_1d = th_interior.mean(axis=0)
    return float(np.mean(th_1d < 1.0 - 1e-6))


def _sigma_exact_array(t_arr: np.ndarray, p0: float = 0.025) -> np.ndarray:
    """
    Vectorised analytic Sigma(t). Returns NaN outside the validity range
    (h' <= 0 or radicand leaving [0, 1]).
    """
    t_arr = np.asarray(t_arr, dtype=np.float64)
    h = 0.125 * np.cos(4.0 * np.pi * t_arr) + 0.375
    hp = -0.125 * 4.0 * np.pi * np.sin(4.0 * np.pi * t_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        rad = p0 * h ** 3 / hp
    out = 1.0 - np.sqrt(rad)
    mask = (hp <= 0.0) | (rad < 0.0) | (rad > 1.0)
    out[mask] = np.nan
    return out


def _find_t_rup_numerical(t: np.ndarray, sigma_num: np.ndarray) -> float:
    """First time index where sigma_num > 0."""
    mask = sigma_num > 0.0
    if not mask.any():
        return float("nan")
    return float(t[int(np.argmax(mask))])


def _find_t_ref_numerical(t: np.ndarray, sigma_num: np.ndarray,
                          t_rup: float) -> tuple[float, float]:
    """
    Reformation start: first local maximum of sigma_num past t_rup.
    Returns (t_ref, sigma_peak).
    """
    if not np.isfinite(t_rup):
        return float("nan"), float("nan")
    idx0 = int(np.searchsorted(t, t_rup))
    if idx0 >= len(sigma_num) - 2:
        return float("nan"), float("nan")
    sub = sigma_num[idx0:]
    peak = int(np.argmax(sub))
    if peak == len(sub) - 1:
        return float("nan"), float(sub[peak])
    return float(t[idx0 + peak]), float(sub[peak])


def run_validation(
    N1: int = 450,
    N2: int = 4,
    dt: float = 6.6e-4,
    NT: int | None = None,
    p0: float = 0.025,
    omega_p: float = 1.95,
    tol_inner: float = 1e-6,
    max_inner: int = 5000,
    out_dir: str = "results",
    make_plot: bool = True,
) -> dict:
    """
    Run the squeeze benchmark, collect Sigma(t) via a field_callback,
    compute metrics, and save a PNG + JSON.
    """
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        return {"skipped": f"cupy not available: {exc}"}

    from reynolds_solver.cavitation.ausas.benchmark_squeeze_dynamic import (
        run_squeeze_benchmark, analytic_rupture_time,
    )

    # Collect Sigma_num(t) on the fly via a callback — avoids holding
    # NT full theta snapshots in memory.
    sigma_trace: list[float] = []
    t_trace: list[float] = []

    def _cb(n, t_n, P_np, theta_np):
        sigma_trace.append(_sigma_from_theta(theta_np))
        t_trace.append(float(t_n))

    if NT is None:
        NT = int(round(0.5 / dt))

    result = run_squeeze_benchmark(
        N1=N1, N2=N2, dt=dt, NT=NT, p0=p0,
        omega_p=omega_p, tol_inner=tol_inner, max_inner=max_inner,
        field_callback=_cb,
        verbose=False,
    )

    t = np.asarray(t_trace, dtype=np.float64)
    sigma_num = np.asarray(sigma_trace, dtype=np.float64)
    sigma_an = _sigma_exact_array(t, p0=p0)

    # --- Metrics ---
    t_rup_ana = float(analytic_rupture_time(p0=p0))
    t_rup_num = _find_t_rup_numerical(t, sigma_num)
    t_ref_num, sigma_peak_num = _find_t_ref_numerical(t, sigma_num, t_rup_num)
    rel_err_t_rup = (
        abs(t_rup_num - t_rup_ana) / t_rup_ana
        if np.isfinite(t_rup_num) else float("nan")
    )

    # RMS error of numerical vs analytic Sigma on the validity range.
    valid = np.isfinite(sigma_an) & (t >= t_rup_num) & (
        t <= (t_ref_num if np.isfinite(t_ref_num) else t[-1])
    )
    if valid.any():
        err = sigma_num[valid] - sigma_an[valid]
        rms_sigma = float(np.sqrt(np.mean(err ** 2)))
        max_sigma = float(np.max(np.abs(err)))
    else:
        rms_sigma = float("nan")
        max_sigma = float("nan")

    metrics = {
        "N1": N1, "N2": N2, "dt": dt, "NT": NT,
        "p0": p0, "omega_p": omega_p,
        "t_rup_analytical": t_rup_ana,
        "t_rup_numerical": t_rup_num,
        "t_rup_rel_err": rel_err_t_rup,
        "t_ref_numerical": t_ref_num,
        "sigma_peak_numerical": sigma_peak_num,
        "sigma_rms_error_on_rupture_phase": rms_sigma,
        "sigma_max_error_on_rupture_phase": max_sigma,
        "inner_iters_mean": float(result.n_inner.mean()),
        "inner_iters_max": int(result.n_inner.max()),
        "converged_steps": int(result.converged.sum()),
        "total_steps": int(len(result.t)),
    }

    out_path = _ensure_results_dir(out_dir)

    # --- Plot ---
    plot_path = None
    if make_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8.0, 5.0))
            ax.plot(t, sigma_num, drawstyle="steps-post",
                    label=r"$\Sigma_{num}(t)$ (GPU)", color="tab:blue")
            # Analytic: plot only where valid
            t_ana_plot = t.copy()
            s_ana_plot = sigma_an.copy()
            ax.plot(t_ana_plot, s_ana_plot,
                    label=r"$\Sigma_{exact}(t) = 1 - \sqrt{p_0 h^3 / h'}$",
                    color="tab:red", linewidth=1.5)
            if np.isfinite(t_rup_num):
                ax.axvline(t_rup_num, color="k", lw=0.5, ls="--",
                           label=f"t_rup_num = {t_rup_num:.4f}")
            if np.isfinite(t_rup_ana):
                ax.axvline(t_rup_ana, color="grey", lw=0.5, ls=":",
                           label=f"t_rup_analytical = {t_rup_ana:.4f}")
            ax.set_xlabel("time")
            ax.set_ylabel(r"cavitation fraction $\Sigma$")
            ax.set_title(
                f"Squeeze benchmark (Ausas 2008 §3), N1={N1}, dt={dt:.2e}, "
                f"rel-err t_rup = {100 * rel_err_t_rup:.2f}%"
            )
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            plot_path = out_path / f"squeeze_sigma_N1{N1}_dt{dt:.0e}.png"
            fig.savefig(plot_path, dpi=130)
            plt.close(fig)
        except Exception as exc:
            metrics["plot_error"] = str(exc)

    # --- JSON summary ---
    json_path = out_path / f"squeeze_metrics_N1{N1}_dt{dt:.0e}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    metrics["plot_path"] = str(plot_path) if plot_path is not None else None
    metrics["json_path"] = str(json_path)
    return metrics


def main(argv=None):
    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser(
        description="Squeeze benchmark validation (Phase 4 Step 2.1)."
    )
    p.add_argument("--N1", type=int, default=450)
    p.add_argument("--N2", type=int, default=4)
    p.add_argument("--dt", type=float, default=6.6e-4)
    p.add_argument("--NT", type=int, default=None)
    p.add_argument("--omega-p", type=float, default=1.95)
    p.add_argument("--tol-inner", type=float, default=1e-6)
    p.add_argument("--max-inner", type=int, default=5000)
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    m = run_validation(
        N1=args.N1, N2=args.N2, dt=args.dt, NT=args.NT,
        omega_p=args.omega_p,
        tol_inner=args.tol_inner, max_inner=args.max_inner,
        out_dir=args.out_dir,
        make_plot=not args.no_plot,
    )

    print("=== Squeeze validation ===")
    if "skipped" in m:
        print(f"  [SKIP] {m['skipped']}")
        return 0
    print(f"  N1 = {m['N1']}, dt = {m['dt']:.2e}, NT = {m['NT']}")
    print(f"  t_rup analytical    = {m['t_rup_analytical']:.6f}")
    print(f"  t_rup numerical     = {m['t_rup_numerical']:.6f}")
    print(f"  t_rup relative err  = {100.0 * m['t_rup_rel_err']:.3f} %")
    print(f"  t_ref numerical     = {m['t_ref_numerical']}")
    print(f"  Sigma peak          = {m['sigma_peak_numerical']}")
    print(f"  Sigma RMS-err       = {m['sigma_rms_error_on_rupture_phase']}")
    print(f"  inner iters mean    = {m['inner_iters_mean']:.1f}")
    print(f"  results saved       : {m['plot_path']}")
    print(f"  metrics JSON        : {m['json_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
