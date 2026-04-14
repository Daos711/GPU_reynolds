"""
Journal-bearing validation (Phase 4 Step 2.2).

Produces the Ausas-2008-Figs-4..7 analogues on the dynamic journal
solver: orbit X(t)/Y(t), applied vs hydrodynamic loads, p_max(t) and
e(t) on a user-configurable grid-convergence ladder.

Defaults run three grids (100x10, 200x20, 400x40); 800x80 is available
via --include-fine. A single grid is enough to produce the Fig.-4/5
plots, so the CLI accepts a `--single-grid` shortcut for a quick run.

Figures + a metrics JSON are written into `results/`.

Usage:
    # Quick single-grid run (fig 4/5 only, ~2-3 min):
    python -m reynolds_solver.cavitation.ausas.validate_journal --single-grid

    # Full grid convergence (can take 30 min on a modest GPU):
    python -m reynolds_solver.cavitation.ausas.validate_journal

    # Include the fine 800x80 grid as well (slow):
    python -m reynolds_solver.cavitation.ausas.validate_journal --include-fine
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_GRIDS = [(100, 10), (200, 20), (400, 40)]
OPTIONAL_FINE_GRID = (800, 80)


def _ensure_results_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(exist_ok=True, parents=True)
    return p


def _run_single_grid(N1: int, N2: int, NT: int, dt: float,
                     omega_p: float, tol_inner: float, max_inner: int,
                     verbose: bool = False):
    """Return a JournalBenchmarkResult for one grid."""
    from reynolds_solver.cavitation.ausas.benchmark_dynamic_journal import (
        run_journal_benchmark,
    )
    t0 = time.perf_counter()
    res = run_journal_benchmark(
        N1=N1, N2=N2, dt=dt, NT=NT,
        omega_p=omega_p, tol_inner=tol_inner, max_inner=max_inner,
        verbose=verbose,
    )
    dt_run = time.perf_counter() - t0
    return res, dt_run


def _metrics_for_result(res) -> dict:
    """Periodicity + peak statistics for a journal run."""
    t = res.t
    e = res.eccentricity
    last = t > (t[-1] - 1.0)
    prev = (t > (t[-1] - 2.0)) & (t <= (t[-1] - 1.0))

    e_max = float(e.max())
    t_e_max = float(t[int(np.argmax(e))])
    p_max_max = float(res.p_max.max())
    h_min_min = float(res.h_min.min())

    # Periodicity: RMS of (X, Y)_last - (X, Y)_prev.
    if prev.any() and last.any():
        n = min(int(prev.sum()), int(last.sum()))
        Xl, Yl = res.X[last][:n], res.Y[last][:n]
        Xp, Yp = res.X[prev][:n], res.Y[prev][:n]
        rms_orbit = float(
            np.sqrt(np.mean((Xl - Xp) ** 2 + (Yl - Yp) ** 2))
        )
        scale = float(np.sqrt(np.mean(Xl ** 2 + Yl ** 2))) + 1e-30
        period_rel = rms_orbit / scale
    else:
        rms_orbit = float("nan")
        period_rel = float("nan")

    # Load-balance residual on the last period.
    WX_last = float(res.WX[last].mean()) if last.any() else float("nan")
    WaX_last = float(res.WaX[last].mean()) if last.any() else float("nan")
    WaX_amp = float(np.max(np.abs(res.WaX[last]))) if last.any() else 1e-30

    return dict(
        N1=res.N1, N2=res.N2, dt=res.dt, NT=int(len(res.t)),
        e_max=e_max, t_e_max=t_e_max,
        p_max_max=p_max_max, h_min_min=h_min_min,
        period_rms_last_vs_prev=rms_orbit,
        period_rel_residual=period_rel,
        WX_last_mean=WX_last, WaX_last_mean=WaX_last,
        load_balance_residual_X=abs(WX_last + WaX_last) / (WaX_amp + 1e-30),
        inner_iters_mean=float(res.n_inner.mean()),
        inner_iters_max=int(res.n_inner.max()),
    )


def _plot_fig4_trajectory_and_loads(res, out_path: Path, tag: str) -> Path:
    """Fig. 4 analogue: X(t), Y(t), WaX vs -WX, WaY vs -WY."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 8.5), sharex=True)
    t = res.t

    axes[0].plot(t, res.X, label="X(t)", color="tab:blue")
    axes[0].plot(t, res.Y, label="Y(t)", color="tab:orange")
    axes[0].plot(t, res.eccentricity, label="e(t)", color="tab:green",
                 linestyle="--", alpha=0.8)
    axes[0].set_ylabel("shaft position")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, res.WaX, label=r"$W_{aX}$", color="tab:red")
    axes[1].plot(t, -res.WX, label=r"$-W_X$", color="tab:blue",
                 linestyle="--")
    axes[1].set_ylabel("x-load")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, res.WaY, label=r"$W_{aY}$", color="tab:red")
    axes[2].plot(t, -res.WY, label=r"$-W_Y$", color="tab:blue",
                 linestyle="--")
    axes[2].set_ylabel("y-load")
    axes[2].set_xlabel("time")
    axes[2].legend(loc="best", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Journal benchmark (Ausas 2008 §5) — {tag}", fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_path / f"journal_fig4_{tag}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_fig5_load_detail(res, out_path: Path, tag: str) -> Path:
    """Fig. 5 analogue: zoom on the two main load impulses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = res.t
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0))

    # Impulse 1 centre ~ 0.25 + 2 = 2.25 (third period).
    for ax, centre in zip(axes, (2.25, 2.50)):
        mask = (t > centre - 0.08) & (t < centre + 0.08)
        if mask.any():
            ax.plot(t[mask], res.WaX[mask], label=r"$W_{aX}$", color="tab:red")
            ax.plot(t[mask], -res.WX[mask], label=r"$-W_X$", color="tab:blue",
                    linestyle="--")
            ax.plot(t[mask], res.WaY[mask], label=r"$W_{aY}$", color="tab:olive")
            ax.plot(t[mask], -res.WY[mask], label=r"$-W_Y$", color="tab:purple",
                    linestyle="--")
        ax.set_title(f"zoom around t = {centre:.2f}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time")
    fig.suptitle(
        f"Journal benchmark — load detail — {tag}", fontsize=11
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_path / f"journal_fig5_{tag}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_fig6_pmax_grid_convergence(results, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for res in results:
        tag = f"{res.N1}x{res.N2}"
        last_period = res.t > res.t[-1] - 1.0
        ax.plot(res.t[last_period], res.p_max[last_period], label=tag)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$p_{max}(t)$ over last period")
    ax.set_title("Fig. 6 analogue — grid convergence of peak pressure")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_path / "journal_fig6_pmax_grid.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _plot_fig7_e_grid_convergence(results, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for res in results:
        tag = f"{res.N1}x{res.N2}"
        last_period = res.t > res.t[-1] - 1.0
        ax.plot(res.t[last_period], res.eccentricity[last_period], label=tag)
    ax.set_xlabel("time")
    ax.set_ylabel(r"eccentricity $e(t) = \sqrt{X^2 + Y^2}$")
    ax.set_title("Fig. 7 analogue — grid convergence of eccentricity")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_path / "journal_fig7_ecc_grid.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run_validation(
    grids: list | None = None,
    NT: int = 3000,
    dt: float = 1e-3,
    omega_p: float = 1.0,
    tol_inner: float = 1e-6,
    max_inner: int = 5000,
    include_fine: bool = False,
    single_grid: bool = False,
    out_dir: str = "results",
    verbose: bool = False,
) -> dict:
    """
    Run the journal validation.

    grids: iterable of (N1, N2). None => DEFAULT_GRIDS. With include_fine
    appends OPTIONAL_FINE_GRID. With single_grid => only (200, 20).
    """
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        return {"skipped": f"cupy not available: {exc}"}

    if single_grid:
        grids = [(200, 20)]
    elif grids is None:
        grids = list(DEFAULT_GRIDS)
        if include_fine:
            grids.append(OPTIONAL_FINE_GRID)

    out_path = _ensure_results_dir(out_dir)

    results = []
    timings = []
    for (N1, N2) in grids:
        if verbose:
            print(f"--- running journal {N1}x{N2} (NT={NT}) ...")
        res, dt_run = _run_single_grid(
            N1=N1, N2=N2, NT=NT, dt=dt,
            omega_p=omega_p, tol_inner=tol_inner, max_inner=max_inner,
            verbose=verbose,
        )
        if verbose:
            print(f"    finished in {dt_run:.1f} s")
        results.append(res)
        timings.append(dt_run)

    # --- Plots ---
    plot_paths = {}
    for res in results:
        tag = f"{res.N1}x{res.N2}"
        plot_paths[f"fig4_{tag}"] = str(_plot_fig4_trajectory_and_loads(
            res, out_path, tag))
        plot_paths[f"fig5_{tag}"] = str(_plot_fig5_load_detail(
            res, out_path, tag))

    if len(results) > 1:
        plot_paths["fig6"] = str(_plot_fig6_pmax_grid_convergence(
            results, out_path))
        plot_paths["fig7"] = str(_plot_fig7_e_grid_convergence(
            results, out_path))

    # --- Metrics ---
    metrics_per_grid = [_metrics_for_result(r) for r in results]
    for m, dtr in zip(metrics_per_grid, timings):
        m["wall_s"] = dtr

    # Grid convergence: relative difference of e_max against the
    # finest grid.
    if len(metrics_per_grid) > 1:
        ref = metrics_per_grid[-1]["e_max"]
        for m in metrics_per_grid:
            m["e_max_rel_diff_to_finest"] = abs(m["e_max"] - ref) / (ref + 1e-30)

    summary = {
        "grids": [(m["N1"], m["N2"]) for m in metrics_per_grid],
        "metrics": metrics_per_grid,
        "plot_paths": plot_paths,
    }
    json_path = out_path / "journal_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["json_path"] = str(json_path)
    return summary


def main(argv=None):
    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser(
        description="Journal benchmark validation (Phase 4 Step 2.2)."
    )
    p.add_argument("--NT", type=int, default=3000)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--omega-p", type=float, default=1.0)
    p.add_argument("--tol-inner", type=float, default=1e-6)
    p.add_argument("--max-inner", type=int, default=5000)
    p.add_argument("--include-fine", action="store_true",
                   help="Include the 800x80 grid (slow)")
    p.add_argument("--single-grid", action="store_true",
                   help="Run only one (200x20) grid — no grid convergence")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    summary = run_validation(
        NT=args.NT, dt=args.dt,
        omega_p=args.omega_p,
        tol_inner=args.tol_inner, max_inner=args.max_inner,
        include_fine=args.include_fine,
        single_grid=args.single_grid,
        out_dir=args.out_dir,
        verbose=args.verbose,
    )

    print("=== Journal validation ===")
    if "skipped" in summary:
        print(f"  [SKIP] {summary['skipped']}")
        return 0

    for m in summary["metrics"]:
        print(
            f"  grid {m['N1']}x{m['N2']}: e_max={m['e_max']:.3f} "
            f"at t={m['t_e_max']:.3f}, p_max={m['p_max_max']:.3e}, "
            f"period_rel={m['period_rel_residual']:.2e}, "
            f"load_bal_X={m['load_balance_residual_X']:.2e}, "
            f"wall={m['wall_s']:.1f} s"
        )
    print(f"  plots in  : {sorted(summary['plot_paths'].values())[0]}  ...")
    print(f"  JSON      : {summary['json_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
