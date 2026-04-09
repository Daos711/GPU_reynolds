"""
Pseudo-transient Δτ sweep for the Ausas CPU reference.

Runs the Step E `pt_trace` from diagnostic_ausas_seam.py for several
pseudo-time step sizes Δτ, both from HS-warm and cold starts, and prints
a compact warm-vs-cold summary table at the end. The goal is to pick a
single Δτ that:

  * keeps maxP physically bounded (no collapse to 0),
  * lets cav_frac plateau instead of drifting to 1,
  * converges warm and cold starts to the SAME state.

Run:
    python -m reynolds_solver.experiments.diagnostics.diag_pt_sweep
"""
import numpy as np

from reynolds_solver.experiments.diagnostics.diagnostic_ausas_seam import pt_trace


def run_sweep(
    dts=(0.1, 0.01, 0.001),
    epsilon=0.6,
    max_time_steps=500,
    max_inner=200,
    inner_tol=1e-5,
):
    results = []
    for dt in dts:
        print(f"\n{'#' * 60}\n#### Δτ = {dt} ####\n{'#' * 60}")

        P_w, th_w = pt_trace(
            start="hs_warm",
            dt_pseudo=dt,
            max_time_steps=max_time_steps,
            max_inner=max_inner,
            inner_tol=inner_tol,
            epsilon=epsilon,
        )
        P_c, th_c = pt_trace(
            start="cold",
            dt_pseudo=dt,
            max_time_steps=max_time_steps,
            max_inner=max_inner,
            inner_tol=inner_tol,
            epsilon=epsilon,
        )

        maxP_w = float(P_w.max())
        maxP_c = float(P_c.max())
        cav_w = float(np.mean(th_w < 1.0 - 1e-6))
        cav_c = float(np.mean(th_c < 1.0 - 1e-6))
        diff_P = float(np.max(np.abs(P_w - P_c)))
        diff_th = float(np.max(np.abs(th_w - th_c)))

        print(
            f"\n>>> dt={dt}: "
            f"warm maxP={maxP_w:.4e} cav={cav_w:.3f} | "
            f"cold maxP={maxP_c:.4e} cav={cav_c:.3f}"
        )
        print(
            f"    diff: |P|={diff_P:.3e}  |θ|={diff_th:.3e}"
        )

        results.append(
            dict(
                dt=dt,
                maxP_w=maxP_w, maxP_c=maxP_c,
                cav_w=cav_w, cav_c=cav_c,
                diff_P=diff_P, diff_th=diff_th,
            )
        )

    # Compact summary table
    print()
    print("=" * 80)
    print(f"Δτ sweep summary  (ε={epsilon}, "
          f"{max_time_steps} steps × ≤{max_inner} inner)")
    print("=" * 80)
    print(
        f"  {'Δτ':>8s}  "
        f"{'maxP_warm':>11s}  {'maxP_cold':>11s}  "
        f"{'cav_warm':>9s}  {'cav_cold':>9s}  "
        f"{'|ΔP|':>10s}  {'|Δθ|':>10s}"
    )
    for r in results:
        print(
            f"  {r['dt']:>8.4f}  "
            f"{r['maxP_w']:>11.4e}  {r['maxP_c']:>11.4e}  "
            f"{r['cav_w']:>9.3f}  {r['cav_c']:>9.3f}  "
            f"{r['diff_P']:>10.3e}  {r['diff_th']:>10.3e}"
        )
    print("=" * 80)
    print(
        "  Expected: smaller Δτ → stronger anchor β=2·d_phi²/Δτ,\n"
        "            maxP stays near the HS value, warm/cold agree,\n"
        "            cav_frac stops drifting to 1."
    )
    return results


if __name__ == "__main__":
    run_sweep()
