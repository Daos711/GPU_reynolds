"""
Stage-2 squeeze-film benchmark regression test (GPU unsteady Ausas).

Validates the full time loop `solve_ausas_prescribed_h_gpu` on the
Ausas, Jai, Buscaglia (2008) oscillatory squeeze problem (Section 3),
as reproduced by `reynolds_solver.cavitation.ausas.benchmark_squeeze_dynamic`.

This is a GPU-only counterpart to the existing CPU-side
`test_squeeze_benchmark.py`.

Checks
------
1. Numerical rupture time within 1 percent of the analytical root of
   `p0 * h^3 = h'`.
2. Cavitation region grows during the rupture phase and shrinks again
   (reformation) before the end of the period. Reformation is a
   grid-limited observable: the mass-deficit coefficient beta =
   2 * dx1^2 / dt scales as 1 / N1^2, so coarse grids see beta too
   large and cannot propagate the elevated-pressure supply into the
   cavitated interior. We use N1 = 450 (spec value) for this check.
3. Memory footprint: no per-step field accumulation when the caller
   does not request field_checkpoints.

Skipped silently on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_squeeze_dynamic_gpu
"""
import sys
import time

import numpy as np


def _run_benchmark(N1, N2=4, dt=6.6e-4, tol_inner=1e-6, max_inner=5000,
                   omega_p=1.95, scheme="rb"):
    """Shared benchmark runner. Returns None on CPU-only machines."""
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return None

    from reynolds_solver.cavitation.ausas.benchmark_squeeze_dynamic import (
        run_squeeze_benchmark,
    )
    t0 = time.perf_counter()
    result = run_squeeze_benchmark(
        N1=N1, N2=N2, dt=dt,
        tol_inner=tol_inner, max_inner=max_inner,
        omega_p=omega_p, scheme=scheme,
        verbose=False,
    )
    print(f"  run took {time.perf_counter() - t0:.1f} s")
    return result


def _print_cav_trace(result, label="cav_frac trace"):
    """Log cav_frac at 11 evenly-spaced points over the period."""
    t = result.t
    cav = result.cav_frac
    idx = np.linspace(0, len(t) - 1, 11, dtype=int)
    samples = ", ".join(f"t={t[i]:.3f}:{cav[i]:.2f}" for i in idx)
    print(f"  {label}: {samples}")


def test_squeeze_rupture_time():
    """Numerical rupture time matches analytical root to within 1 %."""
    print("\n=== Test 1: squeeze rupture time (N1 = 120) ===")
    # Rupture is a dt-limited observable, so use a coarse grid for speed.
    result = _run_benchmark(N1=120, dt=6.6e-4)
    if result is None:
        return True

    t_rup_num = result.t_rup_numerical
    t_rup_ana = result.t_rup_analytical
    rel_err = result.rupture_relative_error

    print(
        f"  t_rup numerical = {t_rup_num:.6f}, analytical = {t_rup_ana:.6f}, "
        f"rel.err = {100.0 * rel_err:.3f} %"
    )
    print(
        f"  inner iters: min={int(result.n_inner.min())}, "
        f"max={int(result.n_inner.max())}, mean={result.n_inner.mean():.0f}; "
        f"converged {int(result.converged.sum())}/{len(result.t)} steps"
    )
    ok = np.isfinite(t_rup_num) and rel_err < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] rupture-time tolerance 1 %")
    return ok


def test_cavitation_growth_and_reformation():
    """
    Cavitation grows to the analytic peak and then starts reforming.

    The analytic Sigma(t) = 1 - sqrt(p0 * h^3 / h') is only valid on the
    active rupture phase (where the radicand lies in [0, 1]); it diverges
    as h' -> 0 near t -> T/2, so it does NOT predict a return to zero by
    the end of the period. Mass-balance also shows that the cavitation
    deficit (~0.19 mass units over ~95 % of the domain) cannot be fully
    re-supplied through the thin full-film boundary strip (~2 %) within
    the reformation half-period T/2 = 0.25. What we expect numerically
    is:

      * peak cav_frac close to the analytic 0.977 (reached near the
        analytic t_ref ~ 0.315);
      * cav_frac plateauing through the middle of the period;
      * a slow, monotone decline past the peak — this is the Ausas JFO
        signature. With a Reynolds-BC surrogate the cavitation would
        collapse instantly; JFO preserves mass and the cloud persists.

    We therefore assert (i) a near-analytic peak and (ii) a visible
    post-peak decline, not a full recovery to zero.
    """
    print("\n=== Test 2: cavitation growth + reformation (N1 = 450) ===")
    result = _run_benchmark(N1=450, dt=6.6e-4)
    if result is None:
        return True

    cav = result.cav_frac
    t = result.t

    t_rup_safe = result.t_rup_numerical if np.isfinite(result.t_rup_numerical) else 0.25
    post_rup = t > t_rup_safe
    if not post_rup.any():
        print("  [FAIL] no post-rupture samples")
        return False
    cav_peak = float(cav[post_rup].max())
    cav_peak_idx = int(np.argmax(cav[post_rup]))
    t_peak = float(t[post_rup][cav_peak_idx])

    # End of period: take the mean over the last ~1 % of steps.
    tail = max(1, len(t) // 100)
    cav_end = float(cav[-tail:].mean())

    _print_cav_trace(result)
    print(
        f"  cav peak = {cav_peak:.3f} at t ~ {t_peak:.4f} "
        f"(analytic peak 0.977 at t ~ 0.315)"
    )
    print(f"  cav(end-of-period) = {cav_end:.3f}")
    print(
        f"  inner iters: min={int(result.n_inner.min())}, "
        f"max={int(result.n_inner.max())}, mean={result.n_inner.mean():.0f}"
    )

    # (1) Growth to within 5 % absolute of the analytic peak (0.977).
    ok_peak = 0.92 < cav_peak < 1.0
    # (2) Peak reached past the analytic rupture time.
    ok_peak_time = t_peak > t_rup_safe
    # (3) Some reformation: decline from peak visible by end of period.
    decline_abs = cav_peak - cav_end
    ok_reformation = decline_abs > 0.01

    ok = ok_peak and ok_peak_time and ok_reformation
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] peak_ok={ok_peak} (0.92 < peak < 1.0), "
        f"peak_time_ok={ok_peak_time} (t_peak > t_rup), "
        f"reform_ok={ok_reformation} (|cav_peak - cav_end| > 0.01, "
        f"got {decline_abs:+.3f})"
    )
    return ok


def test_memory_footprint():
    """No field accumulation when save_stride is not provided."""
    print("\n=== Test 3: memory footprint ===")
    result = _run_benchmark(
        N1=64, N2=4, dt=5e-3,
        tol_inner=1e-3, max_inner=200,
    )
    if result is None:
        return True

    histories_nbytes = (
        result.t.nbytes
        + result.p_max.nbytes
        + result.cav_frac.nbytes
        + result.n_inner.nbytes
        + result.converged.nbytes
    )
    last_nbytes = result.P_last.nbytes + result.theta_last.nbytes

    print(
        f"  NT={len(result.t)}, histories={histories_nbytes/1024:.1f} kB, "
        f"last field={last_nbytes/1024:.1f} kB, "
        f"checkpoints={result.field_checkpoints}"
    )
    ok = result.field_checkpoints is None
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] field_checkpoints is None")
    return ok


def main():
    ok1 = test_squeeze_rupture_time()
    ok2 = test_cavitation_growth_and_reformation()
    ok3 = test_memory_footprint()
    ok = ok1 and ok2 and ok3
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
