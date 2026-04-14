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
   (reformation) before the end of the period.
3. Memory footprint: no per-step field accumulation when the caller
   does not request field_checkpoints.

Skipped silently on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_squeeze_dynamic_gpu
"""
import sys

import numpy as np


def _run_benchmark(N1=450, N2=4, dt=6.6e-4, tol_inner=1e-5, max_inner=5000,
                   omega_p=1.8, scheme="rb"):
    """Shared benchmark runner. Returns None on CPU-only machines."""
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return None

    from reynolds_solver.cavitation.ausas.benchmark_squeeze_dynamic import (
        run_squeeze_benchmark,
    )
    return run_squeeze_benchmark(
        N1=N1, N2=N2, dt=dt,
        tol_inner=tol_inner, max_inner=max_inner,
        omega_p=omega_p, scheme=scheme,
        verbose=False,
    )


def test_squeeze_rupture_time():
    """
    Numerical rupture time matches analytical root to within 1 %.

    Uses a coarser grid (N1 = 120) so the pure-Jacobi inner solve fits
    comfortably inside `max_inner` on a modest GPU. The rupture time is
    a temporal (dt-limited) observable, not a grid-limited one, so the
    coarser grid does not change the expected accuracy.
    """
    print("\n=== Test 1: squeeze rupture time ===")
    result = _run_benchmark(N1=120, dt=6.6e-4, tol_inner=1e-5, max_inner=5000)
    if result is None:
        return True

    t_rup_num = result.t_rup_numerical
    t_rup_ana = result.t_rup_analytical
    rel_err = result.rupture_relative_error

    print(
        f"  t_rup numerical = {t_rup_num:.6f}, analytical = {t_rup_ana:.6f}, "
        f"rel.err = {100.0 * rel_err:.3f} %"
    )
    ok = np.isfinite(t_rup_num) and rel_err < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] rupture-time tolerance 1 %")
    return ok


def test_cavitation_growth_and_reformation():
    """Cavitation region grows past rupture then shrinks back towards T."""
    print("\n=== Test 2: cavitation growth + reformation ===")
    result = _run_benchmark(N1=120, dt=6.6e-4, tol_inner=1e-5, max_inner=5000)
    if result is None:
        return True

    cav = result.cav_frac
    t = result.t

    # Rupture phase: after t_rup, cav_frac should climb.
    t_rup_safe = result.t_rup_numerical if np.isfinite(result.t_rup_numerical) else 0.25
    post_rup = (t > t_rup_safe) & (t < 0.45)
    if not post_rup.any():
        print("  [FAIL] no post-rupture samples")
        return False
    cav_peak = float(cav[post_rup].max())
    cav_peak_idx = int(np.argmax(cav[post_rup]))
    t_peak = float(t[post_rup][cav_peak_idx])

    # Reformation phase: cav_frac near end of period should be well below
    # the peak.
    end_window = t > 0.48
    cav_end = float(cav[end_window].mean()) if end_window.any() else float(cav[-1])

    print(
        f"  cav peak = {cav_peak:.3f} at t ~ {t_peak:.4f}, "
        f"cav(end-of-period) = {cav_end:.3f}"
    )
    ok_grew = cav_peak > 0.05           # at least a few percent of nodes cavitated
    ok_reformed = cav_end < 0.5 * cav_peak
    ok = ok_grew and ok_reformed
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] grew={ok_grew} (peak>5 %), "
        f"reformed={ok_reformed} (end < 0.5 * peak)"
    )
    return ok


def test_memory_footprint():
    """No field accumulation when save_stride is not provided."""
    print("\n=== Test 3: memory footprint ===")
    # Use a much coarser benchmark for this sanity check so it finishes
    # quickly even on modest GPUs.
    result = _run_benchmark(N1=64, N2=4, dt=5e-3, tol_inner=1e-3, max_inner=200)
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
