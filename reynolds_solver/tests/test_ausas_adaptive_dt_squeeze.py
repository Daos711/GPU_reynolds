"""
Adaptive-dt regression for the Ausas 2008 squeeze benchmark.

Check that the adaptive-dt path in `solve_ausas_prescribed_h_gpu`:

  1. Recovers rupture time to within 1 % of the analytical root.
  2. Uses FEWER accepted steps than the fixed-dt reference when the
     gap-evolution smoothness permits dt growth.

Skipped on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_ausas_adaptive_dt_squeeze
"""
import sys
import time

import numpy as np


def _available():
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return False
    return True


def _run(accel=None, N1=120, dt=6.6e-4, NT=None, tol_inner=1e-5):
    from reynolds_solver.cavitation.ausas.benchmark_squeeze_dynamic import (
        run_squeeze_benchmark,
    )
    t0 = time.perf_counter()
    res = run_squeeze_benchmark(
        N1=N1, N2=4, dt=dt, NT=NT,
        tol_inner=tol_inner, max_inner=5000,
        omega_p=1.95,
        accel=accel,
    )
    wall = time.perf_counter() - t0
    return res, wall


def test_adaptive_squeeze_rupture():
    """Rupture time within 1 % and fewer accepted steps than fixed-dt."""
    print("\n=== Test: adaptive-dt squeeze rupture ===")
    if not _available():
        return True
    from reynolds_solver import AusasAccelerationOptions

    # Fixed baseline.
    res_fix, wall_fix = _run(accel=None, N1=120, dt=6.6e-4)
    # Adaptive: dt can grow up to 4x during smooth phases, shrink near
    # rupture / reformation fronts.
    accel = AusasAccelerationOptions(
        adaptive_dt=True,
        dt_min=1.0e-4,
        dt_max=3.0e-3,
        dt_grow=1.25,
        dt_shrink=0.5,
        target_inner_low=50,
        target_inner_high=200,
        reject_if_not_converged=False,   # squeeze rarely diverges,
                                         # keep accepts generous.
    )
    res_adapt, wall_adapt = _run(accel=accel, N1=120, dt=6.6e-4)

    print(
        f"  fixed    : NT={len(res_fix.t):>4d} wall={wall_fix:.1f} s  "
        f"t_rup={res_fix.t_rup_numerical:.6f}  "
        f"rel_err={100.0*res_fix.rupture_relative_error:.3f}%"
    )
    print(
        f"  adaptive : NT={len(res_adapt.t):>4d} wall={wall_adapt:.1f} s  "
        f"t_rup={res_adapt.t_rup_numerical:.6f}  "
        f"rel_err={100.0*res_adapt.rupture_relative_error:.3f}%"
    )

    rupture_ok = (
        np.isfinite(res_adapt.t_rup_numerical)
        and res_adapt.rupture_relative_error < 0.01
    )
    fewer_steps_ok = len(res_adapt.t) < len(res_fix.t)

    ok = rupture_ok and fewer_steps_ok
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] rupture <1% ({rupture_ok}), "
        f"adaptive_steps < fixed_steps ({fewer_steps_ok})"
    )
    return ok


def main():
    ok = test_adaptive_squeeze_rupture()
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
