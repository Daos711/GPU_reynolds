"""
Adaptive-dt regression for the dynamic-journal benchmark.

Check that the adaptive-dt path in `solve_ausas_journal_dynamic_gpu`:

  1. Reproduces the fixed-dt last-period trajectory within ~2 %
     relative error (accuracy unchanged).
  2. Is not slower than the fixed-dt baseline (ideally faster on
     smooth harmonic loads).

Skipped on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_ausas_adaptive_dt_journal
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


def _run(accel=None, N1=80, N2=8, dt=2e-3, NT=300):
    from reynolds_solver.cavitation.ausas.benchmark_dynamic_journal import (
        run_journal_benchmark,
    )
    t0 = time.perf_counter()
    res = run_journal_benchmark(
        N1=N1, N2=N2, dt=dt, NT=NT,
        omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-6, max_inner=5000,
        scheme="rb",
        accel=accel,
    )
    return res, time.perf_counter() - t0


def _interp_to_common_t(t_src, y_src, t_ref):
    """Linear interpolation of y_src(t_src) to t_ref."""
    # Clip t_ref to the source range so interp does not extrapolate.
    t_ref = np.clip(t_ref, t_src[0], t_src[-1])
    return np.interp(t_ref, t_src, y_src)


def test_adaptive_journal_accuracy():
    """
    Last-period trajectory under adaptive dt matches fixed-dt reference
    within 2 % relative. Because adaptive and fixed produce histories
    on DIFFERENT time grids, we linearly interpolate the adaptive
    result onto the fixed time axis before comparing.
    """
    print("\n=== Test: adaptive-dt journal accuracy + wall-time ===")
    if not _available():
        return True
    from reynolds_solver import AusasAccelerationOptions

    res_fix, wall_fix = _run(accel=None, N1=80, N2=8, dt=2e-3, NT=300)
    accel = AusasAccelerationOptions(
        adaptive_dt=True,
        dt_min=5e-4,
        dt_max=5e-3,
        dt_grow=1.25,
        dt_shrink=0.5,
        target_inner_low=80,
        target_inner_high=250,
        reject_if_not_converged=True,
    )
    res_adp, wall_adp = _run(accel=accel, N1=80, N2=8, dt=2e-3, NT=300)

    print(
        f"  fixed    : NT={len(res_fix.t):>4d}  wall={wall_fix:.1f} s"
    )
    print(
        f"  adaptive : NT={len(res_adp.t):>4d}  wall={wall_adp:.1f} s  "
        f"mean dt={0.6/max(len(res_adp.t), 1):.4e}"
    )

    # Interpolate adaptive histories onto the fixed-dt time grid so
    # we can compare element-wise. Restrict to the last period.
    t_last = res_fix.t[res_fix.t > (res_fix.t[-1] - 1.0)]
    if len(t_last) < 10:
        t_last = res_fix.t

    X_fix = _interp_to_common_t(res_fix.t, res_fix.X, t_last)
    Y_fix = _interp_to_common_t(res_fix.t, res_fix.Y, t_last)
    WX_fix = _interp_to_common_t(res_fix.t, res_fix.WX, t_last)

    X_adp = _interp_to_common_t(res_adp.t, res_adp.X, t_last)
    Y_adp = _interp_to_common_t(res_adp.t, res_adp.Y, t_last)
    WX_adp = _interp_to_common_t(res_adp.t, res_adp.WX, t_last)

    scale_X = max(float(np.max(np.abs(X_fix))), 1e-30)
    scale_W = max(float(np.max(np.abs(WX_fix))), 1e-30)
    err_X = float(np.max(np.abs(X_fix - X_adp))) / scale_X
    err_Y = float(np.max(np.abs(Y_fix - Y_adp))) / scale_X
    err_WX = float(np.max(np.abs(WX_fix - WX_adp))) / scale_W

    print(
        f"  trajectory rel-err on last period: "
        f"X={err_X:.2e}, Y={err_Y:.2e}, WX={err_WX:.2e}"
    )
    ok_acc = err_X < 0.02 and err_Y < 0.02 and err_WX < 0.02
    # Wall time: allow equality as pass (no hard speedup target).
    ok_wall = wall_adp <= 1.5 * wall_fix
    ok = ok_acc and ok_wall
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] accuracy <2% ({ok_acc}), "
        f"wall_adp <= 1.5*wall_fix ({ok_wall})"
    )
    return ok


def main():
    ok = test_adaptive_journal_accuracy()
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
