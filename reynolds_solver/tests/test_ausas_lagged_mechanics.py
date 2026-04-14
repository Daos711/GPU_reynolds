"""
Regression + speedup test for Phase 5 Part 1 (lagged mechanics).

Two cases:

  (1) Physics parity: same dynamic-journal run under
        accel=None                                 (baseline, K=1)
        accel=AusasAccelerationOptions(K=3, ...)   (lagged)
      Compare per-step X(t), Y(t), WX(t), p_max(t). The lagged run
      does NOT have to reproduce baseline bit-for-bit (mechanics are
      held frozen for K-1 iters between refreshes), but it must stay
      within 1 % relative error — the guards (residual stall + cav
      jump) must kick in before physics drifts.

  (2) Speedup: wall-time of the K=3 run must be at least 1.5x the
      K=1 run on a reduced journal benchmark (the TZ's conservative
      lower bound; the target is 1.8x).

Run:
    python -m reynolds_solver.tests.test_ausas_lagged_mechanics

Skipped cleanly on CPU-only machines.
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


def _run(accel, NT=300, N1=100, N2=12, dt=2e-3):
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
        verbose=False,
    )
    return res, time.perf_counter() - t0


def test_lagged_vs_baseline():
    """K=3 run agrees with baseline within 1% on the key scalars."""
    print("\n=== Test 1: lagged (K=3) vs baseline (K=1) physics parity ===")
    if not _available():
        return True
    from reynolds_solver import AusasAccelerationOptions

    res_base, t_base = _run(accel=None)
    res_lag, t_lag = _run(
        accel=AusasAccelerationOptions(mech_update_every=3),
    )

    scale_X = max(float(np.max(np.abs(res_base.X))), 1e-30)
    err_X = float(np.max(np.abs(res_base.X - res_lag.X))) / scale_X
    err_Y = float(np.max(np.abs(res_base.Y - res_lag.Y))) / scale_X

    scale_W = max(float(np.max(np.abs(res_base.WX))), 1e-30)
    err_WX = float(np.max(np.abs(res_base.WX - res_lag.WX))) / scale_W
    err_WY = float(np.max(np.abs(res_base.WY - res_lag.WY))) / scale_W

    scale_p = max(float(np.max(np.abs(res_base.p_max))), 1e-30)
    err_p_max = float(
        np.max(np.abs(res_base.p_max - res_lag.p_max))
    ) / scale_p

    print(
        f"  trajectory rel-err: X={err_X:.2e}, Y={err_Y:.2e}"
    )
    print(
        f"  forces     rel-err: WX={err_WX:.2e}, WY={err_WY:.2e}"
    )
    print(f"  p_max      rel-err: {err_p_max:.2e}")
    # Inner-iter stats for diagnostics.
    print(
        f"  baseline inner iters: mean={res_base.n_inner.mean():.0f}, "
        f"max={int(res_base.n_inner.max())}"
    )
    print(
        f"  lagged   inner iters: mean={res_lag.n_inner.mean():.0f}, "
        f"max={int(res_lag.n_inner.max())}"
    )

    ok = (
        err_X < 1e-2
        and err_Y < 1e-2
        and err_WX < 1e-2
        and err_WY < 1e-2
        and err_p_max < 1e-2
    )
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] all relative errors < 1%")
    return ok


def test_lagged_speedup():
    """K=3 wall time should be >= 1.5x faster than K=1 baseline."""
    print("\n=== Test 2: lagged (K=3) speedup >= 1.5x ===")
    if not _available():
        return True
    from reynolds_solver import AusasAccelerationOptions

    # Slightly larger run so launch/sync overheads don't dominate the
    # measurement and the mechanics skip is visible.
    NT = 500
    res_base, t_base = _run(accel=None, NT=NT)
    res_lag, t_lag = _run(
        accel=AusasAccelerationOptions(mech_update_every=3),
        NT=NT,
    )

    speedup = t_base / max(t_lag, 1e-9)
    print(
        f"  baseline  (K=1): {t_base:.1f} s, "
        f"mean inner={res_base.n_inner.mean():.0f}"
    )
    print(
        f"  lagged    (K=3): {t_lag:.1f} s, "
        f"mean inner={res_lag.n_inner.mean():.0f}"
    )
    print(f"  speedup        : {speedup:.2f} x")

    ok = speedup >= 1.5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] speedup >= 1.5x (target 1.8x)")
    return ok


def test_baseline_unchanged():
    """`accel=None` and `accel=AusasAccelerationOptions()` match bit-exact."""
    print("\n=== Test 3: baseline invariance (accel=None vs default options) ===")
    if not _available():
        return True
    from reynolds_solver import AusasAccelerationOptions

    res_none, _ = _run(accel=None, NT=80)
    res_def, _ = _run(accel=AusasAccelerationOptions(), NT=80)
    err_X = float(np.max(np.abs(res_none.X - res_def.X)))
    err_WX = float(np.max(np.abs(res_none.WX - res_def.WX)))
    err_pmax = float(np.max(np.abs(res_none.p_max - res_def.p_max)))
    print(
        f"  X max|Δ| = {err_X:.2e}, WX max|Δ| = {err_WX:.2e}, "
        f"p_max max|Δ| = {err_pmax:.2e}"
    )
    # Tight: should be bit-exact since K=1 path is identical.
    ok = err_X < 1e-12 and err_WX < 1e-12 and err_pmax < 1e-12
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] accel=None and default-options match to ~1e-12")
    return ok


def main():
    ok = True
    ok = test_baseline_unchanged() and ok
    ok = test_lagged_vs_baseline() and ok
    ok = test_lagged_speedup() and ok
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
