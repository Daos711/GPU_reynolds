"""
Regression for Phase 5 Part 3 — dynamic check cadence.

Two checks:

  1. Baseline invariance: with `dynamic_check_every=False` (or no
     accel at all), the solver produces bit-identical histories to
     the pre-Phase-5.3 baseline. We verify by running the same
     problem twice: once with `accel=None` and once with
     `accel=AusasAccelerationOptions()` (defaults — dynamic_check_every
     is False by default). Trajectories must agree to machine zero.

  2. Physics parity with the flag ON: the dynamic-cadence variant
     must produce the same physical trajectory within 1 % relative
     error (only residual-check schedule changes; accept decisions
     and sweep updates are identical).

Skipped on CPU-only machines.
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


def _run(accel, N1=80, N2=8, dt=2e-3, NT=200):
    from reynolds_solver.cavitation.ausas.benchmark_dynamic_journal import (
        run_journal_benchmark,
    )
    t0 = time.perf_counter()
    res = run_journal_benchmark(
        N1=N1, N2=N2, dt=dt, NT=NT,
        omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-6, max_inner=5000,
        scheme="rb", accel=accel,
    )
    return res, time.perf_counter() - t0


def test_baseline_invariance():
    """accel=None and default AusasAccelerationOptions() are bit-equal."""
    print("\n=== Test 1: baseline invariance (flag OFF) ===")
    if not _available():
        return True
    from reynolds_solver import AusasAccelerationOptions

    res_none, _ = _run(accel=None, NT=200)
    res_def, _ = _run(accel=AusasAccelerationOptions(), NT=200)

    err_X = float(np.max(np.abs(res_none.X - res_def.X)))
    err_WX = float(np.max(np.abs(res_none.WX - res_def.WX)))
    err_pmax = float(np.max(np.abs(res_none.p_max - res_def.p_max)))
    print(
        f"  X max|Δ| = {err_X:.2e}, WX max|Δ| = {err_WX:.2e}, "
        f"p_max max|Δ| = {err_pmax:.2e}"
    )
    ok = err_X < 1e-12 and err_WX < 1e-12 and err_pmax < 1e-12
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] default options match accel=None bit-for-bit")
    return ok


def test_dynamic_cadence_parity():
    """Dynamic cadence variant reproduces fixed-cadence trajectory."""
    print("\n=== Test 2: dynamic_check_every ON vs OFF parity ===")
    if not _available():
        return True
    from reynolds_solver import AusasAccelerationOptions

    res_fix, wall_fix = _run(accel=None, NT=200)
    res_dyn, wall_dyn = _run(
        accel=AusasAccelerationOptions(
            dynamic_check_every=True,
            check_every_min=5, check_every_max=25,
        ),
        NT=200,
    )

    scale_X = max(float(np.max(np.abs(res_fix.X))), 1e-30)
    scale_W = max(float(np.max(np.abs(res_fix.WX))), 1e-30)
    err_X = float(np.max(np.abs(res_fix.X - res_dyn.X))) / scale_X
    err_WX = float(np.max(np.abs(res_fix.WX - res_dyn.WX))) / scale_W

    print(
        f"  fixed-cadence   : wall={wall_fix:.1f} s, "
        f"mean inner={res_fix.n_inner.mean():.0f}"
    )
    print(
        f"  dynamic-cadence : wall={wall_dyn:.1f} s, "
        f"mean inner={res_dyn.n_inner.mean():.0f}"
    )
    print(
        f"  trajectory rel-err: X={err_X:.2e}, WX={err_WX:.2e}"
    )

    # Residual is measured less often with dynamic cadence, which
    # means the convergence decision is taken slightly later in some
    # steps — this can nudge the final state by the convergence
    # tolerance. We allow 1 % rel-err.
    ok = err_X < 1e-2 and err_WX < 1e-2
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] dynamic cadence trajectory within 1 %")
    return ok


def main():
    ok = True
    ok = test_baseline_invariance() and ok
    ok = test_dynamic_cadence_parity() and ok
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
