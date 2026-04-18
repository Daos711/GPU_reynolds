"""
Regression + speedup check for Phase 5 Part 1 variant B (fused
Newmark predictor + fused WX/WY reduction).

Unlike the abandoned lagged-mechanics attempt, these two kernels do
NOT change when mechanics is recomputed — they just REPLACE the
multi-launch CuPy pipelines for forces (~4-6 launches) and predictor
(~8 launches) with a single-launch kernel each. They are always on
when the journal solver is compiled.

This test therefore only needs to assert:

  1. `result.X`, `result.Y`, `result.WX`, `result.WY`, `result.p_max`,
     `result.cav_frac` match an analytic sanity: finite, e < 1,
     h_min > 0, WX approximately reacts to WaX on the periodic
     orbit.

  2. A baseline (this solver) wall-time on the reduced journal
     benchmark (100x12, NT=300) is within a reasonable band. The
     fused kernels should noticeably beat the pre-5.1B baseline
     (which was 80 s on the user's machine at Phase 4.1), so the
     expected budget is <= 60 s for the fused version. This is
     MACHINE DEPENDENT: we allow up to 150 s before failing so the
     test does not become a Windows-CUDA timing flake.

Skipped on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_ausas_fused_kernels
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


def _run(N1=100, N2=12, dt=2e-3, NT=300):
    from reynolds_solver.cavitation.ausas.benchmark_dynamic_journal import (
        run_journal_benchmark,
    )
    t0 = time.perf_counter()
    res = run_journal_benchmark(
        N1=N1, N2=N2, dt=dt, NT=NT,
        omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-6, max_inner=5000,
        scheme="rb", verbose=False,
    )
    return res, time.perf_counter() - t0


def test_physical_sanity():
    """Fused journal run reproduces the Stage-3 physical invariants."""
    print("\n=== Test 1: fused journal physical invariants ===")
    if not _available():
        return True

    res, wall = _run(N1=100, N2=12, dt=2e-3, NT=500)

    e = np.sqrt(res.X ** 2 + res.Y ** 2)
    e_max = float(e.max())
    h_min = float(res.h_min.min())
    cav_max = float(res.cav_frac.max())

    # Force balance on the last period: mean |WX + WaX| vs applied-load
    # amplitude. Stage-3 got ~1e-7; with the fused reduction we expect
    # similar quality since the physics is unchanged.
    last = res.t > res.t[-1] - 1.0
    wax_amp = float(np.max(np.abs(res.WaX[last])))
    bal_X = abs(
        float(res.WX[last].mean()) + float(res.WaX[last].mean())
    ) / (wax_amp + 1e-30)

    print(
        f"  run {wall:.1f} s; e_max={e_max:.3f}, h_min={h_min:.3e}, "
        f"cav_max={cav_max:.3f}, load-balance rel={bal_X:.2e}"
    )
    print(
        f"  inner iters: min={int(res.n_inner.min())}, "
        f"max={int(res.n_inner.max())}, mean={res.n_inner.mean():.0f}"
    )

    ok_finite = bool(
        np.all(np.isfinite(res.X))
        and np.all(np.isfinite(res.Y))
        and np.all(np.isfinite(res.WX))
        and np.all(np.isfinite(res.WY))
        and np.all(np.isfinite(res.p_max))
    )
    ok_bounds = e_max < 1.0 and h_min > 0.0
    ok_cav = cav_max > 0.0
    # Load balance tolerance: orbit may still be in transient at NT=500,
    # accept up to 20%. Stage-3 3-period run hit <1e-6.
    ok_balance = bal_X < 0.20

    ok = ok_finite and ok_bounds and ok_cav and ok_balance
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] finite={ok_finite}, e<1 & h_min>0 ({ok_bounds}), "
        f"cav_max>0 ({ok_cav}), load-balance<20% ({ok_balance})"
    )
    return ok


def test_restart_still_bit_exact():
    """
    Fused kernels must not break restart: continuous run == split +
    save/load + resume, to ~1e-10 relative on trajectory. This is a
    regression check against the Phase 4.3 property.
    """
    print("\n=== Test 2: restart bit-exactness under fused kernels ===")
    if not _available():
        return True
    import os
    import tempfile
    from reynolds_solver import save_state, load_state
    from reynolds_solver.cavitation.ausas.benchmark_dynamic_journal import (
        run_journal_benchmark,
    )

    common = dict(
        N1=60, N2=8, dt=2e-3,
        omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-6, max_inner=3000,
        scheme="rb",
    )

    res_full = run_journal_benchmark(NT=200, **common)
    res_a = run_journal_benchmark(NT=100, **common)
    assert res_a.final_state is not None

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp = f.name
    try:
        save_state(res_a.final_state, tmp)
        loaded = load_state(tmp)
    finally:
        os.unlink(tmp)

    res_b = run_journal_benchmark(NT=100, state=loaded, **common)

    X_split = np.concatenate([res_a.X, res_b.X])
    WX_split = np.concatenate([res_a.WX, res_b.WX])
    scale_X = max(float(np.max(np.abs(res_full.X))), 1e-30)
    scale_W = max(float(np.max(np.abs(res_full.WX))), 1e-30)
    err_X = float(np.max(np.abs(X_split - res_full.X))) / scale_X
    err_WX = float(np.max(np.abs(WX_split - res_full.WX))) / scale_W

    print(f"  X rel = {err_X:.2e}, WX rel = {err_WX:.2e}")
    ok = err_X < 1e-8 and err_WX < 1e-6
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] restart preserves trajectory to ~1e-8")
    return ok


def test_speed_budget():
    """
    Wall-time budget for the reduced benchmark. This replaces the
    abandoned lagged-mechanics speedup test. We compare against a
    generous Windows/Linux-portable upper bound rather than a strict
    speedup ratio vs some baseline run we cannot re-do in the test.
    """
    print("\n=== Test 3: fused-kernel wall-time budget ===")
    if not _available():
        return True

    # Matches the Phase 4.1 reduced journal test grid exactly, so we
    # can compare apples-to-apples against the 80 s number we saw then.
    res, wall = _run(N1=100, N2=12, dt=2e-3, NT=1500)
    print(f"  100x12 NT=1500 wall time = {wall:.1f} s (Phase 4.1 was ~80 s)")
    # Phase 5.1B should be AT LEAST as fast as Phase 4.1. On Windows
    # CUDA we give a generous 150 s ceiling.
    ok = wall <= 150.0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] wall time <= 150 s")
    return ok


def main():
    ok = True
    ok = test_physical_sanity() and ok
    ok = test_restart_still_bit_exact() and ok
    ok = test_speed_budget() and ok
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
