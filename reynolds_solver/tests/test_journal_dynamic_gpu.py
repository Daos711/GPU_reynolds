"""
Stage-3 dynamic-journal-bearing regression test.

Validates `solve_ausas_journal_dynamic_gpu` on the Ausas, Jai, Buscaglia
(2008) Section-5 benchmark. The full spec-grid run takes several minutes
even on a fast GPU (3000 time steps, each with an inner loop that
rebuilds the gap, coefficients, and does two RB sweeps per iteration),
so the test uses a reduced NT / grid by default. Run the full benchmark
via `python -m reynolds_solver.cavitation.ausas.benchmark_dynamic_journal`.

Checks
------
1. Shaft trajectory stays physical: eccentricity e = sqrt(X^2 + Y^2) < 1
   at all times; interior gap h_min > 0.
2. Cavitation forms during the loaded phase (cav_frac > 0 for at least
   one step).
3. Second-period trajectory is close to the third period: the shaft
   settles into a periodic orbit after the initial transient.
4. Hydrodynamic force WX approximately cancels the applied load WaX on
   average over the periodic orbit (Newton's law: the bearing must
   react to the applied load for periodicity to hold).

Skipped silently on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_journal_dynamic_gpu
"""
import sys
import time

import numpy as np


def _run(
    N1=200, N2=20, dt=1e-3, NT=3000,
    tol_inner=1e-6, max_inner=5000,
    omega_p=1.0, scheme="rb",
):
    """Run the benchmark; returns None on CPU-only machines."""
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return None

    from reynolds_solver.cavitation.ausas.benchmark_dynamic_journal import (
        run_journal_benchmark,
    )
    t0 = time.perf_counter()
    res = run_journal_benchmark(
        N1=N1, N2=N2, dt=dt, NT=NT,
        tol_inner=tol_inner, max_inner=max_inner,
        omega_p=omega_p, scheme=scheme,
        verbose=False,
    )
    print(f"  run took {time.perf_counter() - t0:.1f} s "
          f"(N1={N1}, N2={N2}, NT={NT})")
    return res


# A single reduced-grid run shared between tests to keep wall time sane.
_shared_run = None

def _get_shared_run():
    global _shared_run
    if _shared_run is not None:
        return _shared_run
    # Reduced grid + 3-period horizon so the test can compare the
    # transient-free last two periods. Full benchmark (N1=200, NT=3000)
    # is run via the CLI, not the unit test.
    _shared_run = _run(N1=100, N2=12, dt=2e-3, NT=1500)
    return _shared_run


def test_physical_bounds():
    """e < 1 at every step and h_min > 0 on the interior."""
    print("\n=== Test 1: physical bounds (e < 1, h_min > 0) ===")
    res = _get_shared_run()
    if res is None:
        return True

    e_max = float(res.eccentricity.max())
    h_min = float(res.h_min.min())
    print(f"  eccentricity max = {e_max:.3f} (bound < 1.0)")
    print(f"  h_min over run   = {h_min:.3e} (bound > 0)")
    ok = e_max < 1.0 and h_min > 0.0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}]")
    return ok


def test_cavitation_forms():
    """Cavitation must appear at some point during the loaded phase."""
    print("\n=== Test 2: cavitation forms ===")
    res = _get_shared_run()
    if res is None:
        return True
    cav_max = float(res.cav_frac.max())
    cav_at = float(res.t[int(np.argmax(res.cav_frac))])
    print(f"  cav_frac max = {cav_max:.3f} at t = {cav_at:.3f}")
    ok = cav_max > 0.0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] cav_max > 0")
    return ok


def test_trajectory_periodic():
    """
    After the initial transient, the orbit should repeat — compare the
    X(t), Y(t) samples of the last period against the penultimate period.
    The expected error is O(percent) (numerics + single-period transient
    tail), not machine precision.
    """
    print("\n=== Test 3: periodic orbit (last vs penultimate period) ===")
    res = _get_shared_run()
    if res is None:
        return True

    t = res.t
    T_period = 1.0
    t_end = t[-1]
    if t_end < 2.0 * T_period:
        print(f"  [SKIP] simulated horizon {t_end:.2f} < 2 T_period")
        return True

    last = (t > t_end - T_period) & (t <= t_end)
    prev = (t > t_end - 2.0 * T_period) & (t <= t_end - T_period)
    n = min(last.sum(), prev.sum())
    if n < 10:
        print(f"  [FAIL] too few samples per period (got {n})")
        return False

    # Align lengths (truncate trailing samples if necessary).
    Xl = res.X[last][:n]
    Yl = res.Y[last][:n]
    Xp = res.X[prev][:n]
    Yp = res.Y[prev][:n]

    diff = np.sqrt(np.mean((Xl - Xp) ** 2 + (Yl - Yp) ** 2))
    scale = np.sqrt(np.mean(Xl ** 2 + Yl ** 2)) + 1e-30
    rel = float(diff / scale)

    print(f"  RMS |(X,Y)_last - (X,Y)_prev| / scale = {rel:.3e}")
    ok = rel < 0.10    # 10 % tolerance — qualitative periodicity check
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] rel.diff between last two periods < 10 %")
    return ok


def test_force_balance():
    """
    Over the periodic orbit the hydrodynamic force must react to the
    applied load. The average of (WX + WaX) over the last period should
    be much smaller than either force individually.
    """
    print("\n=== Test 4: hydrodynamic force balances applied load ===")
    res = _get_shared_run()
    if res is None:
        return True

    t = res.t
    T_period = 1.0
    last = t > t[-1] - T_period

    WX_mean = float(res.WX[last].mean())
    WaX_mean = float(res.WaX[last].mean())
    WY_mean = float(res.WY[last].mean())
    WaY_mean = float(res.WaY[last].mean())

    # The shaft is in a periodic orbit, so over one period the net
    # impulse from (WX + WaX) must be small (otherwise the shaft drifts).
    # Compare the integrated mismatch against the applied-load amplitude.
    WaX_amp = float(np.max(np.abs(res.WaX[last])))
    WaY_amp = float(np.max(np.abs(res.WaY[last])))
    res_X = abs(WX_mean + WaX_mean) / (WaX_amp + 1e-30)
    res_Y = abs(WY_mean + WaY_mean) / (WaY_amp + 1e-30)

    print(
        f"  last-period means: WX={WX_mean:+.3e} WaX={WaX_mean:+.3e} "
        f"relative-residual={res_X:.3f}"
    )
    print(
        f"  last-period means: WY={WY_mean:+.3e} WaY={WaY_mean:+.3e} "
        f"relative-residual={res_Y:.3f}"
    )
    ok = res_X < 0.5 and res_Y < 0.5
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] |mean(WX + WaX)| < 0.5 * WaX_amp (and same for Y)")
    return ok


def main():
    ok1 = test_physical_bounds()
    ok2 = test_cavitation_forms()
    ok3 = test_trajectory_periodic()
    ok4 = test_force_balance()
    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
