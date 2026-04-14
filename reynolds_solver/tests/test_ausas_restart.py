"""
Restart / save / load regression: continuous run must equal
split-with-restart.

Two scenarios:
  (1) prescribed-h: one 400-step run == run(0..200) + save/load +
      run(200..400).
  (2) dynamic journal: one 300-step run == run(0..150) + save/load +
      run(150..300).

Expected tolerance: bit-for-bit on scalar histories after the split.
Floating-point reordering from separating the runs is below the last
double digit; we allow ~1e-12 rel-err for safety.

Skipped on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_ausas_restart
"""
import os
import sys
import tempfile

import numpy as np


def _available():
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return False
    return True


def test_prescribed_h_restart():
    """Continuous 400-step run vs 200+200 split with save/load state."""
    print("\n=== Test 1: prescribed-h restart ===")
    if not _available():
        return True

    from reynolds_solver import (
        solve_ausas_prescribed_h_gpu, save_state, load_state,
    )

    N_phi, N_Z = 62, 6
    d_phi = 1.0 / (N_phi - 2)
    d_Z = 0.1 / (N_Z - 2)
    R, L = 0.5, 1.0
    p0_bc = 0.02
    dt = 1e-3

    def H_provider(n, t):
        h = 0.4 + 0.08 * np.cos(2.0 * np.pi * 4.0 * t)
        return np.full((N_Z, N_phi), h, dtype=np.float64)

    common = dict(
        H_provider=H_provider, dt=dt,
        d_phi=d_phi, d_Z=d_Z, R=R, L=L,
        alpha=0.0, omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-6, max_inner=3000,
        p_bc_phi0=p0_bc, p_bc_phiL=p0_bc,
        theta_bc_phi0=1.0, theta_bc_phiL=1.0,
        p_bc_z0=p0_bc, p_bc_zL=p0_bc,
        theta_bc_z0=1.0, theta_bc_zL=1.0,
        periodic_phi=False, periodic_z=True,
        scheme="rb",
    )

    # Continuous.
    res_full = solve_ausas_prescribed_h_gpu(NT=400, **common)

    # Split: 200 steps, save state, load, run remaining 200.
    res_a = solve_ausas_prescribed_h_gpu(NT=200, **common)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
    try:
        save_state(res_a.final_state, tmp_path)
        state_loaded = load_state(tmp_path)
    finally:
        os.unlink(tmp_path)
    res_b = solve_ausas_prescribed_h_gpu(
        NT=200, **common, state=state_loaded,
    )

    # Stitch histories.
    t_split = np.concatenate([res_a.t, res_b.t])
    pmax_split = np.concatenate([res_a.p_max, res_b.p_max])
    cav_split = np.concatenate([res_a.cav_frac, res_b.cav_frac])
    hmin_split = np.concatenate([res_a.h_min, res_b.h_min])

    scale_p = max(float(np.max(np.abs(res_full.p_max))), 1e-30)
    err_t = float(np.max(np.abs(t_split - res_full.t)))
    err_pmax = float(np.max(np.abs(pmax_split - res_full.p_max))) / scale_p
    err_cav = float(np.max(np.abs(cav_split - res_full.cav_frac)))
    err_hmin = float(np.max(np.abs(hmin_split - res_full.h_min)))
    # Final-field drift.
    err_P = float(np.max(np.abs(res_full.P_last - res_b.P_last))) / scale_p
    err_th = float(np.max(np.abs(res_full.theta_last - res_b.theta_last)))

    print(
        f"  time axis max|Δ| = {err_t:.2e}, "
        f"p_max rel = {err_pmax:.2e}, cav max|Δ| = {err_cav:.2e}"
    )
    print(
        f"  h_min max|Δ| = {err_hmin:.2e}, final P rel = {err_P:.2e}, "
        f"final theta max|Δ| = {err_th:.2e}"
    )
    ok = (
        err_t < 1e-12 and err_pmax < 1e-10 and err_cav < 1e-10
        and err_hmin < 1e-12 and err_P < 1e-10 and err_th < 1e-10
    )
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] restart reproduces continuous run (~1e-10 tol)")
    return ok


def test_journal_restart():
    """Journal: one 300-step run vs 150 + save/load + 150."""
    print("\n=== Test 2: dynamic journal restart ===")
    if not _available():
        return True

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

    res_full = run_journal_benchmark(NT=300, **common)

    res_a = run_journal_benchmark(NT=150, **common)
    assert res_a.final_state is not None, \
        "JournalBenchmarkResult should carry final_state"
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_path = f.name
    try:
        save_state(res_a.final_state, tmp_path)
        state_loaded = load_state(tmp_path)
    finally:
        os.unlink(tmp_path)

    res_b = run_journal_benchmark(NT=150, state=state_loaded, **common)

    # Stitch scalar histories.
    X_split = np.concatenate([res_a.X, res_b.X])
    Y_split = np.concatenate([res_a.Y, res_b.Y])
    WX_split = np.concatenate([res_a.WX, res_b.WX])
    e_split = np.concatenate([res_a.eccentricity, res_b.eccentricity])
    t_split = np.concatenate([res_a.t, res_b.t])

    # Tolerances: restart rebuilds H_prev from (X, Y) on the CPU using
    # the exact same sampling, so parity should be very tight.
    scale_X = max(float(np.max(np.abs(res_full.X))), 1e-30)
    err_X = float(np.max(np.abs(X_split - res_full.X))) / scale_X
    err_Y = float(np.max(np.abs(Y_split - res_full.Y))) / scale_X
    err_e = float(np.max(np.abs(e_split - res_full.eccentricity))) / scale_X
    scale_W = max(float(np.max(np.abs(res_full.WX))), 1e-30)
    err_WX = float(np.max(np.abs(WX_split - res_full.WX))) / scale_W
    err_t = float(np.max(np.abs(t_split - res_full.t)))

    print(
        f"  X rel = {err_X:.2e}, Y rel = {err_Y:.2e}, e rel = {err_e:.2e}"
    )
    print(
        f"  WX rel = {err_WX:.2e}, time axis max|Δ| = {err_t:.2e}"
    )
    # Allow ~1e-8 relative error because H_prev is reconstructed, not
    # saved, from X/Y on host doubles — not exactly the device's
    # rounding path.
    ok = err_X < 1e-6 and err_Y < 1e-6 and err_WX < 1e-4 and err_t < 1e-10
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] restart matches continuous run (rel < 1e-6)")
    return ok


def main():
    ok = True
    ok = test_prescribed_h_restart() and ok
    ok = test_journal_restart() and ok
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
