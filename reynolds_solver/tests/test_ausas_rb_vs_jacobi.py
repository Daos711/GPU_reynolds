"""
Regression: RB (red-black GS/SOR) vs Jacobi parity test.

Stage-1 proved GPU/CPU parity for the Jacobi kernel. Stage-2/3 rely on
the RB kernel for speed. This test confirms the two schemes converge
to the same physical solution (within discretisation / iteration
tolerance, NOT bit-for-bit) on three representative cases:

  (1) prescribed-h, a handful of time steps on a small grid;
  (2) a short squeeze run (50 time steps) past the rupture onset;
  (3) a short dynamic-journal run (150 time steps) past the first
      load impulse.

All three runs use the SAME inputs in both schemes; the only
difference is `scheme='jacobi'` vs `scheme='rb'`.

Skipped silently on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_ausas_rb_vs_jacobi
"""
import sys

import numpy as np


def _available():
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return False
    return True


def _h_uniform(N_Z, N_phi, h_val=0.5):
    return np.full((N_Z, N_phi), h_val, dtype=np.float64)


def test_prescribed_h_parity():
    """Same prescribed-h problem under Jacobi and RB."""
    print("\n=== Test 1: prescribed-h, Jacobi vs RB ===")
    if not _available():
        return True
    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        solve_ausas_prescribed_h_gpu,
    )

    N_phi, N_Z = 52, 6
    d_phi = 1.0 / (N_phi - 2)
    d_Z = 0.1 / (N_Z - 2)
    R, L = 0.5, 1.0

    # Oscillating gap: a coarse squeeze.
    def H_provider(n, t):
        h = 0.4 + 0.08 * np.cos(2.0 * np.pi * 5.0 * t)
        return np.full((N_Z, N_phi), h, dtype=np.float64)

    NT = 30
    dt = 2e-3
    p0_bc = 0.02

    P0 = p0_bc * np.ones((N_Z, N_phi), dtype=np.float64)
    theta0 = np.ones((N_Z, N_phi), dtype=np.float64)

    common = dict(
        H_provider=H_provider, NT=NT, dt=dt,
        d_phi=d_phi, d_Z=d_Z, R=R, L=L,
        alpha=0.0, omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-7, max_inner=5000,
        P0=P0, theta0=theta0,
        p_bc_phi0=p0_bc, p_bc_phiL=p0_bc,
        theta_bc_phi0=1.0, theta_bc_phiL=1.0,
        p_bc_z0=p0_bc, p_bc_zL=p0_bc,
        theta_bc_z0=1.0, theta_bc_zL=1.0,
        periodic_phi=False, periodic_z=True,
    )

    res_j = solve_ausas_prescribed_h_gpu(scheme="jacobi", **common)
    res_r = solve_ausas_prescribed_h_gpu(scheme="rb", **common)

    # Compare per-step scalars.
    err_pmax = float(
        np.max(np.abs(res_j.p_max - res_r.p_max))
        / (np.max(np.abs(res_j.p_max)) + 1e-30)
    )
    err_cav = float(np.max(np.abs(res_j.cav_frac - res_r.cav_frac)))

    # Compare last fields.
    P_ref_scale = max(float(np.max(np.abs(res_j.P_last))), 1e-30)
    err_P = float(np.max(np.abs(res_j.P_last - res_r.P_last))) / P_ref_scale
    err_th = float(np.max(np.abs(res_j.theta_last - res_r.theta_last)))

    print(
        f"  per-step p_max rel-err = {err_pmax:.2e}, "
        f"cav_frac max|Δ| = {err_cav:.2e}"
    )
    print(
        f"  final P rel-err = {err_P:.2e}, "
        f"final theta max|Δ| = {err_th:.2e}"
    )
    # Jacobi and RB converge to the same fixed point up to
    # iteration-tolerance error; we expect parity at ~1e-4 level.
    ok = err_pmax < 1e-3 and err_cav < 1e-3 and err_P < 1e-3 and err_th < 1e-3
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] all relative errors < 1e-3")
    return ok


def test_short_squeeze_parity():
    """Short squeeze run (50 steps past rupture) under Jacobi and RB."""
    print("\n=== Test 2: short squeeze, Jacobi vs RB ===")
    if not _available():
        return True
    from reynolds_solver.cavitation.ausas.benchmark_squeeze_dynamic import (
        run_squeeze_benchmark,
    )

    common = dict(
        N1=80, N2=4, dt=6.6e-4,
        NT=420,                      # well past t_rup (~0.2501)
        omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-6, max_inner=5000,
    )
    res_j = run_squeeze_benchmark(scheme="jacobi", **common)
    res_r = run_squeeze_benchmark(scheme="rb", **common)

    # Rupture-time detection should match exactly.
    rup_match = abs(res_j.t_rup_numerical - res_r.t_rup_numerical) < 1.5 * common["dt"]

    # Per-step cav fraction and p_max close in L2.
    cav_err = float(
        np.sqrt(np.mean((res_j.cav_frac - res_r.cav_frac) ** 2))
    )
    pmax_err = float(
        np.sqrt(np.mean((res_j.p_max - res_r.p_max) ** 2))
        / (np.max(np.abs(res_j.p_max)) + 1e-30)
    )

    print(
        f"  t_rup jacobi = {res_j.t_rup_numerical:.6f}, "
        f"rb = {res_r.t_rup_numerical:.6f}"
    )
    print(f"  cav_frac L2 = {cav_err:.2e}")
    print(f"  p_max L2 rel = {pmax_err:.2e}")
    ok = rup_match and cav_err < 2e-2 and pmax_err < 2e-2
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] rupture matches + field L2 < 2%")
    return ok


def test_short_journal_parity():
    """Short dynamic-journal run (150 steps) under Jacobi and RB."""
    print("\n=== Test 3: short journal, Jacobi vs RB ===")
    if not _available():
        return True
    from reynolds_solver.cavitation.ausas.benchmark_dynamic_journal import (
        run_journal_benchmark,
    )

    common = dict(
        N1=60, N2=8, dt=2e-3, NT=150,
        omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-6, max_inner=5000,
    )
    res_j = run_journal_benchmark(scheme="jacobi", **common)
    res_r = run_journal_benchmark(scheme="rb", **common)

    # Compare trajectory + forces.
    X_scale = max(float(np.max(np.abs(res_j.X))), 1e-30)
    err_X = float(np.max(np.abs(res_j.X - res_r.X))) / X_scale
    err_Y = float(np.max(np.abs(res_j.Y - res_r.Y))) / X_scale

    WX_scale = max(float(np.max(np.abs(res_j.WX))), 1e-30)
    err_WX = float(np.max(np.abs(res_j.WX - res_r.WX))) / WX_scale
    err_WY = float(np.max(np.abs(res_j.WY - res_r.WY))) / WX_scale

    print(
        f"  trajectory rel-err: X={err_X:.2e}, Y={err_Y:.2e}"
    )
    print(
        f"  forces rel-err: WX={err_WX:.2e}, WY={err_WY:.2e}"
    )
    ok = err_X < 5e-2 and err_Y < 5e-2 and err_WX < 5e-2 and err_WY < 5e-2
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] trajectory + forces rel-err < 5 %")
    return ok


def main():
    ok = True
    ok = test_prescribed_h_parity() and ok
    ok = test_short_squeeze_parity() and ok
    ok = test_short_journal_parity() and ok
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
