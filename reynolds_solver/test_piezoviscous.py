"""
Tests for piezoviscous (Barus) Reynolds solver.

Tests:
  1. Regression: alpha=0 matches laminar
  2. Monotonicity: higher alpha → higher load
  3. Oil discrimination: mineral (α=18e-9) > rapeseed (α=12e-9) in W
  4. Convergence: typical params converge in ≤15 outer iterations
  5. Overflow protection: extreme alpha doesn't produce inf/nan
  6. Backward compatibility: existing API calls unchanged
  7. Squeeze compatibility: alpha=0 + squeeze matches dynamic solver

Run: python -m reynolds_solver.test_piezoviscous
"""

import sys
import numpy as np

R = 0.035
L = 0.056
ETA = 0.01105
C_CLEAR = 50e-6
N_RPM = 2980
OMEGA_SHAFT = 2 * np.pi * N_RPM / 60
U = OMEGA_SHAFT * (R - C_CLEAR)

# Pressure scale (same as pipeline_gpu.py)
P_SCALE = (6 * ETA * U * R) / (C_CLEAR**2)


def run_test(name, passed, details=""):
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")
    if details:
        print(f"         {details}")
    return passed


def generate_grid(N, epsilon=0.6):
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z


# -----------------------------------------------------------------------
# Test 1: Regression (alpha=0 = laminar)
# -----------------------------------------------------------------------
def test_regression():
    print("\n=== Test 1: Regression (alpha=0 = laminar) ===")
    from reynolds_solver import solve_reynolds

    N = 200
    H, d_phi, d_Z = generate_grid(N, 0.6)

    P_lam, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure="laminar", cavitation="half_sommerfeld",
        omega=1.5, tol=1e-6, max_iter=50000,
    )

    P_pv, _, _, n_outer = solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure="laminar", cavitation="half_sommerfeld",
        omega=1.5, tol=1e-6, max_iter=50000,
        alpha_pv=0.0, p_scale=P_SCALE,
    )

    rel_diff = np.max(np.abs(P_pv - P_lam)) / (np.max(P_lam) + 1e-30)
    print(f"    rel_diff = {rel_diff:.4e}, n_outer = {n_outer}")

    return run_test(
        "alpha=0 matches laminar (< 1e-6)",
        rel_diff < 1e-6,
        f"rel_diff = {rel_diff:.4e}",
    )


# -----------------------------------------------------------------------
# Test 2: Monotonicity (higher alpha → higher W)
# -----------------------------------------------------------------------
def test_monotonicity():
    print("\n=== Test 2: Monotonicity (alpha → W) ===")
    from reynolds_solver import solve_reynolds

    N = 200
    H, d_phi, d_Z = generate_grid(N, 0.6)

    alphas = [0.0, 5e-9, 10e-9, 15e-9, 20e-9]
    loads = []

    for alpha in alphas:
        P, _, _, n_out = solve_reynolds(
            H, d_phi, d_Z, R, L,
            alpha_pv=alpha, p_scale=P_SCALE,
            tol=1e-5, max_iter=50000,
            verbose=False,
        )
        W = np.sum(P)
        loads.append(W)
        print(f"    alpha={alpha:.0e}: W={W:.4f}, n_outer={n_out}")

    # Check monotonically increasing
    monotone = all(loads[i] <= loads[i+1] for i in range(len(loads)-1))
    return run_test(
        "W increases with alpha",
        monotone,
        f"loads = {[f'{w:.2f}' for w in loads]}",
    )


# -----------------------------------------------------------------------
# Test 3: Oil discrimination (mineral vs rapeseed)
# -----------------------------------------------------------------------
def test_oil_discrimination():
    print("\n=== Test 3: Oil discrimination ===")
    from reynolds_solver import solve_reynolds

    N = 200
    H, d_phi, d_Z = generate_grid(N, 0.6)

    alpha_mineral = 18e-9
    alpha_rapeseed = 12e-9

    P_min, _, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        alpha_pv=alpha_mineral, p_scale=P_SCALE,
        tol=1e-5, verbose=False,
    )
    P_rap, _, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        alpha_pv=alpha_rapeseed, p_scale=P_SCALE,
        tol=1e-5, verbose=False,
    )

    W_min = np.sum(P_min)
    W_rap = np.sum(P_rap)
    diff_pct = (W_min - W_rap) / W_rap * 100

    print(f"    W_mineral  = {W_min:.4f} (alpha={alpha_mineral:.0e})")
    print(f"    W_rapeseed = {W_rap:.4f} (alpha={alpha_rapeseed:.0e})")
    print(f"    difference = {diff_pct:.2f}%")

    return run_test(
        "Mineral (α=18e-9) gives higher W than rapeseed (α=12e-9)",
        W_min > W_rap,
        f"W_min={W_min:.4f}, W_rap={W_rap:.4f}",
    )


# -----------------------------------------------------------------------
# Test 4: Convergence (≤15 outer iterations)
# -----------------------------------------------------------------------
def test_convergence():
    print("\n=== Test 4: Convergence (≤15 outer) ===")
    from reynolds_solver import solve_reynolds

    N = 200
    H, d_phi, d_Z = generate_grid(N, 0.6)

    _, _, _, n_outer = solve_reynolds(
        H, d_phi, d_Z, R, L,
        alpha_pv=15e-9, p_scale=P_SCALE,
        tol=1e-5, tol_outer=1e-3, max_outer_pv=20,
        verbose=True,
    )

    return run_test(
        f"Converged in {n_outer} outer iterations (≤15)",
        n_outer <= 15,
        f"n_outer = {n_outer}",
    )


# -----------------------------------------------------------------------
# Test 5: Overflow protection
# -----------------------------------------------------------------------
def test_overflow():
    print("\n=== Test 5: Overflow protection ===")
    from reynolds_solver import solve_reynolds

    N = 100
    H, d_phi, d_Z = generate_grid(N, 0.6)

    # Extreme alpha: should clamp, not crash
    P, _, _, n_outer = solve_reynolds(
        H, d_phi, d_Z, R, L,
        alpha_pv=1e-3,  # absurdly high
        p_scale=P_SCALE,
        tol=1e-5, max_outer_pv=5,
        verbose=True,
    )

    has_nan = np.any(np.isnan(P))
    has_inf = np.any(np.isinf(P))
    max_P = np.max(np.abs(P))
    print(f"    has_nan = {has_nan}, has_inf = {has_inf}, max|P| = {max_P:.4e}")

    all_ok = True
    all_ok &= run_test(
        "No inf/nan with extreme alpha",
        not has_nan and not has_inf,
    )
    all_ok &= run_test(
        "max|P| < 1e10 (clamp prevents blowup)",
        max_P < 1e10,
        f"max|P| = {max_P:.4e}",
    )
    return all_ok


# -----------------------------------------------------------------------
# Test 5b: Roelands formula correctness
# -----------------------------------------------------------------------
def test_roelands_formula():
    print("\n=== Test 5b: Roelands formula correctness ===")
    import cupy as cp
    from reynolds_solver.solver_piezoviscous import _compute_mu_ratio_gpu

    alpha = 18e-9
    p0 = 1.98e8
    z = 0.6

    # At p=0: mu_ratio should be exactly 1.0
    P_zero = cp.zeros((3, 3), dtype=cp.float64)
    mu_zero, _ = _compute_mu_ratio_gpu(P_zero, alpha, 1.0, p0, z)
    err_zero = float(cp.max(cp.abs(mu_zero - 1.0)))

    # At p=p0: log_mu = (alpha*p0/z) * (2^z - 1)
    P_p0 = cp.ones((3, 3), dtype=cp.float64) * p0  # p_dim = P * p_scale, use p_scale=1
    mu_p0, _ = _compute_mu_ratio_gpu(P_p0, alpha, 1.0, p0, z)
    log_expected = (alpha * p0 / z) * (2.0**z - 1.0)
    mu_expected = np.exp(log_expected)
    err_p0 = abs(float(cp.mean(mu_p0)) - mu_expected) / mu_expected

    # At low pressure (p << p0): should approximate Barus
    p_low = 1e6  # 1 MPa
    P_low = cp.ones((3, 3), dtype=cp.float64) * p_low
    mu_roel, _ = _compute_mu_ratio_gpu(P_low, alpha, 1.0, p0, z)
    mu_barus = np.exp(alpha * p_low)
    err_barus = abs(float(cp.mean(mu_roel)) - mu_barus) / mu_barus

    print(f"    p=0: mu_ratio={float(cp.mean(mu_zero)):.6f}, err={err_zero:.2e}")
    print(f"    p=p0: mu_ratio={float(cp.mean(mu_p0)):.6f}, expected={mu_expected:.6f}, "
          f"err={err_p0:.2e}")
    print(f"    p=1MPa: Roelands={float(cp.mean(mu_roel)):.6f}, Barus={mu_barus:.6f}, "
          f"err={err_barus:.2e}")

    all_ok = True
    all_ok &= run_test("p=0 → mu_ratio=1.0 exactly", err_zero < 1e-14)
    all_ok &= run_test("p=p0 matches analytic", err_p0 < 1e-10,
                       f"err={err_p0:.2e}")
    all_ok &= run_test("p<<p0 ≈ Barus (<1%)", err_barus < 0.01,
                       f"err={err_barus:.2e}")
    return all_ok


# -----------------------------------------------------------------------
# Test 6: Backward compatibility
# -----------------------------------------------------------------------
def test_backward_compat():
    print("\n=== Test 6: Backward compatibility ===")
    from reynolds_solver import solve_reynolds

    N = 100
    H, d_phi, d_Z = generate_grid(N, 0.6)

    # Standard call without alpha_pv — should work unchanged
    result = solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure="laminar", cavitation="half_sommerfeld",
    )

    ok = len(result) == 3  # (P, delta, n_iter) — 3-tuple
    return run_test(
        "Standard call returns 3-tuple",
        ok,
        f"got {len(result)} elements",
    )


# -----------------------------------------------------------------------
# Test 7: Squeeze compatibility (alpha=0 + squeeze = dynamic solver)
# -----------------------------------------------------------------------
def test_squeeze_compat():
    print("\n=== Test 7: Squeeze compatibility ===")
    from reynolds_solver import solve_reynolds

    N = 200
    H, d_phi, d_Z = generate_grid(N, 0.6)
    xp, yp = 1e-3, 2e-3

    # Dynamic solver (no piezoviscosity)
    P_dyn, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        xprime=xp, yprime=yp, beta=2.0,
        tol=1e-6, max_iter=50000,
    )

    # Piezoviscous with alpha=0 + same squeeze
    P_pv, _, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        xprime=xp, yprime=yp, beta=2.0,
        alpha_pv=0.0, p_scale=P_SCALE,
        tol=1e-6, max_iter=50000,
    )

    rel_diff = np.max(np.abs(P_pv - P_dyn)) / (np.max(P_dyn) + 1e-30)
    print(f"    rel_diff = {rel_diff:.4e}")

    return run_test(
        "alpha=0 + squeeze matches dynamic solver (< 1e-6)",
        rel_diff < 1e-6,
        f"rel_diff = {rel_diff:.4e}",
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Piezoviscous (Roelands) tests")
    print("=" * 60)

    results = []
    results.append(test_regression())
    results.append(test_monotonicity())
    results.append(test_oil_discrimination())
    results.append(test_convergence())
    results.append(test_overflow())
    results.append(test_roelands_formula())
    results.append(test_backward_compat())
    results.append(test_squeeze_compat())

    print("\n" + "=" * 60)
    all_ok = all(results)
    if all_ok:
        print("  ALL PIEZOVISCOUS (ROELANDS) TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        for i, r in enumerate(results, 1):
            if not r:
                print(f"    Test {i} FAILED")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
