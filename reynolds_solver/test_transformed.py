"""
Tests for transformed-pressure piezoviscous solver (Barus).

Tests:
  1. Regression: α=0 → P = Φ (matches laminar)
  2. Agreement with iterative Barus at low α
  3. Clamp protection: no inf/nan at high α
  4. Squeeze compatibility: α=0 + squeeze = dynamic solver
  5. Monotonicity: higher α → higher W

Run: python -m reynolds_solver.test_transformed
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
# Test 1: Regression (α=0 = laminar)
# -----------------------------------------------------------------------
def test_regression():
    print("\n=== Test 1: Regression (α=0 = laminar) ===")
    from reynolds_solver import solve_reynolds

    N = 200
    H, d_phi, d_Z = generate_grid(N, 0.6)

    P_lam, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure="laminar", cavitation="half_sommerfeld",
        omega=1.5, tol=1e-6, max_iter=50000,
    )

    P_tr, _, n_iter = solve_reynolds(
        H, d_phi, d_Z, R, L,
        alpha_pv=0.0, p_scale=P_SCALE, pv_method="transformed",
        omega=1.5, tol=1e-6, max_iter=50000,
    )

    rel_diff = np.max(np.abs(P_tr - P_lam)) / (np.max(P_lam) + 1e-30)
    print(f"    rel_diff = {rel_diff:.4e}, n_iter = {n_iter}")

    return run_test(
        "α=0 transformed matches laminar (< 1e-6)",
        rel_diff < 1e-6,
        f"rel_diff = {rel_diff:.4e}",
    )


# -----------------------------------------------------------------------
# Test 2: Agreement with iterative Barus
# -----------------------------------------------------------------------
def test_vs_iterative():
    print("\n=== Test 2: Agreement with iterative Barus ===")
    from reynolds_solver import solve_reynolds

    N = 200
    H, d_phi, d_Z = generate_grid(N, 0.6)
    alpha = 15e-9

    # Iterative (Roelands, but at low p ≈ Barus)
    P_iter, _, _, n_outer = solve_reynolds(
        H, d_phi, d_Z, R, L,
        alpha_pv=alpha, p_scale=P_SCALE, pv_method="iterative",
        tol=1e-6, verbose=False,
    )

    # Transformed (exact Barus)
    P_tr, _, n_iter = solve_reynolds(
        H, d_phi, d_Z, R, L,
        alpha_pv=alpha, p_scale=P_SCALE, pv_method="transformed",
        tol=1e-6, verbose=True,
    )

    W_iter = np.sum(P_iter)
    W_tr = np.sum(P_tr)
    rel_W = abs(W_tr - W_iter) / (abs(W_iter) + 1e-30)

    maxP_diff = np.max(np.abs(P_tr - P_iter)) / (np.max(P_iter) + 1e-30)

    print(f"    W_iterative = {W_iter:.4f} (n_outer={n_outer})")
    print(f"    W_transformed = {W_tr:.4f} (n_iter={n_iter})")
    print(f"    rel_W = {rel_W:.4e}, max_P_diff = {maxP_diff:.4e}")

    # At low pressures, Roelands ≈ Barus, so results should be close
    return run_test(
        "W matches iterative within 1%",
        rel_W < 0.01,
        f"rel_W = {rel_W:.4e}",
    )


# -----------------------------------------------------------------------
# Test 3: Clamp protection
# -----------------------------------------------------------------------
def test_clamp():
    print("\n=== Test 3: Clamp protection (high α) ===")
    from reynolds_solver import solve_reynolds

    N = 100
    H, d_phi, d_Z = generate_grid(N, 0.6)

    P, _, n_iter = solve_reynolds(
        H, d_phi, d_Z, R, L,
        alpha_pv=1e-6, p_scale=P_SCALE, pv_method="transformed",
        tol=1e-5, verbose=True,
    )

    has_nan = np.any(np.isnan(P))
    has_inf = np.any(np.isinf(P))
    max_P = np.max(P)
    print(f"    has_nan={has_nan}, has_inf={has_inf}, max(P)={max_P:.4e}")

    all_ok = True
    all_ok &= run_test("No inf/nan", not has_nan and not has_inf)
    all_ok &= run_test("max(P) finite and reasonable",
                       max_P < 1e10, f"max(P)={max_P:.4e}")
    return all_ok


# -----------------------------------------------------------------------
# Test 4: Squeeze compatibility
# -----------------------------------------------------------------------
def test_squeeze():
    print("\n=== Test 4: Squeeze compatibility (α=0) ===")
    from reynolds_solver import solve_reynolds

    N = 200
    H, d_phi, d_Z = generate_grid(N, 0.6)
    xp, yp = 1e-3, 2e-3

    P_dyn, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        xprime=xp, yprime=yp, beta=2.0,
        tol=1e-6, max_iter=50000,
    )

    P_tr, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        xprime=xp, yprime=yp, beta=2.0,
        alpha_pv=0.0, p_scale=P_SCALE, pv_method="transformed",
        tol=1e-6, max_iter=50000,
    )

    rel_diff = np.max(np.abs(P_tr - P_dyn)) / (np.max(P_dyn) + 1e-30)
    print(f"    rel_diff = {rel_diff:.4e}")

    return run_test(
        "α=0 + squeeze matches dynamic (< 1e-6)",
        rel_diff < 1e-6,
        f"rel_diff = {rel_diff:.4e}",
    )


# -----------------------------------------------------------------------
# Test 5: Monotonicity
# -----------------------------------------------------------------------
def test_monotonicity():
    print("\n=== Test 5: Monotonicity (α → W) ===")
    from reynolds_solver import solve_reynolds

    N = 200
    H, d_phi, d_Z = generate_grid(N, 0.6)

    alphas = [0.0, 5e-9, 10e-9, 15e-9, 20e-9]
    loads = []

    for alpha in alphas:
        P, _, n_iter = solve_reynolds(
            H, d_phi, d_Z, R, L,
            alpha_pv=alpha, p_scale=P_SCALE, pv_method="transformed",
            tol=1e-5,
        )
        W = np.sum(P)
        loads.append(W)
        print(f"    α={alpha:.0e}: W={W:.4f}, n_iter={n_iter}")

    monotone = all(loads[i] <= loads[i+1] for i in range(len(loads)-1))
    return run_test(
        "W increases with α",
        monotone,
        f"loads = {[f'{w:.2f}' for w in loads]}",
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Transformed-pressure (Barus) tests")
    print("=" * 60)

    results = []
    results.append(test_regression())
    results.append(test_vs_iterative())
    results.append(test_clamp())
    results.append(test_squeeze())
    results.append(test_monotonicity())

    print("\n" + "=" * 60)
    all_ok = all(results)
    if all_ok:
        print("  ALL TRANSFORMED TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        for i, r in enumerate(results, 1):
            if not r:
                print(f"    Test {i} FAILED")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
