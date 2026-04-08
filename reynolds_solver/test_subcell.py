"""
Tests for subcell quadrature conductance.

Test 1: Smooth bearing — subcell_quad=True ≈ subcell_quad=False (<0.5%)
Test 2: Backward compatibility — default call unchanged

Run: python -m reynolds_solver.test_subcell
"""

import sys
import numpy as np

R = 0.035
L = 0.056


def run_test(name, passed, details=""):
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")
    if details:
        print(f"         {details}")
    return passed


def test_smooth_bearing():
    """Smooth bearing: subcell ≈ standard (<0.5% in W, p_max)."""
    print("\n=== Test 1: Smooth bearing (subcell ≈ standard) ===")
    from reynolds_solver import solve_reynolds

    N = 250
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + 0.6 * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]

    P_std, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure="laminar", cavitation="half_sommerfeld",
        subcell_quad=False,
    )

    P_sub, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure="laminar", cavitation="half_sommerfeld",
        subcell_quad=True, n_sub=4,
    )

    W_std = np.sum(P_std)
    W_sub = np.sum(P_sub)
    rel_W = abs(W_sub - W_std) / (abs(W_std) + 1e-30)

    pmax_std = np.max(P_std)
    pmax_sub = np.max(P_sub)
    rel_pmax = abs(pmax_sub - pmax_std) / (pmax_std + 1e-30)

    print(f"    W_std = {W_std:.4f}, W_sub = {W_sub:.4f}, rel = {rel_W:.4e}")
    print(f"    pmax_std = {pmax_std:.6f}, pmax_sub = {pmax_sub:.6f}, rel = {rel_pmax:.4e}")

    all_ok = True
    all_ok &= run_test("W difference < 0.5%", rel_W < 0.005, f"rel_W = {rel_W:.4e}")
    all_ok &= run_test("pmax difference < 0.5%", rel_pmax < 0.005, f"rel_pmax = {rel_pmax:.4e}")
    return all_ok


def test_backward_compat():
    """Default call (no subcell params) works unchanged."""
    print("\n=== Test 2: Backward compatibility ===")
    from reynolds_solver import solve_reynolds

    N = 100
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + 0.6 * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]

    result = solve_reynolds(H, d_phi, d_Z, R, L)
    ok = len(result) == 3 and np.max(result[0]) > 0
    return run_test("Default call returns valid result", ok)


def main():
    print("=" * 60)
    print("  Subcell quadrature tests")
    print("=" * 60)

    results = []
    results.append(test_smooth_bearing())
    results.append(test_backward_compat())

    print("\n" + "=" * 60)
    if all(results):
        print("  ALL SUBCELL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
