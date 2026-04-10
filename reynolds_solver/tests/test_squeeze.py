"""
Tests for squeeze-film support.

Tests:
  1. Regression: zero squeeze = static solver
  2. RHS unit test: squeeze adds correct coefficient to F
  3. Smoke test: pure squeeze (no wedge) produces symmetric pressure
  4. Sign test: closing gap → higher P, opening gap → lower P

Run: python -m reynolds_solver.test_squeeze
"""

import sys
import numpy as np


def run_test(name, passed, details=""):
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")
    if details:
        print(f"         {details}")
    return passed


def generate_grid(N, epsilon=0.6):
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z, phi_1D, Z, Phi_mesh


R, L = 0.035, 0.056


# -----------------------------------------------------------------------
# Test 1: Regression — zero squeeze = static
# -----------------------------------------------------------------------
def test_regression():
    print("\n=== Test 1: Regression (zero squeeze = static) ===")
    from reynolds_solver import solve_reynolds
    from reynolds_solver.dynamic.squeeze import solve_reynolds_squeeze

    N = 200
    H, d_phi, d_Z, *_ = generate_grid(N, epsilon=0.6)

    P_static, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure="laminar", cavitation="half_sommerfeld",
        omega=1.5, tol=1e-6, max_iter=50000,
    )

    P_squeeze, _, _ = solve_reynolds_squeeze(
        H, d_phi, d_Z, R, L,
        v_x=0.0, v_y=0.0,
        omega=1.5, tol=1e-6, max_iter=50000,
    )

    max_diff = np.max(np.abs(P_squeeze - P_static))
    max_P = np.max(P_static)
    rel_diff = max_diff / (max_P + 1e-30)

    print(f"    max|P_squeeze - P_static| = {max_diff:.4e}")
    print(f"    rel_diff = {rel_diff:.4e}")

    return run_test(
        "Zero squeeze matches static",
        rel_diff < 1e-10,
        f"rel_diff = {rel_diff:.4e}",
    )


# -----------------------------------------------------------------------
# Test 2: RHS unit test — squeeze_to_api_params
# -----------------------------------------------------------------------
def test_rhs_unit():
    print("\n=== Test 2: RHS unit test (squeeze_to_api_params) ===")
    from reynolds_solver.dynamic.squeeze import squeeze_to_api_params

    c = 50e-6
    omega_shaft = 312.0
    d_phi = 2 * np.pi / 200
    v_x = 1.0   # 1 m/s in x
    v_y = 0.0
    beta = 2.0
    Lambda = 1.0

    xprime, yprime, beta_out = squeeze_to_api_params(
        v_x, v_y, c, omega_shaft, d_phi, beta, Lambda
    )

    # Expected: eps_dot_x = v_x / (c * omega) = 1.0 / (50e-6 * 312) ≈ 64.1
    eps_dot_x = v_x / (c * omega_shaft)

    # yprime = Lambda * d_phi² * eps_dot_x / beta
    yprime_expected = Lambda * d_phi**2 * eps_dot_x / beta

    # xprime should be 0 (no v_y)
    print(f"    eps_dot_x = {eps_dot_x:.2f}")
    print(f"    xprime = {xprime:.6e} (expected ~0)")
    print(f"    yprime = {yprime:.6e} (expected {yprime_expected:.6e})")
    print(f"    beta = {beta_out}")

    all_ok = True
    all_ok &= run_test(
        "xprime ≈ 0 for v_y=0",
        abs(xprime) < 1e-30,
        f"xprime = {xprime:.4e}",
    )
    all_ok &= run_test(
        "yprime matches formula",
        abs(yprime - yprime_expected) / (abs(yprime_expected) + 1e-30) < 1e-12,
        f"yprime = {yprime:.6e}, expected = {yprime_expected:.6e}",
    )
    all_ok &= run_test("beta passed through", beta_out == 2.0)
    return all_ok


# -----------------------------------------------------------------------
# Test 3: Smoke test — pure squeeze, no wedge
# -----------------------------------------------------------------------
def test_pure_squeeze():
    print("\n=== Test 3: Pure squeeze (H=const, dH/dt<0) ===")
    from reynolds_solver.dynamic.squeeze import solve_reynolds_squeeze

    N = 200
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]

    # Uniform gap (no wedge effect)
    H = np.ones((N, N))

    c = 50e-6
    omega_shaft = 312.0

    # Pure x-velocity (closing gap = negative v_x at phi=pi → ∂H/∂t < 0 there)
    # Actually for uniform squeeze: v_x closing means ε̇_x > 0,
    # ∂H/∂t = ε̇_x·cos(φ) > 0 at φ=0 and < 0 at φ=π
    # With half-Sommerfeld cavitation, pressure builds where gap closes (φ≈π)
    P, _, _ = solve_reynolds_squeeze(
        H, d_phi, d_Z, R, L,
        v_x=0.5, v_y=0.0,
        c=c, omega_shaft=omega_shaft,
        omega=1.5, tol=1e-6, max_iter=50000,
    )

    max_P = np.max(P)
    # P should be > 0 somewhere (squeeze creates pressure)
    print(f"    max(P) = {max_P:.6e}")

    # Check symmetry in Z (P should be symmetric about Z=0)
    mid = N // 2
    P_top = P[:mid, :]
    P_bot = P[mid:, :][::-1, :]
    z_sym = np.max(np.abs(P_top - P_bot)) / (max_P + 1e-30)
    print(f"    Z-symmetry error = {z_sym:.4e}")

    all_ok = True
    all_ok &= run_test(
        "Squeeze creates pressure (max(P) > 0)",
        max_P > 1e-8,
        f"max(P) = {max_P:.6e}",
    )
    all_ok &= run_test(
        "Z-symmetric (< 1%)",
        z_sym < 0.01,
        f"z_sym = {z_sym:.4e}",
    )
    return all_ok


# -----------------------------------------------------------------------
# Test 4: Sign test — closing vs opening
# -----------------------------------------------------------------------
def test_sign():
    print("\n=== Test 4: Sign test (squeeze opposes motion) ===")
    from reynolds_solver.dynamic.squeeze import solve_reynolds_squeeze

    N = 200
    # Pure squeeze: H=const (no wedge), so force is entirely from squeeze
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    H = np.ones((N, N))

    c = 50e-6
    omega_shaft = 312.0
    v_mag = 0.5

    # v_x > 0 → squeeze force Fx should be < 0 (opposes motion)
    P_px, _, _ = solve_reynolds_squeeze(
        H, d_phi, d_Z, R, L,
        v_x=v_mag, v_y=0.0,
        c=c, omega_shaft=omega_shaft,
        omega=1.5, tol=1e-6, max_iter=50000,
    )
    Fx_px = np.trapezoid(np.trapezoid(P_px * np.cos(Phi_mesh), phi_1D, axis=1), Z)

    # v_x < 0 → squeeze force Fx should be > 0 (opposes motion)
    P_mx, _, _ = solve_reynolds_squeeze(
        H, d_phi, d_Z, R, L,
        v_x=-v_mag, v_y=0.0,
        c=c, omega_shaft=omega_shaft,
        omega=1.5, tol=1e-6, max_iter=50000,
    )
    Fx_mx = np.trapezoid(np.trapezoid(P_mx * np.cos(Phi_mesh), phi_1D, axis=1), Z)

    print(f"    v_x=+{v_mag}: Fx = {Fx_px:.4e} (should be < 0)")
    print(f"    v_x=-{v_mag}: Fx = {Fx_mx:.4e} (should be > 0)")
    print(f"    Squeeze stronger at thin film: |Fx(+v)| > |Fx(-v)|: "
          f"{abs(Fx_px):.4e} vs {abs(Fx_mx):.4e}")

    all_ok = True
    all_ok &= run_test(
        "v_x>0 → Fx<0 (squeeze opposes +x motion)",
        Fx_px < 0,
        f"Fx = {Fx_px:.4e}",
    )
    all_ok &= run_test(
        "v_x<0 → Fx>0 (squeeze opposes -x motion)",
        Fx_mx > 0,
        f"Fx = {Fx_mx:.4e}",
    )
    return all_ok


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Squeeze-film tests")
    print("=" * 60)

    results = []
    results.append(test_regression())
    results.append(test_rhs_unit())
    results.append(test_pure_squeeze())
    results.append(test_sign())

    print("\n" + "=" * 60)
    all_ok = all(results)
    if all_ok:
        print("  ALL SQUEEZE TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
        for i, r in enumerate(results, 1):
            if not r:
                print(f"    Test {i} FAILED")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
