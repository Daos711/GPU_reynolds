"""
Tests for piezoviscous + Payvar-Salant combined solver.

Tests:
  A. alpha_pv=0 matches plain PS
  B. Tiny alpha_pv ≈ plain PS (load integral)
  C. Pump bearing convergence (mineral oil, alpha=18e-9)
  D. Physical invariants after PV+PS
  E. coefficients_ext input validation

Run:
    python -m reynolds_solver.tests.test_pv_payvar_salant
"""
import sys
import numpy as np


def run_test(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if details:
        print(f"         {details}")
    return passed


def generate_test_case(N_phi, N_Z, epsilon=0.6):
    phi_1D = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)
    Phi, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z, phi_1D, Z


R, L = 0.035, 0.056


# -----------------------------------------------------------------------
# Test A: alpha=0 → same as plain PS
# -----------------------------------------------------------------------
def test_pv_ps_alpha_zero():
    print("\n=== Test A: alpha_pv=0 matches plain PS ===")
    from reynolds_solver.cavitation.payvar_salant import (
        solve_payvar_salant_gpu,
    )
    from reynolds_solver.piezoviscous.solver_pv_payvar_salant import (
        solve_payvar_salant_piezoviscous,
    )

    N_phi, N_Z = 100, 40
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.6)

    P_ps, th_ps, _, _ = solve_payvar_salant_gpu(
        H, d_phi, d_Z, R, L, tol=1e-7,
    )
    P_pv, th_pv, _, _ = solve_payvar_salant_piezoviscous(
        H, d_phi, d_Z, R, L,
        alpha_pv=0.0, p_scale=1e6, tol=1e-7,
    )

    dP = float(np.max(np.abs(P_pv - P_ps)))
    dth = float(np.max(np.abs(th_pv - th_ps)))
    print(f"    max|ΔP|={dP:.2e}, max|Δθ|={dth:.2e}")

    return run_test(
        "alpha=0: PV+PS == PS",
        dP < 1e-8 and dth < 1e-8,
        f"dP={dP:.2e}, dth={dth:.2e}",
    )


# -----------------------------------------------------------------------
# Test B: tiny alpha → ≈ plain PS
# -----------------------------------------------------------------------
def test_pv_ps_tiny_alpha():
    print("\n=== Test B: tiny alpha_pv ≈ plain PS ===")
    from reynolds_solver.cavitation.payvar_salant import (
        solve_payvar_salant_gpu,
    )
    from reynolds_solver.piezoviscous.solver_pv_payvar_salant import (
        solve_payvar_salant_piezoviscous,
    )

    N_phi, N_Z = 100, 40
    epsilon = 0.6
    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N_phi, N_Z, epsilon)
    Phi, _ = np.meshgrid(phi_1D, Z)

    P_ps, th_ps, _, _ = solve_payvar_salant_gpu(
        H, d_phi, d_Z, R, L, tol=1e-7,
    )
    P_pv, th_pv, _, _ = solve_payvar_salant_piezoviscous(
        H, d_phi, d_Z, R, L,
        alpha_pv=1e-15, p_scale=1e6, tol=1e-7,
    )

    W_ps = float(np.trapezoid(
        np.trapezoid(P_ps * np.cos(Phi), phi_1D, axis=1), Z
    ))
    W_pv = float(np.trapezoid(
        np.trapezoid(P_pv * np.cos(Phi), phi_1D, axis=1), Z
    ))
    rel_W = abs(W_pv - W_ps) / (abs(W_ps) + 1e-30)
    print(f"    W_ps={W_ps:.4e}, W_pv={W_pv:.4e}, rel={rel_W:.2e}")

    return run_test(
        "tiny alpha: load within 1% of plain PS",
        rel_W < 0.01,
        f"rel_W={rel_W:.2e}",
    )


# -----------------------------------------------------------------------
# Test C: pump bearing convergence
# -----------------------------------------------------------------------
def test_pv_ps_pump_convergence():
    print("\n=== Test C: pump bearing PV+PS convergence ===")
    from reynolds_solver.piezoviscous.solver_pv_payvar_salant import (
        solve_payvar_salant_piezoviscous,
    )

    N_phi, N_Z = 100, 40
    epsilon = 0.6
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon)

    # Pump bearing: R=35mm, c=50μm, n=3000rpm, η=0.022 Pa·s, α=18e-9 Pa⁻¹
    c = 50e-6
    n_rpm = 3000
    eta = 0.022
    omega_shaft = 2 * np.pi * n_rpm / 60
    p_scale = 6 * eta * omega_shaft * (R / c) ** 2

    P, theta, res, n_iter, diag = solve_payvar_salant_piezoviscous(
        H, d_phi, d_Z, R, L,
        alpha_pv=18e-9, p_scale=p_scale,
        verbose=True, return_diagnostics=True,
    )

    print(f"    n_outer={diag['n_outer']}, converged={diag['converged']}, "
          f"dP_rel={diag['dP_rel_final']:.2e}, mu_max={diag['mu_max']:.3f}")
    print(f"    maxP={P.max():.4e}, cav={np.mean(theta[1:-1,1:-1]<1-1e-6):.3f}")

    return run_test(
        "Pump: PV+PS converges within 30 outer iterations",
        diag["converged"] and diag["n_outer"] <= 30,
        f"n_outer={diag['n_outer']}, dP_rel={diag['dP_rel_final']:.2e}",
    )


# -----------------------------------------------------------------------
# Test D: physical invariants
# -----------------------------------------------------------------------
def test_pv_ps_invariants():
    print("\n=== Test D: PV+PS physical invariants ===")
    from reynolds_solver.piezoviscous.solver_pv_payvar_salant import (
        solve_payvar_salant_piezoviscous,
    )

    N_phi, N_Z = 100, 40
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.6)

    c = 50e-6
    eta = 0.022
    omega_shaft = 2 * np.pi * 3000 / 60
    p_scale = 6 * eta * omega_shaft * (R / c) ** 2

    P, theta, _, _ = solve_payvar_salant_piezoviscous(
        H, d_phi, d_Z, R, L,
        alpha_pv=18e-9, p_scale=p_scale,
    )

    p_ok = np.all(P >= -1e-15)
    th_ok = np.all(theta >= -1e-15) and np.all(theta <= 1.0 + 1e-15)
    not_collapsed = P.max() > 1e-3

    print(f"    P=[{P.min():.2e}, {P.max():.4e}], "
          f"θ=[{theta.min():.4f}, {theta.max():.4f}]")

    return run_test(
        "PV+PS invariants: P≥0, θ∈[0,1], not collapsed",
        p_ok and th_ok and not_collapsed,
        f"P_min={P.min():.2e}, P_max={P.max():.2e}, "
        f"θ_min={theta.min():.4f}",
    )


# -----------------------------------------------------------------------
# Test E: coefficients_ext validation
# -----------------------------------------------------------------------
def test_coefficients_ext_validation():
    print("\n=== Test E: coefficients_ext input validation ===")
    import cupy as cp
    from reynolds_solver.cavitation.payvar_salant.solver_gpu import (
        solve_payvar_salant_gpu,
        _build_coefficients_gpu,
    )

    N_phi, N_Z = 50, 30
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.3)
    H_np = H.copy()
    H_np[:, 0] = H_np[:, -2]
    H_np[:, -1] = H_np[:, 1]
    H_gpu = cp.asarray(H_np)
    A, B, C, D, E = _build_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)

    all_ok = True

    # Incomplete tuple
    try:
        solve_payvar_salant_gpu(
            H, d_phi, d_Z, R, L, coefficients_ext=(A, B, C),
        )
        print("    incomplete tuple: no error (FAIL)")
        all_ok = False
    except ValueError:
        print("    incomplete tuple: ValueError raised (OK)")

    # numpy instead of cupy
    try:
        solve_payvar_salant_gpu(
            H, d_phi, d_Z, R, L,
            coefficients_ext=(
                cp.asnumpy(A), cp.asnumpy(B), cp.asnumpy(C),
                cp.asnumpy(D), cp.asnumpy(E),
            ),
        )
        print("    numpy arrays: no error (FAIL)")
        all_ok = False
    except TypeError:
        print("    numpy arrays: TypeError raised (OK)")

    # Wrong shape
    try:
        A_wrong = cp.zeros((10, 10), dtype=cp.float64)
        solve_payvar_salant_gpu(
            H, d_phi, d_Z, R, L,
            coefficients_ext=(A_wrong, B, C, D, E),
        )
        print("    wrong shape: no error (FAIL)")
        all_ok = False
    except ValueError:
        print("    wrong shape: ValueError raised (OK)")

    return run_test("coefficients_ext validation", all_ok)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Piezoviscous + Payvar-Salant tests")
    print("=" * 60)

    tests = [
        ("A", test_pv_ps_alpha_zero),
        ("B", test_pv_ps_tiny_alpha),
        ("C", test_pv_ps_pump_convergence),
        ("D", test_pv_ps_invariants),
        ("E", test_coefficients_ext_validation),
    ]

    results = []
    for name, func in tests:
        try:
            results.append((name, func()))
        except Exception as e:
            print(f"  [FAIL] Test {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print()
    print("=" * 60)
    all_ok = all(r for _, r in results)
    if all_ok:
        print("  ALL PV+PS TESTS PASSED")
    else:
        print("  SOME PV+PS TESTS FAILED")
        for name, r in results:
            if not r:
                print(f"    Test {name} FAILED")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
