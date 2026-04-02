"""
Validation tests for JFO cavitation model.

Tests:
  0a. F_theta == F_orig when theta=1 (invariant check, no GPU solver needed)
  0b. Frozen theta=1: JFO == HS (update_mask=False, run_theta_sweep=False)
  1.  Mass conservativity (discrete flux balance)
  2.  P >= 0 everywhere
  3.  theta in [0, 1] everywhere
  4.  In active zone theta == 1
  5.  Symmetric case (epsilon=0): P ~ 0
  6.  Cavitation zone non-empty at epsilon=0.6
  7.  Warm start no worse than cold start
  8.  Backward compatibility (old tests unaffected)
  9.  JFO ~ Half-Sommerfeld at small epsilon (loads within 5%)

Run:
    python -m reynolds_solver.test_jfo
"""

import os
import sys
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_test(test_name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}")
    if details:
        print(f"         {details}")
    return passed


def generate_test_case(N, epsilon=0.6):
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z, phi_1D, Z


# -----------------------------------------------------------------------
# Test 0a: F_theta == F_orig when theta=1
# -----------------------------------------------------------------------
def test_f_theta_equals_f_orig():
    print("\n=== Test 0a: F_theta == F_orig when theta=1 ===")
    import cupy as cp
    from reynolds_solver.utils import precompute_coefficients_gpu, build_F_theta_gpu

    R, L = 0.035, 0.056
    N = 250
    epsilon = 0.6

    H, d_phi, d_Z, _, _ = generate_test_case(N, epsilon)
    H_gpu = cp.asarray(H, dtype=cp.float64)

    _, _, _, _, _, F_orig = precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)

    theta_ones = cp.ones_like(H_gpu)
    F_theta = build_F_theta_gpu(H_gpu, theta_ones, d_phi)

    max_diff = float(cp.max(cp.abs(F_theta - F_orig)))
    print(f"    max|F_theta - F_orig| = {max_diff:.4e}")

    return run_test(
        "F_theta == F_orig when theta=1 (< 1e-12)",
        max_diff < 1e-12,
        f"max_diff = {max_diff:.4e}"
    )


# -----------------------------------------------------------------------
# Test 0b: Frozen theta=1 -> JFO == HS
# -----------------------------------------------------------------------
def test_frozen_theta_jfo_equals_hs():
    print("\n=== Test 0b: Frozen theta=1 -> JFO == HS ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N = 250
    epsilon = 0.6

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="half_sommerfeld")

    P_jfo, theta, residual, n_outer, n_inner = solve_reynolds(
        H, d_phi, d_Z, R, L, cavitation="jfo",
        update_mask=False, run_theta_sweep=False,
        verbose=True,
    )

    # theta should remain all-ones
    theta_err = np.max(np.abs(theta - 1.0))

    # P should match HS solution
    max_P = max(np.max(np.abs(P_hs)), 1e-30)
    p_diff = np.max(np.abs(P_jfo - P_hs)) / max_P

    print(f"    max|theta - 1| = {theta_err:.4e}")
    print(f"    max|P_jfo - P_hs| / max|P_hs| = {p_diff:.4e}")

    all_passed = True
    all_passed &= run_test(
        "Frozen theta stays 1.0",
        theta_err < 1e-12,
        f"max|theta - 1| = {theta_err:.4e}"
    )
    all_passed &= run_test(
        "Frozen JFO pressure matches HS (< 1%)",
        p_diff < 0.01,
        f"rel_diff = {p_diff:.4e}"
    )
    return all_passed


# -----------------------------------------------------------------------
# Test 1: Mass conservativity
# -----------------------------------------------------------------------
def test_mass_conservativity():
    print("\n=== Test 1: Mass conservativity ===")
    from reynolds_solver import solve_reynolds

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 250

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    P, theta, residual, n_outer, n_inner = solve_reynolds(
        H, d_phi, d_Z, R, L, cavitation="jfo", verbose=True,
    )
    print(f"    Converged: {n_outer} outer, {n_inner} inner total, residual={residual:.2e}")

    # Discrete flux: Q = H*theta - 0.5*H^3*dP/dphi
    # Interior faces
    H_face = 0.5 * (H[:, :-1] + H[:, 1:])
    theta_face = 0.5 * (theta[:, :-1] + theta[:, 1:])
    dP_face = (P[:, 1:] - P[:, :-1]) / d_phi
    Q_inner = H_face * theta_face - 0.5 * H_face ** 3 * dP_face

    # Wrap-around face: column -1 -> column 0
    H_wrap = 0.5 * (H[:, -1] + H[:, 0])
    theta_wrap = 0.5 * (theta[:, -1] + theta[:, 0])
    dP_wrap = (P[:, 0] - P[:, -1]) / d_phi
    Q_wrap = (H_wrap * theta_wrap - 0.5 * H_wrap ** 3 * dP_wrap)[:, None]

    Q_all = np.concatenate([Q_inner, Q_wrap], axis=1)
    Q_integrated = np.sum(Q_all, axis=0) * d_Z

    mass_err = (np.max(Q_integrated) - np.min(Q_integrated)) / (np.mean(np.abs(Q_integrated)) + 1e-12)
    print(f"    Mass error (relative flux variation): {mass_err:.4e}")

    return run_test(
        "Mass conservativity: flux variation < 1%",
        mass_err < 0.01,
        f"mass_err = {mass_err:.4e}"
    )


# -----------------------------------------------------------------------
# Test 2: P >= 0 everywhere
# -----------------------------------------------------------------------
def test_pressure_nonneg():
    print("\n=== Test 2: P >= 0 everywhere ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N = 250

    H, d_phi, d_Z, _, _ = generate_test_case(N, epsilon=0.6)
    P, theta, *_ = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo")

    p_min = np.min(P)
    return run_test("P >= 0 everywhere", p_min >= 0.0, f"min(P) = {p_min:.4e}")


# -----------------------------------------------------------------------
# Test 3: theta in [0, 1] everywhere
# -----------------------------------------------------------------------
def test_theta_bounds():
    print("\n=== Test 3: theta in [0, 1] everywhere ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N = 250

    H, d_phi, d_Z, _, _ = generate_test_case(N, epsilon=0.6)
    _, theta, *_ = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo")

    t_min = np.min(theta)
    t_max = np.max(theta)
    passed = t_min >= 0.0 and t_max <= 1.0
    return run_test(
        "theta in [0, 1]",
        passed,
        f"min(theta) = {t_min:.4e}, max(theta) = {t_max:.4e}"
    )


# -----------------------------------------------------------------------
# Test 4: In active zone theta == 1
# -----------------------------------------------------------------------
def test_theta_active_zone():
    print("\n=== Test 4: In active zone theta == 1 ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N = 250

    H, d_phi, d_Z, _, _ = generate_test_case(N, epsilon=0.6)
    P, theta, *_ = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo")

    active = P > 0
    if np.any(active):
        theta_active = theta[active]
        all_one = np.allclose(theta_active, 1.0)
        return run_test(
            "theta == 1 in active zone",
            all_one,
            f"min(theta[active]) = {np.min(theta_active):.6f}, "
            f"max(theta[active]) = {np.max(theta_active):.6f}"
        )
    else:
        return run_test("theta == 1 in active zone", False, "No active zone found")


# -----------------------------------------------------------------------
# Test 5: Symmetric case (epsilon=0)
# -----------------------------------------------------------------------
def test_symmetric_case():
    print("\n=== Test 5: Symmetric case (epsilon=0) ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N = 100

    H, d_phi, d_Z, _, _ = generate_test_case(N, epsilon=0.0)

    P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="half_sommerfeld")
    P_jfo, theta, *_ = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo")

    all_passed = True
    all_passed &= run_test(
        "HS: P ~ 0 for uniform gap",
        np.max(np.abs(P_hs)) < 1e-8,
        f"max|P_hs| = {np.max(np.abs(P_hs)):.2e}"
    )
    all_passed &= run_test(
        "JFO: P ~ 0 for uniform gap",
        np.max(np.abs(P_jfo)) < 1e-8,
        f"max|P_jfo| = {np.max(np.abs(P_jfo)):.2e}"
    )
    return all_passed


# -----------------------------------------------------------------------
# Test 6: Cavitation zone non-empty at epsilon=0.6
# -----------------------------------------------------------------------
def test_cavitation_zone_nonempty():
    print("\n=== Test 6: Cavitation zone non-empty (epsilon=0.6) ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N = 250

    H, d_phi, d_Z, _, _ = generate_test_case(N, epsilon=0.6)
    P, theta, *_ = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo")

    cav_frac = np.mean(P == 0)
    passed = 0.05 < cav_frac < 0.95
    return run_test(
        "Cavitation fraction in (5%, 95%)",
        passed,
        f"cavitation_fraction = {cav_frac:.3f} ({cav_frac*100:.1f}%)"
    )


# -----------------------------------------------------------------------
# Test 7: Warm start no worse than cold
# -----------------------------------------------------------------------
def test_warm_start():
    print("\n=== Test 7: Warm start ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N = 250

    H, d_phi, d_Z, _, _ = generate_test_case(N, epsilon=0.6)

    # Cold start
    P_cold, theta_cold, _, n_out_cold, _ = solve_reynolds(
        H, d_phi, d_Z, R, L, cavitation="jfo",
    )
    print(f"    Cold start: {n_out_cold} outer iterations")

    # Warm start with converged state
    mask_cold = (P_cold > 0).astype(np.int32)
    P_warm, _, _, n_out_warm, _ = solve_reynolds(
        H, d_phi, d_Z, R, L, cavitation="jfo",
        P_init=P_cold, theta_init=theta_cold, mask_init=mask_cold,
    )
    print(f"    Warm start: {n_out_warm} outer iterations")

    all_passed = True
    all_passed &= run_test(
        "Warm start converges no worse",
        n_out_warm <= n_out_cold,
        f"warm={n_out_warm}, cold={n_out_cold}"
    )

    # Shape validation
    try:
        solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo",
                       P_init=np.zeros((3, 3)))
        all_passed &= run_test("P_init wrong shape -> ValueError", False)
    except ValueError:
        all_passed &= run_test("P_init wrong shape -> ValueError", True)

    # theta_init out of range
    try:
        solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo",
                       theta_init=np.full_like(H, 2.0))
        all_passed &= run_test("theta_init > 1 -> ValueError", False)
    except ValueError:
        all_passed &= run_test("theta_init > 1 -> ValueError", True)

    # mask_init bad values
    try:
        solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo",
                       mask_init=np.full(H.shape, 5, dtype=np.int32))
        all_passed &= run_test("mask_init bad values -> ValueError", False)
    except ValueError:
        all_passed &= run_test("mask_init bad values -> ValueError", True)

    return all_passed


# -----------------------------------------------------------------------
# Test 8: Backward compatibility
# -----------------------------------------------------------------------
def test_backward_compatibility():
    print("\n=== Test 8: Backward compatibility ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N = 100

    H, d_phi, d_Z, _, _ = generate_test_case(N, epsilon=0.6)

    all_passed = True

    # Half-Sommerfeld still returns 3-tuple
    result_hs = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="half_sommerfeld")
    all_passed &= run_test(
        "HS returns 3-tuple",
        len(result_hs) == 3,
        f"got {len(result_hs)} elements"
    )

    # JFO returns 5-tuple
    result_jfo = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo")
    all_passed &= run_test(
        "JFO returns 5-tuple",
        len(result_jfo) == 5,
        f"got {len(result_jfo)} elements"
    )

    # JFO + non-laminar -> NotImplementedError
    try:
        solve_reynolds(H, d_phi, d_Z, R, L,
                       cavitation="jfo", closure="constantinescu",
                       rho=860.0, U_velocity=10.0, mu=0.03, c_clearance=50e-6)
        all_passed &= run_test("JFO + constantinescu -> NotImplementedError", False)
    except NotImplementedError:
        all_passed &= run_test("JFO + constantinescu -> NotImplementedError", True)

    # Unknown cavitation -> NotImplementedError
    try:
        solve_reynolds(H, d_phi, d_Z, R, L, cavitation="reynolds")
        all_passed &= run_test("Unknown cavitation -> NotImplementedError", False)
    except NotImplementedError:
        all_passed &= run_test("Unknown cavitation -> NotImplementedError", True)

    return all_passed


# -----------------------------------------------------------------------
# Test 9: JFO ~ Half-Sommerfeld at small epsilon
# -----------------------------------------------------------------------
def test_jfo_vs_hs_small_epsilon():
    print("\n=== Test 9: JFO ~ HS at small epsilon ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N = 250
    epsilon = 0.1

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)

    P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="half_sommerfeld")
    P_jfo, theta, *_ = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo")

    W_hs = np.trapezoid(np.trapezoid(P_hs * np.cos(Phi_mesh), phi_1D, axis=1), Z)
    W_jfo = np.trapezoid(np.trapezoid(P_jfo * np.cos(Phi_mesh), phi_1D, axis=1), Z)

    rel_diff = abs(W_jfo - W_hs) / (abs(W_hs) + 1e-12)
    print(f"    W_hs = {W_hs:.6e}, W_jfo = {W_jfo:.6e}, rel_diff = {rel_diff:.4f}")

    return run_test(
        "JFO and HS loads within 5% at epsilon=0.1",
        rel_diff < 0.05,
        f"rel_diff = {rel_diff:.4f}"
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  JFO cavitation model validation")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    results.append(test_f_theta_equals_f_orig())
    results.append(test_frozen_theta_jfo_equals_hs())
    results.append(test_mass_conservativity())
    results.append(test_pressure_nonneg())
    results.append(test_theta_bounds())
    results.append(test_theta_active_zone())
    results.append(test_symmetric_case())
    results.append(test_cavitation_zone_nonempty())
    results.append(test_warm_start())
    results.append(test_backward_compatibility())
    results.append(test_jfo_vs_hs_small_epsilon())

    print("\n" + "=" * 60)
    all_ok = all(results)
    if all_ok:
        print("  ALL JFO TESTS PASSED")
    else:
        print("  SOME JFO TESTS FAILED")
        for i, r in enumerate(results, 1):
            if not r:
                print(f"    Test {i} FAILED")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
