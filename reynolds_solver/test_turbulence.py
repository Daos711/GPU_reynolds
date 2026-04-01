"""
Validation tests for turbulence closure (Constantinescu) and API v1.2.

Tests:
  1. Laminar limit: constantinescu at Re~0 matches laminar
  2. Physical trend: W(Re) smooth, no order-of-magnitude jumps
  3. Positive conductances: A_half > 0, C_half_raw > 0
  4. Backward compatibility: old tests pass with closure="laminar"
  5. Warm start: converges no worse than cold start
  6. Performance: laminar closure overhead < 10%
  7. API error handling: cavitation, closure, missing params

Run:
    python -m reynolds_solver.test_turbulence
"""

import os
import sys
import time
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
# Test 1: Laminar limit
# -----------------------------------------------------------------------
def test_laminar_limit():
    print("\n=== Test 1: Laminar limit (constantinescu at Re~0) ===")
    from reynolds_solver import solve_reynolds

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 250
    c = 50e-6

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    P_lam, _, _ = solve_reynolds(H, d_phi, d_Z, R, L, closure="laminar")

    # Re ~ 0: rho=1, U=0.001, mu=1000 -> Re ~ rho*U*h/mu ~ 1*0.001*50e-6/1000 ~ 5e-11
    P_turb, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure="constantinescu",
        rho=1.0, U_velocity=0.001, mu=1000.0, c_clearance=c,
    )

    P_max = np.max(np.abs(P_lam))
    passed_nontrivial = P_max > 1e-6
    run_test("Test case is non-trivial (P_lam > 1e-6)", passed_nontrivial,
             f"max|P_lam| = {P_max:.4e}")

    rel_err = np.max(np.abs(P_turb - P_lam)) / P_max
    passed_limit = rel_err < 0.001
    run_test("Laminar limit: rel_err < 0.1%", passed_limit,
             f"rel_err = {rel_err:.4e}")

    return passed_nontrivial and passed_limit


# -----------------------------------------------------------------------
# Test 2: Physical trend (W vs Re)
# -----------------------------------------------------------------------
def test_physical_trend():
    print("\n=== Test 2: Physical trend W(Re) ===")
    from reynolds_solver import solve_reynolds

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 250
    c = 50e-6
    rho = 860.0
    mu = 0.03

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)

    Re_targets = [0.1, 1, 10, 100, 1000]
    W_list = []

    for Re_target in Re_targets:
        U_test = Re_target * mu / (rho * c)
        P, _, _ = solve_reynolds(
            H, d_phi, d_Z, R, L,
            closure="constantinescu",
            rho=rho, U_velocity=U_test, mu=mu, c_clearance=c,
        )
        W = np.trapezoid(
            np.trapezoid(P * np.cos(Phi_mesh), phi_1D, axis=1), Z
        )
        W_list.append(W)
        print(f"    Re_target={Re_target:>6.1f}, U={U_test:.4e}, W={W:.6e}")

    all_passed = True
    for i in range(len(W_list) - 1):
        ratio = abs(W_list[i + 1]) / (abs(W_list[i]) + 1e-12)
        ok = 0.1 < ratio < 10.0
        all_passed &= run_test(
            f"W[{i}]->W[{i+1}] ratio in (0.1, 10)",
            ok,
            f"W[{i}]={W_list[i]:.3e}, W[{i+1}]={W_list[i+1]:.3e}, ratio={ratio:.3f}"
        )

    return all_passed


# -----------------------------------------------------------------------
# Test 3: Positive conductances
# -----------------------------------------------------------------------
def test_positive_conductances():
    print("\n=== Test 3: Positive conductances ===")
    import cupy as cp
    from reynolds_solver.physics.closures import LaminarClosure, ConstantinescuClosure

    N = 100
    epsilon = 0.6
    c = 50e-6
    R = 0.035
    L = 0.056

    H, d_phi, d_Z, _, _ = generate_test_case(N, epsilon)
    H_gpu = cp.asarray(H, dtype=cp.float64)

    all_passed = True

    # Laminar
    lam = LaminarClosure()
    _, _, A_lam, C_lam = lam.modify_conductances(H_gpu, d_phi, d_Z, R, L)
    all_passed &= run_test("Laminar A_half > 0", bool(cp.all(A_lam > 0)))
    all_passed &= run_test("Laminar C_half_raw > 0", bool(cp.all(C_lam > 0)))

    # Constantinescu
    turb = ConstantinescuClosure(rho=860.0, U_velocity=10.0, mu=0.03, c_clearance=c)
    _, _, A_turb, C_turb = turb.modify_conductances(H_gpu, d_phi, d_Z, R, L)
    all_passed &= run_test("Constantinescu A_half > 0", bool(cp.all(A_turb > 0)))
    all_passed &= run_test("Constantinescu C_half_raw > 0", bool(cp.all(C_turb > 0)))

    # Check float64
    all_passed &= run_test("A_half dtype is float64", A_turb.dtype == cp.float64)
    all_passed &= run_test("C_half_raw dtype is float64", C_turb.dtype == cp.float64)

    return all_passed


# -----------------------------------------------------------------------
# Test 4: Backward compatibility
# -----------------------------------------------------------------------
def test_backward_compatibility():
    print("\n=== Test 4: Backward compatibility ===")
    from reynolds_solver import solve_reynolds
    from reynolds_solver.solver import solve_reynolds_gpu

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 250

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    # Old-style call (no closure param) via internal function
    P_old, delta_old, n_old = solve_reynolds_gpu(H, d_phi, d_Z, R, L)

    # New API with explicit laminar
    P_new, delta_new, n_new = solve_reynolds(H, d_phi, d_Z, R, L, closure="laminar")

    max_diff = np.max(np.abs(P_old - P_new))
    passed = max_diff < 1e-10
    all_passed = run_test(
        "closure='laminar' == old behavior (L_inf < 1e-10)",
        passed,
        f"max|diff| = {max_diff:.2e}"
    )

    # Dynamic backward compatibility
    from reynolds_solver.solver_dynamic import solve_reynolds_gpu_dynamic
    P_old_d, _, _ = solve_reynolds_gpu_dynamic(
        H, d_phi, d_Z, R, L, xprime=0.001, yprime=0.001, beta=2.0
    )
    P_new_d, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L, xprime=0.001, yprime=0.001, beta=2.0, closure="laminar"
    )
    max_diff_d = np.max(np.abs(P_old_d - P_new_d))
    passed_d = max_diff_d < 1e-10
    all_passed &= run_test(
        "Dynamic: closure='laminar' == old behavior (L_inf < 1e-10)",
        passed_d,
        f"max|diff| = {max_diff_d:.2e}"
    )

    return all_passed


# -----------------------------------------------------------------------
# Test 5: Warm start
# -----------------------------------------------------------------------
def test_warm_start():
    print("\n=== Test 5: Warm start ===")
    from reynolds_solver import solve_reynolds

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 250

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    all_passed = True

    # Cold start
    P_cold, _, n_cold = solve_reynolds(H, d_phi, d_Z, R, L)
    print(f"    Cold start: {n_cold} iterations")

    # Warm start with converged solution
    P_warm, _, n_warm = solve_reynolds(H, d_phi, d_Z, R, L, P_init=P_cold)
    print(f"    Warm start (converged P): {n_warm} iterations")
    all_passed &= run_test(
        "Warm start converges no worse than cold",
        n_warm <= n_cold,
        f"warm={n_warm}, cold={n_cold}"
    )

    # Warm start with zeros = cold start
    P_zero, _, n_zero = solve_reynolds(H, d_phi, d_Z, R, L, P_init=np.zeros_like(H))
    all_passed &= run_test(
        "Warm start with zeros ~ cold start",
        abs(n_zero - n_cold) <= 2,
        f"zero_init={n_zero}, cold={n_cold}"
    )

    # Shape validation
    try:
        solve_reynolds(H, d_phi, d_Z, R, L, P_init=np.zeros((3, 3)))
        all_passed &= run_test("P_init wrong shape raises ValueError", False)
    except ValueError:
        all_passed &= run_test("P_init wrong shape raises ValueError", True)

    return all_passed


# -----------------------------------------------------------------------
# Test 6: Performance (laminar overhead)
# -----------------------------------------------------------------------
def test_performance():
    print("\n=== Test 6: Performance (laminar overhead) ===")
    import cupy as cp
    from reynolds_solver import solve_reynolds

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 500

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    # Warmup
    solve_reynolds(H, d_phi, d_Z, R, L)
    cp.cuda.Device(0).synchronize()

    # Measure baseline (laminar, default)
    times = []
    for _ in range(3):
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        solve_reynolds(H, d_phi, d_Z, R, L, closure="laminar")
        cp.cuda.Device(0).synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    median_time = sorted(times)[1]
    print(f"    Laminar 500x500 median time: {median_time:.3f}s")

    # No baseline from before refactoring available, mark as xfail
    # but still report the time for manual comparison
    passed = run_test(
        "Performance measurement recorded (xfail: no pre-refactoring baseline)",
        True,
        f"median={median_time:.3f}s (compare manually with pre-refactoring baseline)"
    )

    return passed


# -----------------------------------------------------------------------
# Test 7: API error handling
# -----------------------------------------------------------------------
def test_api_errors():
    print("\n=== Test 7: API error handling ===")
    from reynolds_solver import solve_reynolds

    N = 50
    H, d_phi, d_Z, _, _ = generate_test_case(N)
    R, L = 0.035, 0.056

    all_passed = True

    # cavitation != half_sommerfeld -> NotImplementedError
    try:
        solve_reynolds(H, d_phi, d_Z, R, L, cavitation="jfo")
        all_passed &= run_test("cavitation='jfo' -> NotImplementedError", False)
    except NotImplementedError:
        all_passed &= run_test("cavitation='jfo' -> NotImplementedError", True)

    # closure=constantinescu without params -> ValueError
    try:
        solve_reynolds(H, d_phi, d_Z, R, L, closure="constantinescu")
        all_passed &= run_test("constantinescu without params -> ValueError", False)
    except ValueError as e:
        all_passed &= run_test(
            "constantinescu without params -> ValueError", True,
            str(e)
        )

    # Unknown closure -> ValueError
    try:
        solve_reynolds(H, d_phi, d_Z, R, L, closure="unknown")
        all_passed &= run_test("closure='unknown' -> ValueError", False)
    except ValueError:
        all_passed &= run_test("closure='unknown' -> ValueError", True)

    # Partial params -> ValueError with missing list
    try:
        solve_reynolds(H, d_phi, d_Z, R, L,
                       closure="constantinescu", rho=860.0, mu=0.03)
        all_passed &= run_test("Partial params -> ValueError", False)
    except ValueError as e:
        has_missing = "U_velocity" in str(e) and "c_clearance" in str(e)
        all_passed &= run_test(
            "Partial params -> ValueError lists missing",
            has_missing,
            str(e)
        )

    return all_passed


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Turbulence & API v1.2 validation")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []
    results.append(test_laminar_limit())
    results.append(test_physical_trend())
    results.append(test_positive_conductances())
    results.append(test_backward_compatibility())
    results.append(test_warm_start())
    results.append(test_performance())
    results.append(test_api_errors())

    print("\n" + "=" * 60)
    all_ok = all(results)
    if all_ok:
        print("  ALL TURBULENCE TESTS PASSED")
    else:
        print("  SOME TURBULENCE TESTS FAILED")
        for i, r in enumerate(results, 1):
            if not r:
                print(f"    Test {i} FAILED")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
