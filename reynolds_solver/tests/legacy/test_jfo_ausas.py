"""
Validation tests for Ausas-style mass-conserving JFO solver.

Tests:
  0. CPU reference: ε=0.6, smooth bearing, sanity check
  1. JFO ≈ HS at small ε (loads within 5%)
  2. Symmetric case (ε=0): P ≈ 0, θ ≈ 1
  3. P ≥ 0 and 0 ≤ θ ≤ 1 invariants
  4. GPU vs CPU reference (smooth bearing, ε=0.6)
  5. Stable convergence: residual decreases monotonically (informational)

Run:
    python -m reynolds_solver.test_jfo_ausas
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


# -----------------------------------------------------------------------
# Test 0: CPU reference works on smooth bearing ε=0.6
# -----------------------------------------------------------------------
def test_cpu_reference():
    print("\n=== Test 0: CPU reference (ε=0.6) ===")
    from reynolds_solver.cavitation.ausas.solver_cpu import solve_jfo_ausas_cpu

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.6)

    P, theta, residual, n_iter = solve_jfo_ausas_cpu(
        H, d_phi, d_Z, R, L,
        omega_p=1.0, omega_theta=1.0,
        tol=1e-6, max_iter=20000, check_every=200,
        verbose=False,
    )

    cav_frac = float(np.mean(theta < 1.0 - 1e-6))
    p_min, p_max = float(np.min(P)), float(np.max(P))
    th_min, th_max = float(np.min(theta)), float(np.max(theta))

    print(f"    n_iter={n_iter}, residual={residual:.2e}, cav_frac={cav_frac:.3f}")
    print(f"    P range: [{p_min:.4e}, {p_max:.4e}]")
    print(f"    theta range: [{th_min:.4e}, {th_max:.4e}]")

    converged = residual < 1e-3
    physical = p_min >= -1e-12 and p_max > 0 and 0.0 <= th_min and th_max <= 1.0
    cav_reasonable = 0.1 < cav_frac < 0.6

    return run_test(
        "CPU reference: converges with physical solution",
        converged and physical and cav_reasonable,
        f"converged={converged}, physical={physical}, cav_reasonable={cav_reasonable}",
    )


# -----------------------------------------------------------------------
# Test 1: JFO_ausas ≈ HS at small ε
# -----------------------------------------------------------------------
def test_ausas_vs_hs_small_epsilon():
    print("\n=== Test 1: JFO_ausas ≈ HS at ε=0.1 ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N_phi, N_Z = 250, 250
    epsilon = 0.1

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N_phi, N_Z, epsilon)
    Phi, _ = np.meshgrid(phi_1D, Z)

    P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="half_sommerfeld")
    P_a, theta_a, residual, n_iter = solve_reynolds(
        H, d_phi, d_Z, R, L, cavitation="jfo_ausas",
        jfo_ausas_max_iter=20000, tol=1e-6,
    )

    print(f"    n_iter={n_iter}, residual={residual:.2e}")

    W_hs = np.trapezoid(np.trapezoid(P_hs * np.cos(Phi), phi_1D, axis=1), Z)
    W_a = np.trapezoid(np.trapezoid(P_a * np.cos(Phi), phi_1D, axis=1), Z)
    rel_diff = abs(W_a - W_hs) / (abs(W_hs) + 1e-12)

    print(f"    W_hs={W_hs:.6e}, W_ausas={W_a:.6e}, rel_diff={rel_diff:.4f}")
    return run_test(
        "JFO_ausas ≈ HS at ε=0.1 (rel_diff < 5%)",
        rel_diff < 0.05,
        f"rel_diff={rel_diff:.4f}",
    )


# -----------------------------------------------------------------------
# Test 2: Symmetric case (ε=0)
# -----------------------------------------------------------------------
def test_symmetric():
    print("\n=== Test 2: Symmetric case (ε=0) ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.0)

    P, theta, residual, n_iter = solve_reynolds(
        H, d_phi, d_Z, R, L, cavitation="jfo_ausas",
        jfo_ausas_max_iter=10000, tol=1e-8,
    )

    p_max = float(np.max(np.abs(P)))
    th_min = float(np.min(theta))

    print(f"    n_iter={n_iter}, residual={residual:.2e}")
    print(f"    max|P|={p_max:.2e}, min(theta)={th_min:.4f}")

    p_zero = p_max < 1e-6
    th_one = th_min > 1.0 - 1e-6
    return run_test(
        "Symmetric: P ≈ 0 and theta ≈ 1",
        p_zero and th_one,
        f"max|P|={p_max:.2e}, min(theta)={th_min:.4f}",
    )


# -----------------------------------------------------------------------
# Test 3: Physical invariants (P >= 0, theta in [0,1])
# -----------------------------------------------------------------------
def test_invariants():
    print("\n=== Test 3: Physical invariants ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N_phi, N_Z = 250, 100
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.6)

    P, theta, residual, n_iter = solve_reynolds(
        H, d_phi, d_Z, R, L, cavitation="jfo_ausas",
        jfo_ausas_max_iter=20000, tol=1e-6,
    )
    print(f"    n_iter={n_iter}, residual={residual:.2e}")

    p_min = float(np.min(P))
    th_min = float(np.min(theta))
    th_max = float(np.max(theta))

    p_ok = p_min >= -1e-12
    th_ok = th_min >= -1e-12 and th_max <= 1.0 + 1e-12

    print(f"    min(P)={p_min:.4e}, theta range=[{th_min:.4e}, {th_max:.4e}]")
    return run_test(
        "P >= 0 and theta in [0, 1]",
        p_ok and th_ok,
        f"p_min={p_min:.2e}, th=[{th_min:.4f},{th_max:.4f}]",
    )


# -----------------------------------------------------------------------
# Test 4: GPU vs CPU reference (ε=0.6, smooth bearing)
# -----------------------------------------------------------------------
def test_gpu_vs_cpu():
    print("\n=== Test 4: GPU vs CPU reference (ε=0.6) ===")
    from reynolds_solver import solve_reynolds
    from reynolds_solver.cavitation.ausas.solver_cpu import solve_jfo_ausas_cpu

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N_phi, N_Z, epsilon=0.6)
    Phi, _ = np.meshgrid(phi_1D, Z)

    # CPU reference
    P_cpu, theta_cpu, res_cpu, n_cpu = solve_jfo_ausas_cpu(
        H, d_phi, d_Z, R, L,
        omega_p=1.0, omega_theta=1.0,
        tol=1e-7, max_iter=30000, check_every=500, verbose=False,
    )

    # GPU
    P_gpu, theta_gpu, res_gpu, n_gpu = solve_reynolds(
        H, d_phi, d_Z, R, L, cavitation="jfo_ausas",
        jfo_ausas_omega_p=1.0, jfo_ausas_omega_theta=1.0,
        tol=1e-7, jfo_ausas_max_iter=30000, jfo_ausas_check_every=500,
    )

    print(f"    CPU: n_iter={n_cpu}, residual={res_cpu:.2e}")
    print(f"    GPU: n_iter={n_gpu}, residual={res_gpu:.2e}")

    # Compare loads
    W_cpu = np.trapezoid(np.trapezoid(P_cpu * np.cos(Phi), phi_1D, axis=1), Z)
    W_gpu = np.trapezoid(np.trapezoid(P_gpu * np.cos(Phi), phi_1D, axis=1), Z)
    rel_W = abs(W_gpu - W_cpu) / (abs(W_cpu) + 1e-12)

    # Compare max P
    maxP_cpu = float(np.max(P_cpu))
    maxP_gpu = float(np.max(P_gpu))
    rel_maxP = abs(maxP_gpu - maxP_cpu) / (maxP_cpu + 1e-12)

    print(f"    W_cpu={W_cpu:.4e}, W_gpu={W_gpu:.4e}, rel={rel_W:.4f}")
    print(f"    maxP: cpu={maxP_cpu:.4e}, gpu={maxP_gpu:.4e}, rel={rel_maxP:.4f}")

    return run_test(
        "GPU vs CPU: loads agree within 2%",
        rel_W < 0.02 and rel_maxP < 0.05,
        f"rel_W={rel_W:.4f}, rel_maxP={rel_maxP:.4f}",
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Ausas-style JFO cavitation model validation")
    print("=" * 60)

    tests = [
        ("0", test_cpu_reference),
        ("1", test_ausas_vs_hs_small_epsilon),
        ("2", test_symmetric),
        ("3", test_invariants),
        ("4", test_gpu_vs_cpu),
    ]

    results = []
    for name, func in tests:
        try:
            results.append((name, func()))
        except Exception as e:
            print(f"  [FAIL] Test {name}: exception {type(e).__name__}: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    all_ok = all(r for _, r in results)
    if all_ok:
        print("  ALL AUSAS JFO TESTS PASSED")
    else:
        print("  SOME AUSAS JFO TESTS FAILED")
        for name, r in results:
            if not r:
                print(f"    Test {name} FAILED")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
