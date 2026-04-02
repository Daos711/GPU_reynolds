"""
Validation tests for JFO-unified (Elrod 1981) cavitation model.

Tests:
  0a. Algebraic full-film ≡ HS  (force_full_film=True → P matches HS)
  0b. Physical small-ε           (unified-JFO close to HS at ε=0.05)
  1.  Mass conservation (A)      (local divergence of face fluxes ≈ 0)
  2.  Mass conservation (B)      (integrated flux variation across φ-sections)
  3.  P ≥ 0 and θ ∈ [0, 1]      (physical constraints)
  4.  Cavitation fraction         (DIAGNOSTIC — not PASS/FAIL)
  5.  JFO ≈ HS at small ε        (loads within 10%)

Run:
    python -m reynolds_solver.test_jfo_unified
"""

import os
import sys
import numpy as np


RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)

# -----------------------------------------------------------------------
# Shared parameters
# -----------------------------------------------------------------------
R = 0.035
L = 0.056


def run_test(test_name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}")
    if details:
        print(f"         {details}")
    return passed


def generate_test_case(N, epsilon=0.6):
    """Create gap field H with ghost columns already set."""
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z, phi_1D, Z, Phi_mesh


# -----------------------------------------------------------------------
# Test 0a: Algebraic full-film ≡ HS
# -----------------------------------------------------------------------
def test_fullfilm_algebraic():
    """force_full_film=True → unified-JFO ≡ HS."""
    print("\n=== Test 0a: Algebraic full-film ≡ HS ===")
    from reynolds_solver import solve_reynolds
    from reynolds_solver.solver_jfo_unified_cpu import solve_jfo_unified_cpu

    N = 250
    epsilon = 0.6
    H, d_phi, d_Z, phi_1D, Z, _ = generate_test_case(N, epsilon)

    # HS reference (GPU)
    P_hs, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        cavitation="half_sommerfeld",
        max_iter=50000, tol=1e-6,
    )

    # Unified-JFO with forced full film (CPU)
    P_jfo, theta_jfo, res, n_out, n_in = solve_jfo_unified_cpu(
        H, d_phi, d_Z, R, L,
        force_full_film=True,
        omega_sor=1.5,
        tol=1e-7,
        max_outer=1,
        max_inner=100000,
        tol_inner=1e-8,
        verbose=True,
    )

    # theta must be 1 everywhere in force_full_film mode
    theta_ok = np.allclose(theta_jfo, 1.0)

    # Pressure comparison
    max_P = max(np.max(np.abs(P_hs)), 1e-30)
    rel_diff = np.max(np.abs(P_jfo - P_hs)) / max_P

    print(f"    max|theta - 1| = {np.max(np.abs(theta_jfo - 1.0)):.2e}")
    print(f"    max|P_jfo - P_hs| / max|P_hs| = {rel_diff:.4e}")
    print(f"    SOR iterations = {n_in}, residual = {res:.2e}")

    all_passed = True
    all_passed &= run_test(
        "theta == 1 in full-film mode",
        theta_ok,
    )
    all_passed &= run_test(
        "Full-film pressure matches HS (< 0.5%)",
        rel_diff < 0.005,
        f"rel_diff = {rel_diff:.4e}",
    )
    return all_passed


# -----------------------------------------------------------------------
# Test 0b: Physical small-ε check
# -----------------------------------------------------------------------
def test_small_epsilon_physical():
    """At small ε, unified-JFO close to HS."""
    print("\n=== Test 0b: Physical small-ε check ===")
    from reynolds_solver import solve_reynolds
    from reynolds_solver.solver_jfo_unified_cpu import solve_jfo_unified_cpu

    N = 250
    epsilon = 0.05
    H, d_phi, d_Z, phi_1D, Z, _ = generate_test_case(N, epsilon)

    P_hs, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        cavitation="half_sommerfeld",
        max_iter=50000, tol=1e-6,
    )

    P_jfo, theta_jfo, *_ = solve_jfo_unified_cpu(
        H, d_phi, d_Z, R, L,
        omega_sor=1.0,
        tol=1e-6,
        max_outer=200,
        max_inner=500,
        tol_inner=1e-7,
        verbose=True,
    )

    cav_frac = np.mean(theta_jfo[1:-1, 1:-1] < 1.0)
    print(f"    cav_frac at eps=0.05: {cav_frac:.4f}")

    W_hs = np.sum(P_hs)
    W_jfo = np.sum(P_jfo)
    rel_diff = abs(W_jfo - W_hs) / (abs(W_hs) + 1e-30)
    print(f"    W_hs = {W_hs:.6e}, W_jfo = {W_jfo:.6e}, rel_diff = {rel_diff:.4f}")

    return run_test(
        "Small-ε: loads within 10%",
        rel_diff < 0.10,
        f"rel_diff = {rel_diff:.4f}",
    )


# -----------------------------------------------------------------------
# Test 1: Mass conservation (A) — local divergence check
# -----------------------------------------------------------------------
def test_mass_conservation_local():
    """Local divergence of face fluxes ≈ 0."""
    print("\n=== Test 1: Mass conservation (A) — local divergence ===")
    from reynolds_solver.solver_jfo_unified_cpu import solve_jfo_unified_cpu

    N = 250
    epsilon = 0.6
    H, d_phi, d_Z, phi_1D, Z, _ = generate_test_case(N, epsilon)

    P, theta, res, n_out, n_in = solve_jfo_unified_cpu(
        H, d_phi, d_Z, R, L,
        omega_sor=1.0,
        tol=1e-6,
        max_outer=300,
        max_inner=500,
        tol_inner=1e-7,
        verbose=True,
    )

    psi = np.where(theta < 1.0, theta - 1.0, P)
    alpha_sq = (2 * R / L * d_phi / d_Z) ** 2
    g = (psi >= 0).astype(np.float64)

    # Poiseuille flux on φ-faces (between columns j and j+1)
    g_face_phi = g[:, :-1] * g[:, 1:]
    H_face_phi = 0.5 * (H[:, :-1] + H[:, 1:])
    q_pois_phi = -g_face_phi * H_face_phi ** 3 * np.diff(psi, axis=1) / d_phi

    # Couette flux on φ-faces (upwind = left neighbour)
    theta_upwind = theta[:, :-1]
    q_couette_phi = H_face_phi * theta_upwind

    q_phi = q_pois_phi + q_couette_phi

    # Poiseuille flux on Z-faces
    g_face_z = g[:-1, :] * g[1:, :]
    H_face_z = 0.5 * (H[:-1, :] + H[1:, :])
    q_z = -g_face_z * alpha_sq * H_face_z ** 3 * np.diff(psi, axis=0) / d_Z

    # Divergence over interior control volumes
    # q_phi has N_phi-1 faces → diff gives N_phi-2 values (columns 0..N_phi-3)
    # q_z   has N_Z-1  faces → diff gives N_Z-2  values (rows 0..N_Z-3)
    div_phi = np.diff(q_phi, axis=1) / d_phi    # (N_Z, N_phi-2)
    div_z   = np.diff(q_z,   axis=0) / d_Z      # (N_Z-2, N_phi)

    # Interior: rows 1..N_Z-2, physical columns 1..N_phi-2
    # div_phi[:, k] corresponds to column k+1 in the original grid (face k+1 - face k)
    # For physical columns j=1..N_phi-2: div_phi column index = j-1 → j=1..N_phi-2 → k=0..N_phi-3
    # div_z[k, :] corresponds to row k+1 → for rows 1..N_Z-2: k=0..N_Z-3
    # Overlap: div_phi rows 1..N_Z-2 (all), div_z columns 1..N_phi-2
    Nz_int = N_Z - 2
    Np_int = N_phi - 2

    # div_phi interior: rows 1:-1, all columns (N_phi-2 of them)
    dp = div_phi[1:-1, :]  # shape (N_Z-2, N_phi-2)
    # div_z interior: all rows (N_Z-2 of them), columns 1:-1
    dz = div_z[:, 1:-1]    # shape (N_Z-2, N_phi-2)

    interior_div = dp + dz

    total_source = np.sum(np.abs(interior_div))
    total_flux = (np.sum(np.abs(q_phi[1:-1, :])) +
                  np.sum(np.abs(q_z[:, 1:-1])))
    flux_err = total_source / (total_flux + 1e-30)

    print(f"    Converged: {n_out} outer, {n_in} inner, res={res:.2e}")
    print(f"    flux_err = {flux_err:.4e}")

    return run_test(
        "Local divergence flux error < 1%",
        flux_err < 0.01,
        f"flux_err = {flux_err:.4e}",
    )


# -----------------------------------------------------------------------
# Test 2: Mass conservation (B) — integrated flux variation
# -----------------------------------------------------------------------
def test_mass_conservation_independent():
    """Integrated flux Q(φ) variation across φ-sections."""
    print("\n=== Test 2: Mass conservation (B) — integrated flux variation ===")
    from reynolds_solver.solver_jfo_unified_cpu import solve_jfo_unified_cpu

    N = 250
    epsilon = 0.6
    H, d_phi, d_Z, phi_1D, Z, _ = generate_test_case(N, epsilon)

    P, theta, res, n_out, n_in = solve_jfo_unified_cpu(
        H, d_phi, d_Z, R, L,
        omega_sor=1.0,
        tol=1e-6,
        max_outer=300,
        max_inner=500,
        tol_inner=1e-7,
        verbose=True,
    )

    # q_phi = H·θ − 0.5·H³·dP/dφ  (project normalisation, same as test_jfo.py)
    dP_dphi = np.zeros_like(P)
    dP_dphi[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * d_phi)
    # Periodic wrap for ghost columns
    dP_dphi[:, 0]  = (P[:, 1] - P[:, -2]) / (2 * d_phi)
    dP_dphi[:, -1] = dP_dphi[:, 0]

    q_local = H * theta - 0.5 * H ** 3 * dP_dphi

    # Integrated flux Q(φ) = ∫ q_phi dZ
    Q_of_phi = np.sum(q_local, axis=0) * d_Z

    # Physical columns only
    Q_phys = Q_of_phi[1:-1]
    Q_mean = np.mean(Q_phys)
    Q_var = np.max(np.abs(Q_phys - Q_mean)) / (np.abs(Q_mean) + 1e-30)

    print(f"    Q_mean = {Q_mean:.6e}, Q_var = {Q_var:.4e}")

    return run_test(
        "Integrated flux variation < 1%",
        Q_var < 0.01,
        f"Q_var = {Q_var:.4e}",
    )


# -----------------------------------------------------------------------
# Test 3: P ≥ 0 and θ ∈ [0, 1]
# -----------------------------------------------------------------------
def test_physical_constraints():
    """P ≥ 0 and θ ∈ [0, 1]."""
    print("\n=== Test 3: P ≥ 0 and θ ∈ [0, 1] ===")
    from reynolds_solver.solver_jfo_unified_cpu import solve_jfo_unified_cpu

    N = 250
    epsilon = 0.6
    H, d_phi, d_Z, *_ = generate_test_case(N, epsilon)

    P, theta, *_ = solve_jfo_unified_cpu(
        H, d_phi, d_Z, R, L,
        omega_sor=1.0,
        tol=1e-6,
        max_outer=300,
        max_inner=500,
        tol_inner=1e-7,
    )

    p_min = np.min(P)
    t_min = np.min(theta)
    t_max = np.max(theta)

    all_passed = True
    all_passed &= run_test(
        "P >= 0 everywhere",
        p_min >= 0.0,
        f"min(P) = {p_min:.4e}",
    )
    all_passed &= run_test(
        "theta in [0, 1]",
        t_min >= 0.0 and t_max <= 1.0,
        f"min(theta) = {t_min:.4e}, max(theta) = {t_max:.4e}",
    )
    return all_passed


# -----------------------------------------------------------------------
# Test 4: Cavitation fraction (DIAGNOSTIC)
# -----------------------------------------------------------------------
def test_cavitation_fraction_diagnostic():
    """Sanity check: cavitation fraction reasonable at ε=0.6."""
    print("\n=== Test 4: Cavitation fraction (DIAGNOSTIC) ===")
    from reynolds_solver.solver_jfo_unified_cpu import solve_jfo_unified_cpu

    N = 250
    epsilon = 0.6
    H, d_phi, d_Z, *_ = generate_test_case(N, epsilon)

    P, theta, res, n_out, n_in = solve_jfo_unified_cpu(
        H, d_phi, d_Z, R, L,
        omega_sor=1.0,
        tol=1e-6,
        max_outer=300,
        max_inner=500,
        tol_inner=1e-7,
        verbose=True,
    )

    cav_frac = np.mean(theta[1:-1, 1:-1] < 1.0)
    print(f"  [DIAGNOSTIC] Cavitation fraction = {cav_frac:.3f} ({cav_frac * 100:.1f}%)")
    print(f"               Outer = {n_out}, Inner total = {n_in}, residual = {res:.2e}")
    print(f"               max(P) = {np.max(P):.4e}")

    if cav_frac < 0.01:
        print("  [WARNING] No cavitation at ε=0.6 — suspicious")
    if cav_frac > 0.80:
        print("  [WARNING] Cavitation > 80% — possible instability (same issue as active-set)")

    # DIAGNOSTIC — always returns True (not a PASS/FAIL gate)
    return True


# -----------------------------------------------------------------------
# Test 5: JFO ≈ HS at small ε
# -----------------------------------------------------------------------
def test_jfo_vs_hs_small_epsilon():
    """JFO and HS loads within 10% at small ε."""
    print("\n=== Test 5: JFO ≈ HS at small ε ===")
    from reynolds_solver import solve_reynolds
    from reynolds_solver.solver_jfo_unified_cpu import solve_jfo_unified_cpu

    N = 250
    epsilon = 0.1
    H, d_phi, d_Z, phi_1D, Z, Phi_mesh = generate_test_case(N, epsilon)

    P_hs, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        cavitation="half_sommerfeld",
        max_iter=50000, tol=1e-6,
    )

    P_jfo, theta_jfo, *_ = solve_jfo_unified_cpu(
        H, d_phi, d_Z, R, L,
        omega_sor=1.0,
        tol=1e-6,
        max_outer=200,
        max_inner=500,
        tol_inner=1e-7,
        verbose=True,
    )

    W_hs  = np.trapezoid(np.trapezoid(P_hs  * np.cos(Phi_mesh), phi_1D, axis=1), Z)
    W_jfo = np.trapezoid(np.trapezoid(P_jfo * np.cos(Phi_mesh), phi_1D, axis=1), Z)

    rel_diff = abs(W_jfo - W_hs) / (abs(W_hs) + 1e-12)
    print(f"    W_hs = {W_hs:.6e}, W_jfo = {W_jfo:.6e}, rel_diff = {rel_diff:.4f}")

    return run_test(
        "JFO and HS loads within 10% at epsilon=0.1",
        rel_diff < 0.10,
        f"rel_diff = {rel_diff:.4f}",
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  JFO-unified (Elrod 1981) validation")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {}

    # --- Test 0a FIRST (gate test) ---
    results["0a"] = test_fullfilm_algebraic()

    if not results["0a"]:
        print("\n" + "=" * 60)
        print("  TEST 0a FAILED — algebraic mismatch with HS.")
        print("  DO NOT proceed until this is fixed.")
        print("=" * 60)
        sys.exit(1)

    # --- Remaining tests ---
    results["0b"] = test_small_epsilon_physical()
    results["1"]  = test_mass_conservation_local()
    results["2"]  = test_mass_conservation_independent()
    results["3"]  = test_physical_constraints()
    results["4"]  = test_cavitation_fraction_diagnostic()
    results["5"]  = test_jfo_vs_hs_small_epsilon()

    # --- Summary ---
    print("\n" + "=" * 60)
    passfail_tests = {k: v for k, v in results.items() if k != "4"}
    all_ok = all(passfail_tests.values())

    if all_ok:
        print("  ALL PASS/FAIL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED:")
        for k, v in passfail_tests.items():
            if not v:
                print(f"    Test {k} FAILED")

    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
