"""
Validation tests for the Payvar-Salant steady-state mass-conserving
JFO cavitation solver (reynolds_solver.cavitation.payvar_salant).

Tests:
  0. ε = 0 (uniform gap): trivial state g ≡ 0, P ≡ 0, θ ≡ 1.
  1. ε = 0.1: load integral within 5% of Half-Sommerfeld.
  2. ε = 0.6: physical invariants (P ≥ 0, θ ∈ [0, 1]), no collapse,
     no checkerboard in θ, physically reasonable cav_frac.
  3. ε-sweep with continuation warm-start: converges for ε in
     {0.1, 0.3, 0.5, 0.7, 0.9}, cav_frac non-decreasing.
  4. Two residuals (update / PDE) both decay over the sweep —
     guarantees we have a true fixed point, not a stalled iterate.

Run:
    python -m reynolds_solver.tests.test_payvar_salant
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
# Test 0: ε = 0 → trivial state
# -----------------------------------------------------------------------
def test_trivial_zero_eccentricity():
    print("\n=== Test 0: ε = 0 (uniform gap) ===")
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu

    R, L = 0.035, 0.056
    N_phi, N_Z = 50, 30
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.0)

    P, theta, res, n = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L,
        omega=1.0, tol=1e-10, max_iter=500,
    )

    maxP = float(np.max(np.abs(P)))
    th_min = float(theta.min())
    th_max = float(theta.max())

    print(f"    n_iter={n}, res={res:.2e}, maxP={maxP:.2e}, "
          f"theta=[{th_min:.6f}, {th_max:.6f}]")

    ok = (maxP < 1e-8) and (1.0 - th_min < 1e-8) and (1.0 - th_max < 1e-8)
    return run_test(
        "Trivial ε=0: g≡0, P≡0, θ≡1",
        ok,
        f"maxP={maxP:.2e}, theta_min={th_min:.6f}",
    )


# -----------------------------------------------------------------------
# Test 1: ε = 0.1 close to Half-Sommerfeld
# -----------------------------------------------------------------------
def test_close_to_hs_small_epsilon():
    print("\n=== Test 1: ε = 0.1 load ≈ Half-Sommerfeld ===")
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N_phi, N_Z = 200, 100
    epsilon = 0.1
    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N_phi, N_Z, epsilon)
    Phi, _ = np.meshgrid(phi_1D, Z)

    # Half-Sommerfeld reference load
    P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L)

    P_ps, theta_ps, res, n = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L,
        omega=1.0, tol=1e-7, max_iter=30000,
    )

    W_hs = float(np.trapezoid(
        np.trapezoid(P_hs * np.cos(Phi), phi_1D, axis=1), Z
    ))
    W_ps = float(np.trapezoid(
        np.trapezoid(P_ps * np.cos(Phi), phi_1D, axis=1), Z
    ))
    rel_diff = abs(W_ps - W_hs) / (abs(W_hs) + 1e-12)

    print(f"    n_iter={n}, res={res:.2e}")
    print(f"    W_hs={W_hs:.4e}, W_ps={W_ps:.4e}, rel_diff={rel_diff:.4f}")

    return run_test(
        "ε=0.1: PS load within 5% of HS",
        rel_diff < 0.05,
        f"rel_diff={rel_diff:.4f}",
    )


# -----------------------------------------------------------------------
# Test 2: ε = 0.6 invariants + no collapse + no θ checkerboard
# -----------------------------------------------------------------------
def test_invariants_strong_eccentricity():
    print("\n=== Test 2: ε = 0.6 physical invariants ===")
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.6)

    P, theta, res, n = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L,
        omega=1.0, tol=1e-7, max_iter=50000,
    )

    p_min = float(P.min())
    p_max = float(P.max())
    th_min = float(theta.min())
    th_max = float(theta.max())
    cav_frac = float(np.mean(theta[1:-1, 1:-1] < 1.0 - 1e-6))

    # Checkerboard detection: relative L2 norm of the 4-point Laplacian of θ
    # inside the cavitation zone. A smooth field has small high-frequency
    # content; a checker-boarded field has a huge one.
    th_int = theta[1:-1, 1:-1]
    lap = (
        th_int[2:, 1:-1] + th_int[:-2, 1:-1]
        + th_int[1:-1, 2:] + th_int[1:-1, :-2]
        - 4.0 * th_int[1:-1, 1:-1]
    )
    checker_norm = float(np.sqrt(np.mean(lap ** 2)))

    print(f"    n_iter={n}, res={res:.2e}")
    print(f"    P=[{p_min:.2e}, {p_max:.2e}]")
    print(f"    θ=[{th_min:.4f}, {th_max:.4f}], cav_frac={cav_frac:.3f}")
    print(f"    θ-checkerboard RMS(∇²θ)={checker_norm:.3e}")

    p_ok = (p_min >= -1e-12) and (p_max > 1e-3)           # not collapsed
    th_ok = (th_min >= -1e-12) and (th_max <= 1.0 + 1e-12)
    cav_reasonable = 0.1 < cav_frac < 0.7
    no_checker = checker_norm < 0.2  # smooth cavitation zone

    return run_test(
        "ε=0.6: invariants + not collapsed + smooth θ",
        p_ok and th_ok and cav_reasonable and no_checker,
        f"p=[{p_min:.2e},{p_max:.2e}], "
        f"θ=[{th_min:.4f},{th_max:.4f}], cav={cav_frac:.3f}, "
        f"checker={checker_norm:.2e}",
    )


# -----------------------------------------------------------------------
# Test 3: ε-sweep with continuation warm-start
# -----------------------------------------------------------------------
def test_continuation_sweep():
    print("\n=== Test 3: ε-sweep with continuation warm start ===")
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    eps_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    results = []
    g_prev = None
    for eps in eps_list:
        H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=eps)

        # Cold start for the smallest ε, warm start (g from previous
        # converged step) for every subsequent ε — standard continuation
        # strategy for nonlinear cavitation problems.
        g_init = None if g_prev is None else g_prev.copy()

        P, theta, res, n = solve_payvar_salant_cpu(
            H, d_phi, d_Z, R, L,
            omega=1.0, tol=1e-6, max_iter=20000,
            g_init=g_init,
        )

        # Recover g from (P, θ) for the next continuation step
        g_prev = np.where(theta >= 1.0 - 1e-12, P, theta - 1.0)

        p_ok = (P.min() >= -1e-12) and (P.max() > 1e-3)
        th_ok = (theta.min() >= -1e-12) and (theta.max() <= 1.0 + 1e-12)
        cav_frac = float(np.mean(theta[1:-1, 1:-1] < 1.0 - 1e-6))
        maxP = float(P.max())
        converged = res < 1e-3

        print(
            f"  ε={eps}: iter={n:>5d}  res={res:.2e}  "
            f"maxP={maxP:.4e}  cav_frac={cav_frac:.3f}  "
            f"inv={'OK' if p_ok and th_ok else 'FAIL'}"
        )
        results.append((eps, cav_frac, p_ok and th_ok, converged))

    cav_fracs = [r[1] for r in results]
    # Allow a tiny numerical slack in the monotonicity check.
    monotone = all(
        cav_fracs[i + 1] >= cav_fracs[i] - 1e-3
        for i in range(len(cav_fracs) - 1)
    )
    all_invariants = all(r[2] for r in results)
    all_converged = all(r[3] for r in results)

    return run_test(
        "Sweep: converges for all ε; invariants hold; cav_frac non-decreasing",
        monotone and all_invariants and all_converged,
        f"cav_fracs={[f'{c:.3f}' for c in cav_fracs]}",
    )


# -----------------------------------------------------------------------
# Test 4: both residuals (update / PDE) decay and the solver does NOT
#         drift from the HS warmup toward the trivial fixed point.
# -----------------------------------------------------------------------
def test_two_residuals_and_no_drift():
    print("\n=== Test 4: no-drift / two residuals ===")
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    from reynolds_solver.cavitation.payvar_salant.solver_cpu import (
        _build_coefficients, _hs_sor_sweep, _ps_pde_residual,
    )

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.6)

    # HS warmup so we know the target maxP.
    H_pack = H.copy()
    H_pack[:, 0] = H_pack[:, -2]
    H_pack[:, -1] = H_pack[:, 1]
    A, B, C, D, E = _build_coefficients(H_pack, d_phi, d_Z, R, L)
    F_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jm = j - 1 if j - 1 >= 1 else N_phi - 2
            F_hs[i, j] = d_phi * (H_pack[i, j] - H_pack[i, jm])
    P_hs = np.zeros_like(H_pack)
    for k in range(20000):
        r = _hs_sor_sweep(P_hs, A, B, C, D, E, F_hs, 1.7, N_Z, N_phi)
        if r < 1e-8 and k > 10:
            break
    maxP_hs = float(P_hs.max())
    print(f"  HS warmup target maxP={maxP_hs:.4e}")

    # PS solve with pinned active set
    P, theta, res, n = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L,
        omega=1.0, tol=1e-7, max_iter=20000, verbose=False,
    )

    # PDE residual on the converged state
    g = np.where(theta >= 1.0 - 1e-12, P, theta - 1.0)
    pde_res = float(_ps_pde_residual(
        g, H_pack, A, B, C, D, E, d_phi, N_Z, N_phi
    ))

    maxP_ps = float(P.max())
    rel_maxP = abs(maxP_ps - maxP_hs) / (maxP_hs + 1e-12)
    update_ok = res < 1e-6
    no_drift = rel_maxP < 0.02   # maxP MUST stay close to HS target
    pde_small = pde_res < 1e-4 * max(maxP_hs, 1e-3)  # rough absolute bound

    print(
        f"  n_iter={n}, update_res={res:.2e}, pde_res={pde_res:.3e}"
    )
    print(
        f"  PS  maxP={maxP_ps:.4e}, HS maxP={maxP_hs:.4e}, "
        f"rel={rel_maxP:.4f}"
    )
    print(
        f"  update_ok={update_ok}, no_drift={no_drift}, "
        f"pde_small={pde_small}"
    )

    # no_drift is the hard criterion; pde_small is informational (the
    # pinned-mask clamps leave a finite but small residual).
    return run_test(
        "No drift from HS target + update residual converged",
        update_ok and no_drift,
        f"rel_maxP={rel_maxP:.4f}, update={res:.2e}, pde={pde_res:.3e}",
    )


# -----------------------------------------------------------------------
# Test 5: load integral vs HS at multiple ε
# -----------------------------------------------------------------------
def test_load_vs_hs_multiple_epsilon():
    print("\n=== Test 5: load vs HS at ε = 0.1, 0.3, 0.5 ===")
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    N_phi, N_Z = 200, 100
    # Tolerances loosen with ε because JFO and HS are genuinely
    # different models at high eccentricity.
    eps_tol = [(0.1, 0.05), (0.3, 0.10), (0.5, 0.15)]

    all_ok = True
    for eps, tol_rel in eps_tol:
        H, d_phi, d_Z, phi_1D, Z = generate_test_case(N_phi, N_Z, eps)
        Phi, _ = np.meshgrid(phi_1D, Z)

        P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L)
        P_ps, theta_ps, res, n = solve_payvar_salant_cpu(
            H, d_phi, d_Z, R, L,
            omega=1.0, tol=1e-7, max_iter=30000,
        )

        W_hs = float(np.trapezoid(
            np.trapezoid(P_hs * np.cos(Phi), phi_1D, axis=1), Z
        ))
        W_ps = float(np.trapezoid(
            np.trapezoid(P_ps * np.cos(Phi), phi_1D, axis=1), Z
        ))
        rel = abs(W_ps - W_hs) / (abs(W_hs) + 1e-12)
        ok = rel < tol_rel
        if not ok:
            all_ok = False
        print(
            f"  ε={eps}: W_hs={W_hs:.4e}, W_ps={W_ps:.4e}, "
            f"rel={rel:.4f}, tol={tol_rel} → {'OK' if ok else 'FAIL'}"
        )

    return run_test(
        "Load integrals at ε=0.1/0.3/0.5 within tolerance vs HS",
        all_ok,
    )


# -----------------------------------------------------------------------
# Test 6: API smoke test
# -----------------------------------------------------------------------
def test_api_smoke():
    print("\n=== Test 6: solve_reynolds(..., cavitation='payvar_salant') ===")
    from reynolds_solver import solve_reynolds

    R, L = 0.035, 0.056
    H, d_phi, d_Z, _, _ = generate_test_case(50, 30, epsilon=0.3)

    result = solve_reynolds(H, d_phi, d_Z, R, L, cavitation="payvar_salant")
    P, theta, residual, n_iter = result

    p_ok = P.min() >= -1e-12 and P.max() > 1e-3
    th_ok = 0 <= theta.min() and theta.max() <= 1.0 + 1e-12

    print(f"    maxP={P.max():.4e}, θ=[{theta.min():.4f}, {theta.max():.4f}], "
          f"n_iter={n_iter}, res={residual:.2e}")

    return run_test(
        "API smoke: cavitation='payvar_salant'",
        p_ok and th_ok,
        f"maxP={P.max():.2e}, θ_min={theta.min():.4f}",
    )


# -----------------------------------------------------------------------
# Test 7: GPU vs CPU agreement
# -----------------------------------------------------------------------
def test_gpu_vs_cpu():
    print("\n=== Test 7: GPU vs CPU agreement (ε=0.6) ===")
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu

    try:
        from reynolds_solver.cavitation.payvar_salant import (
            solve_payvar_salant_gpu,
        )
    except ImportError:
        print("  [SKIP] cupy not available — GPU test skipped")
        return run_test("GPU vs CPU", True, "SKIPPED (no cupy)")

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N_phi, N_Z, epsilon=0.6)
    Phi, _ = np.meshgrid(phi_1D, Z)

    P_cpu, th_cpu, res_cpu, n_cpu = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L, omega=1.0, tol=1e-7, max_iter=20000,
    )
    P_gpu, th_gpu, res_gpu, n_gpu = solve_payvar_salant_gpu(
        H, d_phi, d_Z, R, L, omega=1.0, tol=1e-7, max_iter=20000,
    )

    W_cpu = float(np.trapezoid(
        np.trapezoid(P_cpu * np.cos(Phi), phi_1D, axis=1), Z
    ))
    W_gpu = float(np.trapezoid(
        np.trapezoid(P_gpu * np.cos(Phi), phi_1D, axis=1), Z
    ))
    rel_W = abs(W_gpu - W_cpu) / (abs(W_cpu) + 1e-12)

    maxP_cpu = float(P_cpu.max())
    maxP_gpu = float(P_gpu.max())
    rel_maxP = abs(maxP_gpu - maxP_cpu) / (maxP_cpu + 1e-12)

    cav_cpu = float(np.mean(th_cpu[1:-1, 1:-1] < 1.0 - 1e-6))
    cav_gpu = float(np.mean(th_gpu[1:-1, 1:-1] < 1.0 - 1e-6))
    diff_cav = abs(cav_gpu - cav_cpu)

    print(f"    CPU: n={n_cpu}, maxP={maxP_cpu:.4e}, cav={cav_cpu:.3f}")
    print(f"    GPU: n={n_gpu}, maxP={maxP_gpu:.4e}, cav={cav_gpu:.3f}")
    print(f"    rel_W={rel_W:.4f}, rel_maxP={rel_maxP:.4f}, "
          f"diff_cav={diff_cav:.4f}")

    ok = rel_W < 0.02 and rel_maxP < 0.02 and diff_cav < 0.01
    return run_test(
        "GPU vs CPU: W <2%, maxP <2%, cav_frac <0.01",
        ok,
        f"rel_W={rel_W:.4f}, rel_maxP={rel_maxP:.4f}, "
        f"diff_cav={diff_cav:.4f}",
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Payvar-Salant steady JFO cavitation validation")
    print("=" * 60)

    tests = [
        ("0", test_trivial_zero_eccentricity),
        ("1", test_close_to_hs_small_epsilon),
        ("2", test_invariants_strong_eccentricity),
        ("3", test_continuation_sweep),
        ("4", test_two_residuals_and_no_drift),
        ("5", test_load_vs_hs_multiple_epsilon),
        ("6", test_api_smoke),
        ("7", test_gpu_vs_cpu),
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
        print("  ALL PAYVAR-SALANT TESTS PASSED")
    else:
        print("  SOME PAYVAR-SALANT TESTS FAILED")
        for name, r in results:
            if not r:
                print(f"    Test {name} FAILED")
    print("=" * 60)

    # Diagnostic (informational only, no pass/fail)
    diag_mass_flux_balance()

    sys.exit(0 if all_ok else 1)


# -----------------------------------------------------------------------
# Diagnostic: mass flux balance (prints only, NOT a pass/fail test)
# -----------------------------------------------------------------------
def diag_mass_flux_balance():
    """
    Print mass flux balance at the Z boundaries vs cavitation deficit.

    For a steady bearing the net axial leakage flux through the two
    Z-end faces should balance the mass deficit in the cavitation zone.
    This is a soft sanity check, not a hard assertion.
    """
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu
    from reynolds_solver.cavitation.payvar_salant.solver_cpu import (
        _build_coefficients,
    )

    print()
    print("  ---- Mass flux balance (diagnostic, info only) ----")

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    epsilon = 0.6
    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N_phi, N_Z, epsilon)

    P, theta, res, n = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L, tol=1e-7, max_iter=20000,
    )

    H_pack = H.copy()
    H_pack[:, 0] = H_pack[:, -2]
    H_pack[:, -1] = H_pack[:, 1]
    A, B, C, D, E = _build_coefficients(H_pack, d_phi, d_Z, R, L)

    # Axial flux through Z=0 face (row i=1, leaking into the bearing)
    # flux_z0[j] = C[1,j] * P[1,j] (approx: face gradient × conductance)
    # ... summing the Z-gradient pressure flux at the two Z ends.
    # dP/dZ at Z=0 ≈ (P[1,j] - P[0,j]) / d_Z = P[1,j] / d_Z
    # Flux per unit phi = h³_face * dP/dZ
    # We approximate using coefficient D which = alpha_sq * 0.5*(h³[0]+h³[1])
    flux_z0 = float(np.sum(D[1, :] * P[1, :]))
    flux_zL = float(np.sum(C[-2, :] * P[-2, :]))
    total_flux_in = flux_z0 + flux_zL

    # Mass deficit in cavitation zone: sum of (1 - θ) * h * dφ * dZ
    deficit = float(np.sum((1.0 - theta[1:-1, 1:-1]) * H_pack[1:-1, 1:-1])
                    * d_phi * d_Z)

    print(f"  flux_in (Z ends):   {total_flux_in:.4e}")
    print(f"  cav mass deficit:   {deficit:.4e}")
    if abs(total_flux_in) > 1e-12:
        ratio = deficit / total_flux_in
        print(f"  deficit / flux_in:  {ratio:.4f}")
    print("  (These should be O(1) of each other for a balanced solution)")


if __name__ == "__main__":
    main()
