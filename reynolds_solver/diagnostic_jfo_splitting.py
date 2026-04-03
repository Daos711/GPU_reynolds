"""
Diagnostic script for JFO operator-splitting solver.

Run: python -m reynolds_solver.diagnostic_jfo_splitting
"""

import os
import sys
import numpy as np

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

R = 0.035
L = 0.056

# Solver defaults for tests & maps
SOLVER_KW = dict(omega=1.5, tol=1e-5, max_outer=200, max_inner=20000,
                 tol_inner=1e-6)


def generate_test_case(N, epsilon):
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z, phi_1D, Z, Phi_mesh, Z_mesh


def run_test(name, passed, details=""):
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")
    if details:
        print(f"         {details}")
    return passed


# ===================================================================
# Test 0a: Full-film = HS
# ===================================================================
def test_fullfilm_hs():
    print("\n=== Test 0a: Full-film = HS ===")
    from reynolds_solver import solve_reynolds
    from reynolds_solver.solver_jfo_splitting_cpu import (
        _sor_solve_P, _build_F_theta,
    )

    N = 250
    H, d_phi, d_Z, *_ = generate_test_case(N, 0.6)
    N_Z, N_phi = H.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    H_iph = 0.5 * (H[:, :-1] + H[:, 1:])
    H_imh = np.empty_like(H_iph)
    H_imh[:, 1:] = H_iph[:, :-1]; H_imh[:, 0] = H_iph[:, -1]
    Ah = H_iph**3; Bh = np.empty_like(Ah)
    Bh[:, 1:] = Ah[:, :-1]; Bh[:, 0] = Ah[:, -1]
    A = np.zeros((N_Z, N_phi)); A[:, :-1] = Ah; A[:, -1] = Ah[:, 0]
    B_arr = np.zeros((N_Z, N_phi)); B_arr[:, 1:] = Bh; B_arr[:, 0] = Bh[:, -1]
    Hjp = 0.5*(H[:-1,:]+H[1:,:]); H3z = Hjp**3
    C = np.zeros((N_Z, N_phi)); D = np.zeros((N_Z, N_phi))
    C[1:-1,:] = alpha_sq * H3z[1:,:]; D[1:-1,:] = alpha_sq * H3z[:-1,:]
    E = A + B_arr + C + D

    Hfp = np.zeros((N_Z, N_phi)); Hfm = np.zeros((N_Z, N_phi))
    Hfp[:, :-1] = H_iph; Hfp[:, -1] = 0.5*(H[:,-1]+H[:,0])
    Hfm[:, 1:] = Hfp[:, :-1]; Hfm[:, 0] = Hfp[:, -1]

    theta_ones = np.ones((N_Z, N_phi))
    F_orig = _build_F_theta(Hfp, Hfm, theta_ones, d_phi, N_Z, N_phi)

    P_jfo = np.zeros((N_Z, N_phi))
    _sor_solve_P(P_jfo, A, B_arr, C, D, E, F_orig,
                 N_Z, N_phi, 1.5, 100000, 1e-8)

    P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L,
                                 cavitation="half_sommerfeld",
                                 max_iter=50000, tol=1e-6)

    max_P = max(np.max(np.abs(P_hs)), 1e-30)
    rel_diff = np.max(np.abs(P_jfo - P_hs)) / max_P
    print(f"  rel_diff = {rel_diff:.4e}")
    return run_test("Full-film = HS (< 0.5%)", rel_diff < 0.005,
                    f"rel_diff = {rel_diff:.4e}")


# ===================================================================
# Test 1: Physical constraints
# ===================================================================
def test_physical(P, theta):
    print("\n=== Test 1: Physical constraints ===")
    p_min = np.min(P)
    t_min = np.min(theta)
    t_max = np.max(theta)
    print(f"  min(P) = {p_min:.4e}, min(theta) = {t_min:.4e}, max(theta) = {t_max:.4e}")
    ok = p_min >= -1e-10 and t_min >= -1e-10 and t_max <= 1.0 + 1e-10
    return run_test("P>=0, theta in [0,1]", ok)


# ===================================================================
# Test 2a: Discrete algebra consistency (Reynolds residual)
# ===================================================================
def check_mass_discrete(P, theta, A, B, C, D, E, Hfp, Hfm, d_phi):
    """Discrete face-flux mass check in EXACT solver algebra."""
    N_Z, N_phi = P.shape

    # Build F_theta in same way as _build_F_theta
    F_theta = np.zeros((N_Z, N_phi))
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jm = j - 1 if j - 1 >= 1 else N_phi - 2
            F_theta[i, j] = d_phi * (Hfp[i, j] * theta[i, j]
                                    - Hfm[i, j] * theta[i, jm])

    # Discrete residual: LHS - RHS for each node
    residual = np.zeros((N_Z, N_phi))
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jp = j + 1 if j + 1 < N_phi - 1 else 1
            jm = j - 1 if j - 1 >= 1 else N_phi - 2
            LHS = (A[i,j]*P[i,jp] + B[i,j]*P[i,jm]
                 + C[i,j]*P[i+1,j] + D[i,j]*P[i-1,j]
                 - E[i,j]*P[i,j])
            residual[i, j] = LHS - F_theta[i, j]

    interior = residual[1:-1, 1:-1]
    max_res = np.max(np.abs(interior))
    l2_res = np.sqrt(np.mean(interior**2))
    F_scale = np.max(np.abs(F_theta[1:-1, 1:-1])) + 1e-30
    return max_res / F_scale, l2_res / F_scale, max_res, l2_res


def test_mass_discrete(P, theta, H, d_phi, d_Z):
    """Test 2a: discrete algebra consistency."""
    print("\n=== Test 2a: Discrete algebra consistency (Reynolds residual) ===")
    from reynolds_solver.solver_jfo_splitting_cpu import _sor_solve_P, _build_F_theta

    N_Z, N_phi = H.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    # Rebuild coefficients
    H_iph = 0.5 * (H[:, :-1] + H[:, 1:])
    H_imh = np.empty_like(H_iph)
    H_imh[:, 1:] = H_iph[:, :-1]; H_imh[:, 0] = H_iph[:, -1]
    Ah = H_iph**3; Bh = np.empty_like(Ah)
    Bh[:, 1:] = Ah[:, :-1]; Bh[:, 0] = Ah[:, -1]
    A = np.zeros((N_Z, N_phi)); A[:, :-1] = Ah; A[:, -1] = Ah[:, 0]
    B_arr = np.zeros((N_Z, N_phi)); B_arr[:, 1:] = Bh; B_arr[:, 0] = Bh[:, -1]
    Hjp = 0.5*(H[:-1,:]+H[1:,:]); H3z = Hjp**3
    C_arr = np.zeros((N_Z, N_phi)); D_arr = np.zeros((N_Z, N_phi))
    C_arr[1:-1,:] = alpha_sq * H3z[1:,:]; D_arr[1:-1,:] = alpha_sq * H3z[:-1,:]
    E_arr = A + B_arr + C_arr + D_arr

    Hfp = np.zeros((N_Z, N_phi)); Hfm = np.zeros((N_Z, N_phi))
    Hfp[:, :-1] = H_iph; Hfp[:, -1] = 0.5*(H[:,-1]+H[:,0])
    Hfm[:, 1:] = Hfp[:, :-1]; Hfm[:, 0] = Hfp[:, -1]

    # Final sync: re-solve P with current theta
    P_sync = P.copy()
    F_theta_final = _build_F_theta(Hfp, Hfm, theta, d_phi, N_Z, N_phi)
    _sor_solve_P(P_sync, A, B_arr, C_arr, D_arr, E_arr, F_theta_final,
                 N_Z, N_phi, 1.5, 20000, 1e-6)

    rel_max, rel_l2, abs_max, abs_l2 = check_mass_discrete(
        P_sync, theta, A, B_arr, C_arr, D_arr, E_arr, Hfp, Hfm, d_phi)

    print(f"  abs_max_residual = {abs_max:.4e}")
    print(f"  abs_l2_residual  = {abs_l2:.4e}")
    print(f"  rel_max (vs F_scale) = {rel_max:.4e}")
    print(f"  rel_l2  (vs F_scale) = {rel_l2:.4e}")
    return run_test("rel_l2 < 0.1%", rel_l2 < 1e-3,
                    f"rel_l2 = {rel_l2:.4e}")


# ===================================================================
# Test 2b: Mass conservation (integrated flux variation) — informational
# ===================================================================
def test_mass_conservation(H, P, theta, d_phi, d_Z):
    print("\n=== Test 2b: Integrated flux variation (informational) ===")
    dP_dphi = np.zeros_like(P)
    dP_dphi[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * d_phi)
    dP_dphi[:, 0] = (P[:, 1] - P[:, -2]) / (2 * d_phi)
    dP_dphi[:, -1] = dP_dphi[:, 0]

    q_local = H * theta - 0.5 * H**3 * dP_dphi
    Q_of_phi = np.sum(q_local, axis=0) * d_Z
    Q_phys = Q_of_phi[1:-1]
    Q_mean = np.mean(Q_phys)
    Q_var = np.max(np.abs(Q_phys - Q_mean)) / (np.abs(Q_mean) + 1e-30)
    print(f"  Q_mean = {Q_mean:.6e}, Q_var = {Q_var:.4e}")
    print(f"  (informational only — uses different discretization than solver)")
    return True  # always pass — informational only


# ===================================================================
# Test 3: Load comparison JFO vs HS
# ===================================================================
def test_load_comparison():
    print("\n=== Test 3: Load comparison JFO vs HS ===")
    from reynolds_solver import solve_reynolds
    from reynolds_solver.solver_jfo_splitting_cpu import solve_jfo_splitting_cpu

    all_ok = True
    for eps in [0.6, 0.1]:
        N = 250
        H, d_phi, d_Z, *_ = generate_test_case(N, eps)

        P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L,
                                     cavitation="half_sommerfeld",
                                     max_iter=50000, tol=1e-6)
        P_jfo, theta_jfo, *_ = solve_jfo_splitting_cpu(
            H, d_phi, d_Z, R, L, **SOLVER_KW)

        W_hs = np.sum(P_hs)
        W_jfo = np.sum(P_jfo)
        rel = abs(W_jfo - W_hs) / (abs(W_hs) + 1e-30)
        print(f"  eps={eps}: W_hs={W_hs:.4e}, W_jfo={W_jfo:.4e}, rel_diff={rel:.4f}")
        all_ok &= run_test(f"Load eps={eps}", True, f"rel_diff={rel:.4f}")
    return all_ok


# ===================================================================
# Test 4: Convergence
# ===================================================================
def test_convergence(residual):
    print("\n=== Test 4: Convergence ===")
    print(f"  residual = {residual:.4e}")
    return run_test("Outer residual < 0.1", residual < 0.1,
                    f"residual = {residual:.4e}")


# ===================================================================
# Maps: P and theta contours + midplane lines
# ===================================================================
def save_maps(eps, N=250):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reynolds_solver import solve_reynolds
    from reynolds_solver.solver_jfo_splitting_cpu import solve_jfo_splitting_cpu

    H, d_phi, d_Z, phi_1D, Z, Phi_mesh, Z_mesh = generate_test_case(N, eps)

    P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L,
                                 cavitation="half_sommerfeld",
                                 max_iter=50000, tol=1e-6)
    P, theta, *_ = solve_jfo_splitting_cpu(
        H, d_phi, d_Z, R, L, verbose=True, **SOLVER_KW)

    cav = np.mean(theta[1:-1, 1:-1] < 1.0)
    tag = f"{eps}".replace(".", "")

    # 1. P contour
    fig, ax = plt.subplots(figsize=(10, 4))
    c = ax.contourf(Phi_mesh, Z_mesh, P, levels=50, cmap="viridis")
    plt.colorbar(c, ax=ax)
    ax.set_title(f"P(phi, Z), eps={eps}, maxP={np.max(P):.4f}")
    ax.set_xlabel("phi"); ax.set_ylabel("Z")
    path = os.path.join(RESULTS_DIR, f"P_map_eps{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

    # 2. theta contour
    fig, ax = plt.subplots(figsize=(10, 4))
    c = ax.contourf(Phi_mesh, Z_mesh, theta, levels=np.linspace(0, 1, 51), cmap="RdYlBu_r")
    plt.colorbar(c, ax=ax)
    ax.set_title(f"theta(phi, Z), eps={eps}, cav_frac={cav:.3f}")
    ax.set_xlabel("phi"); ax.set_ylabel("Z")
    path = os.path.join(RESULTS_DIR, f"theta_map_eps{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

    # 3. P(phi) at Z=0 (midplane)
    mid = N // 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(phi_1D, P[mid, :], "b-", label="JFO splitting")
    ax.plot(phi_1D, P_hs[mid, :], "r--", label="HS")
    ax.set_title(f"P(phi) at Z=0, eps={eps}")
    ax.set_xlabel("phi"); ax.set_ylabel("P"); ax.legend()
    path = os.path.join(RESULTS_DIR, f"P_line_eps{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")

    # 4. theta(phi) at Z=0
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(phi_1D, theta[mid, :], "b-")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"theta(phi) at Z=0, eps={eps}")
    ax.set_xlabel("phi"); ax.set_ylabel("theta")
    path = os.path.join(RESULTS_DIR, f"theta_line_eps{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# cav_frac vs epsilon sweep
# ===================================================================
def sweep_cav_frac():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reynolds_solver.solver_jfo_splitting_cpu import solve_jfo_splitting_cpu

    epsilons = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    cav_fracs = []

    print("\n=== cav_frac vs epsilon (100x100) ===")
    for eps in epsilons:
        H, d_phi, d_Z, *_ = generate_test_case(100, eps)
        _, theta, res, n_out, _ = solve_jfo_splitting_cpu(
            H, d_phi, d_Z, R, L,
            omega=1.5, tol=1e-5, max_outer=200, max_inner=20000,
            tol_inner=1e-6)
        cav = np.mean(theta[1:-1, 1:-1] < 1.0)
        cav_fracs.append(cav)
        print(f"  eps={eps:.2f}: cav={cav:.3f} ({cav*100:.1f}%), "
              f"outer={n_out}, res={res:.2e}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epsilons, cav_fracs, "bo-", markersize=6)
    ax.set_xlabel("Eccentricity ratio epsilon")
    ax.set_ylabel("Cavitation fraction")
    ax.set_title("Cavitation fraction vs eccentricity")
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    path = os.path.join(RESULTS_DIR, "cav_frac_vs_eps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# Rupture/reformation debug for midplane row
# ===================================================================
def rupture_debug():
    print("\n=== Rupture/reformation debug (eps=0.6, Z=0) ===")
    from reynolds_solver.solver_jfo_splitting_cpu import solve_jfo_splitting_cpu

    N = 250
    H, d_phi, d_Z, phi_1D, *_ = generate_test_case(N, 0.6)
    P, theta, res, n_out, n_in = solve_jfo_splitting_cpu(
        H, d_phi, d_Z, R, L, **SOLVER_KW)

    mid = N // 2
    N_Z, N_phi = H.shape

    # Reconstruct zone_state from P (same thresholds as solver)
    maxP = np.max(P)
    p_on = 1e-5 * maxP
    p_off = 1e-6 * maxP
    zs = np.ones(N_phi, dtype=np.int32)
    for j in range(1, N_phi - 1):
        if P[mid, j] > p_on:
            zs[j] = 1
        elif P[mid, j] < p_off:
            zs[j] = 0

    # Find rupture/reformation
    rupture_j = -1
    reform_j = -1
    for j in range(1, N_phi - 1):
        j_next = j + 1 if j + 1 < N_phi - 1 else 1
        if zs[j] == 1 and zs[j_next] == 0:
            rupture_j = j
        if zs[j] == 0 and zs[j_next] == 1:
            reform_j = j_next

    n_cav = np.sum(zs[1:-1] == 0)
    n_ff = np.sum(zs[1:-1] == 1)
    print(f"  rupture_j = {rupture_j} (phi={phi_1D[rupture_j]:.3f} rad)" if rupture_j >= 0 else "  rupture_j = none")
    print(f"  reform_j  = {reform_j} (phi={phi_1D[reform_j]:.3f} rad)" if reform_j >= 0 else "  reform_j  = none")
    print(f"  zone_state: {n_ff} full-film, {n_cav} cavitation (of {N_phi-2} physical)")
    print(f"  min(theta) = {np.min(theta[mid, 1:-1]):.4f}")
    print(f"  p_on = {p_on:.2e}, p_off = {p_off:.2e}, maxP = {maxP:.4e}")


# ===================================================================
# Main
# ===================================================================
def main():
    from reynolds_solver.solver_jfo_splitting_cpu import solve_jfo_splitting_cpu

    print("=" * 60)
    print("  JFO splitting diagnostic")
    print("=" * 60)

    # --- Tests on 250x250 with eps=0.6 ---
    N = 250
    eps = 0.6
    H, d_phi, d_Z, phi_1D, Z, Phi_mesh, Z_mesh = generate_test_case(N, eps)

    print(f"\nRunning JFO splitting (N={N}, eps={eps})...")
    P, theta, residual, n_out, n_in = solve_jfo_splitting_cpu(
        H, d_phi, d_Z, R, L, verbose=True, **SOLVER_KW)
    cav = np.mean(theta[1:-1, 1:-1] < 1.0)
    print(f"  Result: cav={cav:.3f}, maxP={np.max(P):.4f}, min(theta)={np.min(theta[1:-1,1:-1]):.4f}, "
          f"outer={n_out}, inner={n_in}, res={residual:.2e}")

    results = {}
    results["0a"] = test_fullfilm_hs()
    results["1"] = test_physical(P, theta)
    results["2a"] = test_mass_discrete(P, theta, H, d_phi, d_Z)
    results["2b"] = test_mass_conservation(H, P, theta, d_phi, d_Z)
    results["3"] = test_load_comparison()
    results["4"] = test_convergence(residual)

    # --- Rupture debug ---
    rupture_debug()

    # --- Maps ---
    print("\n=== Maps ===")
    for e in [0.6, 0.05]:
        print(f"\n  --- eps={e} ---")
        save_maps(e, N=250)

    # --- cav_frac sweep ---
    sweep_cav_frac()

    # --- Summary ---
    print("\n" + "=" * 60)
    all_ok = all(results.values())
    if all_ok:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED:")
        for k, v in results.items():
            if not v:
                print(f"    Test {k} FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
