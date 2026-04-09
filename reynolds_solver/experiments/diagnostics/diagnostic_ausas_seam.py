"""
Diagnostic for the Ausas-JFO CPU solver.

Checks several hypotheses from the Ausas validation TЗ:

  A. Periodic-seam inconsistency in H during coefficient assembly.
     Compares vectorized `_build_coefficients()` with an explicit
     jm/jp-wrap loop, element-by-element on the interior.

  B. Cascade collapse / slow drift with the in-place GS scheme: runs the
     HS warmup alone, then a fixed number of Ausas sweeps and prints
     per-iteration {maxP, cav_frac, zombie, ff, cav_nodes}.

  D. Frozen-iterate (Jacobi) sweep: same trace, but with the new
     `_ausas_relax_sweep_jacobi`. Two starts:
        - HS-warm: HS warmup → Jacobi
        - Cold:    P=0, θ=1 → Jacobi (no HS warmup)
     Both starts should converge to the same physical state if Jacobi is
     a valid fix.

Run:
    python -m reynolds_solver.experiments.diagnostics.diagnostic_ausas_seam
"""
import numpy as np

from reynolds_solver.cavitation.ausas.solver_cpu import (
    _build_coefficients,
    _hs_sor_sweep,
    _ausas_relax_sweep,
    _ausas_relax_sweep_jacobi,
    _ausas_relax_sweep_pt,
)


# ----------------------------------------------------------------------------
# Test case (matches test_jfo_ausas.py::generate_test_case)
# ----------------------------------------------------------------------------
def generate_test_case(N_phi, N_Z, epsilon=0.6):
    phi_1D = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)
    Phi, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z


# ----------------------------------------------------------------------------
# Reference coefficient assembly: explicit loop with jm/jp wrap.
# ----------------------------------------------------------------------------
def build_coefficients_explicit(H, d_phi, d_Z, R, L):
    """
    Reference: build A/B/C/D/E by explicit loop with jm/jp wrap on
    physical columns [1, N_phi-2] only. H[:, 0] and H[:, N_phi-1] are
    treated as ghost columns and never read.
    """
    N_Z, N_phi = H.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    A = np.zeros((N_Z, N_phi))
    B = np.zeros((N_Z, N_phi))
    C = np.zeros((N_Z, N_phi))
    D = np.zeros((N_Z, N_phi))

    for i in range(N_Z):
        for j in range(1, N_phi - 1):
            jp = j + 1 if j + 1 < N_phi - 1 else 1
            jm = j - 1 if j - 1 >= 1 else N_phi - 2
            A[i, j] = 0.5 * (H[i, j] ** 3 + H[i, jp] ** 3)
            B[i, j] = 0.5 * (H[i, jm] ** 3 + H[i, j] ** 3)

    for i in range(1, N_Z - 1):
        for j in range(N_phi):
            C[i, j] = alpha_sq * 0.5 * (H[i, j] ** 3 + H[i + 1, j] ** 3)
            D[i, j] = alpha_sq * 0.5 * (H[i - 1, j] ** 3 + H[i, j] ** 3)

    E = A + B + C + D
    return A, B, C, D, E


def pack_ghost(H):
    """Apply periodic ghost packing: H[:, 0] = H[:, -2], H[:, -1] = H[:, 1]."""
    H = H.copy()
    H[:, 0] = H[:, -2]
    H[:, -1] = H[:, 1]
    return H


# ----------------------------------------------------------------------------
# Zombie / ff / cav counting
# ----------------------------------------------------------------------------
def count_states(P, theta):
    """Counts on INTERIOR only (excluding Z-boundary rows and ghost cols)."""
    P_int = P[1:-1, 1:-1]
    th_int = theta[1:-1, 1:-1]
    full_film = np.sum((th_int > 1.0 - 1e-8) & (P_int > 1e-12))
    zombie = np.sum((th_int > 1.0 - 1e-8) & (P_int <= 1e-12))
    cav = np.sum(th_int < 1.0 - 1e-8)
    return int(zombie), int(full_film), int(cav)


# ----------------------------------------------------------------------------
# Step A: coefficient seam-check (vectorized vs explicit-wrap)
# ----------------------------------------------------------------------------
def coefficient_seam_check():
    print("=" * 60)
    print("Step A: coefficient seam-check (vectorized vs explicit-wrap)")
    print("=" * 60)

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    H, d_phi, d_Z = generate_test_case(N_phi, N_Z, epsilon=0.6)

    # Path 1: vectorized, H as-produced by the test (no ghost packing)
    A1, B1, C1, D1, E1 = _build_coefficients(H, d_phi, d_Z, R, L)

    # Path 2: explicit loop with jm/jp wrap, H as-is (ghost irrelevant here)
    A2, B2, C2, D2, E2 = build_coefficients_explicit(H, d_phi, d_Z, R, L)

    # Path 3: vectorized with properly packed H (ghost copies of physical cols)
    H_packed = pack_ghost(H)
    A3, B3, C3, D3, E3 = _build_coefficients(H_packed, d_phi, d_Z, R, L)

    def maxdiff(X, Y, name):
        # Compare on interior (physical columns and interior Z rows).
        d = np.max(np.abs(X[1:-1, 1:-1] - Y[1:-1, 1:-1]))
        print(f"    max|{name}_vec - {name}_ref| (interior) = {d:.6e}")
        return d

    print("\n[1] vectorized(H_raw) vs explicit-wrap(H_raw):")
    dmax_raw = 0.0
    for name, X, Y in [
        ("A", A1, A2), ("B", B1, B2), ("C", C1, C2),
        ("D", D1, D2), ("E", E1, E2),
    ]:
        dmax_raw = max(dmax_raw, maxdiff(X, Y, name))
    print(f"    OVERALL max diff = {dmax_raw:.6e}")

    print("\n[2] vectorized(H_packed) vs explicit-wrap(H_raw):")
    dmax_pack = 0.0
    for name, X, Y in [
        ("A", A3, A2), ("B", B3, B2), ("C", C3, C2),
        ("D", D3, D2), ("E", E3, E2),
    ]:
        dmax_pack = max(dmax_pack, maxdiff(X, Y, name))
    print(f"    OVERALL max diff = {dmax_pack:.6e}")

    print()
    if dmax_raw > 1e-12:
        print(f"  [ISSUE] vectorized(H_raw) and explicit-wrap disagree by "
              f"{dmax_raw:.2e} on the seam.")
        if dmax_pack < 1e-12:
            print("  [FIX] packing H ghost cols fixes the discrepancy.")
        else:
            print("  [WARNING] packing does not fully resolve it — check C/D.")
    else:
        print("  vectorized and explicit-wrap agree — H ghost packing is "
              "not strictly needed for this test case.")

    return dmax_raw


# ----------------------------------------------------------------------------
# Step B: HS warmup + Ausas cascade trace
# ----------------------------------------------------------------------------
def cascade_trace(apply_ghost_pack=True, n_warmup=2000, n_ausas=200,
                  omega_hs=1.7, omega_p=1.0, omega_theta=1.0):
    print()
    print("=" * 60)
    print(f"Step B: cascade trace (ghost_pack={apply_ghost_pack}, "
          f"omega_p={omega_p}, omega_theta={omega_theta})")
    print("=" * 60)

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    H, d_phi, d_Z = generate_test_case(N_phi, N_Z, epsilon=0.6)
    if apply_ghost_pack:
        H = pack_ghost(H)

    A, B, C, D, E = _build_coefficients(H, d_phi, d_Z, R, L)

    # ---- HS warmup ----
    P = np.zeros((N_Z, N_phi), dtype=np.float64)
    F_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jm = j - 1 if j - 1 >= 1 else N_phi - 2
            F_hs[i, j] = d_phi * (H[i, j] - H[i, jm])

    print(f"\n  HS warmup (omega={omega_hs}, up to {n_warmup} iters, tol=1e-7):")
    last_dP = 0.0
    for k in range(n_warmup):
        last_dP = _hs_sor_sweep(P, A, B, C, D, E, F_hs, omega_hs, N_Z, N_phi)
        if k % 200 == 0 or k in (0, 1, 2):
            print(f"    iter={k:>5d}  dP={last_dP:.3e}  maxP={P.max():.4e}")
        if last_dP < 1e-7 and k > 5:
            print(f"    [converged] iter={k}, dP={last_dP:.3e}, "
                  f"maxP={P.max():.4e}")
            break

    # ---- Ausas relaxation trace ----
    print(f"\n  Ausas relaxation (omega_p={omega_p}, omega_theta={omega_theta}, "
          f"{n_ausas} iters):")
    theta = np.ones((N_Z, N_phi), dtype=np.float64)
    theta[0, :] = 1.0
    theta[-1, :] = 1.0
    P[0, :] = 0.0
    P[-1, :] = 0.0
    flooded_flag = 1

    print(f"    {'iter':>5s}  {'residual':>12s}  {'maxP':>10s}  "
          f"{'cav_frac':>9s}  {'zombie':>7s}  {'ff':>7s}  {'cav':>7s}")
    for k in range(n_ausas):
        residual = _ausas_relax_sweep(
            P, theta, H, A, B, C, D, E,
            d_phi, omega_p, omega_theta, N_Z, N_phi, flooded_flag,
        )
        if k < 5 or k % 10 == 0 or k == n_ausas - 1:
            cav_frac = float(np.mean(theta < 1.0 - 1e-6))
            zombie, ff, cav = count_states(P, theta)
            print(f"    {k:>5d}  {residual:>12.3e}  {P.max():>10.4e}  "
                  f"{cav_frac:>9.3f}  {zombie:>7d}  {ff:>7d}  {cav:>7d}")


# ----------------------------------------------------------------------------
# Step D: frozen-iterate (Jacobi) Ausas trace
# ----------------------------------------------------------------------------
def jacobi_trace(start="hs_warm", n_warmup=2000, n_jacobi=1000,
                 omega_p=1.0, omega_theta=1.0, epsilon=0.6):
    print()
    print("=" * 60)
    print(f"Step D: Jacobi trace (start={start}, "
          f"omega_p={omega_p}, omega_theta={omega_theta}, eps={epsilon})")
    print("=" * 60)

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    H, d_phi, d_Z = generate_test_case(N_phi, N_Z, epsilon=epsilon)
    H = pack_ghost(H)

    A, B, C, D, E = _build_coefficients(H, d_phi, d_Z, R, L)

    P = np.zeros((N_Z, N_phi), dtype=np.float64)

    if start == "hs_warm":
        F_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                jm = j - 1 if j - 1 >= 1 else N_phi - 2
                F_hs[i, j] = d_phi * (H[i, j] - H[i, jm])

        last_dP = 0.0
        for k in range(n_warmup):
            last_dP = _hs_sor_sweep(P, A, B, C, D, E, F_hs, 1.7, N_Z, N_phi)
            if last_dP < 1e-7 and k > 5:
                break
        print(f"  HS warmup done: iter={k}, dP={last_dP:.3e}, "
              f"maxP={P.max():.4e}")
    elif start == "cold":
        print(f"  Cold start: P=0, theta=1, maxP={P.max():.4e}")
    else:
        raise ValueError(f"Unknown start: {start}")

    theta = np.ones((N_Z, N_phi), dtype=np.float64)
    P[0, :] = 0.0
    P[-1, :] = 0.0
    theta[0, :] = 1.0
    theta[-1, :] = 1.0
    flooded_flag = 1

    P_new = P.copy()
    theta_new = theta.copy()

    print(f"\n  Jacobi relaxation ({n_jacobi} iters):")
    print(f"    {'iter':>5s}  {'residual':>12s}  {'maxP':>10s}  "
          f"{'cav_frac':>9s}  {'zombie':>7s}  {'ff':>7s}  {'cav':>7s}")
    for k in range(n_jacobi):
        residual = _ausas_relax_sweep_jacobi(
            P_new, P, theta_new, theta, H, A, B, C, D, E,
            d_phi, omega_p, omega_theta, N_Z, N_phi, flooded_flag,
        )
        # Swap
        P, P_new = P_new, P
        theta, theta_new = theta_new, theta

        if k < 5 or k % 50 == 0 or k == n_jacobi - 1:
            cav_frac = float(np.mean(theta < 1.0 - 1e-6))
            zombie, ff, cav = count_states(P, theta)
            print(f"    {k:>5d}  {residual:>12.3e}  {P.max():>10.4e}  "
                  f"{cav_frac:>9.3f}  {zombie:>7d}  {ff:>7d}  {cav:>7d}")

    return P, theta


# ----------------------------------------------------------------------------
# Step E: pseudo-transient (Ausas eq. 12 with time term) trace
# ----------------------------------------------------------------------------
def pt_trace(start="hs_warm", n_warmup=2000,
             dt_pseudo=0.1, max_time_steps=200, max_inner=50,
             inner_tol=1e-4, omega_p=1.0, omega_theta=1.0, epsilon=0.6):
    """
    Pseudo-transient Ausas trace: marches in pseudo-time τ with an inner
    Jacobi Ausas relaxation at each step. The temporal term
    β (c^n - c^{n-1}), β = 2 d_phi² / Δτ, anchors θ to the previous time
    slab and enlarges the θ denominator from d_phi·h_ij to
    (β + d_phi)·h_ij, damping the drift to the trivial c = θh ≈ const.
    """
    print()
    print("=" * 60)
    print(
        f"Step E: pseudo-transient trace "
        f"(start={start}, dt={dt_pseudo}, "
        f"ω_p={omega_p}, ω_θ={omega_theta}, eps={epsilon})"
    )
    print("=" * 60)

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    H, d_phi, d_Z = generate_test_case(N_phi, N_Z, epsilon=epsilon)
    H = pack_ghost(H)

    A, B, C, D, E = _build_coefficients(H, d_phi, d_Z, R, L)
    beta_pt = 2.0 * d_phi * d_phi / dt_pseudo
    print(f"  d_phi={d_phi:.4f}, d_phi^2={d_phi**2:.4e}, "
          f"β = 2 d_phi² / dτ = {beta_pt:.4e}")

    P = np.zeros((N_Z, N_phi), dtype=np.float64)

    if start == "hs_warm":
        F_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                jm = j - 1 if j - 1 >= 1 else N_phi - 2
                F_hs[i, j] = d_phi * (H[i, j] - H[i, jm])

        last_dP = 0.0
        for k in range(n_warmup):
            last_dP = _hs_sor_sweep(P, A, B, C, D, E, F_hs, 1.7, N_Z, N_phi)
            if last_dP < 1e-7 and k > 5:
                break
        print(f"  HS warmup done: iter={k}, dP={last_dP:.3e}, "
              f"maxP={P.max():.4e}")
    elif start == "cold":
        print(f"  Cold start: P=0, theta=1, maxP={P.max():.4e}")
    else:
        raise ValueError(f"Unknown start: {start}")

    theta = np.ones((N_Z, N_phi), dtype=np.float64)
    P[0, :] = 0.0
    P[-1, :] = 0.0
    theta[0, :] = 1.0
    theta[-1, :] = 1.0
    flooded_flag = 1

    P_new = P.copy()
    theta_new = theta.copy()
    c_prev = theta * H

    print(
        f"\n  Pseudo-transient march "
        f"({max_time_steps} steps × ≤{max_inner} inner):"
    )
    print(
        f"    {'step':>5s}  {'inner_k':>7s}  {'inner_res':>11s}  "
        f"{'steady':>11s}  {'maxP':>10s}  {'cav_frac':>9s}  "
        f"{'zombie':>7s}  {'ff':>7s}  {'cav':>7s}"
    )

    for n_step in range(max_time_steps):
        P_prev_step = P.copy()
        theta_prev_step = theta.copy()

        inner_res = 1.0
        inner_k = 0
        for inner_k in range(max_inner):
            inner_res = _ausas_relax_sweep_pt(
                P_new, P, theta_new, theta, H, c_prev,
                A, B, C, D, E,
                d_phi, beta_pt, omega_p, omega_theta,
                N_Z, N_phi, flooded_flag,
            )
            P, P_new = P_new, P
            theta, theta_new = theta_new, theta
            if inner_res < inner_tol and inner_k > 2:
                break

        c_prev = theta * H

        steady_res = (
            float(np.sqrt(np.sum((P - P_prev_step) ** 2)))
            + float(np.sqrt(np.sum((theta - theta_prev_step) ** 2)))
        )

        if n_step < 5 or n_step % 10 == 0 or n_step == max_time_steps - 1:
            cav_frac = float(np.mean(theta < 1.0 - 1e-6))
            zombie, ff, cav = count_states(P, theta)
            print(
                f"    {n_step:>5d}  {inner_k + 1:>7d}  {inner_res:>11.3e}  "
                f"{steady_res:>11.3e}  {P.max():>10.4e}  "
                f"{cav_frac:>9.3f}  {zombie:>7d}  {ff:>7d}  {cav:>7d}"
            )

    return P, theta


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    dmax = coefficient_seam_check()

    # Run cascade trace both without and with ghost packing so we can
    # compare. If vectorized seam error is non-trivial, the no-pack run
    # shows it; the packed run is the "fixed" case.
    cascade_trace(apply_ghost_pack=False, n_ausas=200)
    cascade_trace(apply_ghost_pack=True, n_ausas=200)

    # If cascade persists with ghost packing, try under-relaxation.
    print()
    print("=" * 60)
    print("Step C (if collapse persists with pack=True): under-relaxation")
    print("=" * 60)
    for omega in (0.7, 0.5, 0.3):
        cascade_trace(apply_ghost_pack=True, n_warmup=2000, n_ausas=200,
                      omega_p=omega, omega_theta=omega)

    # Frozen-iterate (Jacobi) reference: HS-warm and cold starts at ε=0.6.
    P_warm, th_warm = jacobi_trace(start="hs_warm", n_jacobi=1000,
                                   omega_p=1.0, omega_theta=1.0,
                                   epsilon=0.6)
    P_cold, th_cold = jacobi_trace(start="cold", n_jacobi=1000,
                                   omega_p=1.0, omega_theta=1.0,
                                   epsilon=0.6)

    print()
    print("=" * 60)
    print("Step D summary: warm vs cold final states (eps=0.6, 1000 iters)")
    print("=" * 60)
    print(f"  warm:  maxP={P_warm.max():.4e}, cav_frac="
          f"{float(np.mean(th_warm < 1.0 - 1e-6)):.3f}")
    print(f"  cold:  maxP={P_cold.max():.4e}, cav_frac="
          f"{float(np.mean(th_cold < 1.0 - 1e-6)):.3f}")
    diff_P = float(np.max(np.abs(P_warm - P_cold)))
    diff_th = float(np.max(np.abs(th_warm - th_cold)))
    print(f"  max|P_warm - P_cold| = {diff_P:.3e}")
    print(f"  max|θ_warm - θ_cold| = {diff_th:.3e}")

    # Quick sanity at ε=0.1: should be close to HS load (test 1's check).
    print()
    print("=" * 60)
    print("Step D sanity at eps=0.1 (Jacobi, HS-warm, 1000 iters)")
    print("=" * 60)
    jacobi_trace(start="hs_warm", n_jacobi=1000,
                 omega_p=1.0, omega_theta=1.0, epsilon=0.1)

    # ---- Step E: pseudo-transient Ausas (ε=0.6) ----
    # Compare warm vs cold at ε=0.6. If PT is a valid fix, both starts
    # should converge to the SAME stationary state (unlike the
    # stationary GS/Jacobi solver).
    P_warm_pt, th_warm_pt = pt_trace(
        start="hs_warm",
        dt_pseudo=0.1, max_time_steps=200, max_inner=50,
        omega_p=1.0, omega_theta=1.0, epsilon=0.6,
    )
    P_cold_pt, th_cold_pt = pt_trace(
        start="cold",
        dt_pseudo=0.1, max_time_steps=200, max_inner=50,
        omega_p=1.0, omega_theta=1.0, epsilon=0.6,
    )

    print()
    print("=" * 60)
    print("Step E summary: PT warm vs cold final states (eps=0.6)")
    print("=" * 60)
    print(
        f"  warm:  maxP={P_warm_pt.max():.4e}, cav_frac="
        f"{float(np.mean(th_warm_pt < 1.0 - 1e-6)):.3f}"
    )
    print(
        f"  cold:  maxP={P_cold_pt.max():.4e}, cav_frac="
        f"{float(np.mean(th_cold_pt < 1.0 - 1e-6)):.3f}"
    )
    diff_P = float(np.max(np.abs(P_warm_pt - P_cold_pt)))
    diff_th = float(np.max(np.abs(th_warm_pt - th_cold_pt)))
    print(f"  max|P_warm - P_cold| = {diff_P:.3e}")
    print(f"  max|θ_warm - θ_cold| = {diff_th:.3e}")

    # PT sanity at ε=0.1.
    print()
    print("=" * 60)
    print("Step E sanity at eps=0.1 (pseudo-transient, HS-warm)")
    print("=" * 60)
    pt_trace(
        start="hs_warm",
        dt_pseudo=0.1, max_time_steps=200, max_inner=50,
        omega_p=1.0, omega_theta=1.0, epsilon=0.1,
    )


if __name__ == "__main__":
    main()
