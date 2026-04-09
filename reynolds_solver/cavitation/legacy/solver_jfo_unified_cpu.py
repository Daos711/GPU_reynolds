"""
JFO cavitation via unified variable ψ (Elrod 1981). CPU reference.

Key principle (correct Elrod 1981):
  - Diffusion coefficients = H³ ALWAYS (no face indicators)
  - Diffusion operates on P = max(ψ, 0), not ψ directly
  - Full-film/cavitation switch is dynamic: try full-film diagonal first,
    if ψ_trial < 0 switch to cavitation (Couette-only) diagonal.
  - Semi-implicit Couette: for cavitation, θ_c = 1+ψ_c, the ψ_c part
    contributes d_phi·H_{j+1/2} to the diagonal.

This avoids:
  1. Face-indicator product killing diffusion at boundaries (v3.1 bug)
  2. Frozen gc classification + large diffusion numerator → NaN (v3.2 bug)
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Nonlinear Gauss-Seidel inner loop
# ---------------------------------------------------------------------------
@njit
def _inner_sor_nonlinear(psi, H3_face_p, H3_face_m, H3_face_zp, H3_face_zm,
                          H_face_p, H_face_m, E_full,
                          d_phi, N_Z, N_phi, omega_sor, max_inner, tol_inner):
    """
    Nonlinear Gauss-Seidel / SOR for the unified variable ψ.

    Each node update:
      1. Compute diff_num = sum(H³_face · max(ψ_neighbor, 0))
      2. Compute conv_rhs = d_phi · (H_p - H_m · θ_upstream)
         where θ_upstream is from the CURRENT ψ (not frozen).
      3. numerator = diff_num - conv_rhs
      4. Try full-film: ψ_trial = numerator / E_full
      5. If ψ_trial ≥ 0: accept (full-film)
         If ψ_trial < 0: cavitation ψ = numerator / (d_phi · H_p)
      6. Clamp ψ ≥ -1, apply SOR relaxation.
    """
    for iteration in range(max_inner):
        max_delta = 0.0

        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                j_plus  = j + 1 if j + 1 < N_phi - 1 else 1
                j_minus = j - 1 if j - 1 >= 1 else N_phi - 2

                psi_old = psi[i, j]

                # Diffusion numerator: P = max(ψ, 0) for all neighbours
                P_jp = max(psi[i, j_plus],  0.0)
                P_jm = max(psi[i, j_minus], 0.0)
                P_ip = max(psi[i + 1, j],   0.0)
                P_im = max(psi[i - 1, j],   0.0)

                diff_num = (H3_face_p[i, j] * P_jp
                          + H3_face_m[i, j] * P_jm
                          + H3_face_zp[i, j] * P_ip
                          + H3_face_zm[i, j] * P_im)

                # Upstream θ from CURRENT ψ (Gauss-Seidel, not frozen)
                psi_jm = psi[i, j_minus]
                theta_jm = 1.0 if psi_jm >= 0.0 else max(1.0 + psi_jm, 0.0)

                # Couette RHS (constant part: θ_c = 1)
                conv_rhs = d_phi * (H_face_p[i, j] - H_face_m[i, j] * theta_jm)

                numerator = diff_num - conv_rhs

                # Try full-film first (large diagonal = E)
                E = E_full[i, j]
                psi_trial = numerator / (E + 1e-30)

                if psi_trial >= 0.0:
                    psi_new = psi_trial
                else:
                    # Cavitation: only Couette diagonal
                    conv_diag = d_phi * H_face_p[i, j]
                    psi_new = numerator / (conv_diag + 1e-30)
                    if psi_new < -1.0:
                        psi_new = -1.0

                # SOR relaxation
                psi[i, j] = psi_old + omega_sor * (psi_new - psi_old)

                delta = abs(psi[i, j] - psi_old)
                if delta > max_delta:
                    max_delta = delta

        # Periodic BC in φ
        for i in range(N_Z):
            psi[i, 0] = psi[i, N_phi - 2]
            psi[i, N_phi - 1] = psi[i, 1]

        # Dirichlet BC in Z
        for j in range(N_phi):
            psi[0, j] = 0.0
            psi[N_Z - 1, j] = 0.0

        if max_delta < tol_inner:
            return iteration + 1, max_delta

    return max_inner, max_delta


# ---------------------------------------------------------------------------
# Inner SOR loop for force_full_film (standard HS-compatible)
# ---------------------------------------------------------------------------
@njit
def _inner_sor_loop_fullfilm(psi, A_coeff, B_coeff, C_coeff, D_coeff,
                              E_coeff, F_coeff,
                              N_Z, N_phi, omega_sor, max_inner, tol_inner):
    """
    Standard SOR for force_full_film mode.
    Uses ψ directly (not max(ψ,0)) to match HS path exactly.
    Clamps ψ ≥ 0 (half-Sommerfeld).
    """
    for iteration in range(max_inner):
        max_delta = 0.0

        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                j_plus  = j + 1 if j + 1 < N_phi - 1 else 1
                j_minus = j - 1 if j - 1 >= 1 else N_phi - 2

                psi_old = psi[i, j]

                psi_new = (A_coeff[i, j] * psi[i, j_plus]
                         + B_coeff[i, j] * psi[i, j_minus]
                         + C_coeff[i, j] * psi[i + 1, j]
                         + D_coeff[i, j] * psi[i - 1, j]
                         - F_coeff[i, j]) / (E_coeff[i, j] + 1e-30)

                if psi_new < 0.0:
                    psi_new = 0.0

                psi[i, j] = psi_old + omega_sor * (psi_new - psi_old)

                delta = abs(psi[i, j] - psi_old)
                if delta > max_delta:
                    max_delta = delta

        # Periodic BC in φ
        for i in range(N_Z):
            psi[i, 0] = psi[i, N_phi - 2]
            psi[i, N_phi - 1] = psi[i, 1]

        # Dirichlet BC in Z
        for j in range(N_phi):
            psi[0, j] = 0.0
            psi[N_Z - 1, j] = 0.0

        if max_delta < tol_inner:
            return iteration + 1, max_delta

    return max_inner, max_delta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def solve_jfo_unified_cpu(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    omega_sor: float = 1.0,
    tol: float = 1e-6,
    max_outer: int = 200,
    max_inner: int = 500,
    tol_inner: float = 1e-7,
    verbose: bool = False,
    force_full_film: bool = False,
) -> tuple:
    """
    JFO cavitation via unified variable ψ (Elrod 1981).  CPU reference.

    Returns
    -------
    P : np.ndarray (N_Z, N_phi) — dimensionless pressure
    theta : np.ndarray (N_Z, N_phi) — fill fraction
    residual : float — final ||Δψ||_inf
    n_outer : int — outer iterations used
    n_inner_total : int — total inner iterations
    """
    N_Z, N_phi = H.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    # ------------------------------------------------------------------
    # Precompute face data (node-indexed, full-sized arrays)
    # ------------------------------------------------------------------
    H_face_p = np.zeros((N_Z, N_phi), dtype=np.float64)
    H_face_m = np.zeros((N_Z, N_phi), dtype=np.float64)

    H_face_p[:, :-1] = 0.5 * (H[:, :-1] + H[:, 1:])
    H_face_p[:, -1]  = 0.5 * (H[:, -1]  + H[:, 0])

    H_face_m[:, 1:]  = H_face_p[:, :-1]
    H_face_m[:, 0]   = H_face_p[:, -1]

    H3_face_p = H_face_p ** 3
    H3_face_m = H_face_m ** 3

    # Z-direction faces (include alpha_sq)
    H3_face_zp = np.zeros((N_Z, N_phi), dtype=np.float64)
    H3_face_zm = np.zeros((N_Z, N_phi), dtype=np.float64)
    H_jph = 0.5 * (H[:-1, :] + H[1:, :])
    H3_jph = H_jph ** 3
    H3_face_zp[1:-1, :] = alpha_sq * H3_jph[1:, :]
    H3_face_zm[1:-1, :] = alpha_sq * H3_jph[:-1, :]

    # Full diffusion diagonal (constant, precomputed once)
    E_full = H3_face_p + H3_face_m + H3_face_zp + H3_face_zm

    # ------------------------------------------------------------------
    # Initialise ψ
    # ------------------------------------------------------------------
    psi = np.zeros((N_Z, N_phi), dtype=np.float64)

    # ------------------------------------------------------------------
    # force_full_film: replicate precompute_coefficients_gpu() EXACTLY
    # ------------------------------------------------------------------
    if force_full_film:
        H_iph_half = 0.5 * (H[:, :-1] + H[:, 1:])
        H_imh_half = np.empty_like(H_iph_half)
        H_imh_half[:, 1:] = H_iph_half[:, :-1]
        H_imh_half[:, 0]  = H_iph_half[:, -1]

        A_half = H_iph_half ** 3
        B_half = np.empty_like(A_half)
        B_half[:, 1:] = A_half[:, :-1]
        B_half[:, 0]  = A_half[:, -1]

        A_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        A_ff[:, :-1] = A_half;  A_ff[:, -1] = A_half[:, 0]

        B_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        B_ff[:, 1:] = B_half;  B_ff[:, 0] = B_half[:, -1]

        C_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        D_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        C_ff[1:-1, :] = alpha_sq * H3_jph[1:, :]
        D_ff[1:-1, :] = alpha_sq * H3_jph[:-1, :]

        E_ff = A_ff + B_ff + C_ff + D_ff

        F_half = d_phi * (H_iph_half - H_imh_half)
        F_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        F_ff[:, :-1] = F_half;  F_ff[:, -1] = F_half[:, 0]

        n_inner, inner_res = _inner_sor_loop_fullfilm(
            psi, A_ff, B_ff, C_ff, D_ff, E_ff, F_ff,
            N_Z, N_phi, omega_sor, max_inner * max_outer, tol_inner,
        )

        P = np.maximum(psi, 0.0)
        theta = np.ones_like(psi)
        return P, theta, inner_res, 1, n_inner

    # ------------------------------------------------------------------
    # JFO mode: outer loop wrapping nonlinear GS inner loop
    # ------------------------------------------------------------------
    n_inner_total = 0
    residual = 1.0
    n_outer_done = 0

    for outer in range(max_outer):
        psi_old = psi.copy()

        # Inner: nonlinear GS (dynamic full-film/cavitation switching)
        n_inner, inner_res = _inner_sor_nonlinear(
            psi, H3_face_p, H3_face_m, H3_face_zp, H3_face_zm,
            H_face_p, H_face_m, E_full,
            d_phi, N_Z, N_phi, omega_sor, max_inner, tol_inner,
        )
        n_inner_total += n_inner

        # Outer convergence
        residual = np.max(np.abs(psi - psi_old))
        n_outer_done = outer + 1

        if verbose and (outer % 10 == 0 or outer < 5):
            cav_frac = np.mean(psi[1:-1, 1:-1] < 0)
            print(f"  outer={outer:>4d}: res={residual:.2e}, "
                  f"inner={n_inner}, cav={cav_frac:.3f}")

        if residual < tol:
            if verbose:
                print(f"  Converged at outer={outer}")
            break

    # ------------------------------------------------------------------
    # Recover P, θ from ψ
    # ------------------------------------------------------------------
    P = np.maximum(psi, 0.0)
    theta = np.where(psi >= 0.0, 1.0, 1.0 + psi)
    theta = np.clip(theta, 0.0, 1.0)

    return P, theta, residual, n_outer_done, n_inner_total
