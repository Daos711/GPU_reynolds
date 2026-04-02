"""
JFO cavitation via unified variable ψ (Elrod 1981). CPU reference.

Iterative scheme: honest Picard outer loop.
  Outer: classify nodes (full-film vs cavitation), rebuild frozen coefficients.
  Inner: linear SOR with frozen coefficients.

Key discretization principle (Elrod 1981):
  Diffusion always uses H³ coefficients (NO face indicators).
  Diffusion operates on P = max(ψ, 0), NOT on ψ directly.
  This preserves pressure coupling at film/cavitation boundaries,
  allowing reformation (return from cavitation to full film).

Semi-implicit Couette: for cavitation nodes (ψ < 0), θ_c = 1 + ψ_c.
  The ψ_c-dependent part contributes d_phi · H_{j+1/2} to the diagonal.

Non-dimensionalization, ghost columns, face indexing — identical to the
existing HS path via precompute_coefficients_gpu() in utils.py.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Frozen-coefficient builder (called once per Picard outer iteration)
# ---------------------------------------------------------------------------
@njit
def _build_frozen_coefficients(psi, H3_face_p, H3_face_m,
                                H3_face_zp, H3_face_zm,
                                H_face_p, H_face_m,
                                d_phi, N_Z, N_phi):
    """
    Build frozen arrays from the current ψ field.

    KEY DIFFERENCE from indicator-product approach:
      - Diffusion coefficients A,B,C,D = H³ ALWAYS (no face indicators).
      - gc classifies nodes: full-film (gc=1) or cavitation (gc=0).
      - diff_diag = A+B+C+D for full-film nodes, 0 for cavitation.
      - Inner loop uses P = max(ψ, 0) in the numerator, not ψ.

    Returns
    -------
    conv_rhs  : (N_Z, N_phi) frozen Couette RHS (constant part, θ_c=1)
    conv_diag : (N_Z, N_phi) Couette diagonal (>0 in cavitation only)
    diff_diag : (N_Z, N_phi) diffusion diagonal (>0 in full-film only)
    gc        : (N_Z, N_phi) node classification (1=full-film, 0=cavitation)
    """
    conv_rhs  = np.zeros((N_Z, N_phi), dtype=np.float64)
    conv_diag = np.zeros((N_Z, N_phi), dtype=np.float64)
    diff_diag = np.zeros((N_Z, N_phi), dtype=np.float64)
    gc        = np.zeros((N_Z, N_phi), dtype=np.float64)

    # Classify nodes
    for i in range(N_Z):
        for j in range(N_phi):
            if psi[i, j] >= 0.0:
                gc[i, j] = 1.0

    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            j_minus = j - 1 if j - 1 >= 1 else N_phi - 2

            # Upstream θ (frozen, from j-1 node)
            psi_jm = psi[i, j_minus]
            theta_jm = 1.0 if psi_jm >= 0.0 else max(1.0 + psi_jm, 0.0)

            # Couette RHS: always uses θ_c = 1 (constant part)
            conv_rhs[i, j] = d_phi * (H_face_p[i, j] * 1.0
                                     - H_face_m[i, j] * theta_jm)

            if gc[i, j] > 0.5:
                # Full-film: ψ enters diffusion diagonal, no conv_diag
                diff_diag[i, j] = (H3_face_p[i, j] + H3_face_m[i, j]
                                 + H3_face_zp[i, j] + H3_face_zm[i, j])
                conv_diag[i, j] = 0.0
            else:
                # Cavitation: P=0 so ψ not in diffusion, but θ_c=1+ψ_c
                # gives conv_diag = d_phi·H_face_p
                diff_diag[i, j] = 0.0
                conv_diag[i, j] = d_phi * H_face_p[i, j]

    return conv_rhs, conv_diag, diff_diag, gc


# ---------------------------------------------------------------------------
# Inner SOR loop (all coefficients FROZEN — no g/θ recomputation)
# ---------------------------------------------------------------------------
@njit
def _inner_sor_loop(psi, H3_face_p, H3_face_m, H3_face_zp, H3_face_zm,
                    conv_rhs, conv_diag, diff_diag,
                    N_Z, N_phi, omega_sor, max_inner, tol_inner,
                    clamp_min):
    """
    Sequential Gauss-Seidel / SOR for the unified variable ψ.

    KEY: diffusion numerator uses P_neighbor = max(ψ_neighbor, 0),
    NOT ψ_neighbor.  This preserves pressure coupling at the
    film/cavitation boundary and enables reformation.

    Parameters
    ----------
    clamp_min : float
        Lower bound for ψ.  -1.0 for JFO mode, 0.0 for force_full_film.
    """
    for iteration in range(max_inner):
        max_delta = 0.0

        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                j_plus  = j + 1 if j + 1 < N_phi - 1 else 1
                j_minus = j - 1 if j - 1 >= 1 else N_phi - 2

                psi_old = psi[i, j]

                # KEY: use P = max(ψ, 0) for neighbours in diffusion
                P_jp = max(psi[i, j_plus],  0.0)
                P_jm = max(psi[i, j_minus], 0.0)
                P_ip = max(psi[i + 1, j],   0.0)
                P_im = max(psi[i - 1, j],   0.0)

                diff_num = (H3_face_p[i, j] * P_jp
                          + H3_face_m[i, j] * P_jm
                          + H3_face_zp[i, j] * P_ip
                          + H3_face_zm[i, j] * P_im)

                # Total diagonal: diffusion (full-film) + Couette (cavitation)
                diag = diff_diag[i, j] + conv_diag[i, j]

                psi_new = (diff_num - conv_rhs[i, j]) / (diag + 1e-30)

                # Lower bound on ψ
                if psi_new < clamp_min:
                    psi_new = clamp_min

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

    Parameters
    ----------
    H : np.ndarray, shape (N_Z, N_phi), float64
        Dimensionless gap (ghost columns already set).
    d_phi, d_Z : float
        Grid spacing.
    R, L : float
        Bearing radius and length (m).
    omega_sor : float
        SOR relaxation (1.0–1.4).
    tol : float
        Outer-loop convergence on ||Δψ||_inf.
    max_outer : int
        Max Picard iterations.
    max_inner : int
        Max SOR iterations per outer step.
    tol_inner : float
        Inner SOR convergence tolerance.
    verbose : bool
        Print convergence info.
    force_full_film : bool
        If True: standard HS path (no cavitation physics), clamp ψ≥0.
        Used for test 0a (algebraic full-film ≡ HS).

    Returns
    -------
    P : np.ndarray (N_Z, N_phi) — dimensionless pressure
    theta : np.ndarray (N_Z, N_phi) — fill fraction
    residual : float — final ||Δψ||_inf
    n_outer : int — Picard iterations used
    n_inner_total : int — total SOR iterations
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

    # Z-direction faces (node-indexed, include alpha_sq)
    H3_face_zp = np.zeros((N_Z, N_phi), dtype=np.float64)
    H3_face_zm = np.zeros((N_Z, N_phi), dtype=np.float64)
    H_jph = 0.5 * (H[:-1, :] + H[1:, :])
    H3_jph = H_jph ** 3
    H3_face_zp[1:-1, :] = alpha_sq * H3_jph[1:, :]
    H3_face_zm[1:-1, :] = alpha_sq * H3_jph[:-1, :]

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
        A_ff[:, :-1] = A_half
        A_ff[:, -1]  = A_half[:, 0]

        B_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        B_ff[:, 1:] = B_half
        B_ff[:, 0]  = B_half[:, -1]

        C_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        D_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        C_ff[1:-1, :] = alpha_sq * H3_jph[1:, :]
        D_ff[1:-1, :] = alpha_sq * H3_jph[:-1, :]

        E_ff = A_ff + B_ff + C_ff + D_ff

        F_half = d_phi * (H_iph_half - H_imh_half)
        F_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        F_ff[:, :-1] = F_half
        F_ff[:, -1]  = F_half[:, 0]

        # Single inner loop (no outer needed for full-film)
        n_inner, inner_res = _inner_sor_loop_fullfilm(
            psi, A_ff, B_ff, C_ff, D_ff, E_ff, F_ff,
            N_Z, N_phi, omega_sor, max_inner * max_outer, tol_inner,
        )

        P = np.maximum(psi, 0.0)
        theta = np.ones_like(psi)
        residual = inner_res
        return P, theta, residual, 1, n_inner

    # ------------------------------------------------------------------
    # JFO mode: Picard outer loop
    # ------------------------------------------------------------------
    n_inner_total = 0
    residual = 1.0
    n_outer_done = 0

    for outer in range(max_outer):
        psi_old = psi.copy()

        # Step 1: build frozen coefficients from current ψ
        conv_rhs, conv_diag, diff_diag, gc = \
            _build_frozen_coefficients(
                psi, H3_face_p, H3_face_m,
                H3_face_zp, H3_face_zm,
                H_face_p, H_face_m,
                d_phi, N_Z, N_phi,
            )

        # Step 2: inner SOR with frozen coefficients
        n_inner, inner_res = _inner_sor_loop(
            psi, H3_face_p, H3_face_m, H3_face_zp, H3_face_zm,
            conv_rhs, conv_diag, diff_diag,
            N_Z, N_phi, omega_sor, max_inner, tol_inner,
            -1.0,  # clamp_min
        )
        n_inner_total += n_inner

        # Step 3: outer convergence check
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
