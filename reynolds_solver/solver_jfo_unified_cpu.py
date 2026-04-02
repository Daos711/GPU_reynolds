"""
JFO cavitation via unified variable ψ (Elrod 1981). CPU reference.

Iterative scheme: honest Picard outer loop.
  Outer: rebuild frozen face indicators gf_* and upwind θ from current ψ.
  Inner: linear SOR with frozen coefficients (no g/θ recomputation).

Discretization — face-flux divergence form:
  Poiseuille (diffusion): g_{face} · H³_{face} · Δψ / Δφ
  Couette   (convection): H_{face} · θ_upwind
  g_{face} = product of node indicators on both sides (ψ ≥ 0 → 1, else 0)
  θ(ψ) = 1 if ψ ≥ 0, else 1 + ψ  (clipped to [0, 1])

Semi-implicit Couette treatment: for cavitation nodes (ψ < 0), θ_c = 1 + ψ_c.
The ψ_c-dependent part contributes to the diagonal (d_phi · H_{j+1/2}),
preventing zero-diagonal singularity in the SOR update.

Non-dimensionalization, ghost columns, face indexing — identical to the
existing HS path via precompute_coefficients_gpu() in utils.py.
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Frozen-coefficient builder (called once per Picard outer iteration)
# ---------------------------------------------------------------------------
@njit
def _build_frozen_coefficients(psi, H_face_p, H_face_m,
                                H3_face_p, H3_face_m,
                                H3_face_zp, H3_face_zm,
                                alpha_sq, d_phi,
                                N_Z, N_phi):
    """
    Build frozen arrays from the current ψ field.

    Semi-implicit Couette: for cavitation nodes (gc=0), θ_c = 1 + ψ_c.
    The constant part (1) goes to conv_rhs, the ψ_c part adds d_phi·H_p
    to the diagonal via conv_diag.  This ensures diag > 0 everywhere.

    Returns
    -------
    A, B, C, D : (N_Z, N_phi) frozen diffusion coefficients
    conv_rhs   : (N_Z, N_phi) frozen Couette RHS (uses θ_c=1 constant part)
    conv_diag  : (N_Z, N_phi) Couette diagonal contribution (>0 in cavitation)
    """
    A_coeff   = np.zeros((N_Z, N_phi), dtype=np.float64)
    B_coeff   = np.zeros((N_Z, N_phi), dtype=np.float64)
    C_coeff   = np.zeros((N_Z, N_phi), dtype=np.float64)
    D_coeff   = np.zeros((N_Z, N_phi), dtype=np.float64)
    conv_rhs  = np.zeros((N_Z, N_phi), dtype=np.float64)
    conv_diag = np.zeros((N_Z, N_phi), dtype=np.float64)

    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            # Periodic neighbours in φ
            j_plus  = j + 1 if j + 1 < N_phi - 1 else 1
            j_minus = j - 1 if j - 1 >= 1 else N_phi - 2

            psi_c  = psi[i, j]
            psi_jp = psi[i, j_plus]
            psi_jm = psi[i, j_minus]
            psi_ip = psi[i + 1, j]
            psi_im = psi[i - 1, j]

            # Node indicators
            gc   = 1.0 if psi_c  >= 0.0 else 0.0
            g_jp = 1.0 if psi_jp >= 0.0 else 0.0
            g_jm = 1.0 if psi_jm >= 0.0 else 0.0
            g_ip = 1.0 if psi_ip >= 0.0 else 0.0
            g_im = 1.0 if psi_im >= 0.0 else 0.0

            # Face indicators (product of neighbours)
            gf_jp = gc * g_jp
            gf_jm = gc * g_jm
            gf_ip = gc * g_ip
            gf_im = gc * g_im

            # Frozen diffusion coefficients
            A_coeff[i, j] = gf_jp * H3_face_p[i, j]
            B_coeff[i, j] = gf_jm * H3_face_m[i, j]
            C_coeff[i, j] = gf_ip * H3_face_zp[i, j]
            D_coeff[i, j] = gf_im * H3_face_zm[i, j]

            # --- Semi-implicit Couette treatment ---
            # θ_c: for active nodes (gc=1), θ_c=1 → entirely in RHS.
            #       for cavitation (gc=0), θ_c=1+ψ_c → constant "1" in RHS,
            #       ψ_c-dependent part → diagonal (d_phi · H_face_p).
            # θ_jm: always frozen (upstream node).
            theta_jm = 1.0 if psi_jm >= 0.0 else max(1.0 + psi_jm, 0.0)

            # conv_rhs uses θ_c = 1 (constant part) for BOTH active & cavitation
            conv_rhs[i, j] = d_phi * (H_face_p[i, j] * 1.0
                                     - H_face_m[i, j] * theta_jm)

            # Convective diagonal: only for cavitation nodes (gc=0)
            conv_diag[i, j] = (1.0 - gc) * d_phi * H_face_p[i, j]

    return A_coeff, B_coeff, C_coeff, D_coeff, conv_rhs, conv_diag


# ---------------------------------------------------------------------------
# Inner SOR loop (all coefficients FROZEN — no g/θ recomputation)
# ---------------------------------------------------------------------------
@njit
def _inner_sor_loop(psi, A_coeff, B_coeff, C_coeff, D_coeff,
                    conv_rhs, conv_diag,
                    N_Z, N_phi, omega_sor, max_inner, tol_inner,
                    clamp_min):
    """
    Sequential Gauss-Seidel / SOR for the unified variable ψ.

    Parameters
    ----------
    conv_diag : (N_Z, N_phi) — Couette contribution to diagonal
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

                # Diffusion numerator (frozen coefficients)
                diff_num = (A_coeff[i, j] * psi[i, j_plus]
                          + B_coeff[i, j] * psi[i, j_minus]
                          + C_coeff[i, j] * psi[i + 1, j]
                          + D_coeff[i, j] * psi[i - 1, j])

                # Total diagonal: diffusion + Couette semi-implicit
                diag = (A_coeff[i, j] + B_coeff[i, j]
                      + C_coeff[i, j] + D_coeff[i, j]
                      + conv_diag[i, j])

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
        If True: all gf=1, θ=1, conv_rhs=F_orig, clamp ψ≥0.
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
    # Precompute face data for JFO mode (node-indexed, correct physics)
    # H_face_p[i,j] = H at j+1/2 face of node j
    # H_face_m[i,j] = H at j-1/2 face of node j
    # ------------------------------------------------------------------
    H_face_p = np.zeros((N_Z, N_phi), dtype=np.float64)
    H_face_m = np.zeros((N_Z, N_phi), dtype=np.float64)

    H_face_p[:, :-1] = 0.5 * (H[:, :-1] + H[:, 1:])
    H_face_p[:, -1]  = 0.5 * (H[:, -1]  + H[:, 0])

    H_face_m[:, 1:]  = H_face_p[:, :-1]
    H_face_m[:, 0]   = H_face_p[:, -1]

    H3_face_p = H_face_p ** 3
    H3_face_m = H_face_m ** 3

    # Z-direction faces (node-indexed)
    H3_face_zp = np.zeros((N_Z, N_phi), dtype=np.float64)
    H3_face_zm = np.zeros((N_Z, N_phi), dtype=np.float64)
    H_jph = 0.5 * (H[:-1, :] + H[1:, :])
    H3_jph = H_jph ** 3
    # face i+1/2 for interior nodes i=1..N_Z-2
    H3_face_zp[1:-1, :] = alpha_sq * H3_jph[1:, :]
    # face i-1/2 for interior nodes i=1..N_Z-2
    H3_face_zm[1:-1, :] = alpha_sq * H3_jph[:-1, :]

    # ------------------------------------------------------------------
    # force_full_film: replicate precompute_coefficients_gpu() EXACTLY
    # so that test 0a (algebraic HS equivalence) passes.
    # ------------------------------------------------------------------
    if force_full_film:
        # --- phi faces (half-indexed, then mapped to full) ---
        H_iph_half = 0.5 * (H[:, :-1] + H[:, 1:])      # (N_Z, N_phi-1)
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

        # --- Z faces (same as precompute_coefficients_gpu) ---
        C_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        D_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        C_ff[1:-1, :] = alpha_sq * H3_jph[1:, :]
        D_ff[1:-1, :] = alpha_sq * H3_jph[:-1, :]

        # --- RHS = d_phi * (H_{j+1/2} - H_{j-1/2}) ---
        F_half = d_phi * (H_iph_half - H_imh_half)
        F_ff = np.zeros((N_Z, N_phi), dtype=np.float64)
        F_ff[:, :-1] = F_half
        F_ff[:, -1]  = F_half[:, 0]

        # No convective diagonal for full-film (all active, gc=1)
        conv_diag_ff = np.zeros((N_Z, N_phi), dtype=np.float64)

    # ------------------------------------------------------------------
    # Initialise ψ
    # ------------------------------------------------------------------
    psi = np.zeros((N_Z, N_phi), dtype=np.float64)

    clamp_min = 0.0 if force_full_film else -1.0

    # ------------------------------------------------------------------
    # Picard outer loop
    # ------------------------------------------------------------------
    n_inner_total = 0
    residual = 1.0
    n_outer_done = 0

    for outer in range(max_outer):
        psi_old = psi.copy()

        # Step 1: build frozen coefficients
        if force_full_film:
            A_coeff = A_ff
            B_coeff = B_ff
            C_coeff = C_ff
            D_coeff = D_ff
            conv_rhs = F_ff
            conv_diag_arr = conv_diag_ff
        else:
            A_coeff, B_coeff, C_coeff, D_coeff, conv_rhs, conv_diag_arr = \
                _build_frozen_coefficients(
                    psi, H_face_p, H_face_m,
                    H3_face_p, H3_face_m,
                    H3_face_zp, H3_face_zm,
                    alpha_sq, d_phi,
                    N_Z, N_phi,
                )

        # Step 2: inner SOR with frozen coefficients
        n_inner, inner_res = _inner_sor_loop(
            psi, A_coeff, B_coeff, C_coeff, D_coeff,
            conv_rhs, conv_diag_arr,
            N_Z, N_phi, omega_sor, max_inner, tol_inner,
            clamp_min,
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

    if force_full_film:
        theta = np.ones_like(psi)
    else:
        theta = np.where(psi >= 0.0, 1.0, 1.0 + psi)
        theta = np.clip(theta, 0.0, 1.0)

    return P, theta, residual, n_outer_done, n_inner_total
