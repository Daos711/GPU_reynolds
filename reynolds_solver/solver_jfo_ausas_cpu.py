"""
CPU reference for Ausas-style mass-conserving JFO cavitation solver.

Reference: Ausas, Jai, Buscaglia (2009),
"A Mass-Conserving Algorithm for Dynamical Lubrication Problems With
Cavitation", ASME J. Tribology, 131(3), 031702.

This is the splitting-free version that updates P and theta at each
node simultaneously with complementarity check (Table 1 of the paper).

Discretization (Ausas eq. 13, cell-centered mass content):
  A*P_{j+1} + B*P_{j-1} + C*P_{i+1} + D*P_{i-1} - E*P_{i,j} = F_theta
where (average-of-cubes conductance):
  A = 0.5 * (h^3_{i,j}   + h^3_{i,j+1}) at phi face +
  B = 0.5 * (h^3_{i,j-1} + h^3_{i,j}  ) at phi face -
  C = alpha_sq * 0.5 * (h^3_{i,j}   + h^3_{i+1,j}) at Z face +
  D = alpha_sq * 0.5 * (h^3_{i-1,j} + h^3_{i,j}  ) at Z face -
  E = A + B + C + D
  alpha_sq = (2R/L * d_phi/d_Z)^2
and the (upwind) mass-content RHS uses cell-centered h (NOT face):
  F_theta = d_phi * (h_{i,j} * theta_{i,j} - h_{i,j-1} * theta_{i,j-1})

Per-node update rule:
  Branch 1: if (P > 0 OR theta == 1):
      P_trial = stencil_solve_for_P(theta_{i,j} = 1)
      P_new   = omega_p * P_trial + (1 - omega_p) * P_old
      if P_new >= 0: theta = 1
      else:          P = 0
  Branch 2: if (P <= 0 OR theta < 1):
      Theta_trial = stencil_solve_for_theta(P_{i,j} = current)
      theta_new   = omega_theta * Theta_trial + (1 - omega_theta) * theta_old
      if theta_new < 1: P = 0
      else:             theta = 1
"""
import numpy as np
from numba import njit


@njit(cache=True)
def _hs_sor_sweep(P, A, B, C, D, E, F_hs, omega, N_Z, N_phi):
    """One Gauss-Seidel sweep of HS-like Reynolds solver (theta=1 fixed)."""
    max_dP = 0.0
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jp = j + 1 if j + 1 < N_phi - 1 else 1
            jm = j - 1 if j - 1 >= 1 else N_phi - 2

            P_old = P[i, j]
            P_new = (A[i,j]*P[i,jp] + B[i,j]*P[i,jm]
                   + C[i,j]*P[i+1,j] + D[i,j]*P[i-1,j]
                   - F_hs[i,j]) / (E[i,j] + 1e-30)
            if P_new < 0.0:
                P_new = 0.0
            P[i, j] = P_old + omega * (P_new - P_old)

            d = abs(P[i,j] - P_old)
            if d > max_dP:
                max_dP = d

    for i in range(N_Z):
        P[i, 0] = P[i, N_phi - 2]
        P[i, N_phi - 1] = P[i, 1]
    for j in range(N_phi):
        P[0, j] = 0.0
        P[N_Z - 1, j] = 0.0

    return max_dP


@njit(cache=True)
def _ausas_relax_sweep(P, theta, H, A, B, C, D, E,
                       d_phi, omega_p, omega_theta, N_Z, N_phi, flooded_ends):
    """
    One full lexicographic Gauss-Seidel sweep (Ausas update rule).
    Returns sqrt(sum dP^2) + sqrt(sum dth^2) over interior nodes.

    Mass content is cell-centered: c_{i,j} = theta_{i,j} * h_{i,j} (NOT face).
    """
    dP_sum = 0.0
    dth_sum = 0.0

    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jp = j + 1 if j + 1 < N_phi - 1 else 1
            jm = j - 1 if j - 1 >= 1 else N_phi - 2

            P_old = P[i, j]
            th_old = theta[i, j]
            h_ij = H[i, j]
            h_jm = H[i, jm]
            th_up = theta[i, jm]

            A_l = A[i, j]
            B_l = B[i, j]
            C_l = C[i, j]
            D_l = D[i, j]
            E_l = E[i, j]

            Pjp = P[i, jp]
            Pjm = P[i, jm]
            Pip = P[i + 1, j]
            Pim = P[i - 1, j]

            P_cur = P_old
            th_cur = th_old

            # Branch 1: pressure update if was full-film
            if P_old > 0.0 or th_old >= 1.0 - 1e-12:
                # Cell-centered mass content with theta_{i,j} = 1 locally:
                # F_full = d_phi * (h_{i,j} * 1 - h_{i,j-1} * theta_{i,j-1})
                F_full = d_phi * (h_ij - h_jm * th_up)
                P_trial = (
                    A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim - F_full
                ) / (E_l + 1e-30)
                P_new = omega_p * P_trial + (1.0 - omega_p) * P_old

                if P_new >= 0.0:
                    P_cur = P_new
                    th_cur = 1.0
                else:
                    P_cur = 0.0
                    # th_cur stays as th_old

            # Branch 2: theta update if cavitation or partial
            if P_cur <= 0.0 or th_cur < 1.0 - 1e-12:
                # Cell-centered mass balance, solve for theta_{i,j}:
                # d_phi*h_{i,j}*theta_{i,j} = stencil(P) - E*P_cur + d_phi*h_{i,j-1}*th_up
                stencil = (
                    A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim - E_l * P_cur
                )
                Theta_trial = (stencil + d_phi * h_jm * th_up) / (d_phi * h_ij + 1e-30)
                th_new = omega_theta * Theta_trial + (1.0 - omega_theta) * th_cur

                if th_new < 1.0:
                    if th_new < 0.0:
                        th_new = 0.0
                    th_cur = th_new
                    P_cur = 0.0
                else:
                    th_cur = 1.0
                    # P_cur stays

            P[i, j] = P_cur
            theta[i, j] = th_cur

            dP_sum += (P_cur - P_old) ** 2
            dth_sum += (th_cur - th_old) ** 2

    # Periodic ghost columns (for both P and theta)
    for i in range(N_Z):
        P[i, 0] = P[i, N_phi - 2]
        P[i, N_phi - 1] = P[i, 1]
        theta[i, 0] = theta[i, N_phi - 2]
        theta[i, N_phi - 1] = theta[i, 1]

    # Z boundaries: Dirichlet P=0; theta=1 for flooded bearing (default),
    # otherwise clamped to [0, 1].
    for j in range(N_phi):
        P[0, j] = 0.0
        P[N_Z - 1, j] = 0.0
        if flooded_ends != 0:
            theta[0, j] = 1.0
            theta[N_Z - 1, j] = 1.0
        else:
            if theta[0, j] < 0.0:
                theta[0, j] = 0.0
            elif theta[0, j] > 1.0:
                theta[0, j] = 1.0
            if theta[N_Z - 1, j] < 0.0:
                theta[N_Z - 1, j] = 0.0
            elif theta[N_Z - 1, j] > 1.0:
                theta[N_Z - 1, j] = 1.0

    return np.sqrt(dP_sum) + np.sqrt(dth_sum)


def _build_coefficients(H, d_phi, d_Z, R, L):
    """
    Build Ausas A, B, C, D, E with average-of-cubes conductance (Ausas eq. 13).

    A_{i,j} = 0.5 * (h^3_{i,j}   + h^3_{i,j+1}),   phi face (+)
    B_{i,j} = 0.5 * (h^3_{i,j-1} + h^3_{i,j}  ),   phi face (-)
    C_{i,j} = alpha_sq * 0.5 * (h^3_{i,j}   + h^3_{i+1,j}),  Z face (+)
    D_{i,j} = alpha_sq * 0.5 * (h^3_{i-1,j} + h^3_{i,j}  ),  Z face (-)
    E = A + B + C + D
    """
    N_Z, N_phi = H.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    # phi-direction face conductance (average of cubes, NOT cube of average)
    Ah = 0.5 * (H[:, :-1] ** 3 + H[:, 1:] ** 3)     # shape (N_Z, N_phi-1)
    Bh = np.empty_like(Ah)
    Bh[:, 1:] = Ah[:, :-1]
    Bh[:, 0] = Ah[:, -1]

    A = np.zeros((N_Z, N_phi))
    A[:, :-1] = Ah
    A[:, -1] = Ah[:, 0]

    B = np.zeros((N_Z, N_phi))
    B[:, 1:] = Bh
    B[:, 0] = Bh[:, -1]

    # Z-direction face conductance (average of cubes)
    H_jph3 = 0.5 * (H[:-1, :] ** 3 + H[1:, :] ** 3)  # shape (N_Z-1, N_phi)

    C = np.zeros((N_Z, N_phi))
    D = np.zeros((N_Z, N_phi))
    C[1:-1, :] = alpha_sq * H_jph3[1:, :]
    D[1:-1, :] = alpha_sq * H_jph3[:-1, :]

    E = A + B + C + D

    return A, B, C, D, E


def solve_jfo_ausas_cpu(
    H, d_phi, d_Z, R, L,
    omega_p=1.0, omega_theta=1.0,
    omega_hs=1.7,
    tol=1e-6, max_iter=50000, check_every=50,
    hs_warmup_iter=2000, hs_warmup_tol=1e-7,
    P_init=None, theta_init=None,
    flooded_ends=True,
    verbose=False,
):
    """
    Ausas-style mass-conserving JFO solver — CPU reference (Numba JIT).

    Performs an HS-like warmup (theta=1 fixed) before Ausas relaxation
    to establish a good pressure profile, then runs Ausas iterations.

    Parameters
    ----------
    H : (N_Z, N_phi) float64 — dimensionless gap
    d_phi, d_Z : float — grid spacing
    R, L : float — bearing radius and length (m)
    omega_p, omega_theta : float — Ausas relaxation factors
    omega_hs : float — SOR omega for HS warmup
    tol : float — convergence on ||dP||_2 + ||dtheta||_2 (Ausas)
    max_iter : int — max Ausas relaxation iterations
    check_every : int — convergence check frequency
    hs_warmup_iter : int — max HS warmup iterations (0 to skip)
    hs_warmup_tol : float — HS warmup convergence tolerance
    P_init, theta_init : (N_Z, N_phi) float64 or None — warm start
    flooded_ends : bool — if True (default, flooded bearing) force theta=1
        on Z boundaries; otherwise clamp theta to [0, 1] without forcing.
    verbose : bool

    Returns
    -------
    P, theta : (N_Z, N_phi) float64
    residual : float
    n_iter : int (HS + Ausas iterations combined)
    """
    N_Z, N_phi = H.shape
    # Defensive H ghost packing: ensure column 0 and column N_phi-1 are
    # proper periodic copies of the physical seam (N_phi-2 and 1). The test
    # case generator fills these with H(phi=0) = H(phi=2π), which does not
    # match the ghost-wrap expected by the vectorized coefficient assembly
    # and produces a ~0.1% error on A[:,N_phi-2] and B[:,1]. Defensive and
    # safe: if H already satisfies this, it is a no-op.
    H = np.ascontiguousarray(H, dtype=np.float64).copy()
    H[:, 0] = H[:, N_phi - 2]
    H[:, N_phi - 1] = H[:, 1]

    A, B, C, D, E = _build_coefficients(H, d_phi, d_Z, R, L)

    if P_init is not None:
        P = np.maximum(P_init.copy().astype(np.float64), 0.0)
    else:
        P = np.zeros((N_Z, N_phi), dtype=np.float64)

    if theta_init is not None:
        theta = np.clip(theta_init.copy().astype(np.float64), 0.0, 1.0)
    else:
        theta = np.ones((N_Z, N_phi), dtype=np.float64)

    P[0, :] = 0.0
    P[-1, :] = 0.0
    if flooded_ends:
        theta[0, :] = 1.0
        theta[-1, :] = 1.0

    flooded_flag = 1 if flooded_ends else 0

    n_iter = 0

    # HS warmup: solve P with theta=1 fixed (only if no warm start P provided).
    # RHS uses cell-centered upwind mass content: F_hs = d_phi*(h_{i,j} - h_{i,j-1}).
    if hs_warmup_iter > 0 and P_init is None:
        F_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                jm = j - 1 if j - 1 >= 1 else N_phi - 2
                F_hs[i, j] = d_phi * (H[i, j] - H[i, jm])

        for k in range(hs_warmup_iter):
            dP = _hs_sor_sweep(P, A, B, C, D, E, F_hs, omega_hs, N_Z, N_phi)
            n_iter += 1
            if verbose and (k % 200 == 0 or k < 3):
                print(f"  [HS warmup] iter={k:>5d}: dP={dP:.4e}, maxP={P.max():.4e}")
            if dP < hs_warmup_tol and k > 5:
                if verbose:
                    print(f"  [HS warmup] CONVERGED at iter={k}, dP={dP:.4e}")
                break

    if verbose:
        print(f"  [HS warmup DONE] n_iter={n_iter}, maxP={P.max():.4e}")

    # Ausas relaxation
    residual = 1.0
    for k in range(max_iter):
        residual = _ausas_relax_sweep(
            P, theta, H, A, B, C, D, E,
            d_phi, omega_p, omega_theta, N_Z, N_phi, flooded_flag,
        )
        n_iter += 1

        if verbose and (k % check_every == 0 or k < 5):
            # Diagnostic: zombie (theta~1 & P~0), full-film (theta~1 & P>0),
            # cavitation (theta<1), computed on interior only.
            P_int = P[1:-1, 1:-1]
            th_int = theta[1:-1, 1:-1]
            ff_mask = (th_int > 1.0 - 1e-8) & (P_int > 1e-12)
            zombie_mask = (th_int > 1.0 - 1e-8) & (P_int <= 1e-12)
            cav_mask = th_int < 1.0 - 1e-8
            n_ff = int(np.sum(ff_mask))
            n_zombie = int(np.sum(zombie_mask))
            n_cav = int(np.sum(cav_mask))
            cav_frac = float(np.mean(theta < 1.0 - 1e-6))
            print(
                f"  [Ausas] iter={k:>5d}: residual={residual:.4e}, "
                f"cav={cav_frac:.3f}, maxP={P.max():.4e}, "
                f"zombie={n_zombie}, ff={n_ff}, cav_n={n_cav}"
            )

        if residual < tol and k > 5:
            if verbose:
                print(f"  [Ausas] CONVERGED at iter={k}, residual={residual:.4e}")
            break

    return P, theta, residual, n_iter
