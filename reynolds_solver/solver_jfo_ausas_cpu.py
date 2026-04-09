"""
CPU reference for Ausas-style mass-conserving JFO cavitation solver.

Reference: Ausas, Jai, Buscaglia (2009),
"A Mass-Conserving Algorithm for Dynamical Lubrication Problems With
Cavitation", ASME J. Tribology, 131(3), 031702.

This is the splitting-free version that updates P and theta at each
node simultaneously with complementarity check (Table 1 of the paper).

Discretization is consistent with precompute_coefficients_gpu():
  A*P_{j+1} + B*P_{j-1} + C*P_{i+1} + D*P_{i-1} - E*P_{i,j} = F_theta
where
  A = H_face_p^3 (j+1/2)
  B = H_face_m^3 (j-1/2)
  C = alpha_sq * H_jph^3 (i+1/2)
  D = alpha_sq * H_jmh^3 (i-1/2)
  E = A + B + C + D
  F_theta = d_phi * (H_face_p * theta_{i,j} - H_face_m * theta_{i,j-1})
  alpha_sq = (D/L * d_phi/d_Z)^2

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
def _ausas_relax_sweep(P, theta, A, B, C, D, E, H_face_p, H_face_m,
                       d_phi, omega_p, omega_theta, N_Z, N_phi):
    """
    One full lexicographic Gauss-Seidel sweep (Ausas update rule).
    Returns sqrt(sum dP^2) + sqrt(sum dth^2) over interior nodes.
    """
    dP_sum = 0.0
    dth_sum = 0.0

    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jp = j + 1 if j + 1 < N_phi - 1 else 1
            jm = j - 1 if j - 1 >= 1 else N_phi - 2

            P_old = P[i, j]
            th_old = theta[i, j]
            H_jp = H_face_p[i, j]
            H_jm = H_face_m[i, j]
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
                # Convention: full-film means theta_{i,j} = 1 in the upwind RHS
                F_full = d_phi * (H_jp - H_jm * th_up)
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
                # From mass balance, solve for theta_{i,j}:
                # d_phi*H_jp*theta_{i,j} = stencil(P) - E*P_cur + d_phi*H_jm*th_up
                stencil = (
                    A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim - E_l * P_cur
                )
                Theta_trial = (stencil + d_phi * H_jm * th_up) / (d_phi * H_jp + 1e-30)
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

    # Z boundaries: Dirichlet P=0, theta clamped (NOT forced to 1)
    for j in range(N_phi):
        P[0, j] = 0.0
        P[N_Z - 1, j] = 0.0
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
    """Build A, B, C, D, E and face-H values matching precompute_coefficients_gpu."""
    N_Z, N_phi = H.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    H_iph = 0.5 * (H[:, :-1] + H[:, 1:])
    H_imh = np.empty_like(H_iph)
    H_imh[:, 1:] = H_iph[:, :-1]
    H_imh[:, 0] = H_iph[:, -1]

    Ah = H_iph ** 3
    Bh = np.empty_like(Ah)
    Bh[:, 1:] = Ah[:, :-1]
    Bh[:, 0] = Ah[:, -1]

    A = np.zeros((N_Z, N_phi))
    A[:, :-1] = Ah
    A[:, -1] = Ah[:, 0]

    B = np.zeros((N_Z, N_phi))
    B[:, 1:] = Bh
    B[:, 0] = Bh[:, -1]

    H_jph = 0.5 * (H[:-1, :] + H[1:, :])
    H_jph3 = H_jph ** 3

    C = np.zeros((N_Z, N_phi))
    D = np.zeros((N_Z, N_phi))
    C[1:-1, :] = alpha_sq * H_jph3[1:, :]
    D[1:-1, :] = alpha_sq * H_jph3[:-1, :]

    E = A + B + C + D

    # Face H values for the upwind RHS (matching solver_jfo.py face H)
    H_face_p = np.zeros((N_Z, N_phi))
    H_face_p[:, :-1] = H_iph
    H_face_p[:, -1] = 0.5 * (H[:, -1] + H[:, 0])

    H_face_m = np.zeros((N_Z, N_phi))
    H_face_m[:, 1:] = H_face_p[:, :-1]
    H_face_m[:, 0] = H_face_p[:, -1]

    return A, B, C, D, E, H_face_p, H_face_m


def solve_jfo_ausas_cpu(
    H, d_phi, d_Z, R, L,
    omega_p=1.0, omega_theta=1.0,
    omega_hs=1.7,
    tol=1e-6, max_iter=50000, check_every=50,
    hs_warmup_iter=2000, hs_warmup_tol=1e-7,
    P_init=None, theta_init=None, verbose=False,
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
    verbose : bool

    Returns
    -------
    P, theta : (N_Z, N_phi) float64
    residual : float
    n_iter : int (HS + Ausas iterations combined)
    """
    N_Z, N_phi = H.shape

    A, B, C, D, E, H_face_p, H_face_m = _build_coefficients(H, d_phi, d_Z, R, L)

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

    n_iter = 0

    # HS warmup: solve P with theta=1 fixed (only if no warm start P provided)
    if hs_warmup_iter > 0 and P_init is None:
        F_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                jm = j - 1 if j - 1 >= 1 else N_phi - 2
                F_hs[i, j] = d_phi * (H_face_p[i, j] - H_face_m[i, j])

        for k in range(hs_warmup_iter):
            dP = _hs_sor_sweep(P, A, B, C, D, E, F_hs, omega_hs, N_Z, N_phi)
            n_iter += 1
            if verbose and (k % 200 == 0 or k < 3):
                print(f"  [HS warmup] iter={k:>5d}: dP={dP:.4e}, maxP={P.max():.4e}")
            if dP < hs_warmup_tol and k > 5:
                if verbose:
                    print(f"  [HS warmup] CONVERGED at iter={k}, dP={dP:.4e}")
                break

    # Ausas relaxation
    residual = 1.0
    for k in range(max_iter):
        residual = _ausas_relax_sweep(
            P, theta, A, B, C, D, E, H_face_p, H_face_m,
            d_phi, omega_p, omega_theta, N_Z, N_phi,
        )
        n_iter += 1

        if verbose and (k % check_every == 0 or k < 5):
            cav_frac = float(np.mean(theta < 1.0 - 1e-6))
            print(f"  [Ausas] iter={k:>5d}: residual={residual:.4e}, cav={cav_frac:.3f}, maxP={P.max():.4e}")

        if residual < tol and k > 5:
            if verbose:
                print(f"  [Ausas] CONVERGED at iter={k}, residual={residual:.4e}")
            break

    return P, theta, residual, n_iter
