"""
JFO cavitation — operator splitting approach.

Alternates between:
  Step A: Solve P from Reynolds eq with FIXED θ (HS-like SOR, many iterations)
  Step B: Update θ from converged P (upwind mass transport, single pass)

This avoids the GS instability from mixing P and θ updates in one sweep.
Coefficients A,B,C,D,E — identical to precompute_coefficients_gpu() in utils.py.
"""
import numpy as np
from numba import njit


@njit
def _sor_solve_P(P, A, B, C, D, E, F_theta,
                 N_Z, N_phi, omega, max_iter, tol):
    """Solve A*P_{j+1} + ... - E*P_j = F_theta with P >= 0 clamp. Standard HS SOR."""
    for iteration in range(max_iter):
        max_delta = 0.0
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                jp = j + 1 if j + 1 < N_phi - 1 else 1
                jm = j - 1 if j - 1 >= 1 else N_phi - 2

                P_old = P[i, j]
                P_new = (A[i,j]*P[i,jp] + B[i,j]*P[i,jm]
                       + C[i,j]*P[i+1,j] + D[i,j]*P[i-1,j]
                       - F_theta[i,j]) / (E[i,j] + 1e-30)
                if P_new < 0.0:
                    P_new = 0.0
                P[i,j] = P_old + omega * (P_new - P_old)

                d = abs(P[i,j] - P_old)
                if d > max_delta:
                    max_delta = d

        for i in range(N_Z):
            P[i, 0] = P[i, N_phi - 2]
            P[i, N_phi - 1] = P[i, 1]
        for j in range(N_phi):
            P[0, j] = 0.0
            P[N_Z - 1, j] = 0.0

        if max_delta < tol:
            return iteration + 1, max_delta
    return max_iter, max_delta


@njit
def _update_theta(theta, P, H_face_p, H_face_m, N_Z, N_phi, relax):
    """Update theta from converged P. Single upwind pass with under-relaxation."""
    max_dth = 0.0
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jm = j - 1 if j - 1 >= 1 else N_phi - 2
            th_old = theta[i, j]

            if P[i, j] > 0.0:
                th_target = 1.0
            else:
                H_fp = H_face_p[i, j]
                H_fm = H_face_m[i, j]
                if H_fp > 1e-30:
                    th_target = H_fm * theta[i, jm] / H_fp
                    if th_target < 0.0: th_target = 0.0
                    if th_target > 1.0: th_target = 1.0
                else:
                    th_target = theta[i, jm]

            theta[i, j] = th_old + relax * (th_target - th_old)

            d = abs(theta[i, j] - th_old)
            if d > max_dth:
                max_dth = d

    for i in range(N_Z):
        theta[i, 0] = theta[i, N_phi - 2]
        theta[i, N_phi - 1] = theta[i, 1]
    for j in range(N_phi):
        theta[0, j] = 1.0
        theta[N_Z - 1, j] = 1.0

    return max_dth


@njit
def _build_F_theta(H_face_p, H_face_m, theta, d_phi, N_Z, N_phi):
    """Build RHS: F_theta = d_phi * (H_{j+1/2} * theta_j - H_{j-1/2} * theta_{j-1})"""
    F = np.zeros((N_Z, N_phi), dtype=np.float64)
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jm = j - 1 if j - 1 >= 1 else N_phi - 2
            F[i, j] = d_phi * (H_face_p[i, j] * theta[i, j]
                              - H_face_m[i, j] * theta[i, jm])
    return F


def solve_jfo_splitting_cpu(
    H, d_phi, d_Z, R, L,
    omega=1.5, tol=1e-5, max_outer=100, max_inner=20000,
    tol_inner=1e-6, theta_relax=0.3,
    max_theta_sweeps=5, tol_theta=1e-4,
    verbose=False,
):
    """
    JFO cavitation via operator splitting (P, theta). CPU reference.

    Step A: Solve P from Reynolds with fixed theta (HS-like SOR).
    Step B: Update theta from converged P (upwind transport + under-relaxation).

    Parameters
    ----------
    H : (N_Z, N_phi) float64 — dimensionless gap
    d_phi, d_Z : float — grid spacing
    R, L : float — bearing radius and length (m)
    omega : float — SOR relaxation for pressure (1.0–1.8)
    tol : float — outer convergence on max(dP, dth)
    max_outer : int — max outer iterations
    max_inner : int — max SOR iterations per pressure solve
    tol_inner : float — inner SOR convergence
    theta_relax : float — under-relaxation for theta (0.1–0.5)
    max_theta_sweeps : int — max upwind sweeps per outer step for theta
    tol_theta : float — convergence tolerance for inner theta loop
    verbose : bool

    Returns
    -------
    P : (N_Z, N_phi) float64 — pressure (>= 0)
    theta : (N_Z, N_phi) float64 — fill fraction [0, 1]
    residual : float
    n_outer : int
    n_inner_total : int
    """
    N_Z, N_phi = H.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    # Coefficients — IDENTICAL to precompute_coefficients_gpu()
    H_iph = 0.5 * (H[:, :-1] + H[:, 1:])
    H_imh = np.empty_like(H_iph)
    H_imh[:, 1:] = H_iph[:, :-1]; H_imh[:, 0] = H_iph[:, -1]
    Ah = H_iph**3; Bh = np.empty_like(Ah)
    Bh[:, 1:] = Ah[:, :-1]; Bh[:, 0] = Ah[:, -1]
    A = np.zeros((N_Z, N_phi)); A[:, :-1] = Ah; A[:, -1] = Ah[:, 0]
    B = np.zeros((N_Z, N_phi)); B[:, 1:] = Bh; B[:, 0] = Bh[:, -1]
    Hjp = 0.5*(H[:-1,:]+H[1:,:]); H3z = Hjp**3
    C = np.zeros((N_Z, N_phi)); D = np.zeros((N_Z, N_phi))
    C[1:-1,:] = alpha_sq * H3z[1:,:]; D[1:-1,:] = alpha_sq * H3z[:-1,:]
    E = A + B + C + D

    Hfp = np.zeros((N_Z, N_phi)); Hfm = np.zeros((N_Z, N_phi))
    Hfp[:, :-1] = H_iph; Hfp[:, -1] = 0.5*(H[:,-1]+H[:,0])
    Hfm[:, 1:] = Hfp[:, :-1]; Hfm[:, 0] = Hfp[:, -1]

    P = np.zeros((N_Z, N_phi))
    theta = np.ones((N_Z, N_phi))

    total_inner = 0

    for outer in range(max_outer):
        F_theta = _build_F_theta(Hfp, Hfm, theta, d_phi, N_Z, N_phi)

        P_old = P.copy()
        ni, res_inner = _sor_solve_P(P, A, B, C, D, E, F_theta,
                                      N_Z, N_phi, omega, max_inner, tol_inner)
        total_inner += ni

        # Step B: Inner theta-loop at fixed P (transport H*theta = const)
        # relax=1.0: one sweep propagates through entire cavitation zone.
        theta_old = theta.copy()
        th_sweeps = 0
        for k in range(max_theta_sweeps):
            dth_inner = _update_theta(theta, P, Hfp, Hfm, N_Z, N_phi, 1.0)
            th_sweeps = k + 1
            if dth_inner < tol_theta:
                break

        dP = np.max(np.abs(P - P_old))
        dth_outer = np.max(np.abs(theta - theta_old))
        residual = max(dP, dth_outer)

        if verbose and (outer % 5 == 0 or outer < 3):
            cav = np.mean(theta[1:-1, 1:-1] < 1.0)
            print(f"  outer={outer:>3d}: dP={dP:.2e} dth={dth_outer:.2e} "
                  f"cav={cav:.3f} maxP={np.max(P):.4f} inner={ni} th_sweeps={th_sweeps}")

        if residual < tol:
            if verbose:
                cav = np.mean(theta[1:-1, 1:-1] < 1.0)
                print(f"  CONVERGED outer={outer}: cav={cav:.3f} maxP={np.max(P):.4f}")
            break

    return P, theta, residual, outer + 1, total_inner
