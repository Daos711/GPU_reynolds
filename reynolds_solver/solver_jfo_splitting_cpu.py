"""
JFO cavitation — operator splitting approach.

Alternates between:
  Step A: Solve P from Reynolds eq with FIXED θ (HS-like SOR, many iterations)
  Step B: Update zone_state with hysteresis, then march θ from rupture point

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
def _update_zone_state(zone_state, P, N_Z, N_phi, p_on, p_off):
    """Update zone classification with hysteresis. Only interior nodes."""
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            if P[i, j] > p_on:
                zone_state[i, j] = 1  # full-film
            elif P[i, j] < p_off:
                zone_state[i, j] = 0  # cavitation
            # else: keep previous (hysteresis band)


@njit
def _update_theta_march(theta, zone_state, H_face_p, H_face_m, N_Z, N_phi):
    """Rupture-anchored theta march for each Z-row.

    For each row:
      1. Full-film nodes (zone_state=1): theta = 1
      2. Find rupture point (full-film -> cavitation transition)
      3. March through cavitation zone from rupture with upwind transport
    """
    for i in range(1, N_Z - 1):
        # Physical columns: 1..N_phi-2
        n_phys = N_phi - 2  # number of physical columns

        # Set full-film nodes to theta=1
        any_cav = False
        for j in range(1, N_phi - 1):
            if zone_state[i, j] == 1:
                theta[i, j] = 1.0
            else:
                any_cav = True

        if not any_cav:
            continue  # entire row is full-film

        # Find rupture: last full-film node before a cavitation node (in +phi)
        # Scan physical columns in order, looking for transition 1 -> 0
        rupture_j = -1
        for jj in range(n_phys):
            j = 1 + jj  # physical column
            j_next = j + 1 if j + 1 < N_phi - 1 else 1
            if zone_state[i, j] == 1 and zone_state[i, j_next] == 0:
                rupture_j = j
                break  # take the first rupture

        if rupture_j < 0:
            # All cavitation, no full-film anchor — march from column 1
            # Use current theta[i, N_phi-2] as upstream (periodic)
            rupture_j = N_phi - 2  # "previous" full-film is the last phys column
            # But if it's also cavitation, just start with theta=1
            theta_prev = 1.0
            j = 1
            for _ in range(n_phys):
                if zone_state[i, j] == 1:
                    theta_prev = 1.0
                else:
                    H_fp = H_face_p[i, j]
                    H_fm = H_face_m[i, j]
                    if H_fp > 1e-30:
                        th_new = H_fm * theta_prev / H_fp
                        if th_new < 0.0:
                            th_new = 0.0
                        if th_new > 1.0:
                            th_new = 1.0
                        theta[i, j] = th_new
                        theta_prev = th_new
                    else:
                        theta[i, j] = theta_prev
                j = j + 1
                if j >= N_phi - 1:
                    j = 1
            continue

        # March from rupture point through cavitation zone
        theta_prev = 1.0  # at rupture, entering cavitation from full-film
        j = rupture_j + 1
        if j >= N_phi - 1:
            j = 1  # periodic wrap

        for _ in range(n_phys):
            if zone_state[i, j] == 1:
                break  # reached reformation point

            H_fp = H_face_p[i, j]
            H_fm = H_face_m[i, j]
            if H_fp > 1e-30:
                th_new = H_fm * theta_prev / H_fp
                if th_new < 0.0:
                    th_new = 0.0
                if th_new > 1.0:
                    th_new = 1.0
                theta[i, j] = th_new
                theta_prev = th_new
            else:
                theta[i, j] = theta_prev

            j = j + 1
            if j >= N_phi - 1:
                j = 1  # periodic wrap

    # Ghost columns (periodic)
    for i in range(N_Z):
        theta[i, 0] = theta[i, N_phi - 2]
        theta[i, N_phi - 1] = theta[i, 1]
    # Z boundaries
    for j in range(N_phi):
        theta[0, j] = 1.0
        theta[N_Z - 1, j] = 1.0


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
    tol_inner=1e-6,
    verbose=False,
):
    """
    JFO cavitation via operator splitting (P, theta). CPU reference.

    Step A: Solve P from Reynolds with fixed theta (HS-like SOR).
    Step B: Update zone_state (hysteresis), then march theta from rupture.

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
    zone_state = np.ones((N_Z, N_phi), dtype=np.int32)  # 1=full-film

    total_inner = 0
    residual = 1.0
    W_prev = 0.0
    theta_acc = np.zeros_like(theta)
    n_acc = 0
    acc_start = max(max_outer - 20, 0)

    for outer in range(max_outer):
        # Step A: build F_theta (blended RHS for stability), solve P
        F_theta_new = _build_F_theta(Hfp, Hfm, theta, d_phi, N_Z, N_phi)
        if outer == 0:
            F_theta = F_theta_new
        else:
            F_theta = 0.5 * F_theta_new + 0.5 * F_theta  # blend RHS

        P_old = P.copy()
        ni, res_inner = _sor_solve_P(P, A, B, C, D, E, F_theta,
                                      N_Z, N_phi, omega, max_inner, tol_inner)
        total_inner += ni

        # Step B: update zone_state with hysteresis, then march theta
        maxP = np.max(P)
        p_on = 1e-5 * maxP if maxP > 0.0 else 1e-10
        p_off = 1e-6 * maxP if maxP > 0.0 else 1e-11

        _update_zone_state(zone_state, P, N_Z, N_phi, p_on, p_off)
        _update_theta_march(theta, zone_state, Hfp, Hfm, N_Z, N_phi)

        # Convergence: |ΔW|/W
        W = np.sum(P)
        dP = np.max(np.abs(P - P_old))
        dW_rel = abs(W - W_prev) / (abs(W) + 1e-30) if outer > 0 else 1.0
        W_prev = W
        residual = dW_rel

        # Accumulate theta over last 20 steps
        if outer >= acc_start:
            theta_acc += theta.copy()
            n_acc += 1

        if verbose and (outer % 5 == 0 or outer < 3):
            cav = np.mean(zone_state[1:-1, 1:-1] == 0)
            print(f"  outer={outer:>3d}: dP={dP:.2e} dW={dW_rel:.2e} "
                  f"cav={cav:.3f} maxP={maxP:.4f} inner={ni} W={W:.2f}")

        if dW_rel < tol and outer > 5:
            if verbose:
                cav = np.mean(zone_state[1:-1, 1:-1] == 0)
                print(f"  CONVERGED outer={outer}: cav={cav:.3f} maxP={np.max(P):.4f}")
            break

    # Average theta over accumulated steps
    if n_acc > 0:
        theta = theta_acc / n_acc

    return P, theta, residual, outer + 1, total_inner
