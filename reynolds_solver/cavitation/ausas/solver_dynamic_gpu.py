"""
GPU solver for the UNSTEADY Ausas mass-conserving JFO cavitation problem.

Stage 1 (this file): one-step wrapper `ausas_unsteady_one_step_gpu` that
advances (P, θ) over a single real time step Δt using implicit Euler and
the two-branch Ausas complementarity update. The actual CUDA sweep lives
in `kernels_dynamic.unsteady_ausas_step`.

This solver is intentionally NOT a port of the stationary Payvar–Salant
or Ausas GPU solvers. In particular it does NOT freeze an active set —
each cell may switch between full-film and cavitation on every sweep.

Intended as a building block for the full time-marching loop (Stage 2)
and the dynamic journal-bearing solver (Stage 3).
"""

import numpy as np
import cupy as cp

from reynolds_solver.cavitation.ausas.kernels_dynamic import (
    get_unsteady_ausas_kernel,
    get_unsteady_ausas_bc_kernel,
)


def _build_coefficients_gpu(H_gpu, d_phi, d_Z, R, L):
    """
    Average-of-cubes Poiseuille conductance coefficients (identical to the
    stationary Ausas GPU solver — see cavitation/ausas/solver_gpu.py).

    Requires H_gpu to be ghost-packed in the φ-direction (columns 0 and
    N_phi-1 already contain the periodic copies).
    """
    N_Z, N_phi = H_gpu.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    H3 = H_gpu ** 3

    Ah = 0.5 * (H3[:, :-1] + H3[:, 1:])           # (N_Z, N_phi-1)

    A = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    B = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    A[:, :-1] = Ah
    A[:, -1] = Ah[:, 0]
    B[:, 1:] = Ah
    B[:, 0] = Ah[:, -1]

    H_jph3 = 0.5 * (H3[:-1, :] + H3[1:, :])       # (N_Z-1, N_phi)

    C = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    D = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    C[1:-1, :] = alpha_sq * H_jph3[1:, :]
    D[1:-1, :] = alpha_sq * H_jph3[:-1, :]

    E = A + B + C + D
    return A, B, C, D, E


def ausas_unsteady_one_step_gpu(
    H_curr,
    H_prev,
    P_prev,
    theta_prev,
    dt,
    d_phi,
    d_Z,
    R,
    L,
    alpha=1.0,
    omega_p=1.0,
    omega_theta=1.0,
    tol=1e-6,
    max_inner=5000,
    p_bc_z0=0.0,
    p_bc_zL=0.0,
    theta_bc_z0=1.0,
    theta_bc_zL=1.0,
    periodic_phi=True,
    check_every=50,
    verbose=False,
):
    """
    Advance (P, θ) by one real time step Δt using the unsteady Ausas
    Jacobi kernel.

    Parameters
    ----------
    H_curr : (N_Z, N_phi) ndarray — dimensionless gap at time step n.
    H_prev : (N_Z, N_phi) ndarray — dimensionless gap at time step n-1.
    P_prev : (N_Z, N_phi) ndarray — pressure at n-1 (warm-start seed).
    theta_prev : (N_Z, N_phi) ndarray — θ at n-1.
    dt : float — real time step.
    d_phi, d_Z : float — grid spacings (radians, Z/L).
    R, L : float — bearing radius, half-length.
    alpha : float — dimensionless slip-velocity scaling (= 1 for classical
        journal bearings). Enters only the Couette and time coupling terms.
    omega_p, omega_theta : float — Ausas relaxation factors for the P and θ
        updates.
    tol : float — convergence criterion on ‖P^{k+1}−P^k‖₂ + ‖θ^{k+1}−θ^k‖₂.
    max_inner : int — max Jacobi sweeps inside the step.
    p_bc_z0, p_bc_zL : float — Dirichlet P on the two Z-ends (default 0).
    theta_bc_z0, theta_bc_zL : float — Dirichlet θ on the two Z-ends
        (default 1 = flooded bearing).
    periodic_phi : bool — periodic φ ghost sync.
    check_every, verbose : diagnostics.

    Returns
    -------
    P : (N_Z, N_phi) ndarray (CPU, float64)
    theta : (N_Z, N_phi) ndarray (CPU, float64)
    residual : float — last inner residual at time step n.
    n_inner : int — Jacobi sweeps actually performed.
    """
    H_curr = np.ascontiguousarray(H_curr, dtype=np.float64)
    H_prev = np.ascontiguousarray(H_prev, dtype=np.float64)
    P_prev = np.ascontiguousarray(P_prev, dtype=np.float64)
    theta_prev = np.ascontiguousarray(theta_prev, dtype=np.float64)

    N_Z, N_phi = H_curr.shape
    if H_prev.shape != (N_Z, N_phi):
        raise ValueError("H_prev shape must match H_curr.")
    if P_prev.shape != (N_Z, N_phi):
        raise ValueError("P_prev shape must match H_curr.")
    if theta_prev.shape != (N_Z, N_phi):
        raise ValueError("theta_prev shape must match H_curr.")

    # Upload to GPU and defensively re-pack φ-ghosts.
    H_curr_gpu = cp.asarray(H_curr)
    H_prev_gpu = cp.asarray(H_prev)
    if periodic_phi:
        H_curr_gpu[:, 0] = H_curr_gpu[:, N_phi - 2]
        H_curr_gpu[:, N_phi - 1] = H_curr_gpu[:, 1]
        H_prev_gpu[:, 0] = H_prev_gpu[:, N_phi - 2]
        H_prev_gpu[:, N_phi - 1] = H_prev_gpu[:, 1]

    theta_prev_gpu = cp.asarray(theta_prev)
    C_prev = theta_prev_gpu * H_prev_gpu   # frozen for the whole inner loop

    # Stencil coefficients from H_curr (implicit-Euler).
    A, B, C, D, E = _build_coefficients_gpu(H_curr_gpu, d_phi, d_Z, R, L)

    # Iterate buffers (ping-pong).
    P_old = cp.asarray(P_prev)
    theta_old = cp.asarray(theta_prev)
    cp.maximum(P_old, 0.0, out=P_old)
    cp.clip(theta_old, 0.0, 1.0, out=theta_old)

    # Apply BCs to the initial iterate so ghosts are consistent.
    if periodic_phi:
        P_old[:, 0] = P_old[:, N_phi - 2]
        P_old[:, N_phi - 1] = P_old[:, 1]
        theta_old[:, 0] = theta_old[:, N_phi - 2]
        theta_old[:, N_phi - 1] = theta_old[:, 1]
    P_old[0, :] = p_bc_z0
    P_old[-1, :] = p_bc_zL
    theta_old[0, :] = theta_bc_z0
    theta_old[-1, :] = theta_bc_zL

    P_new = P_old.copy()
    theta_new = theta_old.copy()

    kernel = get_unsteady_ausas_kernel()
    bc_kernel = get_unsteady_ausas_bc_kernel()

    block = (32, 8, 1)
    grid = (
        (N_phi - 2 + block[0] - 1) // block[0],
        (N_Z - 2 + block[1] - 1) // block[1],
        1,
    )
    max_dim = max(N_Z, N_phi)
    bc_block = (256, 1, 1)
    bc_grid = ((max_dim + 255) // 256, 1, 1)

    periodic_flag = 1 if periodic_phi else 0

    residual = float("inf")
    n_inner = 0
    for k in range(max_inner):
        kernel(
            grid, block,
            (
                P_old, P_new, theta_old, theta_new,
                H_curr_gpu, C_prev,
                A, B, C, D, E,
                np.float64(d_phi), np.float64(d_Z),
                np.float64(dt), np.float64(alpha),
                np.float64(omega_p), np.float64(omega_theta),
                np.int32(N_Z), np.int32(N_phi),
                np.int32(periodic_flag),
            ),
        )
        bc_kernel(
            bc_grid, bc_block,
            (
                P_new, theta_new,
                np.int32(N_Z), np.int32(N_phi),
                np.int32(periodic_flag),
                np.float64(p_bc_z0), np.float64(p_bc_zL),
                np.float64(theta_bc_z0), np.float64(theta_bc_zL),
            ),
        )
        n_inner += 1

        if k % check_every == 0 or k < 3:
            dP = float(cp.sqrt(cp.sum((P_new - P_old) ** 2)))
            dth = float(cp.sqrt(cp.sum((theta_new - theta_old) ** 2)))
            residual = dP + dth
            if verbose:
                print(
                    f"  [Ausas-dyn-GPU] inner={k:>5d}: residual={residual:.4e}, "
                    f"dP={dP:.2e}, dth={dth:.2e}, maxP={float(cp.max(P_new)):.4e}"
                )

        # Swap ping-pong buffers (P_new / theta_new hold the new iterate;
        # make them the old iterate for the next sweep).
        P_old, P_new = P_new, P_old
        theta_old, theta_new = theta_new, theta_old

        if residual < tol and k > 2:
            if verbose:
                print(
                    f"  [Ausas-dyn-GPU] CONVERGED at inner={k}, "
                    f"residual={residual:.4e}"
                )
            break

    # After swap, the freshly-computed iterate now lives in P_old / theta_old.
    P_cpu = cp.asnumpy(P_old)
    theta_cpu = cp.asnumpy(theta_old)
    return P_cpu, theta_cpu, float(residual), n_inner
