"""
GPU solver for the Reynolds equation (dynamic version).

Extends the static solver with a dynamic contribution to the RHS:
    F[i,j] += beta * (xprime * sin(phi_global) + yprime * cos(phi_global))
where phi_global = j * d_phi + phase_shift (default 0.0).
"""

import numpy as np
import cupy as cp

from reynolds_solver.solver import _get_solver
from reynolds_solver.utils import precompute_coefficients_gpu, add_dynamic_rhs_gpu


def solve_reynolds_gpu_dynamic(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    xprime: float = 0.0,
    yprime: float = 0.0,
    beta: float = 2.0,
    phase_shift: float = 0.0,
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 500,
) -> tuple:
    """
    Drop-in replacement for solve_reynolds_gauss_seidel_numba_dynamic().

    Solves the dynamic Reynolds equation on GPU via Red-Black SOR.
    RHS includes dynamic contribution from velocities xprime, yprime.

    Parameters
    ----------
    H : np.ndarray, shape (N_Z, N_phi), float64
    d_phi, d_Z : float
    R, L : float
    xprime, yprime : float
    beta : float
    phase_shift : float
        Phase offset for dynamic RHS (default 0.0, no shift).
    omega : float
    tol : float
    max_iter : int
    check_every : int

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64
    delta : float
    n_iter : int
    """
    N_Z, N_phi = H.shape
    solver = _get_solver(N_Z, N_phi)

    # 1. Transfer H to GPU and precompute coefficients
    H_gpu = cp.asarray(H, dtype=cp.float64)
    A, B, C, D, E, F_full = precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)

    # 2. Add dynamic contribution to RHS
    add_dynamic_rhs_gpu(F_full, d_phi, N_Z, N_phi, xprime, yprime, beta, phase_shift)

    # 3. Solve with pre-computed coefficients
    P_gpu, delta, n_iter = solver.solve_with_rhs(
        H_gpu, F_full, A, B, C, D, E,
        omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
    )

    # 4. Transfer result to CPU
    P_cpu = cp.asnumpy(P_gpu)
    return P_cpu, delta, n_iter
