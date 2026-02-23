"""
Unified API: solve_reynolds().

Examples:
    # Static equation:
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L)

    # Dynamic equation:
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L,
                                       xprime=0.001, yprime=0.001)

    # With solver settings:
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L,
                                       omega=1.7, tol=1e-6, max_iter=100000)
"""

import numpy as np
from reynolds_solver.solver import solve_reynolds_gpu
from reynolds_solver.solver_dynamic import solve_reynolds_gpu_dynamic


def solve_reynolds(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 500,
    # Dynamic parameters
    xprime: float = 0.0,
    yprime: float = 0.0,
    beta: float = 2.0,
) -> tuple:
    """
    Solve the Reynolds equation on GPU (Red-Black SOR).

    Parameters
    ----------
    H : np.ndarray, shape (N_Z, N_phi), float64
        Dimensionless gap.
    d_phi, d_Z : float
        Grid steps along phi and Z.
    R, L : float
        Bearing radius and length (m).
    omega : float
        SOR relaxation parameter (1.0-1.9, optimum ~1.5).
    tol : float
        Convergence criterion (relative residual).
    max_iter : int
        Maximum number of iterations.
    check_every : int
        Convergence check frequency.
    xprime, yprime : float
        Dimensionless velocities (for dynamic equation, 0 = static).
    beta : float
        Dynamic term coefficient.

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64
        Dimensionless pressure field.
    delta : float
        Final relative residual.
    n_iter : int
        Number of iterations to convergence.
    """
    is_dynamic = abs(xprime) > 1e-15 or abs(yprime) > 1e-15

    if is_dynamic:
        return solve_reynolds_gpu_dynamic(
            H, d_phi, d_Z, R, L,
            xprime=xprime, yprime=yprime, beta=beta,
            omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
        )
    else:
        return solve_reynolds_gpu(
            H, d_phi, d_Z, R, L,
            omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
        )
