"""
Unified API for solving the Reynolds equation.

from reynolds_solver import solve_reynolds

# Simple call (auto-selects method):
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L)

# With explicit method:
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L, method="amg")
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L, method="sor")
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L, method="direct")

# Dynamic equation:
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L,
                                    method="amg",
                                    xprime=0.001, yprime=0.001, beta=2.0)
"""

import numpy as np
import cupy as cp

from reynolds_solver.physics.standard import StandardReynolds
from reynolds_solver.physics.standard_dynamic import StandardReynoldsDynamic
from reynolds_solver.assembly.sparse_builder import build_sparse_matrix_cpu
from reynolds_solver.linear_solvers.gpu_sor import (
    solve_reynolds_sor,
    solve_reynolds_sor_dynamic,
)
from reynolds_solver.nonlinear.cavitation import solve_with_cavitation_cpu


def _build_stencil(H, d_phi, d_Z, R, L, is_dynamic, xprime, yprime, beta):
    """Build stencil coefficients on GPU, return as cupy arrays."""
    H_gpu = cp.asarray(H, dtype=cp.float64)

    if is_dynamic:
        builder = StandardReynoldsDynamic()
        A, B, C, D, E, F = builder.build(
            H_gpu, d_phi, d_Z, R, L,
            xprime=xprime, yprime=yprime, beta=beta,
        )
    else:
        builder = StandardReynolds()
        A, B, C, D, E, F = builder.build(H_gpu, d_phi, d_Z, R, L)

    return A, B, C, D, E, F


def _reshape_solution(p_vec, N_Z, N_phi, M, f):
    """Reshape solution vector to (N_Z, N_phi) with boundary conditions."""
    N_inner_phi = N_phi - 2
    P_full = np.zeros((N_Z, N_phi), dtype=np.float64)
    P_inner = p_vec.reshape((N_Z - 2, N_inner_phi))
    P_full[1:-1, 1:-1] = P_inner

    # Boundary conditions
    P_full[:, 0] = P_full[:, -2]       # periodic phi
    P_full[:, -1] = P_full[:, 1]
    P_full[0, :] = 0.0                  # Dirichlet Z
    P_full[-1, :] = 0.0

    # Compute residual
    residual = M @ p_vec - f
    delta = float(np.linalg.norm(residual)) / (float(np.linalg.norm(f)) + 1e-12)

    return P_full, delta


def solve_reynolds(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    method: str = "amg",
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 500,
    # Dynamic parameters
    xprime: float = 0.0,
    yprime: float = 0.0,
    beta: float = 2.0,
    # AMG/direct-specific
    amg_tol: float = 1e-8,
    amg_maxiter: int = 200,
    max_cav_iter: int = 20,
    # Backward compatibility aliases
    krylov_tol: float = None,
    krylov_maxiter: int = None,
) -> tuple:
    """
    Universal Reynolds equation solver.

    Parameters
    ----------
    H : np.ndarray, shape (N_Z, N_phi), float64
        Dimensionless gap.
    d_phi, d_Z : float
        Grid step sizes.
    R, L : float
        Bearing radius and length (m).
    method : str
        "sor" -- Red-Black SOR (GPU),
        "amg" -- spsolve with cavitation (matrix assembly via GPU stencil),
        "direct" -- alias for "amg",
        "krylov" -- alias for "amg" (backward compatibility).
    omega : float
        SOR relaxation parameter (only for method="sor").
    tol : float
        Solution tolerance.
    max_iter : int
        Max iterations (SOR).
    xprime, yprime, beta : float
        Dynamic equation parameters (0 = static).
    max_cav_iter : int
        Max outer cavitation iterations.

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64
    delta : float
        Relative residual.
    n_iter : int
        Number of iterations.
    """
    # Backward compatibility aliases
    if method in ("krylov", "direct"):
        method = "amg"

    N_Z, N_phi = H.shape
    is_dynamic = abs(xprime) > 1e-15 or abs(yprime) > 1e-15

    # --- SOR method (existing GPU solver) ---
    if method == "sor":
        if is_dynamic:
            return solve_reynolds_sor_dynamic(
                H, d_phi, d_Z, R, L,
                xprime=xprime, yprime=yprime, beta=beta,
                omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
            )
        else:
            return solve_reynolds_sor(
                H, d_phi, d_Z, R, L,
                omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
            )

    # --- AMG method (spsolve + cavitation) ---
    elif method == "amg":
        A, B, C, D, E, F = _build_stencil(
            H, d_phi, d_Z, R, L, is_dynamic, xprime, yprime, beta,
        )

        M, f = build_sparse_matrix_cpu(A, B, C, D, E, F, N_Z, N_phi)

        p_vec, n_outer = solve_with_cavitation_cpu(
            M, f, max_outer=max_cav_iter,
        )

        P_full, delta = _reshape_solution(p_vec, N_Z, N_phi, M, f)
        return P_full, delta, n_outer

    else:
        raise ValueError(f"Unknown method: {method}. Use 'sor', 'amg', or 'direct'.")
