"""
Unified API for solving the Reynolds equation.

from reynolds_solver import solve_reynolds

# Simple call (auto-selects method):
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L)

# With explicit method:
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L, method="krylov")
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L, method="sor")

# Dynamic equation:
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L,
                                    method="krylov",
                                    xprime=0.001, yprime=0.001, beta=2.0)
"""

import numpy as np
import cupy as cp

from reynolds_solver.physics.standard import StandardReynolds
from reynolds_solver.physics.standard_dynamic import StandardReynoldsDynamic
from reynolds_solver.assembly.sparse_builder import build_sparse_matrix_gpu
from reynolds_solver.linear_solvers.gpu_krylov import GPUKrylovSolver
from reynolds_solver.linear_solvers.gpu_sor import (
    solve_reynolds_sor,
    solve_reynolds_sor_dynamic,
)
from reynolds_solver.nonlinear.cavitation import solve_with_cavitation


def solve_reynolds(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    method: str = "krylov",
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 500,
    # Dynamic parameters
    xprime: float = 0.0,
    yprime: float = 0.0,
    beta: float = 2.0,
    # Krylov-specific
    krylov_tol: float = 1e-6,
    krylov_maxiter: int = 2000,
    max_cav_iter: int = 20,
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
        "sor" -- Red-Black SOR (GPU), "krylov" -- BiCGSTAB (GPU).
    omega : float
        SOR relaxation parameter (only for method="sor").
    tol : float
        Solution tolerance.
    max_iter : int
        Max iterations (SOR) or max Krylov iterations.
    xprime, yprime, beta : float
        Dynamic equation parameters (0 = static).
    krylov_tol : float
        BiCGSTAB tolerance (only for method="krylov").
    krylov_maxiter : int
        BiCGSTAB max iterations.
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
    N_Z, N_phi = H.shape
    is_dynamic = abs(xprime) > 1e-15 or abs(yprime) > 1e-15

    # --- SOR method (existing) ---
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

    # --- Krylov method (new) ---
    elif method == "krylov":
        H_gpu = cp.asarray(H, dtype=cp.float64)

        # 1. Build stencil coefficients
        if is_dynamic:
            builder = StandardReynoldsDynamic()
            A, B, C, D, E, F = builder.build(
                H_gpu, d_phi, d_Z, R, L,
                xprime=xprime, yprime=yprime, beta=beta,
            )
        else:
            builder = StandardReynolds()
            A, B, C, D, E, F = builder.build(H_gpu, d_phi, d_Z, R, L)

        # 2. Assemble sparse matrix
        M, f = build_sparse_matrix_gpu(A, B, C, D, E, F, N_Z, N_phi)

        # 3. Solve with cavitation
        krylov = GPUKrylovSolver(tol=krylov_tol, maxiter=krylov_maxiter)
        p_vec, n_outer = solve_with_cavitation(
            krylov, M, f, max_outer=max_cav_iter,
        )

        # 4. Reshape vector back to (N_Z, N_phi) matrix
        N_inner_phi = N_phi - 2
        P_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
        P_inner = p_vec.reshape((N_Z - 2, N_inner_phi))
        P_full[1:-1, 1:-1] = P_inner

        # Boundary conditions
        P_full[:, 0] = P_full[:, -2]       # periodic phi
        P_full[:, -1] = P_full[:, 1]
        P_full[0, :] = 0.0                  # Dirichlet Z
        P_full[-1, :] = 0.0

        # Compute residual for interface compatibility
        residual = M @ p_vec - f
        delta = float(cp.linalg.norm(residual)) / (float(cp.linalg.norm(f)) + 1e-12)

        P_cpu = cp.asnumpy(P_full)
        return P_cpu, delta, n_outer

    else:
        raise ValueError(f"Unknown method: {method}. Use 'sor' or 'krylov'.")
