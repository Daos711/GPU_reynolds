"""
GPU Krylov solver via cupyx.scipy.sparse.linalg.

Solves M @ p = f where M is a sparse CSR matrix on GPU.
Uses GMRES with Jacobi (diagonal) preconditioning.
"""

import cupy as cp
import cupyx.scipy.sparse as cusparse
import cupyx.scipy.sparse.linalg as cla

from reynolds_solver.linear_solvers.base import LinearSolver


class GPUKrylovSolver(LinearSolver):
    """
    Linear system solver M @ p = f using GMRES on GPU.

    Includes Jacobi preconditioner (diagonal scaling) which reduces
    the condition number from O(N^2) to O(N) for Poisson-like systems.

    Parameters
    ----------
    tol : float
        Solution tolerance (relative residual).
    maxiter : int
        Maximum number of restart cycles.
    restart : int
        Inner iterations before restart (GMRES(restart)).
    """

    def __init__(self, tol=1e-6, maxiter=200, restart=50):
        self.tol = tol
        self.maxiter = maxiter
        self.restart = restart

    def solve(self, M, f):
        """
        Solve M @ p = f with Jacobi-preconditioned GMRES.

        Returns
        -------
        p : cp.ndarray, shape (N,)
        info : int
            0 = success, >0 = did not converge.
        """
        # Jacobi preconditioner: M_precond ~= diag(M)^{-1}
        diag = M.diagonal()
        diag_inv = cp.where(cp.abs(diag) > 1e-14, 1.0 / diag, 0.0)
        M_precond = cusparse.diags(diag_inv)

        p, info = cla.gmres(
            M, f,
            tol=self.tol,
            maxiter=self.maxiter,
            restart=self.restart,
            M=M_precond,
        )
        return p, info
