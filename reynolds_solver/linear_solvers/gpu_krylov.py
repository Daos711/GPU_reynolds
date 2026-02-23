"""
GPU Krylov solver via cupyx.scipy.sparse.linalg.

Solves M @ p = f where M is a sparse CSR matrix on GPU.
Uses GMRES (non-symmetric Krylov method available in all CuPy versions).
"""

import cupy as cp
import cupyx.scipy.sparse.linalg as cla

from reynolds_solver.linear_solvers.base import LinearSolver


class GPUKrylovSolver(LinearSolver):
    """
    Linear system solver M @ p = f using a Krylov method on GPU.

    Tries solvers in order of preference:
      1. gmres  -- robust for non-symmetric systems, always available in CuPy

    Parameters
    ----------
    tol : float
        Solution tolerance (relative residual).
    maxiter : int
        Maximum number of Krylov iterations.
    """

    def __init__(self, tol=1e-6, maxiter=2000):
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, M, f):
        """
        Solve M @ p = f.

        Returns
        -------
        p : cp.ndarray, shape (N,)
        info : int
            0 = success, >0 = did not converge.
        """
        p, info = cla.gmres(M, f, tol=self.tol, maxiter=self.maxiter)
        return p, info
