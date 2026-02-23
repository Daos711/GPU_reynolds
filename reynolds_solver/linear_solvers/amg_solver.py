"""
Algebraic Multigrid solver via PyAMG.

AMG converges in O(N) iterations instead of O(N^2) for SOR/Krylov.
Works on CPU, but solves in 10-30 iterations what SOR takes 15,000 for.

For anisotropic problems (alpha^2 >> 1, typical for Reynolds equation in
bearings) uses Smoothed Aggregation with strength='evolution' which correctly
handles anisotropy.

NOTE: For the cavitation loop (where the matrix changes each iteration),
spsolve is faster than AMG because AMG hierarchy rebuild is expensive.
The AMGSolver class is kept for advanced use cases (no cavitation, or
solving the same matrix many times in parameter studies).
"""

import numpy as np
import pyamg


class AMGSolver:
    """
    Linear system solver M @ p = f using Algebraic Multigrid.

    Parameters
    ----------
    tol : float
        Solution tolerance.
    maxiter : int
        Maximum number of AMG V-cycles.
    """

    def __init__(self, tol=1e-8, maxiter=200):
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, M, f):
        """
        Solve M @ p = f using AMG.

        Parameters
        ----------
        M : scipy.sparse.csr_matrix
        f : np.ndarray, shape (N,)

        Returns
        -------
        p : np.ndarray, shape (N,)
        info : int
            0 = converged, 1 = did not converge.
        """
        ml = pyamg.smoothed_aggregation_solver(
            M,
            strength='evolution',
            smooth=('energy', {'degree': 2}),
            max_coarse=500,
        )

        residuals = []
        p = ml.solve(
            f,
            tol=self.tol,
            maxiter=self.maxiter,
            accel='bicgstab',
            residuals=residuals,
        )

        converged = len(residuals) > 0 and residuals[-1] / (residuals[0] + 1e-15) < self.tol
        info = 0 if converged else 1

        return p, info
