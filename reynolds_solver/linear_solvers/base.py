"""Abstract interface for linear solvers."""

from abc import ABC, abstractmethod
import cupy as cp
import cupyx.scipy.sparse as cusparse


class LinearSolver(ABC):
    """Base class for GPU linear solvers."""

    @abstractmethod
    def solve(self, M: cusparse.csr_matrix, f: cp.ndarray) -> tuple:
        """
        Solve M @ p = f.

        Parameters
        ----------
        M : cupyx.scipy.sparse.csr_matrix
        f : cp.ndarray, shape (N,)

        Returns
        -------
        p : cp.ndarray, shape (N,)
        info : int
            0 = success, >0 = did not converge.
        """
        pass
