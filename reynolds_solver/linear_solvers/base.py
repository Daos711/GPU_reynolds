"""Abstract interface for linear solvers."""

from abc import ABC, abstractmethod


class LinearSolver(ABC):
    """Base class for linear solvers."""

    @abstractmethod
    def solve(self, M, f) -> tuple:
        """
        Solve M @ p = f.

        Parameters
        ----------
        M : sparse matrix (scipy or cupyx)
        f : array, shape (N,)

        Returns
        -------
        p : array, shape (N,)
        info : int
            0 = success, >0 = did not converge.
        """
        pass
