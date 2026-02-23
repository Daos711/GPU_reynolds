"""Abstract interface for building 5-point stencil coefficients."""

from abc import ABC, abstractmethod
import cupy as cp


class StencilBuilder(ABC):
    """
    Base class for building 5-point stencil coefficients.

    Any modification of the Reynolds equation (roughness, thermal, etc.)
    inherits from this class and implements build().

    Stencil:
        A[i,j]*P[i,j+1] + B[i,j]*P[i,j-1] + C[i,j]*P[i+1,j] + D[i,j]*P[i-1,j]
        - E[i,j]*P[i,j] = F[i,j]

    Boundary conditions:
        - phi: periodic (j=0 <-> j=N_phi-2, j=N_phi-1 <-> j=1)
        - Z: Dirichlet P=0 (i=0, i=N_Z-1)
    """

    @abstractmethod
    def build(self, H_gpu: cp.ndarray, d_phi: float, d_Z: float,
              R: float, L: float, **kwargs) -> tuple:
        """
        Compute stencil coefficients on GPU.

        Parameters
        ----------
        H_gpu : cp.ndarray, shape (N_Z, N_phi), float64
        d_phi, d_Z : float
        R, L : float
        **kwargs : additional physics-specific parameters

        Returns
        -------
        A, B, C, D, E, F : cp.ndarray, each shape (N_Z, N_phi), float64
        """
        pass
