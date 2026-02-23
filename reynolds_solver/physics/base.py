from abc import ABC, abstractmethod
import cupy as cp


class StencilBuilder(ABC):
    """
    Base class for building 5-point stencil coefficients.

    Any modification of the Reynolds equation inherits from this class.
    The solver calls build() and receives coefficients A,B,C,D,E,F --
    it doesn't care which physics produced them.

    Stencil: A*P[i,j+1] + B*P[i,j-1] + C*P[i+1,j] + D*P[i-1,j] - E*P[i,j] = F[i,j]
    """

    @abstractmethod
    def build(self, H_gpu: cp.ndarray, d_phi: float, d_Z: float,
              R: float, L: float, **kwargs) -> tuple:
        """
        Returns
        -------
        A, B, C, D, E, F : cp.ndarray, shape (N_Z, N_phi), float64
        """
        pass
