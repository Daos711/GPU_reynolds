import cupy as cp
from reynolds_solver.physics.base import StencilBuilder
from reynolds_solver.utils import precompute_coefficients_gpu


class StandardReynolds(StencilBuilder):
    """Standard Reynolds: d/dphi[H^3 dP/dphi] + (D/L)^2 d/dZ[H^3 dP/dZ] = dH/dphi"""

    def build(self, H_gpu, d_phi, d_Z, R, L, **kwargs):
        return precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)
