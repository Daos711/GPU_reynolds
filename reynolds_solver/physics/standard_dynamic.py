import cupy as cp
from reynolds_solver.physics.standard import StandardReynolds
from reynolds_solver.utils import add_dynamic_rhs_gpu


class StandardReynoldsDynamic(StandardReynolds):
    """Standard Reynolds + dynamic squeeze film."""

    def build(self, H_gpu, d_phi, d_Z, R, L, **kwargs):
        A, B, C, D, E, F = super().build(H_gpu, d_phi, d_Z, R, L)
        xprime = kwargs.get("xprime", 0.0)
        yprime = kwargs.get("yprime", 0.0)
        beta = kwargs.get("beta", 2.0)
        if abs(xprime) > 1e-15 or abs(yprime) > 1e-15:
            N_Z, N_phi = H_gpu.shape
            add_dynamic_rhs_gpu(F, d_phi, N_Z, N_phi, xprime, yprime, beta)
        return A, B, C, D, E, F
