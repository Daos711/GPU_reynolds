"""Dynamic version: F += beta*(xprime*sin(phi_g) + yprime*cos(phi_g))"""

import cupy as cp
from reynolds_solver.physics.standard import StandardReynolds


class StandardReynoldsDynamic(StandardReynolds):
    """Standard Reynolds + dynamic contribution."""

    def build(self, H_gpu, d_phi, d_Z, R, L, **kwargs):
        A, B, C, D, E, F = super().build(H_gpu, d_phi, d_Z, R, L)

        xprime = kwargs.get("xprime", 0.0)
        yprime = kwargs.get("yprime", 0.0)
        beta = kwargs.get("beta", 2.0)

        if abs(xprime) > 1e-15 or abs(yprime) > 1e-15:
            N_Z, N_phi = H_gpu.shape
            j_idx = cp.arange(N_phi, dtype=cp.float64)
            phi_global = j_idx * d_phi + cp.pi / 4.0
            dyn = beta * (xprime * cp.sin(phi_global) + yprime * cp.cos(phi_global))
            F += dyn[cp.newaxis, :]

        return A, B, C, D, E, F
