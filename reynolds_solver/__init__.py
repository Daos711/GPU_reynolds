"""
reynolds_solver -- Universal GPU-accelerated Reynolds equation solver.

Usage:
    from reynolds_solver import solve_reynolds

    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L, method="krylov")
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L, method="sor")
"""

from reynolds_solver.api import solve_reynolds

# Backward compatibility (old names)
from reynolds_solver.linear_solvers.gpu_sor import (
    solve_reynolds_sor as solve_reynolds_gpu,
    solve_reynolds_sor_dynamic as solve_reynolds_gpu_dynamic,
    ReynoldsSolverGPU,
)

__all__ = [
    "solve_reynolds",
    "solve_reynolds_gpu",
    "solve_reynolds_gpu_dynamic",
    "ReynoldsSolverGPU",
]
