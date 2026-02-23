"""
gpu_reynolds — GPU-accelerated Reynolds equation solver for hydrodynamic bearings.

Uses CuPy + Red-Black SOR on NVIDIA GPUs (tested on RTX 4090).
Drop-in replacement for the Numba CPU solver.

Usage:
    from gpu_reynolds.solver import solve_reynolds_gpu
    from gpu_reynolds.solver_dynamic import solve_reynolds_gpu_dynamic

    P, delta, n_iter = solve_reynolds_gpu(H, d_phi, d_Z, R, L)
    P, delta, n_iter = solve_reynolds_gpu_dynamic(H, d_phi, d_Z, R, L, xprime=0.001, yprime=0.001)
"""

from gpu_reynolds.solver import solve_reynolds_gpu, ReynoldsSolverGPU
from gpu_reynolds.solver_dynamic import solve_reynolds_gpu_dynamic

__all__ = [
    "solve_reynolds_gpu",
    "solve_reynolds_gpu_dynamic",
    "ReynoldsSolverGPU",
]
