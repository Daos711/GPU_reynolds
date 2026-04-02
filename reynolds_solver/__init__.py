"""
reynolds_solver -- GPU-accelerated Reynolds equation solver.

Basic usage:
    from reynolds_solver import solve_reynolds
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L)

Advanced usage (cached buffers):
    from reynolds_solver import ReynoldsSolverGPU
    solver = ReynoldsSolverGPU(500, 500)
    P1, _, _ = solver.solve(H1, d_phi, d_Z, R, L)
    P2, _, _ = solver.solve(H2, d_phi, d_Z, R, L)  # buffers reused
"""

from reynolds_solver.api import solve_reynolds
from reynolds_solver.solver import ReynoldsSolverGPU
from reynolds_solver.solver_jfo import SolverJFO

__all__ = ["solve_reynolds", "ReynoldsSolverGPU", "SolverJFO"]
__version__ = "1.3.0"
