"""
reynolds_solver — GPU-accelerated Reynolds equation solver.

Basic usage:
    from reynolds_solver import solve_reynolds
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L)

Advanced usage (cached buffers):
    from reynolds_solver import ReynoldsSolverGPU
    solver = ReynoldsSolverGPU(500, 500)
    P1, _, _ = solver.solve(H1, d_phi, d_Z, R, L)
    P2, _, _ = solver.solve(H2, d_phi, d_Z, R, L)  # buffers reused

Subpackages:
    physics/            — conductance closures (laminar, Constantinescu)
    dynamic/            — unsteady / squeeze Reynolds solvers
    piezoviscous/       — Barus / Roelands piezoviscous solvers
    cavitation.ausas    — dynamic Ausas (2009) JFO solver (validated)
    cavitation.payvar_salant — steady mass-conserving JFO (planned)
    cavitation.legacy   — archived JFO experiments
    tests/              — active validation and regression tests
    experiments/        — diagnostic / exploratory scripts
"""

from reynolds_solver.api import solve_reynolds
from reynolds_solver.solver import ReynoldsSolverGPU

__all__ = ["solve_reynolds", "ReynoldsSolverGPU"]
__version__ = "1.4.0"
