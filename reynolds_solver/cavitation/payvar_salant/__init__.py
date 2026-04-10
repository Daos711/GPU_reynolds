"""
reynolds_solver.cavitation.payvar_salant — steady mass-conserving JFO.

Unified-variable (Payvar-Salant / Elrod 1981) cavitation solver for the
steady Reynolds equation. Unlike dynamic Ausas, Payvar-Salant is
designed as a STATIONARY mass-conserving JFO solver, with a
well-defined fixed point for steady journal-bearing problems.

    g >= 0  ⇒  P = g,    θ = 1       (full-film)
    g <  0  ⇒  P = 0,    θ = 1 + g   (cavitation)
"""
from reynolds_solver.cavitation.payvar_salant.solver_cpu import (
    solve_payvar_salant_cpu,
)

__all__ = ["solve_payvar_salant_cpu"]

# GPU solver requires cupy; import it lazily to avoid breaking
# CPU-only usage of this package.
try:
    from reynolds_solver.cavitation.payvar_salant.solver_gpu import (
        solve_payvar_salant_gpu,
    )
    __all__.append("solve_payvar_salant_gpu")
except ImportError:
    pass
