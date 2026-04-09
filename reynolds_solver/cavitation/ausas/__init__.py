"""
reynolds_solver.cavitation.ausas — dynamic Ausas (2009) JFO cavitation.

Algorithm: Ausas, Jai, Buscaglia (2009), "A Mass-Conserving Algorithm for
Dynamical Lubrication Problems With Cavitation", ASME J. Tribology,
131(3), 031702.

The squeeze benchmark in `benchmark_squeeze.py` reproduces the analytic
rupture-phase solution from Section 3 of the paper and verifies the
formulas (12), (17), (18) end to end.

Intended use: DYNAMIC (time-dependent) lubrication problems. The
stationary reduction has no stable non-trivial fixed point for journal
bearings — use `cavitation.payvar_salant` for steady problems instead.
"""
from reynolds_solver.cavitation.ausas.solver_cpu import solve_jfo_ausas_cpu

__all__ = ["solve_jfo_ausas_cpu"]
