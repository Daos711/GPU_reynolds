"""
reynolds_solver — GPU-accelerated Reynolds equation solver.

Basic usage (stationary):
    from reynolds_solver import solve_reynolds
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L)

Advanced usage (cached buffers):
    from reynolds_solver import ReynoldsSolverGPU
    solver = ReynoldsSolverGPU(500, 500)
    P1, _, _ = solver.solve(H1, d_phi, d_Z, R, L)
    P2, _, _ = solver.solve(H2, d_phi, d_Z, R, L)  # buffers reused

Dynamic Ausas (2008) JFO solvers — public API:
    from reynolds_solver import (
        solve_ausas_prescribed_h_gpu,     # prescribed h(t) time loop
        solve_ausas_journal_dynamic_gpu,  # coupled X,Y dynamics
        AusasTransientResult,             # returned dataclass
        AusasState, save_state, load_state,  # restart / checkpoint
    )

Subpackages:
    physics/                 — conductance closures (laminar, Constantinescu)
    dynamic/                 — unsteady / squeeze Reynolds solvers
    piezoviscous/            — Barus / Roelands piezoviscous solvers
    cavitation.ausas         — dynamic Ausas (2008) JFO solver (validated)
    cavitation.payvar_salant — steady mass-conserving JFO (Elrod-Adams, CPU + GPU)
    cavitation.legacy        — archived JFO experiments
    tests/                   — active validation and regression tests
    experiments/             — diagnostic / exploratory scripts
"""

from reynolds_solver.api import solve_reynolds
from reynolds_solver.solver import ReynoldsSolverGPU

# --- Dynamic Ausas public API (Stages 1-3 + Phase 4) ---
from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
    solve_ausas_prescribed_h_gpu,
    solve_ausas_journal_dynamic_gpu,
    ausas_unsteady_one_step_gpu,
    AusasTransientResult,
)
from reynolds_solver.cavitation.ausas.state_io import (
    AusasState,
    save_state,
    load_state,
)
from reynolds_solver.cavitation.ausas.accel_options import (
    AusasAccelerationOptions,
)

__all__ = [
    "solve_reynolds",
    "ReynoldsSolverGPU",
    "solve_ausas_prescribed_h_gpu",
    "solve_ausas_journal_dynamic_gpu",
    "ausas_unsteady_one_step_gpu",
    "AusasTransientResult",
    "AusasState",
    "save_state",
    "load_state",
    "AusasAccelerationOptions",
]
__version__ = "1.7.0"
