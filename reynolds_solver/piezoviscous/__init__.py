"""
reynolds_solver.piezoviscous — piezoviscous Reynolds solvers.

    solver_piezoviscous       — iterative Barus / Roelands solver
                                (stationary Reynolds, no JFO).
    solver_transformed        — reduced-pressure transformation solver.
    solver_pv_payvar_salant   — Payvar-Salant + PV (steady JFO).
    solver_pv_ausas_dynamic   — dynamic Ausas/JFO + PV (one-step API,
                                ТЗ-2). Default `pv_model='off'`.
"""

# Lazy public re-exports (cupy-aware modules import on first use).
__all__ = [
    "solve_ausas_unsteady_pv_one_step_gpu",
    "solve_ausas_dynamic_pv",
]


def __getattr__(name):
    if name in ("solve_ausas_unsteady_pv_one_step_gpu",
                "solve_ausas_dynamic_pv"):
        from reynolds_solver.piezoviscous.solver_pv_ausas_dynamic import (
            solve_ausas_unsteady_pv_one_step_gpu,
            solve_ausas_dynamic_pv,
        )
        return {
            "solve_ausas_unsteady_pv_one_step_gpu":
                solve_ausas_unsteady_pv_one_step_gpu,
            "solve_ausas_dynamic_pv": solve_ausas_dynamic_pv,
        }[name]
    raise AttributeError(name)
