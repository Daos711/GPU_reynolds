"""
reynolds_solver.thermal — oil-property and global thermal helpers.

This module provides a minimal helper layer for THD (Thermo-Hydro-
Dynamic) lubrication pipelines. It owns ONLY oil-property functions
and single-ODE global thermal estimates; it does NOT own the energy
equation, engine geometry, load cycle, or thermal state management.

Public API:

    from reynolds_solver.thermal import (
        OilModel,
        fit_walther_two_point,
        mu_at_T_C,
        alpha_at_T_C,
        global_static_target_C,
        global_relax_step_C,
    )
"""

from reynolds_solver.thermal.oil_model import (
    OilModel,
    fit_walther_two_point,
    mu_at_T_C,
    alpha_at_T_C,
)
from reynolds_solver.thermal.global_thermal import (
    global_static_target_C,
    global_relax_step_C,
)

__all__ = [
    "OilModel",
    "fit_walther_two_point",
    "mu_at_T_C",
    "alpha_at_T_C",
    "global_static_target_C",
    "global_relax_step_C",
]
