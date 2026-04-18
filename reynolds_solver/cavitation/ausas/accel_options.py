"""
Acceleration-option knobs for the dynamic Ausas journal solver.

An `AusasAccelerationOptions` instance is passed via the `accel=`
kwarg of `solve_ausas_journal_dynamic_gpu`. Leaving it as None (or
leaving `mech_update_every=1`) preserves the BASELINE behaviour of
the solver: mechanics are rebuilt on every inner iteration, exactly
as before Phase 5. Setting `mech_update_every=K>1` switches the
solver to lagged-mechanics mode.

Phase 5 Part 1 only exposes `mech_update_every` and its safety
guards. The adaptive-dt and dynamic-check knobs are plumbed for
forward compatibility (subsequent Parts 2 and 3) but are not yet
consumed by the solver.

Baseline invariance
-------------------
With the default construction (`AusasAccelerationOptions()` — all
defaults, same as the old implicit behaviour) the inner-loop logic
refreshes mechanics on every iteration, so the solver reproduces
the pre-Phase-5 behaviour bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AusasAccelerationOptions:
    """
    Runtime knobs for the journal-solver inner loop.

    Lagged-mechanics fields
    -----------------------
    mech_update_every : int
        Recompute the forces, Newmark predictor, gap H and stencil
        coefficients only every K iterations (K = this value). K = 1
        means "refresh every iteration" — identical to the baseline.
        K >= 2 starts skipping the expensive non-sweep work.
    mech_force_first_iters : int
        Always refresh mechanics for the first N iterations of each
        time step (good-initial-guess insurance). Default 3.
    mech_force_if_residual_stalls : bool
        If the last-measured inner residual is too close to the
        residual at the last refresh (see `mech_residual_stall_ratio`),
        force an out-of-schedule refresh on the next iteration.
    mech_force_if_cav_jumps : bool
        If the cavitation fraction has drifted by more than
        `mech_cav_jump_tol` since the last refresh, force a refresh.
    mech_cav_jump_tol : float
        Cav-fraction change threshold that triggers a forced refresh.
        Checked only on residual-measurement iterations (where we
        already sync to the host).
    mech_residual_stall_ratio : float
        Stall condition: `last_residual > ratio * residual_at_refresh`
        — i.e. residual dropped by less than (1 - ratio) since the
        last refresh. Default 0.9 (stall if drop < 10%).

    Adaptive-dt fields (Part 2, NOT yet consumed)
    ---------------------------------------------
    These are exposed so that existing scripts can start passing
    forward-compatible option sets. The Phase 5 Part 1 solver ignores
    them.

    Dynamic check-every fields (Part 3, NOT yet consumed)
    -----------------------------------------------------
    Same story — plumbed for the future, no-op for now.
    """

    # --- Lagged mechanics (Part 1) ---
    mech_update_every: int = 1
    mech_force_first_iters: int = 3
    mech_force_if_residual_stalls: bool = True
    mech_force_if_cav_jumps: bool = True
    mech_cav_jump_tol: float = 5e-3
    mech_residual_stall_ratio: float = 0.90

    # --- Adaptive dt (Part 2, forward-compat) ---
    adaptive_dt: bool = False
    dt_min: float = 2.5e-4
    dt_max: float = 5.0e-3
    dt_grow: float = 1.25
    dt_shrink: float = 0.5
    target_inner_low: int = 80
    target_inner_high: int = 250
    reject_if_not_converged: bool = True

    # --- Dynamic check cadence (Part 3, forward-compat) ---
    dynamic_check_every: bool = False
    check_every_min: int = 5
    check_every_max: int = 25
