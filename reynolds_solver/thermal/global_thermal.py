"""
Global (lumped-parameter) thermal helpers for THD pipelines.

The temperature model here is a SINGLE-ODE mixing balance over the
entire bearing pad:

    T_eff = T_in + gamma * (T_out - T_in)

where T_out is the adiabatic outlet temperature:

    T_out = T_in + P_loss / (mdot * cp)

and T_eff is the "effective" oil-film temperature used for viscosity.

The transient variant adds a first-order thermal lag:

    dT_eff/dt = (T_target - T_eff) / tau_th

These functions are stateless (no internal memory). The pipeline
owns the state and calls them per-step.
"""
from __future__ import annotations


def global_static_target_C(
    T_in_C: float,
    P_loss_W: float,
    mdot_kg_s: float,
    cp_J_kgK: float,
    gamma: float,
) -> float:
    """
    Steady-state effective oil temperature (°C) for a bearing pad.

    T_target = T_in + gamma * P_loss / (mdot * cp)

    Parameters
    ----------
    T_in_C : float
        Inlet oil temperature (°C).
    P_loss_W : float
        Friction power loss (W) dissipated in the bearing.
    mdot_kg_s : float
        Mass flow rate through the bearing (kg/s).
    cp_J_kgK : float
        Oil specific heat capacity (J/(kg·K)).
    gamma : float
        Mixing factor. 0 → inlet temperature; 1 → outlet temperature;
        0.5 → arithmetic mean (typical for short bearings).

    Returns
    -------
    float — T_target in °C.
    """
    if mdot_kg_s <= 0.0:
        raise ValueError(f"mdot_kg_s must be > 0, got {mdot_kg_s}")
    if cp_J_kgK <= 0.0:
        raise ValueError(f"cp_J_kgK must be > 0, got {cp_J_kgK}")
    delta_T = P_loss_W / (mdot_kg_s * cp_J_kgK)
    return T_in_C + gamma * delta_T


def global_relax_step_C(
    T_prev_C: float,
    T_target_C: float,
    dt_s: float,
    tau_th_s: float,
) -> float:
    """
    One explicit-Euler step of the first-order thermal lag:

        T_new = T_prev + (dt / tau) * (T_target - T_prev)

    When tau → 0 (instant equilibrium), this returns T_target. When
    dt << tau, T_new ≈ T_prev (strong lag).

    Parameters
    ----------
    T_prev_C : float
        Current effective temperature (°C).
    T_target_C : float
        Steady-state target from `global_static_target_C`.
    dt_s : float
        Time step (s).
    tau_th_s : float
        Thermal time constant (s). Must be > 0. Typical 0.1–5 s for
        engine main bearings.

    Returns
    -------
    float — T_new in °C.
    """
    if tau_th_s <= 0.0:
        raise ValueError(f"tau_th_s must be > 0, got {tau_th_s}")
    alpha = dt_s / tau_th_s
    if alpha > 1.0:
        alpha = 1.0
    return T_prev_C + alpha * (T_target_C - T_prev_C)
