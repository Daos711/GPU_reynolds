"""
Oil-property model (Walther viscosity–temperature + Barus PV coefficient).

Pure-numpy, no GPU deps, no engine-specific logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


@dataclass(frozen=True)
class OilModel:
    """
    Immutable carrier for oil properties used by the THD helper layer.

    Walther equation (ASTM D341):
        log10( log10(nu_cSt + 0.7) ) = A_w - B_w * log10(T_K)

    where nu_cSt is kinematic viscosity in cSt (mm²/s), T_K is
    temperature in Kelvin.

    Attributes
    ----------
    A_w, B_w : float
        Walther coefficients (fitted via `fit_walther_two_point`).
    rho_kg_m3 : float
        Reference density (kg/m³). Assumed constant for now — a linear
        ρ(T) model can be layered on top later.
    cp_J_kgK : float
        Specific heat capacity (J/(kg·K)).
    alpha_pv_base : float
        Pressure-viscosity coefficient (Pa⁻¹) at the reference
        temperature. Used as a fallback when no explicit temperature-
        dependent model for α is wired.
    gamma_mix : float
        Mixing parameter for the global lumped-thermal model.
        T_eff = T_in + gamma * (T_out - T_in).  Typical ≈ 0.5.
    """

    A_w: float
    B_w: float
    rho_kg_m3: float = 860.0
    cp_J_kgK: float = 2000.0
    alpha_pv_base: float = 2.2e-8
    gamma_mix: float = 0.5


def fit_walther_two_point(
    T1_C: float,
    nu1_cSt: float,
    T2_C: float,
    nu2_cSt: float,
) -> OilModel:
    """
    Fit Walther A_w, B_w from two (T, nu) data points.

    Parameters
    ----------
    T1_C, T2_C : float
        Temperatures in °C.
    nu1_cSt, nu2_cSt : float
        Kinematic viscosities in cSt (mm²/s).

    Returns
    -------
    OilModel with A_w, B_w populated. Remaining fields use defaults;
    override them via `dataclasses.replace(model, rho_kg_m3=...)`.
    """
    if T1_C == T2_C:
        raise ValueError("T1 and T2 must differ")
    if nu1_cSt <= 0.0 or nu2_cSt <= 0.0:
        raise ValueError("Viscosities must be positive")

    T1_K = T1_C + 273.15
    T2_K = T2_C + 273.15

    Y1 = np.log10(np.log10(nu1_cSt + 0.7))
    Y2 = np.log10(np.log10(nu2_cSt + 0.7))
    X1 = np.log10(T1_K)
    X2 = np.log10(T2_K)

    B_w = (Y1 - Y2) / (X2 - X1)
    A_w = Y1 + B_w * X1

    return OilModel(A_w=float(A_w), B_w=float(B_w))


def mu_at_T_C(T_C: ArrayLike, model: OilModel) -> ArrayLike:
    """
    Dynamic viscosity η (Pa·s) at temperature T (°C) via Walther + ρ.

    Returns a scalar if T_C is scalar, ndarray otherwise.
    """
    T_K = np.asarray(T_C, dtype=np.float64) + 273.15
    logT = np.log10(T_K)
    W = model.A_w - model.B_w * logT          # log10(log10(nu+0.7))
    nu_cSt = 10.0 ** (10.0 ** W) - 0.7        # kinematic viscosity (cSt)
    nu_m2s = nu_cSt * 1e-6                     # m²/s
    eta = nu_m2s * model.rho_kg_m3             # Pa·s
    if np.ndim(T_C) == 0:
        return float(eta)
    return eta


def alpha_at_T_C(
    T_C: ArrayLike,
    model: OilModel,
    *,
    mode: str = "constant",
) -> ArrayLike:
    """
    Pressure-viscosity coefficient α (Pa⁻¹) at temperature T (°C).

    mode
    ----
    "constant" : return model.alpha_pv_base regardless of T. This is
        the Barus-constant fallback; adequate when the temperature
        variation is modest (< 30 K swing) and α(T) is unknown.

    Future extension: "gold" / "roelands" with T-dependent alpha.
    """
    if mode == "constant":
        val = model.alpha_pv_base
        if np.ndim(T_C) == 0:
            return float(val)
        return np.full_like(np.asarray(T_C, dtype=np.float64), val)
    raise ValueError(f"Unknown alpha mode {mode!r}. Use 'constant'.")
