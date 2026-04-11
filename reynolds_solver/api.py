"""
Unified API: solve_reynolds().

Examples:
    # Static equation (Half-Sommerfeld):
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L)

    # Dynamic equation:
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L,
                                       xprime=0.001, yprime=0.001)

    # Turbulent (Constantinescu):
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L,
                                       closure="constantinescu",
                                       rho=860.0, U_velocity=10.0,
                                       mu=0.03, c_clearance=50e-6)

    # JFO cavitation (mass-conserving):
    P, theta, residual, n_outer, n_inner = solve_reynolds(
        H, d_phi, d_Z, R, L, cavitation="jfo")
"""

import numpy as np
from reynolds_solver.solver import solve_reynolds_gpu
from reynolds_solver.dynamic.solver_dynamic import solve_reynolds_gpu_dynamic
from reynolds_solver.piezoviscous.solver_piezoviscous import solve_reynolds_piezoviscous
from reynolds_solver.piezoviscous.solver_transformed import solve_reynolds_transformed
from reynolds_solver.physics.closures import LaminarClosure, ConstantinescuClosure

# Note: cavitation="jfo" and cavitation="jfo_ausas" have been removed
# from the public API. The stationary reduction of both algorithms is
# unstable for steady journal bearings (see experiments/diagnostics).
#   * For dynamic Ausas (squeeze, orbit analysis), import directly:
#         from reynolds_solver.cavitation.ausas.solver_cpu import
#             solve_jfo_ausas_cpu
#   * For steady mass-conserving cavitation, Payvar-Salant is planned
#     under reynolds_solver.cavitation.payvar_salant.


def solve_reynolds(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    closure: str = "laminar",
    cavitation: str = "half_sommerfeld",
    # Dynamic parameters
    xprime: float = 0.0,
    yprime: float = 0.0,
    beta: float = 2.0,
    phase_shift: float = 0.0,
    # Turbulence parameters (required for closure="constantinescu")
    rho: float = None,
    U_velocity: float = None,
    mu: float = None,
    c_clearance: float = None,
    # Solver settings (Half-Sommerfeld)
    omega: float = None,
    tol: float = 1e-5,
    max_iter: int = 200000,
    check_every: int = 500,
    return_converged: bool = False,
    P_init: np.ndarray = None,
    # Subcell quadrature for conductance
    subcell_quad: bool = False,
    n_sub: int = 4,
    H_smooth: np.ndarray = None,
    texture_params: dict = None,
    phi_1D: np.ndarray = None,
    Z_1D: np.ndarray = None,
    verbose: bool = False,
    # Piezoviscous parameters
    alpha_pv: float = None,
    p_scale: float = None,
    p0_roelands: float = 1.98e8,
    z_roelands: float = 0.6,
    pv_method: str = "iterative",
    tol_outer: float = 1e-3,
    max_outer_pv: int = 20,
    relax_pv: float = 0.7,
) -> tuple:
    """
    Solve the Reynolds equation on GPU (Red-Black SOR).

    Parameters
    ----------
    H : np.ndarray, shape (N_Z, N_phi), float64
        Dimensionless gap.
    d_phi, d_Z : float
        Grid steps along phi and Z.
    R, L : float
        Bearing radius and length (m).
    closure : str
        Conductance model: "laminar" or "constantinescu".
    cavitation : str
        Cavitation model:
          "half_sommerfeld" — classic P ≥ 0 clamping (GPU, default).
          "payvar_salant"   — steady mass-conserving JFO via Elrod-Adams
              frozen-active-set method with unified variable g (CPU).
    xprime, yprime : float
        Dimensionless velocities (for dynamic equation, 0 = static).
    beta : float
        Dynamic term coefficient.
    phase_shift : float
        Phase offset for dynamic RHS (default 0.0, no shift).
    rho : float
        Lubricant density, kg/m³ (required for closure="constantinescu").
    U_velocity : float
        Shaft surface velocity, m/s (required for closure="constantinescu").
    mu : float
        Dynamic viscosity, Pa·s (required for closure="constantinescu").
    c_clearance : float
        Radial clearance, m (required for closure="constantinescu").
    omega : float
        SOR relaxation parameter (1.0-1.9, optimum ~1.5).
    tol : float
        Convergence criterion (relative residual).
    max_iter : int
        Maximum SOR iterations.
    check_every : int
        Convergence check frequency.
    P_init : np.ndarray or None
        Initial pressure field for warm start. Shape must match H.

    Returns
    -------
    For cavitation="half_sommerfeld":
        P : np.ndarray, shape (N_Z, N_phi), float64
        delta : float
        n_iter : int

    For cavitation="payvar_salant":
        P : np.ndarray, shape (N_Z, N_phi), float64
        theta : np.ndarray, shape (N_Z, N_phi), float64
        residual : float
        n_iter : int
    """
    # --- Auto omega ---
    if omega is None:
        from reynolds_solver.utils import compute_auto_omega
        N_Z, N_phi = H.shape
        cap = 1.95 if alpha_pv is not None else 1.97
        omega = compute_auto_omega(N_phi, N_Z, R, L, cap=cap)

    # --- Build closure object ---
    if closure == "laminar":
        import cupy as cp
        H_smooth_gpu = cp.asarray(H_smooth, dtype=cp.float64) if H_smooth is not None else None
        closure_obj = LaminarClosure(
            subcell_quad=subcell_quad, n_sub=n_sub,
            H_smooth_gpu=H_smooth_gpu, texture_params=texture_params,
            phi_1D=phi_1D, Z_1D=Z_1D,
        )
    elif closure == "constantinescu":
        missing = [name for name, val in [
            ("rho", rho), ("U_velocity", U_velocity),
            ("mu", mu), ("c_clearance", c_clearance)
        ] if val is None]
        if missing:
            raise ValueError(
                f"closure='constantinescu' requires: {missing}"
            )
        closure_obj = ConstantinescuClosure(rho, U_velocity, mu, c_clearance)
    else:
        raise ValueError(
            f"Unknown closure: '{closure}'. "
            "Valid options: 'laminar', 'constantinescu'."
        )

    # --- Piezoviscous path (overrides normal dispatch) ---
    if alpha_pv is not None:
        if cavitation != "half_sommerfeld":
            raise NotImplementedError(
                "Piezoviscosity only supported with cavitation='half_sommerfeld'."
            )
        if closure != "laminar":
            raise NotImplementedError(
                "Piezoviscosity only supported with closure='laminar'."
            )
        if p_scale is None:
            raise ValueError(
                "p_scale is required when alpha_pv is set."
            )

        if pv_method == "transformed":
            return solve_reynolds_transformed(
                H, d_phi, d_Z, R, L,
                alpha_pv=alpha_pv,
                p_scale=p_scale,
                xprime=xprime, yprime=yprime, beta=beta,
                omega=omega, tol=tol, max_iter=max_iter,
                check_every=check_every,
                P_init=P_init,
                verbose=verbose,
                subcell_quad=subcell_quad,
                n_sub=n_sub,
                H_smooth_gpu=H_smooth_gpu,
                texture_params=texture_params,
                phi_1D=phi_1D,
                Z_1D=Z_1D,
            )
        elif pv_method == "iterative":
            return solve_reynolds_piezoviscous(
                H, d_phi, d_Z, R, L,
                alpha_pv=alpha_pv,
                p_scale=p_scale,
                p0_roelands=p0_roelands,
                z_roelands=z_roelands,
                xprime=xprime, yprime=yprime, beta=beta,
                omega=omega, tol=tol, max_iter=max_iter,
                check_every=check_every,
                tol_outer=tol_outer,
                max_outer=max_outer_pv,
                relax=relax_pv,
                P_init=P_init,
                verbose=verbose,
                subcell_quad=subcell_quad,
                n_sub=n_sub,
                H_smooth_gpu=H_smooth_gpu,
                texture_params=texture_params,
                phi_1D=phi_1D,
                Z_1D=Z_1D,
            )
        else:
            raise ValueError(
                f"Unknown pv_method: '{pv_method}'. "
                "Valid: 'iterative', 'transformed'."
            )

    # --- Helper: strip converged flag for backward compat ---
    def _maybe_strip(result_4):
        """Strip 4th element (converged) if not requested."""
        if return_converged:
            return result_4
        return result_4[:3]

    # --- Dispatch by cavitation model ---
    if cavitation == "half_sommerfeld":
        is_dynamic = abs(xprime) > 1e-15 or abs(yprime) > 1e-15

        if is_dynamic:
            return _maybe_strip(solve_reynolds_gpu_dynamic(
                H, d_phi, d_Z, R, L,
                xprime=xprime, yprime=yprime, beta=beta,
                phase_shift=phase_shift,
                closure=closure_obj,
                omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
                P_init=P_init,
            ))
        else:
            return _maybe_strip(solve_reynolds_gpu(
                H, d_phi, d_Z, R, L,
                closure=closure_obj,
                omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
                P_init=P_init,
            ))

    elif cavitation == "payvar_salant":
        if closure != "laminar":
            raise NotImplementedError(
                "cavitation='payvar_salant' is only supported with "
                "closure='laminar'."
            )

        try:
            from reynolds_solver.cavitation.payvar_salant import (
                solve_payvar_salant_gpu,
            )
            P, theta, residual, n_iter = solve_payvar_salant_gpu(
                H, d_phi, d_Z, R, L,
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
            )
        except (ImportError, ModuleNotFoundError):
            from reynolds_solver.cavitation.payvar_salant import (
                solve_payvar_salant_cpu,
            )
            P, theta, residual, n_iter = solve_payvar_salant_cpu(
                H, d_phi, d_Z, R, L,
                tol=tol,
                max_iter=max_iter,
                verbose=verbose,
            )
        return P, theta, residual, n_iter

    else:
        raise NotImplementedError(
            f"cavitation='{cavitation}' not implemented. "
            "Supported: 'half_sommerfeld', 'payvar_salant'. "
            "For dynamic Ausas import reynolds_solver.cavitation.ausas "
            "directly."
        )
