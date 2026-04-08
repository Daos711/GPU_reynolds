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
from reynolds_solver.solver_dynamic import solve_reynolds_gpu_dynamic
from reynolds_solver.solver_jfo import solve_reynolds_gpu_jfo
from reynolds_solver.solver_piezoviscous import solve_reynolds_piezoviscous
from reynolds_solver.solver_transformed import solve_reynolds_transformed
from reynolds_solver.physics.closures import LaminarClosure, ConstantinescuClosure


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
    # JFO-specific parameters
    jfo_max_outer: int = 500,
    jfo_max_inner: int = 500,
    jfo_p_off: float = 0.0,
    jfo_p_on: float = 1e-6,
    jfo_tol_theta: float = 1e-5,
    jfo_tol_inner: float = None,
    theta_init: np.ndarray = None,
    mask_init: np.ndarray = None,
    verbose: bool = False,
    # JFO debug flags
    use_F_theta: bool = True,
    update_mask: bool = True,
    run_theta_sweep: bool = True,
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
        Cavitation model: "half_sommerfeld" or "jfo".
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
        Convergence criterion (relative residual). For JFO, used as tol_P.
    max_iter : int
        Maximum SOR iterations (Half-Sommerfeld only).
    check_every : int
        Convergence check frequency (Half-Sommerfeld only).
    P_init : np.ndarray or None
        Initial pressure field for warm start. Shape must match H.
    jfo_max_outer : int
        Max outer (active-set) iterations for JFO.
    jfo_max_inner : int
        Max inner SOR iterations per outer step for JFO.
    jfo_p_off : float
        Threshold for entering cavitation zone (P <= p_off).
    jfo_p_on : float
        Threshold for leaving cavitation zone (P_trial > p_on). Must be > p_off.
    jfo_tol_theta : float
        Convergence tolerance for theta in JFO.
    jfo_tol_inner : float or None
        Inner SOR convergence tolerance for JFO. Default: tol * 0.1.
    theta_init : np.ndarray or None
        Initial fill fraction for JFO warm start. Values in [0, 1].
    mask_init : np.ndarray or None
        Initial zone mask for JFO warm start. Values in {0, 1}.
    use_F_theta : bool
        If True (default), use F_theta = d(H*theta)/dphi as SOR RHS.
        If False, use F_orig = dH/dphi (diagnostic mode).
    update_mask : bool
        If True (default), update zone mask each outer iteration.
    run_theta_sweep : bool
        If True (default), run theta line-sweep each outer iteration.

    Returns
    -------
    For cavitation="half_sommerfeld":
        P : np.ndarray, shape (N_Z, N_phi), float64
        delta : float
        n_iter : int

    For cavitation="jfo":
        P : np.ndarray, shape (N_Z, N_phi), float64
        theta : np.ndarray, shape (N_Z, N_phi), float64
        residual : float
        n_outer : int
        n_inner_total : int
    """
    # --- Auto omega ---
    if omega is None:
        from reynolds_solver.utils import compute_auto_omega
        N_Z, N_phi = H.shape
        if alpha_pv is not None:
            cap = 1.95
        elif cavitation == "jfo":
            cap = 1.95
        else:
            cap = 1.97
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

    elif cavitation == "jfo":
        if closure != "laminar":
            raise NotImplementedError(
                "cavitation='jfo' is only supported with closure='laminar' in this version. "
                "Turbulent JFO is planned for step 3."
            )

        return solve_reynolds_gpu_jfo(
            H, d_phi, d_Z, R, L,
            closure=closure_obj,
            omega=omega,
            tol_P=tol,
            tol_theta=jfo_tol_theta,
            tol_inner=jfo_tol_inner,
            max_outer=jfo_max_outer,
            max_inner=jfo_max_inner,
            p_off=jfo_p_off,
            p_on=jfo_p_on,
            P_init=P_init,
            theta_init=theta_init,
            mask_init=mask_init,
            verbose=verbose,
            use_F_theta=use_F_theta,
            update_mask=update_mask,
            run_theta_sweep=run_theta_sweep,
        )

    else:
        raise NotImplementedError(
            f"cavitation='{cavitation}' not implemented. "
            "Supported: 'half_sommerfeld', 'jfo'."
        )
