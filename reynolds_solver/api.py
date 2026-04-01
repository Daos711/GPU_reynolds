"""
Unified API: solve_reynolds().

Examples:
    # Static equation:
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L)

    # Dynamic equation:
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L,
                                       xprime=0.001, yprime=0.001)

    # Turbulent (Constantinescu):
    P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L,
                                       closure="constantinescu",
                                       rho=860.0, U_velocity=10.0,
                                       mu=0.03, c_clearance=50e-6)
"""

import numpy as np
from reynolds_solver.solver import solve_reynolds_gpu
from reynolds_solver.solver_dynamic import solve_reynolds_gpu_dynamic
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
    # Solver settings
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 500,
    P_init: np.ndarray = None,
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
        Cavitation model. Only "half_sommerfeld" is supported.
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
        Maximum number of iterations.
    check_every : int
        Convergence check frequency.
    P_init : np.ndarray or None
        Initial pressure field for warm start. Shape must match H.

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64
        Dimensionless pressure field.
    delta : float
        Final relative residual.
    n_iter : int
        Number of iterations to convergence.
    """
    if cavitation != "half_sommerfeld":
        raise NotImplementedError(
            f"cavitation='{cavitation}' not implemented. "
            "Only 'half_sommerfeld' is supported in this version."
        )

    if closure == "laminar":
        closure_obj = LaminarClosure()
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

    is_dynamic = abs(xprime) > 1e-15 or abs(yprime) > 1e-15

    if is_dynamic:
        return solve_reynolds_gpu_dynamic(
            H, d_phi, d_Z, R, L,
            xprime=xprime, yprime=yprime, beta=beta,
            phase_shift=phase_shift,
            closure=closure_obj,
            omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
            P_init=P_init,
        )
    else:
        return solve_reynolds_gpu(
            H, d_phi, d_Z, R, L,
            closure=closure_obj,
            omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
            P_init=P_init,
        )
