"""
Squeeze-film support for the Reynolds equation solver.

The existing dynamic solver (solver_dynamic.py) already handles squeeze-film
effects through the RHS term:

    F[i,j] += beta * (xprime * sin(φ) + yprime * cos(φ))

This module provides:
1. Documentation of the derivation connecting physical squeeze parameters
   to the solver API (xprime, yprime, beta).
2. A helper function `squeeze_to_api_params()` for the conversion.
3. A convenience wrapper `solve_reynolds_squeeze()`.

Derivation
----------
The dimensionless Reynolds equation with squeeze:

    ∂/∂φ(H³ ∂P/∂φ) + α² ∂/∂Z(H³ ∂P/∂Z) = ∂H/∂φ + Λ · ∂H/∂t

For a journal bearing: H = 1 + εx·cos(φ) + εy·sin(φ)

    ∂H/∂t = ε̇x·cos(φ) + ε̇y·sin(φ)

The SOR stencil operates on the equation multiplied by Δφ²:

    [H³ stencil] = F_wedge + F_squeeze

where F_wedge = dφ·(H_{j+½} - H_{j-½}) ≈ dφ²·∂H/∂φ  (already has dφ²)

The dynamic RHS adds: beta·(xprime·sin(φ) + yprime·cos(φ))  (NO dφ² factor)

Matching coefficients:
    F_squeeze = Λ·dφ²·(ε̇x·cos(φ) + ε̇y·sin(φ))

    → yprime·beta = Λ·dφ²·ε̇x   (cos component)
    → xprime·beta = Λ·dφ²·ε̇y   (sin component)

With beta=2 (convention) and Λ=1 in standard non-dimensionalization:

    yprime = dφ²·ε̇x / beta = dφ²·ε̇x / 2
    xprime = dφ²·ε̇y / beta = dφ²·ε̇y / 2

Physical velocities → dimensionless velocities:
    ε̇x = v_x / (c·ω)    (shaft center x-velocity / (clearance × angular velocity))
    ε̇y = v_y / (c·ω)
"""

import numpy as np


def squeeze_to_api_params(v_x, v_y, c, omega_shaft, d_phi, beta=2.0, Lambda=1.0):
    """
    Convert physical squeeze velocities to solver API parameters.

    Parameters
    ----------
    v_x : float
        Shaft center velocity in x-direction (m/s).
    v_y : float
        Shaft center velocity in y-direction (m/s).
    c : float
        Radial clearance (m).
    omega_shaft : float
        Shaft angular velocity (rad/s).
    d_phi : float
        Grid spacing in circumferential direction (rad).
    beta : float
        Dynamic coefficient (default 2.0, matches solver convention).
    Lambda : float
        Squeeze number (default 1.0 for standard non-dimensionalization).

    Returns
    -------
    xprime : float
        API parameter for solver (sin component).
    yprime : float
        API parameter for solver (cos component).
    beta : float
        API parameter for solver (passed through).

    Notes
    -----
    The mapping accounts for:
    - x-velocity → cos(φ) term → solver yprime
    - y-velocity → sin(φ) term → solver xprime
    - dφ² factor: the dynamic RHS is added without dφ² scaling,
      while the wedge term (F_full) already includes it.
    """
    eps_dot_x = v_x / (c * omega_shaft)
    eps_dot_y = v_y / (c * omega_shaft)

    yprime = Lambda * d_phi**2 * eps_dot_x / beta
    xprime = Lambda * d_phi**2 * eps_dot_y / beta

    return xprime, yprime, beta


def solve_reynolds_squeeze(
    H, d_phi, d_Z, R, L,
    v_x=0.0, v_y=0.0,
    c=None, omega_shaft=None,
    Lambda=1.0, beta=2.0,
    closure="laminar",
    omega=1.5, tol=1e-5, max_iter=50000,
    P_init=None,
):
    """
    Solve Reynolds equation with squeeze-film effect on GPU.

    Convenience wrapper around solve_reynolds() that converts physical
    squeeze velocities (v_x, v_y) to the solver's (xprime, yprime, beta).

    Parameters
    ----------
    H : np.ndarray, shape (N_Z, N_phi), float64
        Dimensionless gap field.
    d_phi, d_Z : float
        Grid spacing.
    R, L : float
        Bearing radius and length (m).
    v_x, v_y : float
        Shaft center velocities (m/s). Positive v_x = shaft moving in +x.
    c : float
        Radial clearance (m). Required if v_x or v_y != 0.
    omega_shaft : float
        Shaft angular velocity (rad/s). Required if v_x or v_y != 0.
    Lambda : float
        Squeeze number (default 1.0).
    beta : float
        Dynamic coefficient (default 2.0).
    closure : str
        Conductance model ("laminar" or "constantinescu").
    omega : float
        SOR relaxation parameter.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum SOR iterations.
    P_init : np.ndarray or None
        Initial pressure for warm start.

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64
    delta : float
    n_iter : int
    """
    from reynolds_solver import solve_reynolds

    has_squeeze = abs(v_x) > 1e-30 or abs(v_y) > 1e-30

    if has_squeeze:
        if c is None or omega_shaft is None:
            raise ValueError(
                "c and omega_shaft are required when v_x or v_y != 0"
            )
        xprime, yprime, beta = squeeze_to_api_params(
            v_x, v_y, c, omega_shaft, d_phi, beta, Lambda
        )
    else:
        xprime, yprime = 0.0, 0.0

    return solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure=closure,
        cavitation="half_sommerfeld",
        xprime=xprime, yprime=yprime, beta=beta,
        omega=omega, tol=tol, max_iter=max_iter,
        P_init=P_init,
    )
