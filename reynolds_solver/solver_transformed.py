"""
Transformed-pressure solver for piezoviscous Reynolds equation (Barus law).

Exact variable substitution:
    Φ = (1 - exp(-α·p_scale·P)) / (α·p_scale)

This transforms the piezoviscous equation into a standard laminar equation
for Φ. No outer iteration needed — single-pass solution.

After solving for Φ, recover P via:
    P = -log1p(-α·p_scale·Φ) / (α·p_scale)

Half-Sommerfeld cavitation: P≥0 ↔ Φ≥0 (monotone transform), so the
standard SOR clamp works directly on Φ.
"""

import numpy as np
import cupy as cp

from reynolds_solver.solver import _get_solver
from reynolds_solver.utils import precompute_coefficients_gpu, add_dynamic_rhs_gpu
from reynolds_solver.physics.closures import LaminarClosure


def solve_reynolds_transformed(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    alpha_pv: float,
    p_scale: float,
    # Squeeze / dynamic
    xprime: float = 0.0,
    yprime: float = 0.0,
    beta: float = 2.0,
    # Solver settings
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 500,
    P_init: np.ndarray = None,
    verbose: bool = False,
    subcell_quad: bool = False,
    n_sub: int = 4,
    H_smooth_gpu=None,
    texture_params=None,
    phi_1D=None,
    Z_1D=None,
) -> tuple:
    """
    Solve piezoviscous Reynolds via transformed pressure (Barus, single pass).

    Parameters
    ----------
    H : np.ndarray, shape (N_Z, N_phi), float64
        Dimensionless gap.
    d_phi, d_Z : float
        Grid spacing.
    R, L : float
        Bearing radius and length (m).
    alpha_pv : float
        Barus pressure-viscosity coefficient (Pa⁻¹).
    p_scale : float
        Pressure scale (Pa).
    xprime, yprime : float
        Dimensionless velocities for squeeze/dynamic.
    beta : float
        Dynamic coefficient.
    omega : float
        SOR relaxation parameter.
    tol : float
        SOR convergence tolerance.
    max_iter : int
        Max SOR iterations.
    check_every : int
        SOR convergence check frequency.
    P_init : np.ndarray or None
        Initial pressure field (transformed to Φ_init internally).
    verbose : bool
        Print info.

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64 — physical pressure
    delta : float — SOR residual
    n_iter : int — SOR iterations
    """
    N_Z, N_phi = H.shape
    solver = _get_solver(N_Z, N_phi)
    H_gpu = cp.asarray(H, dtype=cp.float64)
    closure = LaminarClosure(
        subcell_quad=subcell_quad, n_sub=n_sub,
        H_smooth_gpu=H_smooth_gpu, texture_params=texture_params,
        phi_1D=phi_1D, Z_1D=Z_1D,
    )

    # Step 1: precompute standard laminar coefficients
    A, B, C, D, E, F_full = precompute_coefficients_gpu(
        H_gpu, d_phi, d_Z, R, L, closure=closure
    )

    # Add squeeze/dynamic RHS if needed
    is_dynamic = abs(xprime) > 1e-15 or abs(yprime) > 1e-15
    if is_dynamic:
        add_dynamic_rhs_gpu(F_full, d_phi, N_Z, N_phi, xprime, yprime, beta)

    # Transform P_init to Φ_init if provided
    Phi_init = None
    a = alpha_pv * p_scale
    if P_init is not None and a > 1e-30:
        P_init_gpu = cp.asarray(P_init, dtype=cp.float64)
        Phi_init = (1.0 - cp.exp(-a * P_init_gpu)) / a
    elif P_init is not None:
        Phi_init = cp.asarray(P_init, dtype=cp.float64)

    # Step 2: solve laminar equation for Φ (standard GPU SOR)
    Phi_gpu, delta, n_iter = solver.solve_with_rhs(
        H_gpu, F_full, A, B, C, D, E,
        omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
        P_init=Phi_init,
    )

    # Step 3: inverse transform Φ → P
    if a > 1e-30:
        arg = a * Phi_gpu
        n_clipped = int(cp.sum(arg >= 1.0))
        max_arg = float(cp.max(arg))
        arg = cp.clip(arg, 0.0, 1.0 - 1e-12)
        P_gpu = -cp.log1p(-arg) / a

        if verbose:
            maxP = float(cp.max(P_gpu))
            maxPhi = float(cp.max(Phi_gpu))
            print(f"  transformed: n_iter={n_iter}, maxΦ={maxPhi:.4e}, "
                  f"maxP={maxP:.4e}, max_arg={max_arg:.6f}"
                  + (f", CLIP={n_clipped}" if n_clipped > 0 else ""))
    else:
        P_gpu = Phi_gpu
        if verbose:
            print(f"  transformed (α=0): n_iter={n_iter}, "
                  f"maxP={float(cp.max(P_gpu)):.4e}")

    P_cpu = cp.asnumpy(P_gpu)
    return P_cpu, float(delta), n_iter
