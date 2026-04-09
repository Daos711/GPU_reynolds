"""
Piezoviscous Reynolds solver (Barus law).

Solves the Reynolds equation with pressure-dependent viscosity:
    μ(p) = μ₀ · exp(α · p)

The conductance H³ is replaced by H³ / μ_ratio where
μ_ratio = exp(α · P_nd · p_scale).

Iterative outer loop: solve linear Reynolds → update μ_ratio → re-solve.
Uses the existing GPU SOR infrastructure (precompute_coefficients_gpu
+ solve_with_rhs) — no new CUDA kernels.

Architecture: Variant B (direct conductance modification in outer loop).
Closures remain stateless — piezoviscosity is handled externally.
"""

import numpy as np
import cupy as cp

from reynolds_solver.solver import _get_solver
from reynolds_solver.utils import precompute_coefficients_gpu, add_dynamic_rhs_gpu
from reynolds_solver.physics.closures import LaminarClosure

LOG_MU_MAX = 20.0  # clamp: exp(20) ≈ 5e8, already extreme for lubrication


def _compute_mu_ratio_gpu(P_gpu, alpha_pv, p_scale, p0_roelands, z_roelands):
    """
    Compute μ_ratio via Roelands model on GPU with overflow protection.

    Roelands: μ(p) = μ₀ · exp[ (α·p₀/z) · ((1 + p/p₀)^z - 1) ]
    At p << p₀: reduces to Barus (exp(α·p)).
    At p >> p₀: growth is sub-exponential (more realistic).

    Parameters
    ----------
    P_gpu : cp.ndarray — dimensionless pressure field
    alpha_pv : float — pressure-viscosity coefficient (Pa⁻¹)
    p_scale : float — pressure scale (Pa), converts P_nd to p_dim
    p0_roelands : float — Roelands constant (Pa), default 1.98e8
    z_roelands : float — Roelands exponent, default 0.6

    Returns
    -------
    mu_ratio : cp.ndarray — viscosity ratio μ(p)/μ₀
    n_clamped : int — number of nodes where clamp was applied
    """
    p_dim = P_gpu * p_scale
    log_mu = (alpha_pv * p0_roelands / z_roelands) * (
        (1.0 + p_dim / p0_roelands) ** z_roelands - 1.0
    )
    n_clamped = int(cp.sum(log_mu > LOG_MU_MAX))
    log_mu = cp.clip(log_mu, 0.0, LOG_MU_MAX)
    mu_ratio = cp.exp(log_mu)
    return mu_ratio, n_clamped


def _apply_piezoviscosity(A, B, C, D, E, mu_ratio_gpu, N_Z, N_phi):
    """
    Divide conductance coefficients by face-averaged μ_ratio.

    A, B are phi-direction (face-averaged along phi).
    C, D are Z-direction (face-averaged along Z).

    Modifies A, B, C, D, E in-place. Returns new E.
    """
    # Face-average μ_ratio in phi direction (same shift convention as H faces)
    mu_iph = 0.5 * (mu_ratio_gpu[:, :-1] + mu_ratio_gpu[:, 1:])

    mu_imh = cp.empty_like(mu_iph)
    mu_imh[:, 1:] = mu_iph[:, :-1]
    mu_imh[:, 0] = mu_iph[:, -1]

    # Map to full arrays (same convention as precompute_coefficients_gpu)
    mu_A = cp.ones((N_Z, N_phi), dtype=cp.float64)
    mu_A[:, :-1] = mu_iph
    mu_A[:, -1] = mu_iph[:, 0]

    mu_B = cp.ones((N_Z, N_phi), dtype=cp.float64)
    mu_B_half = cp.empty_like(mu_iph)
    mu_B_half[:, 1:] = mu_iph[:, :-1]
    mu_B_half[:, 0] = mu_iph[:, -1]
    mu_B[:, 1:] = mu_B_half
    mu_B[:, 0] = mu_B_half[:, -1]

    # Face-average μ_ratio in Z direction
    mu_jph = 0.5 * (mu_ratio_gpu[:-1, :] + mu_ratio_gpu[1:, :])

    mu_C = cp.ones((N_Z, N_phi), dtype=cp.float64)
    mu_D = cp.ones((N_Z, N_phi), dtype=cp.float64)
    mu_C[1:-1, :] = mu_jph[1:, :]
    mu_D[1:-1, :] = mu_jph[:-1, :]

    # Divide conductances by μ_ratio (higher viscosity → lower conductance)
    A /= mu_A
    B /= mu_B
    C /= mu_C
    D /= mu_D

    # Recompute diagonal
    E_new = A + B + C + D
    return E_new


def solve_reynolds_piezoviscous(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    alpha_pv: float,
    p_scale: float,
    # Roelands parameters
    p0_roelands: float = 1.98e8,
    z_roelands: float = 0.6,
    # Squeeze / dynamic
    xprime: float = 0.0,
    yprime: float = 0.0,
    beta: float = 2.0,
    # Solver settings
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 500,
    # Outer iteration
    tol_outer: float = 1e-3,
    max_outer: int = 20,
    relax: float = 0.7,
    P_init: np.ndarray = None,
    verbose: bool = False,
    # Subcell quadrature
    subcell_quad: bool = False,
    n_sub: int = 4,
    H_smooth_gpu=None,
    texture_params=None,
    phi_1D=None,
    Z_1D=None,
) -> tuple:
    """
    Solve Reynolds equation with piezoviscosity (Roelands) on GPU.

    Outer loop: solve linear Reynolds → update μ_ratio → re-solve.
    Inner: standard GPU Red-Black SOR.

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
        Pressure scale (Pa) for converting P_nd to dimensional.
    xprime, yprime : float
        Dimensionless velocities for squeeze/dynamic.
    beta : float
        Dynamic coefficient.
    omega : float
        SOR relaxation parameter.
    tol : float
        Inner SOR convergence tolerance.
    max_iter : int
        Max inner SOR iterations.
    check_every : int
        Inner SOR convergence check frequency.
    tol_outer : float
        Outer loop convergence (relative max-norm of ΔP).
    max_outer : int
        Max outer iterations.
    relax : float
        Under-relaxation for P update (1.0 = no damping).
    P_init : np.ndarray or None
        Initial pressure field.
    verbose : bool
        Print convergence info.

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64
    delta : float — final inner SOR residual
    n_iter_inner_total : int — total inner SOR iterations
    n_outer : int — outer iterations used
    """
    N_Z, N_phi = H.shape
    solver = _get_solver(N_Z, N_phi)

    H_gpu = cp.asarray(H, dtype=cp.float64)
    closure = LaminarClosure(
        subcell_quad=subcell_quad, n_sub=n_sub,
        H_smooth_gpu=H_smooth_gpu, texture_params=texture_params,
        phi_1D=phi_1D, Z_1D=Z_1D,
    )

    # Base coefficients (without piezoviscosity)
    A_base, B_base, C_base, D_base, E_base, F_full = \
        precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L, closure=closure)

    # Add dynamic/squeeze RHS if needed
    is_dynamic = abs(xprime) > 1e-15 or abs(yprime) > 1e-15
    if is_dynamic:
        add_dynamic_rhs_gpu(F_full, d_phi, N_Z, N_phi, xprime, yprime, beta)

    # Handle alpha_pv = 0 (pure laminar, no outer loop needed)
    if abs(alpha_pv) < 1e-30:
        if P_init is not None:
            P_init_cp = cp.asarray(P_init, dtype=cp.float64)
        else:
            P_init_cp = None
        P_gpu, delta, n_iter, _ = solver.solve_with_rhs(
            H_gpu, F_full, A_base, B_base, C_base, D_base, E_base,
            omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
            P_init=P_init_cp,
        )
        return cp.asnumpy(P_gpu), float(delta), n_iter, 1

    # --- Outer iteration loop (piezoviscous) ---

    # Step 1: Initial solve with μ = μ₀ (laminar)
    P_gpu, delta, n_iter_total, _ = solver.solve_with_rhs(
        H_gpu, F_full, A_base, B_base, C_base, D_base, E_base,
        omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
        P_init=cp.asarray(P_init, dtype=cp.float64) if P_init is not None else None,
    )

    if verbose:
        maxP = float(cp.max(P_gpu))
        print(f"  pv outer=0: inner={n_iter_total}, maxP={maxP:.4e}, "
              f"mu_ratio_max=1.000")

    n_outer = 0
    for outer in range(1, max_outer + 1):
        P_old = P_gpu.copy()

        # Step 2: compute μ_ratio from current P
        mu_ratio, n_clamped = _compute_mu_ratio_gpu(
            P_gpu, alpha_pv, p_scale, p0_roelands, z_roelands)

        # Step 3: modify conductances
        A_pv = A_base.copy()
        B_pv = B_base.copy()
        C_pv = C_base.copy()
        D_pv = D_base.copy()
        E_pv = _apply_piezoviscosity(A_pv, B_pv, C_pv, D_pv, E_base, mu_ratio,
                                      N_Z, N_phi)

        # Step 4: re-solve with warm start
        P_gpu_new, delta, n_iter, _ = solver.solve_with_rhs(
            H_gpu, F_full, A_pv, B_pv, C_pv, D_pv, E_pv,
            omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
            P_init=P_gpu,  # warm start from previous
        )
        n_iter_total += n_iter

        # Under-relaxation
        P_gpu = relax * P_gpu_new + (1.0 - relax) * P_old

        # Step 5: convergence check
        max_P = float(cp.max(cp.abs(P_gpu)))
        dP = float(cp.max(cp.abs(P_gpu - P_old)))
        rel_change = dP / (max_P + 1e-30)

        mu_max = float(cp.max(mu_ratio))
        n_outer = outer

        if verbose:
            print(f"  pv outer={outer}: inner={n_iter}, dP_rel={rel_change:.2e}, "
                  f"maxP={max_P:.4e}, mu_max={mu_max:.3f}"
                  + (f", CLAMP={n_clamped}" if n_clamped > 0 else ""))

        if rel_change < tol_outer:
            if verbose:
                print(f"  pv CONVERGED at outer={outer}")
            break

    P_cpu = cp.asnumpy(P_gpu)
    return P_cpu, float(delta), n_iter_total, n_outer
