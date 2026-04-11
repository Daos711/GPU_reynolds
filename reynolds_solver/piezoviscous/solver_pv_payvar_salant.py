"""
Piezoviscous + Payvar-Salant combined solver.

Wraps the Payvar-Salant mass-conserving JFO solver in the standard
Roelands piezoviscosity outer loop:

    for each PV iteration:
        μ̄(P) ← Roelands(P)               (with under-relaxation on μ̄)
        A,B,C,D ← A_base/μ̄_face           (conductance ∝ 1/μ)
        P, θ ← PS_solve(H, A,B,C,D,E,     (Couette uses real H)
                         g_init=g_prev)
        g ← relax_g·g_new + (1-relax_g)·g_old   (g-level relaxation)

The conductance (Poiseuille part) is divided by face-averaged μ_ratio.
The Couette term is NOT modified — viscosity does not enter Couette
flow in the standard Newtonian Reynolds formulation.

Between PV iterations the PS solver re-derives its active set from
g_init with full active-set loop (max_outer_active_set=10). The
g-level relaxation prevents cavitation creep by blending the new
solution with the old one in the unified variable space.
"""
import numpy as np

try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "Piezoviscous + Payvar-Salant requires cupy. "
        "Install cupy or use cavitation='half_sommerfeld' for CPU-only."
    )

from reynolds_solver.piezoviscous.solver_piezoviscous import (
    _compute_mu_ratio_gpu,
    _apply_piezoviscosity,
)

# Seed for full-film cells in g_init: must be >> cav_threshold (1e-10)
# so that PS classifies them as full-film unambiguously.
_CAV_THRESHOLD = 1e-10
_SEED = max(100 * _CAV_THRESHOLD, 1e-8)


def solve_payvar_salant_piezoviscous(
    H, d_phi, d_Z, R, L,
    alpha_pv, p_scale,
    p0_roelands=1.98e8,
    z_roelands=0.6,
    tol_outer=1e-3,
    max_outer=30,
    relax_mu=0.5,
    relax_g=0.7,
    tol=1e-6,
    max_iter=50000,
    verbose=False,
    return_diagnostics=False,
):
    """
    Piezoviscous Payvar-Salant solver (Roelands + mass-conserving JFO).

    Parameters
    ----------
    H : (N_Z, N_phi) float64 — dimensionless gap.
    d_phi, d_Z : float — grid spacing.
    R, L : float — bearing radius and length (m).
    alpha_pv : float — pressure-viscosity coefficient (Pa⁻¹).
    p_scale : float — pressure scale (Pa), converts P_nd to p_dim.
    p0_roelands, z_roelands : float — Roelands model parameters.
    tol_outer : float — PV outer loop convergence on max|ΔP|/max|P|.
    max_outer : int — max PV iterations.
    relax_mu : float — under-relaxation for μ̄ (log-space), default 0.5.
    relax_g : float — under-relaxation for g (unified variable), default 0.7.
    tol : float — inner PS convergence tolerance.
    max_iter : int — inner PS max iterations.
    verbose : bool.
    return_diagnostics : bool — if True, append a dict with n_outer,
        converged, dP_rel_final, mu_max to the return tuple.

    Returns
    -------
    P : (N_Z, N_phi) float64
    theta : (N_Z, N_phi) float64
    residual : float — final inner PS residual.
    n_iter : int — total inner iterations across all PV steps.
    [diag : dict] — only if return_diagnostics=True.
    """
    # --- Special case: no piezoviscosity ---
    if abs(alpha_pv) < 1e-30:
        try:
            from reynolds_solver.cavitation.payvar_salant import (
                solve_payvar_salant_gpu,
            )
            result = solve_payvar_salant_gpu(
                H, d_phi, d_Z, R, L, tol=tol, max_iter=max_iter,
                verbose=verbose,
            )
        except (ImportError, ModuleNotFoundError):
            from reynolds_solver.cavitation.payvar_salant import (
                solve_payvar_salant_cpu,
            )
            result = solve_payvar_salant_cpu(
                H, d_phi, d_Z, R, L, tol=tol, max_iter=max_iter,
                verbose=verbose,
            )
        if return_diagnostics:
            return result + ({"n_outer": 0, "converged": True,
                              "dP_rel_final": 0.0, "mu_max": 1.0},)
        return result

    from reynolds_solver.cavitation.payvar_salant.solver_gpu import (
        solve_payvar_salant_gpu,
        _build_coefficients_gpu,
    )

    N_Z, N_phi = H.shape

    # Ghost-pack H and upload
    H_np = np.ascontiguousarray(H, dtype=np.float64).copy()
    H_np[:, 0] = H_np[:, N_phi - 2]
    H_np[:, N_phi - 1] = H_np[:, 1]
    H_gpu = cp.asarray(H_np)

    # Base coefficients (without PV) — built once
    A_base, B_base, C_base, D_base, E_base = _build_coefficients_gpu(
        H_gpu, d_phi, d_Z, R, L,
    )

    # First solve without PV (μ̄ = 1)
    P_np, theta_np, res, n_iter = solve_payvar_salant_gpu(
        H, d_phi, d_Z, R, L,
        tol=tol, max_iter=max_iter, verbose=False,
    )
    n_iter_total = n_iter

    # Move to GPU for all subsequent operations
    P_g = cp.asarray(P_np)
    theta_g = cp.asarray(theta_np)
    mu_ratio = cp.ones((N_Z, N_phi), dtype=cp.float64)

    converged = False
    dP_rel = 1.0
    outer = 0

    # PV outer loop
    for outer in range(1, max_outer + 1):
        # g_old from current state (all on GPU)
        g_old = cp.where(
            theta_g < 1.0 - 1e-8,
            theta_g - 1.0,
            cp.maximum(P_g, _SEED),
        )
        g_old = cp.clip(g_old, -1.0, None)

        # μ̄ with log-space relaxation
        mu_ratio_new, n_clamped = _compute_mu_ratio_gpu(
            P_g, alpha_pv, p_scale, p0_roelands, z_roelands,
        )
        log_mu = (
            (1.0 - relax_mu) * cp.log(cp.clip(mu_ratio, 1e-30, None))
            + relax_mu * cp.log(cp.clip(mu_ratio_new, 1e-30, None))
        )
        mu_ratio = cp.exp(log_mu)

        # Modify conductance coefficients
        A_pv = A_base.copy()
        B_pv = B_base.copy()
        C_pv = C_base.copy()
        D_pv = D_base.copy()
        E_pv = _apply_piezoviscosity(
            A_pv, B_pv, C_pv, D_pv, E_base, mu_ratio, N_Z, N_phi,
        )

        # g_init for PS (PS expects numpy for g_init)
        g_init_np = cp.asnumpy(g_old)

        # PS solve with full active-set loop
        P_new_np, theta_new_np, res, n_inner = solve_payvar_salant_gpu(
            H, d_phi, d_Z, R, L,
            coefficients_ext=(A_pv, B_pv, C_pv, D_pv, E_pv),
            g_init=g_init_np,
            max_outer_active_set=10,
            tol=tol,
            max_iter=max_iter,
            verbose=False,
        )
        n_iter_total += n_inner

        # g_new from PS result (on GPU)
        P_new = cp.asarray(P_new_np)
        theta_new = cp.asarray(theta_new_np)
        g_new = cp.where(
            theta_new < 1.0 - 1e-8,
            theta_new - 1.0,
            cp.maximum(P_new, _SEED),
        )
        g_new = cp.clip(g_new, -1.0, None)

        # g-level relaxation
        g = relax_g * g_new + (1.0 - relax_g) * g_old
        g = cp.clip(g, -1.0, None)

        # Recover consistent P, θ from relaxed g
        P_g = cp.maximum(g, 0.0)
        theta_g = cp.where(g >= 0.0, 1.0, 1.0 + g)
        theta_g = cp.clip(theta_g, 0.0, 1.0)

        # Convergence on relaxed state
        P_old_from_g = cp.maximum(g_old, 0.0)
        max_P = float(cp.max(cp.abs(P_g))) + 1e-30
        dP_rel = float(cp.max(cp.abs(P_g - P_old_from_g))) / max_P
        converged = dP_rel < tol_outer

        if verbose:
            mu_max = float(cp.max(mu_ratio))
            cav_frac = float(cp.mean(
                (theta_g[1:-1, 1:-1] < 1.0 - 1e-6).astype(cp.float64)
            ))
            dg_rel = float(cp.max(cp.abs(g - g_old))) / (
                float(cp.max(cp.abs(g))) + 1e-30
            )
            print(
                f"  pv+ps outer={outer}: dP_rel={dP_rel:.2e}, "
                f"dg_rel={dg_rel:.2e}, mu_max={mu_max:.3f}, "
                f"cav={cav_frac:.3f}, inner={n_inner}"
                + (f", CLAMP={n_clamped}" if n_clamped > 0 else "")
            )

        if converged:
            if verbose:
                print(f"  pv+ps CONVERGED at outer={outer}")
            break

    # Transfer to host
    P = cp.asnumpy(P_g)
    theta = cp.asnumpy(theta_g)

    if return_diagnostics:
        diag = {
            "n_outer": outer,
            "converged": converged,
            "dP_rel_final": dP_rel,
            "mu_max": float(cp.max(mu_ratio)),
        }
        return P, theta, res, n_iter_total, diag
    return P, theta, res, n_iter_total
