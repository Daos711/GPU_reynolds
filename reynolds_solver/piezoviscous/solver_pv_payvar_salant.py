"""
Piezoviscous + Payvar-Salant combined solver.

Wraps the Payvar-Salant mass-conserving JFO solver in the standard
Roelands piezoviscosity outer loop:

    for each PV iteration:
        μ̄(P) ← Roelands(P)               (with under-relaxation)
        A,B,C,D ← A_base/μ̄_face           (conductance ∝ 1/μ)
        P, θ ← PS_solve(H, A,B,C,D,E,     (Couette uses real H)
                         g_init=g_prev)

The conductance (Poiseuille part) is divided by face-averaged μ_ratio.
The Couette term is NOT modified — viscosity does not enter Couette
flow in the standard Newtonian Reynolds formulation.

Between PV iterations the PS solver re-derives its active set from
``g_init`` (previous solution) rather than re-running the HS warmup.
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


def solve_payvar_salant_piezoviscous(
    H, d_phi, d_Z, R, L,
    alpha_pv, p_scale,
    p0_roelands=1.98e8,
    z_roelands=0.6,
    tol_outer=1e-3,
    max_outer=30,
    relax=0.5,
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
    relax : float — under-relaxation for μ̄ update (in log-space).
    tol : float — inner PS convergence tolerance.
    max_iter : int — inner PS max iterations.
    verbose : bool.

    Returns
    -------
    P : (N_Z, N_phi) float64
    theta : (N_Z, N_phi) float64
    residual : float — final inner PS residual.
    n_iter : int — total inner iterations across all PV steps.
    """
    import cupy as cp

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
    P, theta, res, n_iter = solve_payvar_salant_gpu(
        H, d_phi, d_Z, R, L,
        tol=tol, max_iter=max_iter, verbose=False,
    )
    n_iter_total = n_iter

    mu_ratio = cp.ones((N_Z, N_phi), dtype=cp.float64)
    converged = False
    dP_rel = 1.0
    outer = 0

    # PV outer loop
    for outer in range(1, max_outer + 1):
        P_old = P.copy()

        # μ̄ from current P
        P_gpu = cp.asarray(P)
        mu_ratio_new, n_clamped = _compute_mu_ratio_gpu(
            P_gpu, alpha_pv, p_scale, p0_roelands, z_roelands,
        )

        # Under-relaxation in log-space
        log_mu_old = cp.log(cp.clip(mu_ratio, 1e-30, None))
        log_mu_new = cp.log(cp.clip(mu_ratio_new, 1e-30, None))
        log_mu = (1.0 - relax) * log_mu_old + relax * log_mu_new
        mu_ratio = cp.exp(log_mu)

        # Modify conductance coefficients
        A_pv = A_base.copy()
        B_pv = B_base.copy()
        C_pv = C_base.copy()
        D_pv = D_base.copy()
        E_pv = _apply_piezoviscosity(
            A_pv, B_pv, C_pv, D_pv, E_base, mu_ratio, N_Z, N_phi,
        )

        # g_init from previous solution (skip HS warmup)
        g_init = np.where(P > 1e-12, P, theta - 1.0)
        g_init = np.clip(g_init, -1.0, None)

        # Solve PS with modified coefficients
        P_new, theta_new, res, n_inner = solve_payvar_salant_gpu(
            H, d_phi, d_Z, R, L,
            coefficients_ext=(A_pv, B_pv, C_pv, D_pv, E_pv),
            g_init=g_init,
            tol=tol,
            max_iter=max_iter,
            verbose=False,
        )
        n_iter_total += n_inner

        # Convergence check
        max_P = float(np.max(np.abs(P_new))) + 1e-30
        dP_rel = float(np.max(np.abs(P_new - P_old))) / max_P

        if verbose:
            mu_max = float(cp.max(mu_ratio))
            cav_frac = float(np.mean(theta_new[1:-1, 1:-1] < 1.0 - 1e-6))
            print(
                f"  pv+ps outer={outer}: dP_rel={dP_rel:.2e}, "
                f"mu_max={mu_max:.3f}, cav={cav_frac:.3f}, "
                f"inner={n_inner}"
                + (f", CLAMP={n_clamped}" if n_clamped > 0 else "")
            )

        P, theta = P_new, theta_new
        converged = dP_rel < tol_outer

        if converged:
            if verbose:
                print(f"  pv+ps CONVERGED at outer={outer}")
            break

    if return_diagnostics:
        diag = {
            "n_outer": outer,
            "converged": converged,
            "dP_rel_final": dP_rel,
            "mu_max": float(cp.max(mu_ratio)),
        }
        return P, theta, res, n_iter_total, diag
    return P, theta, res, n_iter_total
