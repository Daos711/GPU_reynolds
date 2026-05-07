"""
Piezoviscous + dynamic Ausas/JFO one-step solver (ТЗ-2).

Wraps `ausas_unsteady_one_step_gpu` in the standard outer-iteration
piezoviscosity loop:

    for k in range(pv_max_outer):
        1. Solve Ausas/JFO one step with frozen mu_ratio (PV applied to
           pressure-flow coefficients only).
        2. Compute mu_ratio_raw = mu_model(p_dim) / mu_0,
           with p_dim = max(p_nd * p_scale, 0).
        3. Clamp mu_ratio_raw to [1, pv_mu_ratio_cap].
        4. Relax in log-space:
               log_mu_new = (1 - relax) * log(mu_old) + relax * log(mu_raw)
        5. Stop when max|log(mu_new / mu_old)| < pv_tol
           and (optionally) ||p_new - p_old|| / ||p_new|| < pv_pressure_tol.

Architecture follows the existing reference in this repo:
    * reynolds_solver/piezoviscous/solver_pv_payvar_salant.py
    * reynolds_solver/piezoviscous/solver_piezoviscous.py
The mu_ratio compute and face-averaged conductance correction live in
those modules and are reused as-is.

Conventions
-----------
* Internal API uses the SAME pair (alpha_pv [Pa^-1], p_scale [Pa]) as
  the existing PV/PS solvers. The CLI / high-level wrapper may accept
  alpha in GPa^-1 and convert internally.
* Roelands (default): mu(p) = mu_0 * exp[(alpha*p_0/z) * ((1+p/p_0)^z - 1)]
  with p_0 = 1.98e8 Pa, z = 0.6.
* Barus: mu(p) = mu_0 * exp(alpha*p). Implemented as Roelands with z=1
  (mathematically exact; same numerics path).
* PV is OFF by default (`pv_model='off'`). When OFF or `alpha_pv == 0`
  the wrapper bypasses to the plain Ausas/JFO one-step path; baseline
  bit-for-bit identity.
"""
import numpy as np

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False
    cp = None

from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
    ausas_unsteady_one_step_gpu,
)


_ROELANDS_P0 = 1.98e8
_ROELANDS_Z = 0.6
_BARUS_Z = 1.0
_PV_MODELS = ("off", "roelands", "barus")


def _resolve_alpha(alpha_pv, alpha_GPa_inv):
    """Return alpha in Pa^-1 from either alpha_pv (Pa^-1) or alpha_GPa_inv."""
    if alpha_pv is not None and alpha_GPa_inv is not None:
        raise ValueError(
            "Specify either alpha_pv (Pa^-1) or alpha_GPa_inv, not both."
        )
    if alpha_GPa_inv is not None:
        return float(alpha_GPa_inv) * 1e-9
    if alpha_pv is not None:
        return float(alpha_pv)
    return 0.0


def _model_params(pv_model):
    """Return (p0, z) for the given PV model. Raises on unknown model."""
    if pv_model == "roelands":
        return _ROELANDS_P0, _ROELANDS_Z
    if pv_model == "barus":
        return _ROELANDS_P0, _BARUS_Z
    raise ValueError(
        f"Unknown pv_model {pv_model!r}. Use one of {_PV_MODELS}."
    )


def _neutral_diagnostics(pv_model, alpha_Pa_inv):
    """Default diagnostics for the off / alpha=0 fast path."""
    return {
        "pv_model": pv_model,
        "pv_alpha_Pa_inv": float(alpha_Pa_inv),
        "pv_alpha_GPa_inv": float(alpha_Pa_inv) * 1e9,
        "pv_outer_iters": 0,
        "pv_converged": True,
        "pv_final_delta_log_mu": 0.0,
        "pv_final_delta_p_rel": 0.0,
        "pv_mu_ratio_min": 1.0,
        "pv_mu_ratio_mean": 1.0,
        "pv_mu_ratio_max": 1.0,
        "pv_mu_ratio_cap_hit": False,
        "pv_mu_ratio_cap_hit_frac": 0.0,
        "pv_nonfinite_guard_triggered": False,
    }


def solve_ausas_unsteady_pv_one_step_gpu(
    H_curr,
    H_prev,
    P_prev,
    theta_prev,
    dt,
    d_phi,
    d_Z,
    R,
    L,
    *,
    # --- PV configuration -------------------------------------------------
    pv_model: str = "off",
    alpha_pv: float = None,
    alpha_GPa_inv: float = None,
    p_scale: float = None,
    pv_relax: float = 0.5,
    pv_max_outer: int = 8,
    pv_tol: float = 1e-3,
    pv_pressure_tol: float = 1e-3,
    pv_mu_ratio_cap: float = 20.0,
    pv_warm_start: bool = True,
    pv_initial_mu_ratio=None,
    return_mu_ratio: bool = True,
    p0_roelands: float = _ROELANDS_P0,
    z_roelands: float = _ROELANDS_Z,
    # --- Forwarded to ausas_unsteady_one_step_gpu -------------------------
    alpha: float = 1.0,
    omega_p: float = 1.0,
    omega_theta: float = 1.0,
    tol: float = 1e-6,
    max_inner: int = 5000,
    p_bc_z0: float = 0.0,
    p_bc_zL: float = 0.0,
    theta_bc_z0: float = 1.0,
    theta_bc_zL: float = 1.0,
    p_bc_phi0: float = 0.0,
    p_bc_phiL: float = 0.0,
    theta_bc_phi0: float = 1.0,
    theta_bc_phiL: float = 1.0,
    periodic_phi: bool = True,
    periodic_z: bool = False,
    check_every: int = 50,
    verbose: bool = False,
    scheme: str = "rb",
):
    """
    Advance (P, theta) by ONE real time step under the unsteady Ausas/JFO
    discretization, with optional Roelands / Barus piezoviscosity.

    Returns
    -------
    dict with the standard one-step keys (P, theta, residual, n_inner,
    converged, ...) PLUS the PV diagnostic block:
        pv_model, pv_alpha_Pa_inv, pv_alpha_GPa_inv,
        pv_outer_iters, pv_converged,
        pv_final_delta_log_mu, pv_final_delta_p_rel,
        pv_mu_ratio_min, pv_mu_ratio_mean, pv_mu_ratio_max,
        pv_mu_ratio_cap_hit, pv_mu_ratio_cap_hit_frac,
        pv_nonfinite_guard_triggered,
        pv_mu_ratio_field    (only if return_mu_ratio=True; else None)

    Default `pv_model='off'` ⇒ identical to `ausas_unsteady_one_step_gpu`.
    """
    if pv_model not in _PV_MODELS:
        raise ValueError(
            f"Unknown pv_model {pv_model!r}. Use one of {_PV_MODELS}."
        )

    alpha_Pa_inv = _resolve_alpha(alpha_pv, alpha_GPa_inv)

    one_step_kwargs = dict(
        dt=dt, d_phi=d_phi, d_Z=d_Z, R=R, L=L,
        alpha=alpha, omega_p=omega_p, omega_theta=omega_theta,
        tol=tol, max_inner=max_inner,
        p_bc_z0=p_bc_z0, p_bc_zL=p_bc_zL,
        theta_bc_z0=theta_bc_z0, theta_bc_zL=theta_bc_zL,
        p_bc_phi0=p_bc_phi0, p_bc_phiL=p_bc_phiL,
        theta_bc_phi0=theta_bc_phi0, theta_bc_phiL=theta_bc_phiL,
        periodic_phi=periodic_phi, periodic_z=periodic_z,
        check_every=check_every, verbose=verbose, scheme=scheme,
    )

    # ----------------------------------------------------------------------
    # Off / alpha=0 fast path: bypass to baseline.
    # ----------------------------------------------------------------------
    if pv_model == "off" or abs(alpha_Pa_inv) < 1e-30:
        result = ausas_unsteady_one_step_gpu(
            H_curr, H_prev, P_prev, theta_prev,
            mu_ratio=None,
            **one_step_kwargs,
        )
        result.update(_neutral_diagnostics(pv_model, alpha_Pa_inv))
        if return_mu_ratio:
            result["pv_mu_ratio_field"] = np.ones_like(np.asarray(P_prev),
                                                      dtype=np.float64)
        else:
            result["pv_mu_ratio_field"] = None
        return result

    if not _HAS_CUPY:
        raise ImportError(
            "Piezoviscous Ausas/JFO requires cupy. Install cupy or use "
            "pv_model='off' for the baseline solver."
        )
    if p_scale is None or float(p_scale) <= 0.0:
        raise ValueError(
            "Piezoviscous solver requires p_scale (Pa) > 0 to convert "
            "dimensionless pressure to Pa. Pass p_scale or use the GPa^-1 "
            "+ p_scale_Pa contract from the runner."
        )

    p0, z = _model_params(pv_model)
    if pv_model == "barus":
        z = _BARUS_Z
    else:
        p0 = float(p0_roelands)
        z = float(z_roelands)

    from reynolds_solver.piezoviscous.solver_piezoviscous import (
        _compute_mu_ratio_gpu,
    )

    # --- mu_ratio init (warm-start or ones) -------------------------------
    N_Z, N_phi = np.asarray(P_prev).shape
    if pv_warm_start and pv_initial_mu_ratio is not None:
        mu_init = np.asarray(pv_initial_mu_ratio, dtype=np.float64)
        if mu_init.shape != (N_Z, N_phi):
            raise ValueError(
                f"pv_initial_mu_ratio shape {mu_init.shape} != "
                f"({N_Z}, {N_phi})"
            )
        mu_old_gpu = cp.asarray(mu_init)
        cp.clip(mu_old_gpu, 1.0, pv_mu_ratio_cap, out=mu_old_gpu)
    else:
        mu_old_gpu = cp.ones((N_Z, N_phi), dtype=cp.float64)

    nonfinite_guard = False
    cap_hit = False
    cap_hit_frac = 0.0
    delta_log_mu = float("inf")
    delta_p_rel = float("inf")
    converged = False
    n_outer = 0

    # --- Initial solve with mu_old (cold: ones; warm: previous field) -----
    # This produces the seed P from which the first mu_raw is derived.
    init_result = ausas_unsteady_one_step_gpu(
        H_curr, H_prev, P_prev, theta_prev,
        mu_ratio=mu_old_gpu,
        **one_step_kwargs,
    )
    if not (np.all(np.isfinite(init_result["P"]))
            and np.all(np.isfinite(init_result["theta"]))):
        # Initial solve already produced non-finite state; bypass loop.
        nonfinite_guard = True
        last_finite_result = ausas_unsteady_one_step_gpu(
            H_curr, H_prev, P_prev, theta_prev,
            mu_ratio=None,
            **one_step_kwargs,
        )
        last_finite_mu = cp.ones((N_Z, N_phi), dtype=cp.float64)
    else:
        P_cur_gpu = cp.asarray(init_result["P"])
        last_finite_result = init_result
        last_finite_mu = mu_old_gpu

        # ------------------------------------------------------------------
        # PV outer loop: relax-then-solve, exactly as PS-PV reference.
        # The (P, theta) returned at the end was solved with the RETURNED
        # mu_ratio_field, so caller-side friction stays self-consistent.
        # ------------------------------------------------------------------
        for k in range(1, pv_max_outer + 1):
            n_outer = k

            # 1. Raw mu from current P.
            mu_raw_gpu, _ = _compute_mu_ratio_gpu(
                P_cur_gpu, alpha_Pa_inv, p_scale, p0, z,
            )
            cap_mask = mu_raw_gpu > pv_mu_ratio_cap
            n_cap = int(cp.sum(cap_mask))
            if n_cap > 0:
                cap_hit = True
                cap_hit_frac = float(n_cap) / float(mu_raw_gpu.size)
            cp.clip(mu_raw_gpu, 1.0, pv_mu_ratio_cap, out=mu_raw_gpu)

            if not bool(cp.all(cp.isfinite(mu_raw_gpu))):
                nonfinite_guard = True
                break

            # 2. Log-space relaxation.
            log_mu_old = cp.log(cp.clip(mu_old_gpu, 1e-30, None))
            log_mu_raw = cp.log(cp.clip(mu_raw_gpu, 1e-30, None))
            log_mu_new = (
                (1.0 - pv_relax) * log_mu_old + pv_relax * log_mu_raw
            )
            mu_new_gpu = cp.exp(log_mu_new)
            delta_log_mu = float(cp.max(cp.abs(log_mu_new - log_mu_old)))

            # 3. Re-solve with the relaxed mu.
            result = ausas_unsteady_one_step_gpu(
                H_curr, H_prev, P_prev, theta_prev,
                mu_ratio=mu_new_gpu,
                **one_step_kwargs,
            )
            if not (np.all(np.isfinite(result["P"]))
                    and np.all(np.isfinite(result["theta"]))):
                nonfinite_guard = True
                break

            P_new_gpu = cp.asarray(result["P"])

            # 4. Convergence diagnostics.
            max_p = float(cp.max(cp.abs(P_new_gpu))) + 1e-30
            delta_p_rel = (
                float(cp.max(cp.abs(P_new_gpu - P_cur_gpu))) / max_p
            )

            # 5. Promote new state.
            mu_old_gpu = mu_new_gpu
            P_cur_gpu = P_new_gpu
            last_finite_result = result
            last_finite_mu = mu_new_gpu

            if verbose:
                mu_max_log = float(cp.max(mu_new_gpu))
                print(
                    f"  pv-ausas outer={k}: dlogmu={delta_log_mu:.2e}, "
                    f"dP_rel={delta_p_rel:.2e}, mu_max={mu_max_log:.3f}"
                    + (f", CAP={n_cap}" if n_cap > 0 else "")
                )

            if (delta_log_mu < pv_tol
                    and delta_p_rel < pv_pressure_tol):
                converged = True
                break

    # ----- Diagnostics -----------------------------------------------------
    mu_min = float(cp.min(last_finite_mu))
    mu_mean = float(cp.mean(last_finite_mu))
    mu_max = float(cp.max(last_finite_mu))

    diag = {
        "pv_model": pv_model,
        "pv_alpha_Pa_inv": float(alpha_Pa_inv),
        "pv_alpha_GPa_inv": float(alpha_Pa_inv) * 1e9,
        "pv_outer_iters": int(n_outer),
        "pv_converged": bool(converged and not nonfinite_guard),
        "pv_final_delta_log_mu": float(delta_log_mu),
        "pv_final_delta_p_rel": float(delta_p_rel),
        "pv_mu_ratio_min": mu_min,
        "pv_mu_ratio_mean": mu_mean,
        "pv_mu_ratio_max": mu_max,
        "pv_mu_ratio_cap_hit": bool(cap_hit),
        "pv_mu_ratio_cap_hit_frac": float(cap_hit_frac),
        "pv_nonfinite_guard_triggered": bool(nonfinite_guard),
    }
    last_finite_result.update(diag)
    last_finite_result["pv_mu_ratio_field"] = (
        cp.asnumpy(last_finite_mu) if return_mu_ratio else None
    )
    return last_finite_result


# Convenience alias — single dynamic-step entry point with the API name
# expected by article-dump-truck (ТЗ-2 §13).
solve_ausas_dynamic_pv = solve_ausas_unsteady_pv_one_step_gpu
