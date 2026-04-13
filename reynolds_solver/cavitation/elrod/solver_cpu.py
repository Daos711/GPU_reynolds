"""
CPU reference for compressible Elrod / Vijayaraghavan-Keith cavitation
(Manser 2019 formulation).

Discretization (Ausas / finite-volume style, our Z ∈ [-1, 1] convention
so alpha_sq = (2R/L · d_phi/d_Z)²):

Starting from Manser eq. 4-8 multiplied by 12:

    ∂/∂θ[β̄·g·h³·∂Θ/∂θ] + alpha_sq·∂/∂Z[β̄·g·h³·∂Θ/∂Z] = 6·∂(Θh)/∂θ

Integrate over a cell and collect (upwind for the convective term
on the RHS, centred for the diffusive fluxes on the LHS):

    β̄·g_{i,j}·(A·Θ_{jp} + B·Θ_{jm} + C·Θ_{ip} + D·Θ_{im} - E·Θ_{i,j})
        = 6·d_phi·(h_{i,j}·Θ_{i,j} - h_{i,jm}·Θ_{i,jm})

where A, B (phi faces) and C, D (Z faces) are the same average-of-cubes
face conductances as in the Payvar-Salant solver, and
E = A + B + C + D. Factoring β̄·g_{i,j} in front of the diffusion
operator is the "cell-local" simplification: diffusion is active in
full-film cells (g=1) and shut off in cavitation (g=0). Flux through
an interface between a full-film and a cavitated cell is thus driven
only by the Couette convection on the RHS — the same behaviour as
Payvar-Salant's branch split.

Per-node SOR update:

    Full-film trial (diagonal β̄·E + 6·d_phi·h_ij):
        num  = β̄·(A·Θ_{jp} + B·Θ_{jm} + C·Θ_{ip} + D·Θ_{im})
             + 6·d_phi·h_{i,jm}·Θ_{i,jm}
        den  = β̄·E + 6·d_phi·h_{i,j}
        Θ_try = num / den
        if Θ_try ≥ 1  ⇒  Θ_new = Θ_try,      g_target = 1
        else           ⇒  cavitation branch (see below)

    Cavitation branch (pure upwind transport; β̄·g_ij → 0):
        6·d_phi·h_{i,j}·Θ_{i,j} = 6·d_phi·h_{i,jm}·Θ_{i,jm}
        Θ_new = (h_{i,jm} / h_{i,j}) · Θ_{i,jm}
        g_target = 0

SOR relaxation:
    Θ_{i,j} ← Θ_old + ω·(Θ_new − Θ_old)

Switch function (Fesanghary-Khonsari soft blend):
    g_{i,j} ← g_old + gfactor·(g_target − g_old)

Pressure recovery: P = β̄·g·ln(Θ) after convergence.
"""
import numpy as np
from numba import njit

# Re-use the HS warmup kernel from Payvar-Salant (identical linear HS
# step, same coefficients). Needed as an initial guess for Θ to avoid
# the trivial "Θh = const, P = 0" fixed point that also exists in the
# Elrod discretisation.
from reynolds_solver.cavitation.payvar_salant.solver_cpu import (
    _hs_sor_sweep,
)


# ----------------------------------------------------------------------------
# Average-of-cubes face conductance (shared with PS formulation)
# ----------------------------------------------------------------------------
def _build_coefficients(H, d_phi, d_Z, R, L, groove=False):
    """Average-of-cubes A, B, C, D, E on host (numpy). Same as PS."""
    N_Z, N_phi = H.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    Ah = 0.5 * (H[:, :-1] ** 3 + H[:, 1:] ** 3)  # (N_Z, N_phi - 1)

    A = np.zeros((N_Z, N_phi), dtype=np.float64)
    A[:, :-1] = Ah
    if groove:
        A[:, -1] = 0.0
    else:
        A[:, -1] = Ah[:, 0]

    B = np.zeros((N_Z, N_phi), dtype=np.float64)
    B[:, 1:] = Ah
    if groove:
        B[:, 0] = 0.0
    else:
        B[:, 0] = Ah[:, -1]

    H_jph3 = 0.5 * (H[:-1, :] ** 3 + H[1:, :] ** 3)  # (N_Z - 1, N_phi)
    C = np.zeros((N_Z, N_phi), dtype=np.float64)
    D = np.zeros((N_Z, N_phi), dtype=np.float64)
    C[1:-1, :] = alpha_sq * H_jph3[1:, :]
    D[1:-1, :] = alpha_sq * H_jph3[:-1, :]

    E = A + B + C + D
    return A, B, C, D, E


# ----------------------------------------------------------------------------
# Inner SOR sweep (nonlinear, with Fesanghary-Khonsari switch blend)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _elrod_sor_sweep(
    Theta, g, H, A, B, C, D, E,
    d_phi, beta_bar, omega, gfactor,
    theta_min, N_Z, N_phi, groove,
):
    """
    One lexicographic SOR sweep of the compressible Elrod equation
    with NONLINEAR dynamic dispatch (no pinned active set).

    Per node:
      1. Compute a full-film trial. Diffusive flux face conductances
         are weighted by the NEIGHBOUR's g value (NOT min(g_self,
         g_neighbour) — that would block reformation in the current
         cavitated cell). The current cell is treated as locally
         active in the trial.
      2. If Θ_ff ≥ 1, accept full-film: Θ_new = Θ_ff, g_target = 1.
      3. Else fall back to the cavitation transport branch
         Θ_cav = h_jm·Θ_jm / h_ij. If Θ_cav < 1, accept cavitation.
         If Θ_cav ≥ 1, REFORMATION: snap back to full-film
         (Θ_new = 1, g_target = 1).
      4. SOR relaxation, then a single guard against SOR overshooting
         a cavitated cell back into the full-film region.
      5. Soft FK switch update on g.

    Returns (max|ΔΘ|, n_state_flips). State flips are counted by
    HARD classification of (Θ_old vs 1) → (g_target):
        was full-film  Θ_old >= 1-eps  → flip if g_target == 0
        was cavitated  Θ_old <  1-eps  → flip if g_target == 1
    Counting flips on hard state (not on soft g) keeps the metric
    meaningful while gfactor < 1 keeps g intermediate.
    """
    max_delta = 0.0
    n_flips = 0
    six_dphi = 6.0 * d_phi
    state_eps = 1e-12

    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            if groove != 0:
                jp = j + 1
                jm = j - 1
            else:
                jp = j + 1 if j + 1 < N_phi - 1 else 1
                jm = j - 1 if j - 1 >= 1 else N_phi - 2

            Theta_old = Theta[i, j]

            Tjp = Theta[i, jp]
            Tjm = Theta[i, jm]
            Tip = Theta[i + 1, j]
            Tim = Theta[i - 1, j]

            h_ij = H[i, j]
            h_jm = H[i, jm]

            A_l = A[i, j]
            B_l = B[i, j]
            C_l = C[i, j]
            D_l = D[i, j]

            # Face g-weights from the NEIGHBOUR (HARD 0/1 from the
            # current Θ). Current cell is treated as locally active
            # in the trial — using min(g_self, g_nb) would zero all
            # faces and trap the cell in cavitation forever (no
            # reformation pathway). Hard-state instead of the soft
            # FK g avoids the slow drift toward trivial collapse
            # that intermediate g values cause when neighbours
            # straddle the rupture boundary.
            gf_jp = 1.0 if Tjp >= 1.0 - state_eps else 0.0
            gf_jm = 1.0 if Tjm >= 1.0 - state_eps else 0.0
            gf_ip = 1.0 if Tip >= 1.0 - state_eps else 0.0
            gf_im = 1.0 if Tim >= 1.0 - state_eps else 0.0

            diff = (gf_jp * A_l * Tjp + gf_jm * B_l * Tjm
                    + gf_ip * C_l * Tip + gf_im * D_l * Tim)
            E_eff = (gf_jp * A_l + gf_jm * B_l
                     + gf_ip * C_l + gf_im * D_l)

            # Full-film trial
            Theta_ff = (
                (beta_bar * diff + six_dphi * h_jm * Tjm)
                / (beta_bar * E_eff + six_dphi * h_ij + 1e-30)
            )

            if Theta_ff >= 1.0:
                Theta_candidate = Theta_ff
                g_target = 1.0
            else:
                # Cavitation transport branch
                if h_ij > 1e-30:
                    Theta_cav = h_jm * Tjm / h_ij
                else:
                    Theta_cav = theta_min

                if Theta_cav < 1.0:
                    if Theta_cav < theta_min:
                        Theta_cav = theta_min
                    Theta_candidate = Theta_cav
                    g_target = 0.0
                else:
                    # REFORMATION: cavitation transport says Θ ≥ 1 →
                    # snap back to full-film locally.
                    Theta_candidate = 1.0
                    g_target = 1.0

            # SOR relaxation
            Theta_relax = Theta_old + omega * (Theta_candidate - Theta_old)
            if Theta_relax < theta_min:
                Theta_relax = theta_min
            # Single guard: if cavitation branch was selected, do not
            # let SOR overshoot back above 1 (would cause flip-flop on
            # the next sweep). Do NOT clamp when g_target == 1 — that
            # would block legitimate Θ > 1 in full-film.
            if g_target == 0.0 and Theta_relax >= 1.0:
                Theta_relax = 1.0 - 1e-12

            Theta[i, j] = Theta_relax

            # Hard-state flip count
            was_full = Theta_old >= 1.0 - state_eps
            now_full = g_target > 0.5
            if was_full != now_full:
                n_flips += 1

            # Soft Fesanghary-Khonsari switch update
            g[i, j] = g[i, j] + gfactor * (g_target - g[i, j])

            delta = Theta_relax - Theta_old
            if delta < 0.0:
                delta = -delta
            if delta > max_delta:
                max_delta = delta

    # phi boundary
    if groove != 0:
        for i in range(N_Z):
            Theta[i, 0] = 1.0
            Theta[i, N_phi - 1] = 1.0
            g[i, 0] = 1.0
            g[i, N_phi - 1] = 1.0
    else:
        for i in range(N_Z):
            Theta[i, 0] = Theta[i, N_phi - 2]
            Theta[i, N_phi - 1] = Theta[i, 1]
            g[i, 0] = g[i, N_phi - 2]
            g[i, N_phi - 1] = g[i, 1]

    # Z boundary (flooded, Θ = 1, g = 1)
    for j in range(N_phi):
        Theta[0, j] = 1.0
        Theta[N_Z - 1, j] = 1.0
        g[0, j] = 1.0
        g[N_Z - 1, j] = 1.0

    return max_delta, n_flips


# ----------------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------------
def _solve_elrod_compressible_legacy(
    H, d_phi, d_Z, R, L,
    beta_bar,
    omega=1.0,
    tol=1e-6,
    max_iter=500_000,
    check_every=200,
    phi_bc="periodic",
    gfactor=0.9,
    theta_min=1e-8,
    Theta_init=None,
    hs_warmup_iter=50_000,
    hs_warmup_tol=1e-5,
    hs_warmup_omega=None,
    # Deprecated, kept for backward compat: the dynamic-dispatch
    # sweep no longer uses an outer active-set loop or a frozen
    # cav_mask. These kwargs are accepted as no-ops.
    pin_active_set=True,
    max_outer_active_set=10,
    cav_threshold=1e-10,
    verbose=False,
):
    """
    Compressible Elrod / Vijayaraghavan-Keith cavitation solver.

    Parameters
    ----------
    H : (N_Z, N_phi) float64 — dimensionless gap.
    d_phi, d_Z : float — grid spacing (Z ∈ [-1, 1]).
    R, L : float — bearing radius and length (m).
    beta_bar : float — dimensionless bulk modulus β̄ = β·C²/(μ·U·R).
        Finite value (~30 for mineral oil at N=3000 rpm) introduces the
        Θ↔P coupling that amplifies texture-induced wedge effects.
    omega : float — SOR relaxation (1.0 = plain GS; 1.5 is a good
        default for the nonlinear switch).
    tol : float — convergence tolerance on max|ΔΘ| per sweep.
    max_iter : int — maximum SOR sweeps.
    check_every : int — diagnostic print interval.
    phi_bc : {"periodic", "groove"} — φ boundary condition.
        periodic: standard journal bearing (wrap at φ=0/2π).
        groove : supply groove at φ=0,2π (Θ=1, g=1 Dirichlet on j=0
                 and j=N_phi-1, stencil without wrap).
    gfactor : float — Fesanghary-Khonsari switch soft-blend factor,
        in [0, 1]. 0.9 is standard.
    theta_min : float — floor on Θ for numerical safety before ln(Θ).
    Theta_init : (N_Z, N_phi) or None — warm start for Θ.
    verbose : bool.

    Returns
    -------
    P : (N_Z, N_phi) — dimensionless pressure, = β̄·g·ln(Θ).
    theta : (N_Z, N_phi) — fractional film content Θ. In compressible
        Elrod Θ>1 in the full-film lobe (liquid compressed) and Θ<1
        in cavitation. Clipped to [theta_min, ∞) for numerical safety.
    residual : float — final max|ΔΘ|.
    n_iter : int — number of SOR sweeps performed.
    """
    if phi_bc not in ("periodic", "groove"):
        raise ValueError(
            f"phi_bc must be 'periodic' or 'groove', got {phi_bc!r}"
        )
    groove = 1 if phi_bc == "groove" else 0

    N_Z, N_phi = H.shape

    # Ghost-pack H
    H = np.ascontiguousarray(H, dtype=np.float64).copy()
    if groove:
        H[:, 0] = H[:, 1]
        H[:, N_phi - 1] = H[:, N_phi - 2]
    else:
        H[:, 0] = H[:, N_phi - 2]
        H[:, N_phi - 1] = H[:, 1]

    A, B, C, D, E = _build_coefficients(
        H, d_phi, d_Z, R, L, groove=bool(groove),
    )

    # Note: pin_active_set / max_outer_active_set / cav_threshold are
    # accepted only for backward compatibility — the dynamic-dispatch
    # sweep below does NOT use a frozen active set or an outer loop.
    # The single inner loop solves the nonlinear system to convergence
    # via the Fesanghary-Khonsari soft switch.
    _ = pin_active_set
    _ = max_outer_active_set
    _ = cav_threshold

    # --- HS warmup to seed Θ and g ---
    # Without a good initial guess the Elrod SOR drops into the
    # trivial "Θ·h = const, P = 0" fixed point that also exists in
    # the discrete system. HS gives a full-film pressure lobe in the
    # converging region (P_hs > 0) and zero in the diverging region
    # (P_hs clamped to 0). Seed Θ = exp(P_hs / β̄) and g = (P_hs > eps).
    P_hs = None
    if Theta_init is None and hs_warmup_iter > 0:
        if hs_warmup_omega is None:
            from reynolds_solver.utils import compute_auto_omega
            hs_warmup_omega = compute_auto_omega(
                N_phi, N_Z, R, L, cap=1.97,
            )

        F_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                if groove:
                    jm = j - 1
                else:
                    jm = j - 1 if j - 1 >= 1 else N_phi - 2
                F_hs[i, j] = d_phi * (H[i, j] - H[i, jm])

        P_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
        for k in range(hs_warmup_iter):
            hs_res = _hs_sor_sweep(
                P_hs, A, B, C, D, E, F_hs,
                hs_warmup_omega, N_Z, N_phi, groove,
            )
            if hs_res < hs_warmup_tol and k > 5:
                break

        Theta = np.exp(P_hs / beta_bar)
        if verbose:
            print(
                f"  [Elrod] HS warmup: maxP_hs={P_hs.max():.4e}, "
                f"maxΘ_init={Theta.max():.4f}"
            )
    elif Theta_init is not None:
        Theta = np.ascontiguousarray(Theta_init, dtype=np.float64).copy()
        Theta = np.clip(Theta, theta_min, None)
    else:
        Theta = np.ones((N_Z, N_phi), dtype=np.float64)

    # --- Initial g from HS pressure (preferred) or from Theta state ---
    hs_p_eps = 1e-14
    state_eps = 1e-12
    if P_hs is not None:
        g = (P_hs > hs_p_eps).astype(np.float64)
    else:
        g = (Theta > 1.0 + state_eps).astype(np.float64)

    # Enforce boundary conditions on initial state
    if groove:
        Theta[:, 0] = 1.0
        Theta[:, -1] = 1.0
        g[:, 0] = 1.0
        g[:, -1] = 1.0
    else:
        Theta[:, 0] = Theta[:, -2]
        Theta[:, -1] = Theta[:, 1]
        g[:, 0] = g[:, -2]
        g[:, -1] = g[:, 1]
    Theta[0, :] = 1.0
    Theta[-1, :] = 1.0
    g[0, :] = 1.0
    g[-1, :] = 1.0

    if verbose:
        cav0 = int(np.sum(g[1:-1, 1:-1] < 0.5))
        tot0 = (N_Z - 2) * (N_phi - 2)
        print(
            f"  [Elrod] N_Z={N_Z}, N_phi={N_phi}, β̄={beta_bar:.3e}, "
            f"ω={omega}, gfactor={gfactor}, phi_bc={phi_bc}, "
            f"initial cav (from g)={cav0}/{tot0}"
        )

    residual = 1.0
    n_iter = 0
    flips_total_recent = 0
    for k in range(max_iter):
        residual, n_flips = _elrod_sor_sweep(
            Theta, g, H, A, B, C, D, E,
            d_phi, beta_bar, omega, gfactor,
            theta_min, N_Z, N_phi, groove,
        )
        n_iter += 1
        flips_total_recent += n_flips

        if residual < tol and k > 5:
            if verbose:
                print(
                    f"  [Elrod] CONVERGED at iter={k}, "
                    f"res={residual:.3e}"
                )
            break

        if verbose and k % check_every == 0 and k > 0:
            cav_frac = float(np.mean(Theta[1:-1, 1:-1] < 1.0 - 1e-6))
            print(
                f"    inner={k}: maxΔΘ={residual:.3e}, "
                f"Θ=[{Theta.min():.4f}, {Theta.max():.4f}], "
                f"cav={cav_frac:.3f}, "
                f"flips(last {check_every})={flips_total_recent}"
            )
            flips_total_recent = 0

    # Recover pressure. Use hard g (round to 0/1) to avoid soft-switch
    # smearing in the final P field.
    g_hard = (Theta >= 1.0 - 1e-12).astype(np.float64)
    Theta_safe = np.clip(Theta, theta_min, None)
    P = beta_bar * g_hard * np.log(Theta_safe)
    # P must be non-negative (full-film: Θ≥1 → ln Θ ≥ 0; cav: g=0 → P=0)
    P = np.where(P >= 0.0, P, 0.0)

    # Do NOT clip Θ above 1 — in compressible Elrod, full-film cells
    # have Θ > 1 (liquid compressed). Clipping only the lower bound
    # preserves the P = β̄·ln(Θ) identity in the returned arrays.
    theta_out = np.clip(Theta, theta_min, None)

    return P, theta_out, residual, n_iter


# ============================================================================
# NEW ENGINE: explicit P-Θ solver with K-weighted coefficients and Picard
# outer loop around a pseudo-transient or plain-GS inner sweep.
#
# State variables are (P, Θ). Full-film cells have P > 0 and Θ = exp(P/β̄);
# cavitation cells have P = 0 and Θ < 1. K = ρ·h³ is LAGGED over the inner
# sweep and refreshed in the Picard outer loop — this keeps the inner
# problem linear and restores the Ausas-style branch complementarity.
# ============================================================================


def _build_weighted_coefficients_K(K, d_phi, d_Z, R, L, groove=False):
    """
    Average-of-K face conductances for the compressible Elrod K = ρ·h³
    weighting. Same seam / groove convention as `_build_coefficients`.

    A[i, j] = 0.5·(K[i, j]   + K[i, j+1])       phi face +
    B[i, j] = 0.5·(K[i, j-1] + K[i, j]  )       phi face −
    C[i, j] = alpha_sq · 0.5·(K[i, j]   + K[i+1, j])  Z face +
    D[i, j] = alpha_sq · 0.5·(K[i-1, j] + K[i, j]  )  Z face −
    E = A + B + C + D
    """
    N_Z, N_phi = K.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    Kh = 0.5 * (K[:, :-1] + K[:, 1:])

    A = np.zeros((N_Z, N_phi), dtype=np.float64)
    A[:, :-1] = Kh
    if groove:
        A[:, -1] = 0.0
    else:
        A[:, -1] = Kh[:, 0]

    B = np.zeros((N_Z, N_phi), dtype=np.float64)
    B[:, 1:] = Kh
    if groove:
        B[:, 0] = 0.0
    else:
        B[:, 0] = Kh[:, -1]

    K_jph = 0.5 * (K[:-1, :] + K[1:, :])
    C = np.zeros((N_Z, N_phi), dtype=np.float64)
    D = np.zeros((N_Z, N_phi), dtype=np.float64)
    C[1:-1, :] = alpha_sq * K_jph[1:, :]
    D[1:-1, :] = alpha_sq * K_jph[:-1, :]

    E = A + B + C + D
    return A, B, C, D, E


@njit(cache=True)
def _elrod_ptheta_sweep_pt(
    P, Theta, H, c_prev,
    A, B, C, D, E,
    d_phi, beta_bar, pt_beta,
    omega_p, omega_theta,
    N_Z, N_phi, groove,
    p_state_eps, theta_state_eps, theta_min,
):
    """
    One lexicographic GS sweep of the explicit P-Θ Elrod system WITH the
    pseudo-transient term (Ausas eq. 12 adapted for compressible):

        6·d_phi·(c^n − c_jm^n) + pt_beta·(c^n − c_prev)
            = A·P_jp + B·P_jm + C·P_ip + D·P_im − E·P_ij

    where c = Θ·h and K = ρ·h³ is LAGGED (baked into A..E).

    Per node, two branches with complementarity:

    Branch 1 (pressure trial, Θ locally = exp(P/β̄)):
        F_couette = 6·d_phi·(ρ_ij·h_ij − Θ_up·h_jm)
        F_pt      = pt_beta·(ρ_ij·h_ij − c_prev_ij)
        P_trial   = (A·Pjp + B·Pjm + C·Pip + D·Pim − F_couette − F_pt) / E
        if P_trial > p_state_eps: accept, Θ = exp(P_trial/β̄)
        else: → Branch 2

    Branch 2 (cavitation, P = 0, solve for Θ):
        stencil    = A·Pjp + B·Pjm + C·Pip + D·Pim − E·P_cur
        Θ_trial   = (stencil + 6·d_phi·h_jm·Θ_up + pt_beta·c_prev_ij)
                      / ((6·d_phi + pt_beta)·h_ij)
        if Θ_trial < 1:  accept cavitation
        else:            REFORMATION → Θ = 1, P = 0 (re-opens Branch 1 next sweep)

    Returns (max_delta, n_state_flips). State flip counted on hard
    classification: was_full = (P_old > p_eps) OR (Θ_old ≥ 1-θ_eps).
    """
    max_delta = 0.0
    n_flips = 0
    six_dphi = 6.0 * d_phi

    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            if groove != 0:
                jp = j + 1
                jm = j - 1
            else:
                jp = j + 1 if j + 1 < N_phi - 1 else 1
                jm = j - 1 if j - 1 >= 1 else N_phi - 2

            P_old = P[i, j]
            Theta_old = Theta[i, j]

            Pjp = P[i, jp]
            Pjm = P[i, jm]
            Pip = P[i + 1, j]
            Pim = P[i - 1, j]

            Theta_up = Theta[i, jm]

            h_ij = H[i, j]
            h_jm = H[i, jm]
            cp_ij = c_prev[i, j]

            A_l = A[i, j]
            B_l = B[i, j]
            C_l = C[i, j]
            D_l = D[i, j]
            E_l = E[i, j]

            was_full = (P_old > p_state_eps) or (Theta_old >= 1.0 - theta_state_eps)

            P_cur = P_old
            Theta_cur = Theta_old

            # ---- Branch 1: pressure trial ----
            candidate_full = was_full
            if candidate_full:
                if P_old > p_state_eps:
                    rho_ij = np.exp(P_old / beta_bar)
                    if rho_ij < 1.0:
                        rho_ij = 1.0
                else:
                    rho_ij = 1.0

                F_couette = six_dphi * (rho_ij * h_ij - Theta_up * h_jm)
                F_pt = pt_beta * (rho_ij * h_ij - cp_ij)
                diff = A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim
                P_trial = (diff - F_couette - F_pt) / (E_l + 1e-30)
                P_relax = omega_p * P_trial + (1.0 - omega_p) * P_old

                if P_relax > p_state_eps:
                    P_cur = P_relax
                    Theta_cur = np.exp(P_cur / beta_bar)
                else:
                    P_cur = 0.0
                    # Theta_cur unchanged; Branch 2 will decide

            # ---- Branch 2: cavitation (solve for Θ) ----
            if P_cur <= p_state_eps:
                stencil = (
                    A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim
                    - E_l * P_cur
                )
                Theta_num = (
                    stencil
                    + six_dphi * h_jm * Theta_up
                    + pt_beta * cp_ij
                )
                Theta_den = (six_dphi + pt_beta) * h_ij + 1e-30
                Theta_trial = Theta_num / Theta_den
                Theta_relax = (
                    omega_theta * Theta_trial
                    + (1.0 - omega_theta) * Theta_old
                )

                if Theta_relax < 1.0 - theta_state_eps:
                    # Cavitated
                    if Theta_relax < theta_min:
                        Theta_relax = theta_min
                    Theta_cur = Theta_relax
                    P_cur = 0.0
                else:
                    # Reformation / saturation: cell will re-enter the
                    # pressure branch on the next sweep.
                    Theta_cur = 1.0
                    P_cur = 0.0

            P[i, j] = P_cur
            Theta[i, j] = Theta_cur

            now_full = (P_cur > p_state_eps) or (Theta_cur >= 1.0 - theta_state_eps)
            if was_full != now_full:
                n_flips += 1

            dP = P_cur - P_old
            if dP < 0.0:
                dP = -dP
            dth = Theta_cur - Theta_old
            if dth < 0.0:
                dth = -dth
            d = dP if dP > dth else dth
            if d > max_delta:
                max_delta = d

    # phi boundary
    if groove != 0:
        for i in range(N_Z):
            P[i, 0] = 0.0
            P[i, N_phi - 1] = 0.0
            Theta[i, 0] = 1.0
            Theta[i, N_phi - 1] = 1.0
    else:
        for i in range(N_Z):
            P[i, 0] = P[i, N_phi - 2]
            P[i, N_phi - 1] = P[i, 1]
            Theta[i, 0] = Theta[i, N_phi - 2]
            Theta[i, N_phi - 1] = Theta[i, 1]

    # Z boundary (flooded, P = 0, Θ = 1)
    for j in range(N_phi):
        P[0, j] = 0.0
        P[N_Z - 1, j] = 0.0
        Theta[0, j] = 1.0
        Theta[N_Z - 1, j] = 1.0

    return max_delta, n_flips


@njit(cache=True)
def _elrod_ptheta_sweep_gs(
    P, Theta, H,
    A, B, C, D, E,
    d_phi, beta_bar,
    omega_p, omega_theta,
    N_Z, N_phi, groove,
    p_state_eps, theta_state_eps, theta_min,
):
    """
    Plain GS sweep of the explicit P-Θ system WITHOUT a pseudo-time term
    (scheme="gs" path). Kept as diagnostic fallback — NOT the default.
    Same branch logic as the PT sweep with pt_beta = 0 and c_prev ignored.
    """
    max_delta = 0.0
    n_flips = 0
    six_dphi = 6.0 * d_phi

    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            if groove != 0:
                jp = j + 1
                jm = j - 1
            else:
                jp = j + 1 if j + 1 < N_phi - 1 else 1
                jm = j - 1 if j - 1 >= 1 else N_phi - 2

            P_old = P[i, j]
            Theta_old = Theta[i, j]

            Pjp = P[i, jp]
            Pjm = P[i, jm]
            Pip = P[i + 1, j]
            Pim = P[i - 1, j]
            Theta_up = Theta[i, jm]

            h_ij = H[i, j]
            h_jm = H[i, jm]

            A_l = A[i, j]
            B_l = B[i, j]
            C_l = C[i, j]
            D_l = D[i, j]
            E_l = E[i, j]

            was_full = (P_old > p_state_eps) or (Theta_old >= 1.0 - theta_state_eps)

            P_cur = P_old
            Theta_cur = Theta_old

            if was_full:
                if P_old > p_state_eps:
                    rho_ij = np.exp(P_old / beta_bar)
                    if rho_ij < 1.0:
                        rho_ij = 1.0
                else:
                    rho_ij = 1.0
                F_couette = six_dphi * (rho_ij * h_ij - Theta_up * h_jm)
                diff = A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim
                P_trial = (diff - F_couette) / (E_l + 1e-30)
                P_relax = omega_p * P_trial + (1.0 - omega_p) * P_old

                if P_relax > p_state_eps:
                    P_cur = P_relax
                    Theta_cur = np.exp(P_cur / beta_bar)
                else:
                    P_cur = 0.0

            if P_cur <= p_state_eps:
                stencil = (
                    A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim
                    - E_l * P_cur
                )
                Theta_num = stencil + six_dphi * h_jm * Theta_up
                Theta_den = six_dphi * h_ij + 1e-30
                Theta_trial = Theta_num / Theta_den
                Theta_relax = (
                    omega_theta * Theta_trial
                    + (1.0 - omega_theta) * Theta_old
                )

                if Theta_relax < 1.0 - theta_state_eps:
                    if Theta_relax < theta_min:
                        Theta_relax = theta_min
                    Theta_cur = Theta_relax
                    P_cur = 0.0
                else:
                    Theta_cur = 1.0
                    P_cur = 0.0

            P[i, j] = P_cur
            Theta[i, j] = Theta_cur

            now_full = (P_cur > p_state_eps) or (Theta_cur >= 1.0 - theta_state_eps)
            if was_full != now_full:
                n_flips += 1

            dP = P_cur - P_old
            if dP < 0.0:
                dP = -dP
            dth = Theta_cur - Theta_old
            if dth < 0.0:
                dth = -dth
            d = dP if dP > dth else dth
            if d > max_delta:
                max_delta = d

    # BC (same as PT sweep)
    if groove != 0:
        for i in range(N_Z):
            P[i, 0] = 0.0
            P[i, N_phi - 1] = 0.0
            Theta[i, 0] = 1.0
            Theta[i, N_phi - 1] = 1.0
    else:
        for i in range(N_Z):
            P[i, 0] = P[i, N_phi - 2]
            P[i, N_phi - 1] = P[i, 1]
            Theta[i, 0] = Theta[i, N_phi - 2]
            Theta[i, N_phi - 1] = Theta[i, 1]
    for j in range(N_phi):
        P[0, j] = 0.0
        P[N_Z - 1, j] = 0.0
        Theta[0, j] = 1.0
        Theta[N_Z - 1, j] = 1.0

    return max_delta, n_flips


def _seed_initial_state(
    H, N_Z, N_phi, d_phi, R, L, groove,
    beta_bar, theta_min,
    P_init, Theta_init,
    hs_warmup_iter, hs_warmup_tol, hs_warmup_omega,
    p_state_eps, verbose,
):
    """
    Build initial (P, Θ) per ТЗ §9.2:
      1. If P_init given: use it as seed; Θ = exp(P/β̄) where P > p_eps
         else Θ = 1.
      2. Elif Theta_init given: reconstruct P from Θ (P = β̄·ln(Θ) on
         Θ ≥ 1, else P = 0).
      3. Else if HS warmup available: use P_hs as seed.
      4. Else: P ≡ 0, Θ ≡ 1.

    After seeding, apply BC (phi groove + Z flooded ends).
    """
    # Case 1: explicit P_init
    if P_init is not None:
        P = np.ascontiguousarray(P_init, dtype=np.float64).copy()
        P = np.where(P > 0.0, P, 0.0)
        Theta = np.where(
            P > p_state_eps,
            np.exp(np.clip(P, 0.0, None) / beta_bar),
            1.0,
        )

    # Case 2: Theta_init without P_init
    elif Theta_init is not None:
        Theta = np.ascontiguousarray(Theta_init, dtype=np.float64).copy()
        Theta = np.clip(Theta, theta_min, None)
        P = np.where(
            Theta >= 1.0,
            beta_bar * np.log(np.clip(Theta, 1.0, None)),
            0.0,
        )

    # Case 3: HS warmup
    elif hs_warmup_iter > 0:
        if hs_warmup_omega is None:
            from reynolds_solver.utils import compute_auto_omega
            hs_warmup_omega = compute_auto_omega(
                N_phi, N_Z, R, L, cap=1.97,
            )
        # Build HS (linear Reynolds) coefficients from bare h³.
        A0, B0, C0, D0, E0 = _build_coefficients(
            H, d_phi, 2.0 / (N_Z - 1), R, L, groove=bool(groove),
        )
        F_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                if groove:
                    jm = j - 1
                else:
                    jm = j - 1 if j - 1 >= 1 else N_phi - 2
                F_hs[i, j] = d_phi * (H[i, j] - H[i, jm])
        P_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
        for k in range(hs_warmup_iter):
            hs_res = _hs_sor_sweep(
                P_hs, A0, B0, C0, D0, E0, F_hs,
                hs_warmup_omega, N_Z, N_phi, groove,
            )
            if hs_res < hs_warmup_tol and k > 5:
                break
        P = np.where(P_hs > 0.0, P_hs, 0.0)
        Theta = np.where(
            P > p_state_eps,
            np.exp(P / beta_bar),
            1.0,
        )
        if verbose:
            print(
                f"  [Elrod-PΘ] HS seed: maxP_hs={P.max():.4e}, "
                f"ω_hs={hs_warmup_omega:.4f}"
            )

    # Case 4: cold start
    else:
        P = np.zeros((N_Z, N_phi), dtype=np.float64)
        Theta = np.ones((N_Z, N_phi), dtype=np.float64)

    # Apply BC
    if groove:
        P[:, 0] = 0.0
        P[:, -1] = 0.0
        Theta[:, 0] = 1.0
        Theta[:, -1] = 1.0
    else:
        P[:, 0] = P[:, -2]
        P[:, -1] = P[:, 1]
        Theta[:, 0] = Theta[:, -2]
        Theta[:, -1] = Theta[:, 1]
    P[0, :] = 0.0
    P[-1, :] = 0.0
    Theta[0, :] = 1.0
    Theta[-1, :] = 1.0

    return P, Theta


def _solve_elrod_ptheta_picard(
    H, d_phi, d_Z, R, L,
    beta_bar,
    scheme,
    omega_p, omega_theta,
    tol, max_iter, check_every,
    max_picard, tol_picard,
    pt_dt, pt_max_time_steps, pt_max_inner, pt_inner_tol,
    phi_bc,
    hs_warmup_iter, hs_warmup_tol, hs_warmup_omega,
    theta_min, p_state_eps, theta_state_eps,
    P_init, Theta_init,
    verbose,
):
    """
    Picard outer loop around the inner complementarity solve.

    Outer iteration (freezes K = ρ·h³):
      * build A..E from the lagged K;
      * run the inner sweep (pt or gs) to the inner tolerance;
      * check max|ΔP|, max|ΔΘ| — break when both are below tol_picard.
    """
    if phi_bc not in ("periodic", "groove"):
        raise ValueError(
            f"phi_bc must be 'periodic' or 'groove', got {phi_bc!r}"
        )
    groove = 1 if phi_bc == "groove" else 0

    N_Z, N_phi = H.shape

    # Ghost-pack H
    H = np.ascontiguousarray(H, dtype=np.float64).copy()
    if groove:
        H[:, 0] = H[:, 1]
        H[:, N_phi - 1] = H[:, N_phi - 2]
    else:
        H[:, 0] = H[:, N_phi - 2]
        H[:, N_phi - 1] = H[:, 1]

    # Initial (P, Θ)
    P, Theta = _seed_initial_state(
        H, N_Z, N_phi, d_phi, R, L, bool(groove),
        beta_bar, theta_min,
        P_init, Theta_init,
        hs_warmup_iter, hs_warmup_tol, hs_warmup_omega,
        p_state_eps, verbose,
    )

    # Pseudo-time β = 2·d_phi² / Δτ (Ausas convention)
    pt_beta = 2.0 * d_phi * d_phi / pt_dt if scheme == "pseudo_transient" else 0.0

    if verbose:
        cav0 = int(np.sum(Theta[1:-1, 1:-1] < 1.0 - theta_state_eps))
        tot0 = (N_Z - 2) * (N_phi - 2)
        print(
            f"  [Elrod-PΘ] N_Z={N_Z}, N_phi={N_phi}, β̄={beta_bar:.3e}, "
            f"scheme={scheme}, ω_p={omega_p}, ω_θ={omega_theta}, "
            f"phi_bc={phi_bc}, initial cav={cav0}/{tot0}"
        )
        if scheme == "pseudo_transient":
            print(
                f"  [Elrod-PΘ] pt_dt={pt_dt}, pt_beta={pt_beta:.3e}, "
                f"pt_max_time_steps={pt_max_time_steps}, "
                f"pt_max_inner={pt_max_inner}"
            )

    n_iter_total = 0
    residual_inner_final = 1.0
    picard_delta_final = 1.0

    for picard in range(max_picard):
        P_lag = P.copy()
        Theta_lag = Theta.copy()

        # Build K = ρ·h³ from the LAGGED pressure (ρ = 1 in cavitation,
        # exp(P/β̄) in full-film).
        rho_lag = np.ones_like(P_lag)
        ff_mask = P_lag > p_state_eps
        if ff_mask.any():
            rho_lag[ff_mask] = np.exp(P_lag[ff_mask] / beta_bar)
        K = rho_lag * (H ** 3)

        A, B, C, D, E = _build_weighted_coefficients_K(
            K, d_phi, d_Z, R, L, groove=bool(groove),
        )

        if scheme == "pseudo_transient":
            # Pseudo-time marching within one Picard iteration.
            c_prev = Theta * H
            for pt_step in range(pt_max_time_steps):
                inner_res = 1.0
                for inner_k in range(pt_max_inner):
                    inner_res, n_flips = _elrod_ptheta_sweep_pt(
                        P, Theta, H, c_prev,
                        A, B, C, D, E,
                        d_phi, beta_bar, pt_beta,
                        omega_p, omega_theta,
                        N_Z, N_phi, groove,
                        p_state_eps, theta_state_eps, theta_min,
                    )
                    n_iter_total += 1
                    if inner_res < pt_inner_tol and inner_k > 2:
                        break
                # advance c_prev between pseudo-time steps
                c_prev = Theta * H
                residual_inner_final = inner_res
                if inner_res < tol and pt_step > 2:
                    break
        else:
            # scheme == "gs": plain nonlinear GS, no pseudo-time
            for k in range(max_iter):
                inner_res, n_flips = _elrod_ptheta_sweep_gs(
                    P, Theta, H,
                    A, B, C, D, E,
                    d_phi, beta_bar,
                    omega_p, omega_theta,
                    N_Z, N_phi, groove,
                    p_state_eps, theta_state_eps, theta_min,
                )
                n_iter_total += 1
                residual_inner_final = inner_res

                if inner_res < tol and k > 5:
                    break

                if verbose and k % check_every == 0 and k > 0:
                    cav_frac = float(np.mean(
                        Theta[1:-1, 1:-1] < 1.0 - theta_state_eps
                    ))
                    print(
                        f"    picard={picard} inner={k}: "
                        f"res_inner={inner_res:.3e}, "
                        f"P_max={P.max():.4e}, "
                        f"Θ=[{Theta.min():.4f}, {Theta.max():.4f}], "
                        f"cav={cav_frac:.3f}, flips={n_flips}"
                    )

        # Picard convergence check
        dP_max = float(np.max(np.abs(P - P_lag)))
        dTh_max = float(np.max(np.abs(Theta - Theta_lag)))
        picard_delta = max(dP_max, dTh_max)
        picard_delta_final = picard_delta

        if verbose:
            cav_frac = float(np.mean(
                Theta[1:-1, 1:-1] < 1.0 - theta_state_eps
            ))
            print(
                f"  [Elrod-PΘ] picard={picard}: "
                f"res_inner={residual_inner_final:.3e}, "
                f"res_picard={picard_delta:.3e}, "
                f"P_max={P.max():.4e}, "
                f"Θ=[{Theta.min():.4f}, {Theta.max():.4f}], "
                f"cav={cav_frac:.3f}"
            )

        if picard_delta < tol_picard and picard > 0:
            if verbose:
                print(
                    f"  [Elrod-PΘ] PICARD CONVERGED at picard={picard}"
                )
            break

    theta_out = np.clip(Theta, theta_min, None)
    return P, theta_out, residual_inner_final, n_iter_total


def solve_elrod_compressible(
    H, d_phi, d_Z, R, L,
    beta_bar,
    omega=None,
    omega_p=None,
    omega_theta=None,
    tol=1e-6,
    max_iter=50_000,
    check_every=200,
    phi_bc="periodic",
    theta_min=1e-8,
    P_init=None,
    Theta_init=None,
    hs_warmup_iter=50_000,
    hs_warmup_tol=1e-5,
    hs_warmup_omega=None,
    scheme="pseudo_transient",
    max_picard=50,
    tol_picard=1e-6,
    pt_dt=0.1,
    pt_max_time_steps=200,
    pt_max_inner=50,
    pt_inner_tol=1e-4,
    # Deprecated kwargs, accepted but unused (for backward compat)
    gfactor=None,
    pin_active_set=None,
    max_outer_active_set=None,
    cav_threshold=None,
    verbose=False,
):
    """
    Compressible Elrod / Vijayaraghavan-Keith cavitation solver —
    explicit (P, Θ) formulation with Picard outer loop and
    pseudo-transient inner sweep.

    State variables are P and Θ directly (no separate g array). The
    equation is

        ∇·(K·∇P) = 6·d_phi·∂(Θ·h)/∂φ + pt_beta·(c − c_prev)

    with K = ρ·h³, ρ = exp(P/β̄) in full-film and ρ = 1 in cavitation.
    K is LAGGED over the inner sweep and refreshed each Picard outer
    iteration.

    Per-node complementarity (Ausas-style):
      * Branch 1 (pressure): try P = (stencil − F_couette − F_pt) / E
        with Θ_local = exp(P/β̄); accept if P > p_state_eps.
      * Branch 2 (cavitation): solve for Θ from mass conservation with
        P = 0; accept if Θ < 1, else REFORMATION (Θ = 1, P = 0).

    Pseudo-transient term pt_beta·(c − c_prev) stabilises the inner
    loop against the trivial "Θ·h = const, P = 0" fixed point.

    Parameters
    ----------
    beta_bar : float — dimensionless bulk modulus.
    omega, omega_p, omega_theta : SOR relaxation. If omega is given but
        omega_p / omega_theta are not, both default to omega. If none
        are given, both default to 1.0.
    tol : convergence tolerance on max|ΔP|, max|ΔΘ| per inner sweep.
    scheme : "pseudo_transient" (default) or "gs" (diagnostic).
    max_picard, tol_picard : outer loop controls.
    pt_dt, pt_max_time_steps, pt_max_inner, pt_inner_tol :
        pseudo-transient controls. pt_beta = 2·d_phi²/pt_dt.
    phi_bc : "periodic" or "groove".
    P_init, Theta_init : optional warm starts (ТЗ §9.2).
    hs_warmup_iter/tol/omega : HS warmup seed for pressure.

    Deprecated (accepted as no-op): gfactor, pin_active_set,
    max_outer_active_set, cav_threshold.

    Returns
    -------
    P, theta, residual, n_iter
    """
    # Silence unused deprecated kwargs
    _ = gfactor
    _ = pin_active_set
    _ = max_outer_active_set
    _ = cav_threshold

    # ω alias logic
    if omega_p is None and omega is not None:
        omega_p = omega
    if omega_theta is None and omega is not None:
        omega_theta = omega
    if omega_p is None:
        omega_p = 1.0
    if omega_theta is None:
        omega_theta = 1.0

    if scheme not in ("pseudo_transient", "gs"):
        raise ValueError(
            f"scheme must be 'pseudo_transient' or 'gs', got {scheme!r}"
        )

    p_state_eps = 1e-12
    theta_state_eps = 1e-12

    return _solve_elrod_ptheta_picard(
        H, d_phi, d_Z, R, L,
        beta_bar=beta_bar,
        scheme=scheme,
        omega_p=omega_p, omega_theta=omega_theta,
        tol=tol, max_iter=max_iter, check_every=check_every,
        max_picard=max_picard, tol_picard=tol_picard,
        pt_dt=pt_dt,
        pt_max_time_steps=pt_max_time_steps,
        pt_max_inner=pt_max_inner,
        pt_inner_tol=pt_inner_tol,
        phi_bc=phi_bc,
        hs_warmup_iter=hs_warmup_iter,
        hs_warmup_tol=hs_warmup_tol,
        hs_warmup_omega=hs_warmup_omega,
        theta_min=theta_min,
        p_state_eps=p_state_eps,
        theta_state_eps=theta_state_eps,
        P_init=P_init,
        Theta_init=Theta_init,
        verbose=verbose,
    )
