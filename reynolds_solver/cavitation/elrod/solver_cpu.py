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
    # Engine selector + per-engine controls
    formulation="ptheta",     # "ptheta" (default) or "theta_vk"
    scheme="pseudo_transient",
    max_picard=50,
    tol_picard=1e-6,
    pt_dt=0.1,
    # NOTE: 5000 outer steps is a safe ceiling — theta_vk needs a long
    # tail to lock the interface integral; ptheta exits early via its
    # Picard tol so the larger cap is harmless.
    pt_max_time_steps=5000,
    pt_max_inner=50,
    pt_inner_tol=1e-4,
    # theta_vk controls
    switch_backend="hard",    # "hard" or "fk_soft"
    theta_vk_scheme="gs_symmetric_inline",   # "gs_symmetric_inline" (default) | "pseudo_transient" | "gs_inline_legacy"
    eta_schedule=None,                        # pseudo_transient continuation
    pair_damp=1.0,                            # symmetric_inline
    stages=None,                              # symmetric_inline staged continuation
    gfactor=0.9,
    tol_g=1e-6,
    n_quiescent=3,
    # Deprecated kwargs, accepted but unused (for backward compat)
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

    if formulation not in ("ptheta", "theta_vk"):
        raise ValueError(
            f"formulation must be 'ptheta' or 'theta_vk', got "
            f"{formulation!r}"
        )

    # Dispatch to faithful Θ-form VK/FK engine (alongside default P-Θ)
    if formulation == "theta_vk":
        return _solve_elrod_theta_vk(
            H, d_phi, d_Z, R, L,
            beta_bar=beta_bar,
            omega=omega_p,                # single ω in Θ-form
            gfactor=0.9 if gfactor is None else gfactor,
            switch_backend=switch_backend,
            scheme=theta_vk_scheme,
            pair_damp=pair_damp,
            stages=stages,
            eta_schedule=eta_schedule,
            pt_dt=pt_dt,
            pt_max_time_steps=pt_max_time_steps,
            pt_max_inner=pt_max_inner,
            pt_inner_tol=pt_inner_tol,
            tol_theta=tol,
            tol_g=tol_g,
            max_iter=max_iter,
            check_every=check_every,
            n_quiescent=n_quiescent,
            phi_bc=phi_bc,
            theta_min=theta_min,
            P_init=P_init,
            Theta_init=Theta_init,
            hs_warmup_iter=hs_warmup_iter,
            hs_warmup_tol=hs_warmup_tol,
            hs_warmup_omega=hs_warmup_omega,
            verbose=verbose,
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


# ============================================================================
# Faithful Θ-form Vijayaraghavan-Keith / Fesanghary-Khonsari solver
# (additional engine, NOT the public default).
#
# Single unknown: Θ. State arrays:
#   Theta    — main variable
#   g_state  — hard switch (strictly 0/1) — physical topology
#   g_soft   — soft FK switch (in [0, 1]) — numerical stabilizer
#
# Stencil weight `g_phys` is taken from g_state (switch_backend="hard")
# or g_soft (switch_backend="fk_soft"). Coefficients are bare h³ face
# averages (no K = ρ·h³); compressibility enters only through the
# final P = β̄·g·ln(Θ) recovery.
#
# Local equation (Manser-style):
#   β̄·g_phys·(A·Θ_jp + B·Θ_jm + C·Θ_ip + D·Θ_im - E·Θ_ij)
#       = 6·d_phi·(h_ij·Θ_ij - h_jm·Θ_jm)
# ============================================================================


@njit(cache=True)
def _elrod_theta_vk_sweep_inline(
    Theta, g_state, g_soft, H,
    A, B, C, D, E,
    d_phi, beta_bar, omega, gfactor,
    use_soft_backend,
    N_Z, N_phi, groove,
    state_eps, theta_min,
    reverse_order,
):
    """
    Inline GS sweep of the faithful Θ-form Elrod system with in-place
    g_state / g_soft updates (node-by-node). This is the low-level
    kernel used by both the plain `gs_inline_legacy` driver and the
    `gs_symmetric_inline` symmetric-pair driver.

    `reverse_order`: if 0 → lexicographic forward (Z up, φ up); if 1 →
    full lexicographic reverse (Z down, φ down). The reverse traversal
    is NOT just a φ-flip — the outer Z loop also reverses. This
    removes the directional bias of a single-direction sweep.

    Returns (max_delta_theta, max_delta_gsoft, n_state_flips).
    """
    max_dth = 0.0
    max_dg = 0.0
    n_flips = 0
    six_dphi = 6.0 * d_phi

    if reverse_order == 0:
        i_start, i_stop, i_step = 1, N_Z - 1, 1
        j_start, j_stop, j_step = 1, N_phi - 1, 1
    else:
        i_start, i_stop, i_step = N_Z - 2, 0, -1
        j_start, j_stop, j_step = N_phi - 2, 0, -1

    i = i_start
    while (i_step > 0 and i < i_stop) or (i_step < 0 and i > i_stop):
        j = j_start
        while (j_step > 0 and j < j_stop) or (j_step < 0 and j > j_stop):
            if groove != 0:
                jp = j + 1
                jm = j - 1
            else:
                jp = j + 1 if j + 1 < N_phi - 1 else 1
                jm = j - 1 if j - 1 >= 1 else N_phi - 2

            Theta_old = Theta[i, j]
            gst_old = g_state[i, j]
            gsf_old = g_soft[i, j]

            Tjp = Theta[i, jp]
            Tjm = Theta[i, jm]
            Tip = Theta[i + 1, j]
            Tim = Theta[i - 1, j]
            Theta_up = Tjm

            h_ij = H[i, j]
            h_jm = H[i, jm]

            A_l = A[i, j]
            B_l = B[i, j]
            C_l = C[i, j]
            D_l = D[i, j]
            E_l = E[i, j]

            if use_soft_backend != 0:
                g_phys = gsf_old
            else:
                g_phys = gst_old

            diff = A_l * Tjp + B_l * Tjm + C_l * Tip + D_l * Tim
            num = beta_bar * g_phys * diff + six_dphi * h_jm * Theta_up
            den = beta_bar * g_phys * E_l + six_dphi * h_ij + 1e-30
            Theta_trial = num / den

            if Theta_trial >= 1.0 - state_eps:
                state_target = 1
            else:
                state_target = 0

            Theta_relax = Theta_old + omega * (Theta_trial - Theta_old)
            if Theta_relax < theta_min:
                Theta_relax = theta_min

            if state_target == 0 and Theta_relax >= 1.0:
                Theta_relax = 1.0 - state_eps
            if state_target == 1 and omega < 1.0 and Theta_relax < 1.0:
                Theta_relax = 1.0

            Theta[i, j] = Theta_relax

            new_state = float(state_target)
            new_soft = gsf_old + gfactor * (new_state - gsf_old)
            if new_soft < 0.0:
                new_soft = 0.0
            if new_soft > 1.0:
                new_soft = 1.0

            if (gst_old > 0.5) != (state_target == 1):
                n_flips += 1
            g_state[i, j] = new_state
            g_soft[i, j] = new_soft

            dth = Theta_relax - Theta_old
            if dth < 0.0:
                dth = -dth
            if dth > max_dth:
                max_dth = dth
            dg = new_soft - gsf_old
            if dg < 0.0:
                dg = -dg
            if dg > max_dg:
                max_dg = dg

            j += j_step
        i += i_step

    # phi BC
    if groove != 0:
        for ii in range(N_Z):
            Theta[ii, 0] = 1.0
            Theta[ii, N_phi - 1] = 1.0
            g_state[ii, 0] = 1.0
            g_state[ii, N_phi - 1] = 1.0
            g_soft[ii, 0] = 1.0
            g_soft[ii, N_phi - 1] = 1.0
    else:
        for ii in range(N_Z):
            Theta[ii, 0] = Theta[ii, N_phi - 2]
            Theta[ii, N_phi - 1] = Theta[ii, 1]
            g_state[ii, 0] = g_state[ii, N_phi - 2]
            g_state[ii, N_phi - 1] = g_state[ii, 1]
            g_soft[ii, 0] = g_soft[ii, N_phi - 2]
            g_soft[ii, N_phi - 1] = g_soft[ii, 1]

    # Z BC (flooded)
    for jj in range(N_phi):
        Theta[0, jj] = 1.0
        Theta[N_Z - 1, jj] = 1.0
        g_state[0, jj] = 1.0
        g_state[N_Z - 1, jj] = 1.0
        g_soft[0, jj] = 1.0
        g_soft[N_Z - 1, jj] = 1.0

    return max_dth, max_dg, n_flips


# Backward-compat alias (used by existing "gs_inline_legacy" path)
@njit(cache=True)
def _elrod_theta_vk_sweep_inline_legacy(
    Theta, g_state, g_soft, H,
    A, B, C, D, E,
    d_phi, beta_bar, omega, gfactor,
    use_soft_backend,
    N_Z, N_phi, groove,
    state_eps, theta_min,
):
    """Forward-only inline sweep (pre-symmetric-pair baseline)."""
    return _elrod_theta_vk_sweep_inline(
        Theta, g_state, g_soft, H,
        A, B, C, D, E,
        d_phi, beta_bar, omega, gfactor,
        use_soft_backend,
        N_Z, N_phi, groove,
        state_eps, theta_min,
        0,   # reverse_order=False
    )


# -----------------------------------------------------------------------
# Fixed-point state pack/unpack/project helpers for the nonlinear outer
# accelerators (underrelaxed FP, Anderson). The "state vector" x
# concatenates INTERIOR (Theta, g_soft) values only; g_state is always
# recovered as (g_soft >= 0.5). Boundary ghosts are re-imposed on every
# unpack. See ТЗ §4.1 / §4.3.
# -----------------------------------------------------------------------


def _pack_theta_vk_state(Theta, g_soft):
    """Pack interior (Theta, g_soft) values into a single 1-D vector."""
    T_int = Theta[1:-1, 1:-1].ravel()
    g_int = g_soft[1:-1, 1:-1].ravel()
    return np.concatenate([T_int, g_int])


def _unpack_theta_vk_state(x, Theta, g_soft, g_state, groove,
                           theta_min=1e-8):
    """Write `x` back into (Theta, g_soft), project, rebuild g_state,
    and reapply phi / Z boundary conditions. In-place on the given
    arrays."""
    N_Z, N_phi = Theta.shape
    n_int = (N_Z - 2) * (N_phi - 2)
    T_int = x[:n_int].reshape(N_Z - 2, N_phi - 2)
    g_int = x[n_int:].reshape(N_Z - 2, N_phi - 2)
    Theta[1:-1, 1:-1] = np.maximum(T_int, theta_min)
    g_soft[1:-1, 1:-1] = np.clip(g_int, 0.0, 1.0)
    # Hard topology from soft
    g_state[:] = (g_soft >= 0.5).astype(np.float64)
    # phi BC
    if groove:
        Theta[:, 0] = 1.0
        Theta[:, -1] = 1.0
        g_state[:, 0] = 1.0
        g_state[:, -1] = 1.0
        g_soft[:, 0] = 1.0
        g_soft[:, -1] = 1.0
    else:
        Theta[:, 0] = Theta[:, -2]
        Theta[:, -1] = Theta[:, 1]
        g_state[:, 0] = g_state[:, -2]
        g_state[:, -1] = g_state[:, 1]
        g_soft[:, 0] = g_soft[:, -2]
        g_soft[:, -1] = g_soft[:, 1]
    # Z BC (flooded)
    Theta[0, :] = 1.0
    Theta[-1, :] = 1.0
    g_state[0, :] = 1.0
    g_state[-1, :] = 1.0
    g_soft[0, :] = 1.0
    g_soft[-1, :] = 1.0


def _theta_vk_fixed_point_map(
    x_in, Theta, g_state, g_soft, H,
    A, B, C, D, E,
    d_phi, beta_bar, omega, gfactor,
    use_soft, N_Z, N_phi, groove,
    state_eps, theta_min,
):
    """
    F(x) = one full forward inline sweep of the theta_vk/fk_soft kernel.

    Unpacks x into the working (Theta, g_state, g_soft) fields
    (in-place on the provided scratch arrays), runs one forward inline
    sweep, and packs the result back into a new vector.

    Returns (x_out, max_dth, max_dg, n_flips).
    """
    _unpack_theta_vk_state(x_in, Theta, g_soft, g_state, groove, theta_min)
    max_dth, max_dg, n_flips = _elrod_theta_vk_sweep_inline(
        Theta, g_state, g_soft, H,
        A, B, C, D, E,
        d_phi, beta_bar, omega, gfactor,
        use_soft,
        N_Z, N_phi, groove,
        state_eps, theta_min,
        0,   # forward
    )
    x_out = _pack_theta_vk_state(Theta, g_soft)
    return x_out, float(max_dth), float(max_dg), int(n_flips)


def _pmax_from_state(Theta, g_state, beta_bar):
    """Pmax(recovered from hard topology) for safeguard/diagnostic use."""
    Theta_for_p = np.where(
        g_state > 0.5, np.maximum(Theta, 1.0), 1.0,
    )
    P = beta_bar * g_state * np.log(Theta_for_p)
    return float(np.where(P >= 0.0, P, 0.0).max())


# -----------------------------------------------------------------------
# New lagged + pseudo-transient sweep — current default for theta_vk.
# Key design points (per TZ):
#   1. Stencil weight g_phys is read from LAGGED arrays g_state_lag,
#      g_soft_lag (plus eta-blend); it is NOT updated in-place in this
#      sweep. All new topology decisions are written to the separate
#      array g_state_target.
#   2. A pseudo-transient term pt_beta * (h·Θ - c_prev) is added to
#      the local equation, where c_prev = Θ_prev · H is the "mass"
#      anchor from the outer step. This damps limit cycles on the
#      hard / soft switch interface.
# -----------------------------------------------------------------------


@njit(cache=True)
def _elrod_theta_vk_sweep_pt_lagged(
    Theta, g_state_lag, g_soft_lag, g_state_target,
    c_prev, H,
    A, B, C, D, E,
    d_phi, beta_bar, omega_theta,
    eta,            # continuation parameter: g_phys = (1-eta)*g_hard + eta*g_soft
    pt_beta,        # pseudo-transient β = 1 / Δτ
    N_Z, N_phi, groove,
    state_eps, theta_min,
):
    """
    Lagged + pseudo-transient sweep of the Θ-form VK/FK Elrod system.

    Stencil uses only lagged g fields. Topology targets are written
    to the separate `g_state_target` array. The caller is responsible
    for applying gfactor relaxation to g_soft AFTER this sweep/block.

    Local equation (see TZ §5.3):
        pt_beta·(h·Θ - c_prev)
        + β̄·g_phys·(E·Θ - [A·Θ_jp + B·Θ_jm + C·Θ_ip + D·Θ_im])
        = 6·d_phi·(h_ij·Θ_ij - h_jm·Θ_up)

    Returns (max_delta_theta, n_state_flips).

    `n_state_flips` counts nodes whose target topology differs from
    g_state_lag (the lag at the start of this outer step), which is
    the quantity that matters for the outer quiescent test.
    """
    max_dth = 0.0
    n_flips = 0
    six_dphi = 6.0 * d_phi
    one_minus_eta = 1.0 - eta

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
            Theta_up = Tjm

            h_ij = H[i, j]
            h_jm = H[i, jm]

            A_l = A[i, j]
            B_l = B[i, j]
            C_l = C[i, j]
            D_l = D[i, j]
            E_l = E[i, j]

            # Lagged g_phys: blend hard and soft lag by eta
            g_phys = one_minus_eta * g_state_lag[i, j] + eta * g_soft_lag[i, j]

            diff = A_l * Tjp + B_l * Tjm + C_l * Tip + D_l * Tim

            # Local equation (see docstring). Collect terms on Θ_ij on
            # the LHS and everything else on the RHS.
            #   LHS coeff: β̄·g·E + 6dφ·h_ij + pt_beta·h_ij
            #   RHS     : β̄·g·diff + 6dφ·h_jm·Θ_up + pt_beta·c_prev
            num = (
                beta_bar * g_phys * diff
                + six_dphi * h_jm * Theta_up
                + pt_beta * c_prev[i, j]
            )
            den = (
                beta_bar * g_phys * E_l
                + six_dphi * h_ij
                + pt_beta * h_ij
                + 1e-30
            )
            Theta_trial = num / den

            # Topology target (HARD decision from the trial value)
            if Theta_trial >= 1.0 - state_eps:
                state_target = 1
            else:
                state_target = 0

            # SOR relaxation
            Theta_relax = Theta_old + omega_theta * (Theta_trial - Theta_old)
            if Theta_relax < theta_min:
                Theta_relax = theta_min

            # Topology-consistency clamp
            if state_target == 0 and Theta_relax >= 1.0:
                Theta_relax = 1.0 - state_eps
            if state_target == 1 and Theta_relax < 1.0:
                Theta_relax = 1.0

            Theta[i, j] = Theta_relax

            # Write target topology; do NOT touch g_state/g_soft here
            new_target = float(state_target)
            if (g_state_lag[i, j] > 0.5) != (state_target == 1):
                n_flips += 1
            g_state_target[i, j] = new_target

            dth = Theta_relax - Theta_old
            if dth < 0.0:
                dth = -dth
            if dth > max_dth:
                max_dth = dth

    # phi BC on Theta and g_state_target (so caller's outer step sees BC-clean fields)
    if groove != 0:
        for i in range(N_Z):
            Theta[i, 0] = 1.0
            Theta[i, N_phi - 1] = 1.0
            g_state_target[i, 0] = 1.0
            g_state_target[i, N_phi - 1] = 1.0
    else:
        for i in range(N_Z):
            Theta[i, 0] = Theta[i, N_phi - 2]
            Theta[i, N_phi - 1] = Theta[i, 1]
            g_state_target[i, 0] = g_state_target[i, N_phi - 2]
            g_state_target[i, N_phi - 1] = g_state_target[i, 1]

    # Z BC (flooded)
    for j in range(N_phi):
        Theta[0, j] = 1.0
        Theta[N_Z - 1, j] = 1.0
        g_state_target[0, j] = 1.0
        g_state_target[N_Z - 1, j] = 1.0

    return max_dth, n_flips


def _solve_elrod_theta_vk(
    H, d_phi, d_Z, R, L,
    beta_bar,
    omega=1.0,
    gfactor=0.9,
    switch_backend="hard",
    scheme="gs_symmetric_inline",     # NEW DEFAULT
    # gs_symmetric_inline controls
    pair_damp=1.0,
    stages=None,             # list of {"gfactor":..., "omega_theta":..., "pair_damp":...}
    # Pseudo-transient (outer/inner) controls (scheme="pseudo_transient")
    pt_dt=0.1,
    pt_max_time_steps=5000,
    pt_max_inner=50,
    pt_inner_tol=1e-4,
    # Eta continuation schedule (only used for fk_soft, scheme="pseudo_transient")
    eta_schedule=None,
    tol_theta=1e-6,
    tol_g=1e-6,
    max_iter=200_000,
    check_every=200,
    n_quiescent=3,
    phi_bc="groove",
    theta_min=1e-8,
    P_init=None,
    Theta_init=None,
    hs_warmup_iter=50_000,
    hs_warmup_tol=1e-5,
    hs_warmup_omega=None,
    verbose=False,
):
    """
    Faithful Θ-form VK/FK solver. Single unknown Θ + (g_state, g_soft).

    Three schemes:
      scheme="gs_symmetric_inline" (NEW DEFAULT): one nonlinear
        iteration = forward lexicographic inline sweep + reverse
        lexicographic inline sweep, followed by a pair-level damping
        Θ = Θ_prev + pair_damp · (Θ_new - Θ_prev) (same for g_soft).
        Keeps the inline coefficient-chasing coupling per sweep but
        removes the directional sweep-order bias that made the
        pure-forward path (gs_inline_legacy) potentially
        sweep-artifactual. Staged continuation by gfactor / omega /
        pair_damp is supported (parameter `stages`).
      scheme="gs_inline_legacy": original forward-only inline sweep.
        Kept for A/B sweep-order diagnostics — if the asymmetry
        seen on gs_inline_legacy flips under the symmetric scheme,
        it was an artifact.
      scheme="pseudo_transient": lagged g_phys + pseudo-time anchor
        c_prev = Θ_prev·H. Robust on hard backend but tends to a
        SYMMETRIC attractor on textured fk_soft (Manser asymmetry
        lost — see previous ТЗ §14 fall-back).

    Quiescent convergence rule (symmetric_inline):
      strict: max_dth_pair < tol_theta AND max_dg_pair < tol_g AND
              n_flips_pair == 0
      soft:   ΔPmax/Pmax < 5e-5 AND Δcav < 1e-4 over last 20
              checkpoints AND n_flips_pair < 0.5 % of interior
              cells (only after pt_min_warmup pairs).
    A history-based limit-cycle detector tracks the last 10
    checkpoints of (Pmax, W, phi_rupt, fluid_area): a limit cycle
    is reported if residuals are not falling AND any of these
    integrals oscillates with finite amplitude above 1 % (P_max)
    or 1° (phi_rupt).
    """
    if phi_bc not in ("periodic", "groove"):
        raise ValueError(
            f"phi_bc must be 'periodic' or 'groove', got {phi_bc!r}"
        )
    if switch_backend not in ("hard", "fk_soft"):
        raise ValueError(
            f"switch_backend must be 'hard' or 'fk_soft', got "
            f"{switch_backend!r}"
        )
    if scheme not in (
        "gs_symmetric_inline", "pseudo_transient",
        "gs_inline_legacy", "gs_inline_reverse",
        "fp_plain", "fp_underrelaxed", "fp_anderson",
    ):
        raise ValueError(
            f"scheme must be one of 'gs_symmetric_inline', "
            f"'pseudo_transient', 'gs_inline_legacy', 'gs_inline_reverse', "
            f"'fp_plain', 'fp_underrelaxed', 'fp_anderson', "
            f"got {scheme!r}"
        )
    use_soft = 1 if switch_backend == "fk_soft" else 0
    groove = 1 if phi_bc == "groove" else 0
    state_eps = 1e-12
    p_state_eps = 1e-12

    N_Z, N_phi = H.shape

    # Ghost-pack H
    H = np.ascontiguousarray(H, dtype=np.float64).copy()
    if groove:
        H[:, 0] = H[:, 1]
        H[:, N_phi - 1] = H[:, N_phi - 2]
    else:
        H[:, 0] = H[:, N_phi - 2]
        H[:, N_phi - 1] = H[:, 1]

    # Bare h³ coefficients (Θ-form uses these directly, weighted by g_phys
    # inside the sweep)
    A, B, C, D, E = _build_coefficients(
        H, d_phi, d_Z, R, L, groove=bool(groove),
    )

    # Seed (P, Θ, g_state, g_soft) per ТЗ §4.9
    if P_init is not None:
        P_seed = np.where(P_init > 0.0, P_init, 0.0)
        Theta = np.where(
            P_seed > p_state_eps,
            np.exp(np.clip(P_seed, 0.0, None) / beta_bar),
            1.0,
        )
        g_state = (P_seed > p_state_eps).astype(np.float64)
    elif Theta_init is not None:
        Theta = np.ascontiguousarray(Theta_init, dtype=np.float64).copy()
        Theta = np.clip(Theta, theta_min, None)
        g_state = (Theta >= 1.0 - state_eps).astype(np.float64)
    elif hs_warmup_iter > 0:
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
        P_seed = np.where(P_hs > 0.0, P_hs, 0.0)
        Theta = np.where(
            P_seed > p_state_eps,
            np.exp(P_seed / beta_bar),
            1.0,
        )
        g_state = (P_seed > p_state_eps).astype(np.float64)
        if verbose:
            print(
                f"  [Elrod-Θ-VK] HS seed: maxP_hs={P_seed.max():.4e}, "
                f"ω_hs={hs_warmup_omega:.4f}"
            )
    else:
        Theta = np.ones((N_Z, N_phi), dtype=np.float64)
        g_state = np.ones((N_Z, N_phi), dtype=np.float64)

    g_soft = g_state.copy()

    # Apply BC to initial state
    if groove:
        Theta[:, 0] = 1.0
        Theta[:, -1] = 1.0
        g_state[:, 0] = 1.0
        g_state[:, -1] = 1.0
        g_soft[:, 0] = 1.0
        g_soft[:, -1] = 1.0
    else:
        Theta[:, 0] = Theta[:, -2]
        Theta[:, -1] = Theta[:, 1]
        g_state[:, 0] = g_state[:, -2]
        g_state[:, -1] = g_state[:, 1]
        g_soft[:, 0] = g_soft[:, -2]
        g_soft[:, -1] = g_soft[:, 1]
    Theta[0, :] = 1.0
    Theta[-1, :] = 1.0
    g_state[0, :] = 1.0
    g_state[-1, :] = 1.0
    g_soft[0, :] = 1.0
    g_soft[-1, :] = 1.0

    if verbose:
        print(
            f"  [Elrod-Θ-VK] N_Z={N_Z}, N_phi={N_phi}, "
            f"β̄={beta_bar:.3e}, ω={omega}, gfactor={gfactor}, "
            f"backend={switch_backend}, scheme={scheme}, phi_bc={phi_bc}"
        )

    # -------------------------------------------------------------
    # NEW DEFAULT: symmetric inline pair + pair_damp + staged cont.
    # -------------------------------------------------------------
    if scheme == "gs_symmetric_inline":
        # Default staged continuation schedule: starts from very
        # damped FK state (small gfactor, ω<1, no pair_damp) and
        # walks up toward fully inline coupling. Each stage starts
        # from the solution of the previous stage.
        if stages is None:
            stages = [
                {"gfactor": 0.3, "omega_theta": 0.5, "pair_damp": 1.0},
                {"gfactor": 0.5, "omega_theta": 0.7, "pair_damp": 1.0},
                {"gfactor": 0.7, "omega_theta": 0.7, "pair_damp": 0.7},
                {"gfactor": 0.9, "omega_theta": 1.0, "pair_damp": 0.7},
            ]

        # History buffers for the limit-cycle detector
        hist_res = []     # pair residual (max of forward/reverse)
        hist_Pmax = []
        hist_cav = []
        hist_fluid = []
        k_hist = 10

        total_pairs = 0
        max_dth_pair = 1.0
        max_dg_pair = 1.0
        n_flips_pair = 0
        limit_cycle_detected = False

        # Iterations budget is shared across stages by total pair count
        pairs_per_stage = max(1, max_iter // len(stages))

        for stage_idx, stage in enumerate(stages):
            gf_stage = float(stage.get("gfactor", gfactor))
            om_stage = float(stage.get("omega_theta", omega))
            pd_stage = float(stage.get("pair_damp", pair_damp))

            if verbose:
                print(
                    f"  [Elrod-Θ-VK sym] stage {stage_idx}: "
                    f"gfactor={gf_stage}, omega_theta={om_stage}, "
                    f"pair_damp={pd_stage}"
                )

            quiescent_streak = 0
            for pair_step in range(pairs_per_stage):
                Theta_prev = Theta.copy()
                g_soft_prev = g_soft.copy()

                # Forward sweep
                res_f, dg_f, fl_f = _elrod_theta_vk_sweep_inline(
                    Theta, g_state, g_soft, H,
                    A, B, C, D, E,
                    d_phi, beta_bar, om_stage, gf_stage,
                    use_soft,
                    N_Z, N_phi, groove,
                    state_eps, theta_min,
                    0,   # forward
                )
                # Reverse sweep
                res_r, dg_r, fl_r = _elrod_theta_vk_sweep_inline(
                    Theta, g_state, g_soft, H,
                    A, B, C, D, E,
                    d_phi, beta_bar, om_stage, gf_stage,
                    use_soft,
                    N_Z, N_phi, groove,
                    state_eps, theta_min,
                    1,   # reverse
                )

                # Pair-level damping on BOTH fields
                if pd_stage != 1.0:
                    Theta[:] = Theta_prev + pd_stage * (Theta - Theta_prev)
                    np.maximum(Theta, theta_min, out=Theta)
                    g_soft[:] = (
                        g_soft_prev
                        + pd_stage * (g_soft - g_soft_prev)
                    )
                    np.clip(g_soft, 0.0, 1.0, out=g_soft)
                    g_state[:] = (g_soft >= 0.5).astype(np.float64)
                    # Apply BC after damping
                    if groove:
                        Theta[:, 0] = 1.0; Theta[:, -1] = 1.0
                        g_state[:, 0] = 1.0; g_state[:, -1] = 1.0
                        g_soft[:, 0] = 1.0; g_soft[:, -1] = 1.0
                    else:
                        Theta[:, 0] = Theta[:, -2]
                        Theta[:, -1] = Theta[:, 1]
                        g_state[:, 0] = g_state[:, -2]
                        g_state[:, -1] = g_state[:, 1]
                        g_soft[:, 0] = g_soft[:, -2]
                        g_soft[:, -1] = g_soft[:, 1]
                    Theta[0, :] = 1.0; Theta[-1, :] = 1.0
                    g_state[0, :] = 1.0; g_state[-1, :] = 1.0
                    g_soft[0, :] = 1.0; g_soft[-1, :] = 1.0

                # Pair-level residuals (true Δ from before both sweeps)
                max_dth_pair = float(np.max(np.abs(Theta - Theta_prev)))
                max_dg_pair = float(np.max(np.abs(g_soft - g_soft_prev)))
                n_flips_pair = fl_f + fl_r
                total_pairs += 1

                # Integral diagnostics
                Theta_for_p = np.where(
                    g_state > 0.5, np.maximum(Theta, 1.0), 1.0,
                )
                P_now = beta_bar * g_state * np.log(Theta_for_p)
                P_now = np.where(P_now >= 0.0, P_now, 0.0)
                Pmax_now = float(P_now.max())
                cav_now = float(np.mean(g_state[1:-1, 1:-1] < 0.5))
                fluid_now = float(np.mean(g_state[1:-1, 1:-1] > 0.5))

                hist_res.append(max_dth_pair)
                hist_Pmax.append(Pmax_now)
                hist_cav.append(cav_now)
                hist_fluid.append(fluid_now)
                if len(hist_res) > k_hist:
                    hist_res.pop(0)
                    hist_Pmax.pop(0)
                    hist_cav.pop(0)
                    hist_fluid.pop(0)

                # Convergence check: strict OR soft
                strict_ok = (
                    max_dth_pair < tol_theta
                    and max_dg_pair < tol_g
                    and n_flips_pair == 0
                )
                soft_ok = False
                pt_min_warmup = 30
                if (
                    stage_idx == len(stages) - 1
                    and pair_step >= pt_min_warmup
                    and len(hist_Pmax) >= k_hist
                    and n_flips_pair > 0
                ):
                    Pmax_span = max(hist_Pmax) - min(hist_Pmax)
                    cav_span = max(hist_cav) - min(hist_cav)
                    rel_Pmax = Pmax_span / max(hist_Pmax[-1], 1e-30)
                    max_flip_frac = 0.005 * max(
                        (N_Z - 2) * (N_phi - 2), 1
                    )
                    soft_ok = (
                        rel_Pmax < 5e-5
                        and cav_span < 1e-4
                        and n_flips_pair < max_flip_frac
                    )

                converged_now = strict_ok or soft_ok
                if converged_now:
                    quiescent_streak += 1
                else:
                    quiescent_streak = 0

                # Limit-cycle detection (only after enough history)
                if len(hist_res) == k_hist and pair_step > 50:
                    # residual not falling: recent max not < 0.9 * oldest max
                    res_falling = hist_res[-1] < 0.9 * max(
                        hist_res[0], 1e-30
                    )
                    Pmax_osc = (
                        (max(hist_Pmax) - min(hist_Pmax))
                        / max(hist_Pmax[-1], 1e-30)
                        > 1e-2
                    )
                    if (not res_falling) and Pmax_osc:
                        limit_cycle_detected = True

                if verbose and pair_step % max(1, check_every // 2) == 0:
                    print(
                        f"    stage={stage_idx} pair={pair_step:5d} "
                        f"ΔΘ={max_dth_pair:.2e} Δg={max_dg_pair:.2e} "
                        f"flips={n_flips_pair:4d} "
                        f"Pmax={Pmax_now:.3e} cav={cav_now:.3f} "
                        f"fluid={fluid_now:.3f} "
                        f"quies={quiescent_streak} "
                        f"limit_cycle={limit_cycle_detected}"
                    )

                if quiescent_streak >= n_quiescent:
                    if verbose:
                        print(
                            f"  [Elrod-Θ-VK sym] stage {stage_idx} "
                            f"CONVERGED at pair={pair_step}"
                        )
                    break

                if total_pairs * 2 >= max_iter:
                    break
            # end stage pair loop

            if total_pairs * 2 >= max_iter:
                break
        # end stages

        # Pressure recovery from HARD topology
        g_hard = g_state
        Theta_for_p = np.where(
            g_hard > 0.5, np.maximum(Theta, 1.0), 1.0,
        )
        P = beta_bar * g_hard * np.log(Theta_for_p)
        P = np.where(P >= 0.0, P, 0.0)
        theta_out = np.clip(Theta, theta_min, None)

        if verbose and limit_cycle_detected:
            print(
                f"  [Elrod-Θ-VK sym] WARNING: limit-cycle detected "
                f"(residual stalled, integrals oscillating)."
            )

        # Return: (P, theta, residual, n_sweeps). Total inner sweeps
        # = 2 · total_pairs (forward + reverse per pair).
        return P, theta_out, max_dth_pair, 2 * total_pairs

    # -------------------------------------------------------------
    # Legacy paths: forward-only and reverse-only inline (diagnostics)
    # -------------------------------------------------------------
    if scheme in ("gs_inline_legacy", "gs_inline_reverse"):
        reverse_flag = 1 if scheme == "gs_inline_reverse" else 0
        quiescent_streak = 0
        max_dth = 1.0
        max_dg = 1.0
        n_iter = 0
        for k in range(max_iter):
            max_dth, max_dg, n_flips = _elrod_theta_vk_sweep_inline(
                Theta, g_state, g_soft, H,
                A, B, C, D, E,
                d_phi, beta_bar, omega, gfactor,
                use_soft,
                N_Z, N_phi, groove,
                state_eps, theta_min,
                reverse_flag,
            )
            n_iter += 1
            if k > 5 and k % check_every == 0:
                converged_now = (
                    max_dth < tol_theta and max_dg < tol_g and n_flips == 0
                )
                if converged_now:
                    quiescent_streak += 1
                else:
                    quiescent_streak = 0
                if verbose:
                    cav_frac = float(np.mean(g_state[1:-1, 1:-1] < 0.5))
                    print(
                        f"    iter={k}: ΔΘ={max_dth:.3e}, "
                        f"Δg={max_dg:.3e}, cav={cav_frac:.3f}, "
                        f"flips={n_flips}, quiescent={quiescent_streak}"
                    )
                if quiescent_streak >= n_quiescent:
                    if verbose:
                        print(
                            f"  [Elrod-Θ-VK legacy] CONVERGED at "
                            f"iter={k}"
                        )
                    break
        g_hard = g_state
        Theta_for_p = np.where(
            g_hard > 0.5, np.maximum(Theta, 1.0), 1.0,
        )
        P = beta_bar * g_hard * np.log(Theta_for_p)
        P = np.where(P >= 0.0, P, 0.0)
        theta_out = np.clip(Theta, theta_min, None)
        return P, theta_out, max_dth, n_iter

    # -------------------------------------------------------------
    # Fixed-point acceleration paths — ТЗ: "не менять kernel,
    # стабилизировать F(x) снаружи". One F-eval = one forward inline
    # sweep. Safeguards: residual-based step acceptance + Pmax spike
    # monitor + Theta_min / g_soft∈[0,1] projection after every
    # mixing step.
    # -------------------------------------------------------------
    if scheme in ("fp_plain", "fp_underrelaxed", "fp_anderson"):
        # Pack initial state
        x = _pack_theta_vk_state(Theta, g_soft)

        # History for metric-stationarity / limit-cycle telemetry
        hist_res = []
        hist_Pmax = []
        hist_cav = []
        hist_fluid = []
        k_hist = 10

        # Adaptive under-relaxation state (used by both underrelaxed
        # and as Anderson safeguard fallback).
        lam_cur = 1.0
        prev_r_norm = None

        # Anderson history buffers (kept only while scheme=fp_anderson)
        # X_hist stores x_k; G_hist stores g_k = F(x_k); R_hist stores
        # residuals r_k = g_k - x_k. Always keep paired entries.
        X_hist = []
        G_hist = []
        R_hist = []
        # NOTE: anderson_start is conservatively large. On hard-switch
        # VK/FK maps the residual is dominated by interface flips for
        # the first ~500–1000 iterations; Anderson extrapolating on
        # that noisy residual reliably produces wrong attractors. We
        # let plain FP carry the first 500 iterations so the topology
        # coarsens, and only then engage Anderson as a refinement.
        anderson_m = 3
        anderson_start = 500

        n_iter = 0
        max_dth = 1.0
        max_dg = 1.0
        n_flips = 0
        quiescent_streak = 0
        converged_via_strict = False
        converged_via_soft = False

        # Evaluate initial F once to get (g_0, r_0, x_1_naive)
        x_trial, max_dth, max_dg, n_flips = _theta_vk_fixed_point_map(
            x, Theta, g_state, g_soft, H,
            A, B, C, D, E,
            d_phi, beta_bar, omega, gfactor,
            use_soft, N_Z, N_phi, groove,
            state_eps, theta_min,
        )
        r = x_trial - x
        n_iter += 1

        # Safeguard reference: Pmax on initial F(x) (HS seed + 1 sweep).
        # Hard ceiling: 3× this reference with a generous floor of
        # 10.0 — legit textured Pmax can reach ~7, so we allow up to
        # 10 by default but catch Anderson spikes > 10..15×.
        Pmax_smooth_ref = _pmax_from_state(Theta, g_state, beta_bar)
        Pmax_ceiling = max(3.0 * Pmax_smooth_ref, 10.0)
        Pmax_running = Pmax_smooth_ref

        for k in range(1, max_iter):
            r_norm = float(np.linalg.norm(r))
            x_trial_from_last = x_trial.copy()

            # --- Pick next x_{k+1} based on scheme ---
            if scheme == "fp_plain":
                x_next = x_trial_from_last

            elif scheme == "fp_underrelaxed":
                # Adaptive λ based on residual growth, NOT on Pmax
                # spikes (those are transient inside healthy inline
                # evolution and caused earlier builds to collapse).
                # Rule: if |r_k| grew by >3× from the previous |r|,
                # halve λ (min 0.1). If |r_k| fell, slowly grow λ
                # toward 1.0. The default is lam=1.0 (= fp_plain).
                if prev_r_norm is not None:
                    if r_norm > 3.0 * prev_r_norm:
                        lam_cur = max(lam_cur * 0.5, 0.1)
                    elif r_norm < 0.9 * prev_r_norm:
                        lam_cur = min(lam_cur * 1.1, 1.0)
                prev_r_norm = r_norm
                x_cand = x + lam_cur * r
                # Project
                _unpack_theta_vk_state(
                    x_cand, Theta, g_soft, g_state, groove, theta_min,
                )
                x_next = _pack_theta_vk_state(Theta, g_soft)

            else:  # fp_anderson
                # Append current (x, g=x_trial, r) to history
                X_hist.append(x.copy())
                G_hist.append(x_trial_from_last.copy())
                R_hist.append(r.copy())
                if len(X_hist) > anderson_m + 1:
                    X_hist.pop(0)
                    G_hist.pop(0)
                    R_hist.pop(0)

                use_anderson = (
                    k >= anderson_start and len(R_hist) >= 2
                )
                if use_anderson:
                    # Build F_mat (residual differences) and G_mat
                    # (g differences) for least-squares.
                    m_k = len(R_hist) - 1
                    F_mat = np.column_stack([
                        R_hist[i + 1] - R_hist[i]
                        for i in range(m_k)
                    ])
                    G_mat = np.column_stack([
                        G_hist[i + 1] - G_hist[i]
                        for i in range(m_k)
                    ])
                    try:
                        alpha, *_ = np.linalg.lstsq(
                            F_mat, R_hist[-1], rcond=None,
                        )
                        x_cand = x_trial_from_last - G_mat @ alpha
                    except np.linalg.LinAlgError:
                        x_cand = None

                    if x_cand is not None and np.all(np.isfinite(x_cand)):
                        # Extra safeguard: bound deviation from plain
                        # FP step (reject Anderson that extrapolates
                        # more than 3× the plain FP residual).
                        dev_norm = float(
                            np.linalg.norm(x_cand - x_trial_from_last)
                        )
                        if dev_norm > 3.0 * r_norm:
                            x_cand = None
                            use_anderson = False

                    if x_cand is not None and np.all(np.isfinite(x_cand)):
                        # Safeguard: project, evaluate F on candidate,
                        # accept only if r_cand < r and no Pmax spike.
                        _unpack_theta_vk_state(
                            x_cand, Theta, g_soft, g_state, groove,
                            theta_min,
                        )
                        Pmax_cand = _pmax_from_state(
                            Theta, g_state, beta_bar,
                        )
                        x_cand_proj = _pack_theta_vk_state(Theta, g_soft)
                        if Pmax_cand > Pmax_ceiling:
                            use_anderson = False
                        else:
                            # Evaluate F on the candidate
                            g_cand, _, _, _ = _theta_vk_fixed_point_map(
                                x_cand_proj, Theta, g_state, g_soft, H,
                                A, B, C, D, E,
                                d_phi, beta_bar, omega, gfactor,
                                use_soft, N_Z, N_phi, groove,
                                state_eps, theta_min,
                            )
                            n_iter += 1
                            # Post-F Pmax check (catch candidates that
                            # look OK before F but spike after it)
                            Pmax_postF = _pmax_from_state(
                                Theta, g_state, beta_bar,
                            )
                            r_cand = g_cand - x_cand_proj
                            r_cand_norm = float(np.linalg.norm(r_cand))
                            # Accept only with strict residual decrease
                            # and no post-F spike.
                            if (r_cand_norm < 0.9 * r_norm
                                    and Pmax_postF < Pmax_ceiling):
                                x_next = x_cand_proj
                                x_trial = g_cand
                                r = r_cand
                                # Skip the normal F-eval below — already did it
                                x = x_next
                                max_dth = float(
                                    np.max(np.abs(g_cand - x_cand_proj))
                                )
                                hist_res.append(r_cand_norm)
                                Pmax_now = _pmax_from_state(
                                    Theta, g_state, beta_bar,
                                )
                                cav_now = float(
                                    np.mean(g_state[1:-1, 1:-1] < 0.5)
                                )
                                fluid_now = float(
                                    np.mean(g_state[1:-1, 1:-1] > 0.5)
                                )
                                hist_Pmax.append(Pmax_now)
                                hist_cav.append(cav_now)
                                hist_fluid.append(fluid_now)
                                if len(hist_res) > k_hist:
                                    hist_res.pop(0)
                                    hist_Pmax.pop(0)
                                    hist_cav.pop(0)
                                    hist_fluid.pop(0)
                                prev_r_norm = r_cand_norm
                                continue
                            else:
                                use_anderson = False

                if not use_anderson:
                    # Fallback: safeguarded under-relaxation
                    accepted_lam = None
                    for lam in [1.0, 0.7, 0.5, 0.3, 0.1]:
                        x_cand = x + lam * r
                        _unpack_theta_vk_state(
                            x_cand, Theta, g_soft, g_state, groove,
                            theta_min,
                        )
                        Pmax_cand = _pmax_from_state(
                            Theta, g_state, beta_bar,
                        )
                        if Pmax_cand < 2.5 * max(
                            Pmax_smooth_ref, 1e-9,
                        ):
                            accepted_lam = lam
                            x_next = _pack_theta_vk_state(Theta, g_soft)
                            lam_cur = lam
                            break
                    if accepted_lam is None:
                        lam = 0.1
                        x_cand = x + lam * r
                        _unpack_theta_vk_state(
                            x_cand, Theta, g_soft, g_state, groove,
                            theta_min,
                        )
                        x_next = _pack_theta_vk_state(Theta, g_soft)
                        lam_cur = lam

            # --- Commit: evaluate F at the new x to get next residual ---
            x = x_next
            x_trial, max_dth, max_dg, n_flips = _theta_vk_fixed_point_map(
                x, Theta, g_state, g_soft, H,
                A, B, C, D, E,
                d_phi, beta_bar, omega, gfactor,
                use_soft, N_Z, N_phi, groove,
                state_eps, theta_min,
            )
            r = x_trial - x
            n_iter += 1

            r_norm = float(np.linalg.norm(r))

            # Diagnostics
            Pmax_now = _pmax_from_state(Theta, g_state, beta_bar)
            cav_now = float(np.mean(g_state[1:-1, 1:-1] < 0.5))
            fluid_now = float(np.mean(g_state[1:-1, 1:-1] > 0.5))
            # Let the healthy Pmax reference grow monotonically, but
            # only by what the inline kernel itself produced (the
            # safeguard already kept the accelerated candidate below
            # 1.5x the previous Pmax_running, so committing this
            # iteration's Pmax_now is safe).
            if Pmax_now > Pmax_running:
                Pmax_running = Pmax_now
            hist_res.append(r_norm)
            hist_Pmax.append(Pmax_now)
            hist_cav.append(cav_now)
            hist_fluid.append(fluid_now)
            if len(hist_res) > k_hist:
                hist_res.pop(0)
                hist_Pmax.pop(0)
                hist_cav.pop(0)
                hist_fluid.pop(0)

            # Convergence: strict OR (much tighter) soft quiescent.
            # Soft requires r_norm small too, to prevent exit during
            # transient Pmax plateaus mid-evolution.
            strict_ok = (
                max_dth < tol_theta and max_dg < tol_g and n_flips == 0
            )
            soft_ok = False
            if (
                len(hist_Pmax) == k_hist
                and k > 100
                and r_norm < 1e-2
                and n_flips < 0.01 * (N_Z - 2) * (N_phi - 2)
            ):
                Pmax_span = max(hist_Pmax) - min(hist_Pmax)
                cav_span = max(hist_cav) - min(hist_cav)
                fluid_span = max(hist_fluid) - min(hist_fluid)
                rel_Pmax = Pmax_span / max(hist_Pmax[-1], 1e-30)
                if (rel_Pmax < 1e-4
                        and cav_span < 5e-4
                        and fluid_span < 5e-4):
                    soft_ok = True
            if strict_ok or soft_ok:
                quiescent_streak += 1
                if strict_ok:
                    converged_via_strict = True
                else:
                    converged_via_soft = True
            else:
                quiescent_streak = 0

            if verbose and k % max(1, check_every) == 0:
                fluid_ = float(np.mean(g_state[1:-1, 1:-1] > 0.5))
                print(
                    f"    [{scheme}] k={k:5d} n_F={n_iter:5d} "
                    f"|r|={r_norm:.3e} ΔΘ={max_dth:.2e} "
                    f"flips={n_flips:4d} "
                    f"Pmax={Pmax_now:.3e} cav={cav_now:.3f} "
                    f"fluid={fluid_:.3f} "
                    f"λ={lam_cur if scheme != 'fp_plain' else 1.0:.2f} "
                    f"quies={quiescent_streak}"
                )

            if quiescent_streak >= n_quiescent:
                if verbose:
                    print(
                        f"  [Elrod-Θ-VK {scheme}] CONVERGED at k={k} "
                        f"(strict={converged_via_strict}, "
                        f"soft={converged_via_soft})"
                    )
                break
            if n_iter >= max_iter:
                break

        g_hard = g_state
        Theta_for_p = np.where(
            g_hard > 0.5, np.maximum(Theta, 1.0), 1.0,
        )
        P = beta_bar * g_hard * np.log(Theta_for_p)
        P = np.where(P >= 0.0, P, 0.0)
        theta_out = np.clip(Theta, theta_min, None)
        return P, theta_out, max_dth, n_iter

    # -------------------------------------------------------------
    # Default path: lagged + pseudo-transient, two-level loop
    # -------------------------------------------------------------

    # η-continuation schedule
    if eta_schedule is None:
        if use_soft == 1:
            eta_schedule = [0.0, 0.25, 0.5, 0.75, 1.0]
        else:
            eta_schedule = [0.0]   # pure hard stencil
    eta_schedule = [float(x) for x in eta_schedule]

    pt_beta_val = 1.0 / max(pt_dt, 1e-30)

    g_state_target = np.zeros_like(g_state)
    max_dth_outer = 1.0
    max_dg_outer = 1.0
    n_flips_outer = 1
    total_pt_steps = 0
    total_inner_sweeps = 0

    # Track last 3 checkpoints of integral metrics for drift diagnostic
    hist_Pmax = []
    hist_cav = []

    last_k_inner = 0

    for eta in eta_schedule:
        if verbose:
            print(
                f"  [Elrod-Θ-VK] continuation stage: η={eta:.3f} "
                f"(pt_dt={pt_dt}, max_pt_steps={pt_max_time_steps}, "
                f"max_inner={pt_max_inner})"
            )
        quiescent_streak = 0

        for pt_step in range(pt_max_time_steps):
            # Outer: refresh lag and pseudo-time anchor
            Theta_prev = Theta.copy()
            c_prev = Theta_prev * H
            g_state_lag = g_state.copy()
            g_soft_lag = g_soft.copy()
            g_state_target[:] = g_state_lag    # initial target = current

            # Inner: Picard-like sweep to pt_inner_tol at fixed lag
            inner_max_dth = 1.0
            inner_n_flips = 0
            inner_iter = 0
            for inner in range(pt_max_inner):
                inner_max_dth, inner_n_flips = (
                    _elrod_theta_vk_sweep_pt_lagged(
                        Theta, g_state_lag, g_soft_lag,
                        g_state_target, c_prev, H,
                        A, B, C, D, E,
                        d_phi, beta_bar, omega,
                        eta, pt_beta_val,
                        N_Z, N_phi, groove,
                        state_eps, theta_min,
                    )
                )
                inner_iter += 1
                total_inner_sweeps += 1
                if inner_max_dth < pt_inner_tol:
                    break

            # Switch update (after inner block)
            g_state_new = g_state_target.copy()
            g_soft_new = g_soft_lag + gfactor * (g_state_new - g_soft_lag)
            g_soft_new = np.clip(g_soft_new, 0.0, 1.0)

            # Outer residuals
            max_dg_outer = float(np.max(np.abs(g_soft_new - g_soft_lag)))
            max_dth_outer = float(np.max(np.abs(Theta - Theta_prev)))
            n_flips_outer = int(
                np.sum(np.abs(g_state_new - g_state_lag) > 0.5)
            )

            g_state[:] = g_state_new
            g_soft[:] = g_soft_new

            # BC on committed state
            if groove:
                Theta[:, 0] = 1.0
                Theta[:, -1] = 1.0
                g_state[:, 0] = 1.0
                g_state[:, -1] = 1.0
                g_soft[:, 0] = 1.0
                g_soft[:, -1] = 1.0
            else:
                Theta[:, 0] = Theta[:, -2]
                Theta[:, -1] = Theta[:, 1]
                g_state[:, 0] = g_state[:, -2]
                g_state[:, -1] = g_state[:, 1]
                g_soft[:, 0] = g_soft[:, -2]
                g_soft[:, -1] = g_soft[:, 1]
            Theta[0, :] = 1.0
            Theta[-1, :] = 1.0
            g_state[0, :] = 1.0
            g_state[-1, :] = 1.0
            g_soft[0, :] = 1.0
            g_soft[-1, :] = 1.0

            total_pt_steps += 1

            # Quiescent check: strict (ΔΘ < tol AND Δg < tol AND no flips)
            strict_ok = (
                max_dth_outer < tol_theta
                and max_dg_outer < tol_g
                and n_flips_outer == 0
            )

            # Integral-stability fallback: when the interface oscillates
            # on a small boundary set, strict quiescent is unreachable,
            # but integral metrics lock in. Only consider this route if:
            #   - we are on the final η stage (past any continuation)
            #   - at least pt_min_warmup outer steps have passed
            #   - the interface is actually still oscillating
            #     (n_flips_outer > 0). If no flips, strict_ok will catch
            #     convergence naturally and we should NOT shortcut.
            #   - last 3 P_max / cav values vary by < 5·10⁻⁵ / 1·10⁻⁴
            #     (relative/absolute)
            #   - flips count is small (< 0.5 % of interior cells)
            soft_ok = False
            pt_min_warmup = 30
            is_last_eta = abs(eta - eta_schedule[-1]) < 1e-12
            # Track a larger history window (last 20 checkpoints) so an
            # interface-oscillating state with stable integrals is
            # caught even though max_dth_outer plateaus above tol_theta.
            if (
                is_last_eta
                and pt_step >= pt_min_warmup
                and n_flips_outer > 0
                and len(hist_Pmax) >= 20
            ):
                Pmax_span = max(hist_Pmax[-20:]) - min(hist_Pmax[-20:])
                cav_span = max(hist_cav[-20:]) - min(hist_cav[-20:])
                rel_Pmax = Pmax_span / max(hist_Pmax[-1], 1e-30)
                max_flip_frac = 0.005 * max((N_Z - 2) * (N_phi - 2), 1)
                soft_ok = (
                    rel_Pmax < 5e-5
                    and cav_span < 1e-4
                    and n_flips_outer < max_flip_frac
                )

            converged_now = strict_ok or soft_ok
            if converged_now:
                quiescent_streak += 1
            else:
                quiescent_streak = 0

            # Integral diagnostics
            g_hard_now = g_state
            Theta_for_p_now = np.where(
                g_hard_now > 0.5, np.maximum(Theta, 1.0), 1.0,
            )
            P_now = beta_bar * g_hard_now * np.log(Theta_for_p_now)
            P_now = np.where(P_now >= 0.0, P_now, 0.0)
            Pmax_now = float(P_now.max())
            cav_now = float(np.mean(g_state[1:-1, 1:-1] < 0.5))
            hist_Pmax.append(Pmax_now)
            hist_cav.append(cav_now)
            if len(hist_Pmax) > 30:
                hist_Pmax.pop(0)
                hist_cav.pop(0)

            if verbose and (
                pt_step % max(1, check_every // pt_max_inner) == 0
                or quiescent_streak >= n_quiescent
            ):
                fluid_area = float(np.mean(g_state[1:-1, 1:-1] > 0.5))
                print(
                    f"    η={eta:.2f} pt_step={pt_step:4d} "
                    f"inner_conv_at={inner_iter:3d} "
                    f"ΔΘ_out={max_dth_outer:.2e} Δg_out={max_dg_outer:.2e} "
                    f"flips={n_flips_outer:4d} "
                    f"Θ=[{Theta.min():.3f},{Theta.max():.3f}] "
                    f"cav={cav_now:.3f} fluid={fluid_area:.3f} "
                    f"Pmax={Pmax_now:.3e} "
                    f"quies={quiescent_streak}"
                )

            # Global iteration cap (total inner sweeps)
            if total_inner_sweeps >= max_iter:
                break

            if quiescent_streak >= n_quiescent:
                if verbose:
                    print(
                        f"  [Elrod-Θ-VK] η={eta:.2f} CONVERGED at "
                        f"pt_step={pt_step}, inner sweeps={total_inner_sweeps}"
                    )
                break
        # end pt_step loop

        if total_inner_sweeps >= max_iter:
            break
    # end eta loop

    # Pressure recovery from HARD topology
    g_hard = g_state
    Theta_for_p = np.where(
        g_hard > 0.5, np.maximum(Theta, 1.0), 1.0,
    )
    P = beta_bar * g_hard * np.log(Theta_for_p)
    P = np.where(P >= 0.0, P, 0.0)

    theta_out = np.clip(Theta, theta_min, None)
    # Report outer ΔΘ as the residual (convergence-meaningful)
    return P, theta_out, max_dth_outer, total_inner_sweeps
