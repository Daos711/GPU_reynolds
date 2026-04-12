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
    cav_mask, pinned,
    d_phi, beta_bar, omega, gfactor,
    theta_min, N_Z, N_phi, groove,
):
    """
    One lexicographic SOR sweep of the compressible Elrod equation.

    pinned=0: nonlinear dispatch (try full-film, else cavitation).
    pinned=1: frozen active set from cav_mask (full-film cells always
        use the elliptic diagonal and clamp Θ ≥ 1; cavitation cells
        always use the pure upwind transport and clamp Θ ∈ [θ_min, 1)).
        Prevents the cavitation zone from creeping into the full-film
        lobe via the positive feedback P↓→cav↑→P↓.

    Returns max|ΔΘ| over interior nodes.
    """
    max_delta = 0.0
    six_dphi = 6.0 * d_phi

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
            E_l = E[i, j]

            diff = A_l * Tjp + B_l * Tjm + C_l * Tip + D_l * Tim

            if pinned != 0:
                if cav_mask[i, j] != 0:
                    # Cavitation (pinned)
                    if h_ij > 1e-30:
                        Theta_new = h_jm * Tjm / h_ij
                    else:
                        Theta_new = theta_min
                    if Theta_new >= 1.0:
                        Theta_new = 1.0 - 1e-12
                    g_target = 0.0
                else:
                    # Full-film (pinned)
                    num_full = beta_bar * diff + six_dphi * h_jm * Tjm
                    den_full = beta_bar * E_l + six_dphi * h_ij
                    Theta_new = num_full / (den_full + 1e-30)
                    if Theta_new < 1.0:
                        Theta_new = 1.0
                    g_target = 1.0
            else:
                # Nonlinear dispatch
                num_full = beta_bar * diff + six_dphi * h_jm * Tjm
                den_full = beta_bar * E_l + six_dphi * h_ij
                Theta_try_full = num_full / (den_full + 1e-30)

                if Theta_try_full >= 1.0:
                    Theta_new = Theta_try_full
                    g_target = 1.0
                else:
                    if h_ij > 1e-30:
                        Theta_new = h_jm * Tjm / h_ij
                    else:
                        Theta_new = theta_min
                    if Theta_new >= 1.0:
                        Theta_new = 1.0 - 1e-12
                    g_target = 0.0

            if Theta_new < theta_min:
                Theta_new = theta_min

            # SOR relaxation
            Theta_relax = Theta_old + omega * (Theta_new - Theta_old)
            if Theta_relax < theta_min:
                Theta_relax = theta_min
            if g_target == 0.0 and Theta_relax >= 1.0:
                Theta_relax = 1.0 - 1e-12
            if pinned != 0 and cav_mask[i, j] == 0 and Theta_relax < 1.0:
                Theta_relax = 1.0

            Theta[i, j] = Theta_relax

            # Fesanghary-Khonsari soft switch update
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

    return max_delta


# ----------------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------------
def solve_elrod_compressible(
    H, d_phi, d_Z, R, L,
    beta_bar,
    omega=1.0,
    tol=1e-6,
    max_iter=500_000,
    check_every=200,
    phi_bc="periodic",
    gfactor=1.0,
    theta_min=1e-8,
    Theta_init=None,
    hs_warmup_iter=50_000,
    hs_warmup_tol=1e-5,
    hs_warmup_omega=None,
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
    theta : (N_Z, N_phi) — fractional film content Θ (=1 in full film,
        <1 in cavitation). Clipped to [theta_min, ∞).
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

    # --- HS warmup to seed Θ ---
    # Without a good initial guess the Elrod SOR drops into the
    # trivial "Θ·h = const, P = 0" fixed point that also exists in
    # the discrete system. HS gives a full-film pressure lobe in the
    # converging region (P_hs > 0) and zero in the diverging region
    # (P_hs clamped to 0). Setting Θ_init = exp(P_hs / β̄) seeds the
    # compressed full-film lobe; the cavitated side starts at Θ = 1
    # and quickly drifts below 1 under the first few Elrod sweeps.
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

    g = np.ones((N_Z, N_phi), dtype=np.float64)  # start full-film

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

    # Initial cav mask (from HS warmup: cells with P_hs < eps)
    # Full-film cells have Θ_init > 1, cavitation cells have Θ_init = 1.
    # We classify cav cells as those where Θ_init ≈ 1 AND the local
    # Couette driver pushes Θ below 1 in the first sweep. Simpler:
    # use the HS pressure threshold directly.
    cav_mask = (Theta <= 1.0 + cav_threshold).astype(np.int32)
    cav_mask[0, :] = 0
    cav_mask[-1, :] = 0
    if groove:
        cav_mask[:, 0] = 0
        cav_mask[:, -1] = 0
    else:
        cav_mask[:, 0] = cav_mask[:, -2]
        cav_mask[:, -1] = cav_mask[:, 1]

    # Seed Θ = 1 in the cavitation set (so the first sweep starts clean)
    if pin_active_set:
        for i in range(N_Z):
            for j in range(N_phi):
                if cav_mask[i, j] != 0:
                    Theta[i, j] = 1.0 - 1e-12
                    g[i, j] = 0.0

    pinned_flag = 1 if pin_active_set else 0
    n_outer_loop = max_outer_active_set if pin_active_set else 1

    if verbose:
        cav0 = int(cav_mask[1:-1, 1:-1].sum())
        tot0 = (N_Z - 2) * (N_phi - 2)
        print(
            f"  [Elrod] N_Z={N_Z}, N_phi={N_phi}, β̄={beta_bar:.3e}, "
            f"ω={omega}, phi_bc={phi_bc}, pinned={pinned_flag}, "
            f"initial cav={cav0}/{tot0}"
        )

    residual = 1.0
    n_iter = 0
    for outer in range(n_outer_loop):
        inner_k = 0
        for k in range(max_iter):
            residual = _elrod_sor_sweep(
                Theta, g, H, A, B, C, D, E,
                cav_mask, pinned_flag,
                d_phi, beta_bar, omega, gfactor,
                theta_min, N_Z, N_phi, groove,
            )
            n_iter += 1
            inner_k = k + 1
            if residual < tol and k > 5:
                break

            if verbose and k % check_every == 0 and k > 0:
                cav_frac = float(np.mean(Theta[1:-1, 1:-1] < 1.0 - 1e-6))
                print(
                    f"    outer={outer} inner={k}: maxΔΘ={residual:.3e}, "
                    f"Θ=[{Theta.min():.4f}, {Theta.max():.4f}], "
                    f"cav={cav_frac:.3f}"
                )

        if not pin_active_set:
            break

        # Active-set update
        new_cav = (Theta < 1.0 - 1e-8).astype(np.int32)
        new_cav[0, :] = 0
        new_cav[-1, :] = 0
        if groove:
            new_cav[:, 0] = 0
            new_cav[:, -1] = 0
        else:
            new_cav[:, 0] = new_cav[:, -2]
            new_cav[:, -1] = new_cav[:, 1]

        flips = int(np.sum(new_cav[1:-1, 1:-1] != cav_mask[1:-1, 1:-1]))
        if verbose:
            print(f"  [Elrod] outer={outer}: inner_k={inner_k}, flips={flips}")
        if flips == 0:
            break
        cav_mask = new_cav

    # Recover pressure. Use hard g (round to 0/1) to avoid soft-switch
    # smearing in the final P field.
    g_hard = (Theta >= 1.0 - 1e-12).astype(np.float64)
    Theta_safe = np.clip(Theta, theta_min, None)
    P = beta_bar * g_hard * np.log(Theta_safe)
    # P must be non-negative (full-film: Θ≥1 → ln Θ ≥ 0; cav: g=0 → P=0)
    P = np.where(P >= 0.0, P, 0.0)

    theta_out = np.clip(Theta, theta_min, 1.0)

    return P, theta_out, residual, n_iter
