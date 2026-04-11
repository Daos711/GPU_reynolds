"""
Payvar-Salant / Elrod unified-variable steady-state JFO cavitation solver.

CPU reference (Numba JIT).

Algorithm
---------
Solve for one variable g instead of the (P, θ) pair:

    g >= 0  ⇒  P = g,    θ = 1       (full-film)
    g <  0  ⇒  P = 0,    θ = 1 + g   (cavitation, θ ∈ [0, 1])
    g = -1  ⇒  full cavitation (θ = 0)

The reduced pressure is P = max(g, 0), so diffusion neighbours are read
as `max(g, 0)`. The discrete equation is Ausas eq. (13):

    A·P_{jp} + B·P_{jm} + C·P_{ip} + D·P_{im} − E·P_{ij} =
        d_phi·(h_{i,j}·θ_{i,j} − h_{i,jm}·θ_{i,jm})

With the substitutions above, both branches collapse to the SAME
numerator `num = diff − couette` where
    diff    = A·P_{jp} + B·P_{jm} + C·P_{ip} + D·P_{im}
    couette = d_phi·(h_{i,j} − h_{i,jm}·θ_{i,jm})
and only the diagonal differs:

    full-film (g_ij ≥ 0):     g_ij = num / E
    cavitation (g_ij <  0):   g_ij = num / (d_phi · h_{i,j})

Per node:  try the full-film diagonal first; if g_trial < 0, switch to
the cavitation diagonal and clamp to [-1, +∞).

Three fixes from the Ausas validation campaign are applied here:

  1. Face conductance = AVERAGE-OF-CUBES (not cube-of-average):
         A_{i,j} = 0.5·(h³_{i,j}   + h³_{i,j+1})     phi+
         B_{i,j} = 0.5·(h³_{i,j-1} + h³_{i,j}  )     phi−
         C / D    similar in Z with alpha_sq factor
     The old `solver_jfo_unified_cpu` used ((h_i + h_{i+1})/2)^3 which
     under-weights the thin-film side and destabilises the solver.

  2. Mass-content term uses CELL-CENTERED H (not face H):
         couette = d_phi·(h_{i,j} − h_{i,jm}·θ_{jm})
     The cavitation diagonal is likewise cell-centered:
         diag_cav = d_phi · h_{i,j}
     (The old solver used H_face_p / H_face_m here.)

  3. H is GHOST-PACKED before the coefficient build, and the Z ends are
     flooded (g = 0 ⇔ P = 0, θ = 1) Dirichlet boundaries.

For a supply pressure p_a > 0, set g = p_a on the Z boundaries instead
of 0 (not wired through the API here — tests only use flooded g = 0).
"""
import numpy as np
from numba import njit


# ----------------------------------------------------------------------------
# Coefficient build (same average-of-cubes as cavitation.ausas.solver_cpu)
# ----------------------------------------------------------------------------
def _build_coefficients(H, d_phi, d_Z, R, L):
    """
    Build A, B, C, D, E with average-of-cubes conductance.

        A_{i,j} = 0.5·(h³_{i,j}   + h³_{i,j+1}),           phi face (+)
        B_{i,j} = 0.5·(h³_{i,j-1} + h³_{i,j}  ),           phi face (-)
        C_{i,j} = alpha_sq · 0.5·(h³_{i,j}   + h³_{i+1,j}),  Z face (+)
        D_{i,j} = alpha_sq · 0.5·(h³_{i-1,j} + h³_{i,j}  ),  Z face (-)
        E = A + B + C + D
        alpha_sq = (2R/L · d_phi/d_Z)²

    Requires H to be ghost-packed (H[:, 0] = H[:, N_phi-2],
    H[:, N_phi-1] = H[:, 1]).
    """
    N_Z, N_phi = H.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    # phi-direction face conductance (average-of-cubes)
    Ah = 0.5 * (H[:, :-1] ** 3 + H[:, 1:] ** 3)     # (N_Z, N_phi - 1)

    A = np.zeros((N_Z, N_phi), dtype=np.float64)
    A[:, :-1] = Ah
    A[:, -1] = Ah[:, 0]

    B = np.zeros((N_Z, N_phi), dtype=np.float64)
    B[:, 1:] = Ah
    B[:, 0] = Ah[:, -1]

    # Z-direction face conductance
    H_jph3 = 0.5 * (H[:-1, :] ** 3 + H[1:, :] ** 3)  # (N_Z - 1, N_phi)
    C = np.zeros((N_Z, N_phi), dtype=np.float64)
    D = np.zeros((N_Z, N_phi), dtype=np.float64)
    C[1:-1, :] = alpha_sq * H_jph3[1:, :]
    D[1:-1, :] = alpha_sq * H_jph3[:-1, :]

    E = A + B + C + D
    return A, B, C, D, E


# ----------------------------------------------------------------------------
# Half-Sommerfeld warmup (same discrete operator, θ ≡ 1, clamp P ≥ 0)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _hs_sor_sweep(P, A, B, C, D, E, F_hs, omega, N_Z, N_phi):
    """
    One SOR sweep of the half-Sommerfeld problem with the same
    coefficients as the Payvar-Salant sweep:

        A·P_{jp} + B·P_{jm} + C·P_{ip} + D·P_{im} − E·P_{ij} = F_hs

    with F_hs = d_phi·(h_{i,j} − h_{i,jm}) (the θ ≡ 1 Couette driver)
    and P clamped to ≥ 0 inside the sweep.

    Used as a warm start for solve_payvar_salant_cpu. Without it, a
    g ≡ 0 cold start tends to fall into the trivial "θ·h = const,
    P ≈ 0" fixed point that also exists in the discrete system. With
    it, the cavitated region of the HS field seeds g < 0 and the PS
    iteration just polishes the rupture boundary.

    Returns max|ΔP| over interior nodes.
    """
    max_dP = 0.0
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jp = j + 1 if j + 1 < N_phi - 1 else 1
            jm = j - 1 if j - 1 >= 1 else N_phi - 2

            P_old = P[i, j]
            P_new = (
                A[i, j] * P[i, jp] + B[i, j] * P[i, jm]
                + C[i, j] * P[i + 1, j] + D[i, j] * P[i - 1, j]
                - F_hs[i, j]
            ) / (E[i, j] + 1e-30)
            if P_new < 0.0:
                P_new = 0.0
            P[i, j] = P_old + omega * (P_new - P_old)

            dP = P[i, j] - P_old
            if dP < 0.0:
                dP = -dP
            if dP > max_dP:
                max_dP = dP

    for i in range(N_Z):
        P[i, 0] = P[i, N_phi - 2]
        P[i, N_phi - 1] = P[i, 1]
    for j in range(N_phi):
        P[0, j] = 0.0
        P[N_Z - 1, j] = 0.0

    return max_dP


# ----------------------------------------------------------------------------
# Inner SOR sweep (nonlinear unified-variable Gauss-Seidel)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _ps_sor_sweep(
    g, H, A, B, C, D, E,
    cav_mask, pinned,
    d_phi, omega, N_Z, N_phi,
):
    """
    One lexicographic SOR sweep of the Payvar-Salant unified variable g.

    Two dispatch modes:

    * pinned = 0 (nonlinear, original Elrod): per node, try the
      full-film diagonal first; if g_trial_ff < 0 switch to the
      cavitation diagonal. Active set is free to drift.

    * pinned = 1 (frozen active set, Elrod-Adams style): `cav_mask`
      is fixed from outside (typically = [P_hs < ε] after an HS
      warmup). Full-film cells (cav_mask = 0) always use diagonal E
      and clamp g ≥ 0; cavitation cells (cav_mask = 1) always use
      diagonal d_phi·h_{i,j} and clamp g ∈ [-1, 0]. The iteration is
      then linear, unique, and does NOT drift toward the trivial
      "θ·h = const, P ≈ 0" fixed point.

    Per node (both modes):
        1. Read neighbours as P = max(g, 0).
        2. Read upstream θ_{jm} from the CURRENT g.
        3. numerator = (A·P_jp + B·P_jm + C·P_ip + D·P_im)
                     − d_phi·(h_{i,j} − h_{i,jm}·θ_{jm}).
        4. Apply diagonal dispatch (pinned vs nonlinear).
        5. SOR relaxation: g_relax = g_old + ω·(g_new − g_old).

    Returns max|Δg| over interior nodes.
    """
    max_delta = 0.0

    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jp = j + 1 if j + 1 < N_phi - 1 else 1
            jm = j - 1 if j - 1 >= 1 else N_phi - 2

            g_old = g[i, j]

            g_njp = g[i, jp]
            g_njm = g[i, jm]
            g_nip = g[i + 1, j]
            g_nim = g[i - 1, j]

            P_jp = g_njp if g_njp > 0.0 else 0.0
            P_jm = g_njm if g_njm > 0.0 else 0.0
            P_ip = g_nip if g_nip > 0.0 else 0.0
            P_im = g_nim if g_nim > 0.0 else 0.0

            if g_njm >= 0.0:
                theta_jm = 1.0
            else:
                theta_jm = 1.0 + g_njm
                if theta_jm < 0.0:
                    theta_jm = 0.0

            h_ij = H[i, j]
            h_jm = H[i, jm]

            diff = (
                A[i, j] * P_jp + B[i, j] * P_jm
                + C[i, j] * P_ip + D[i, j] * P_im
            )
            couette = d_phi * (h_ij - h_jm * theta_jm)
            numerator = diff - couette

            E_l = E[i, j]
            cav_diag = d_phi * h_ij

            if pinned != 0:
                # Frozen active set from cav_mask
                if cav_mask[i, j] != 0:
                    # Cavitation: Branch 2 only
                    g_new = numerator / (cav_diag + 1e-30)
                    if g_new > 0.0:
                        g_new = 0.0
                    elif g_new < -1.0:
                        g_new = -1.0
                else:
                    # Full-film: Branch 1 only
                    g_new = numerator / (E_l + 1e-30)
                    if g_new < 0.0:
                        g_new = 0.0
            else:
                # Original nonlinear Elrod dispatch
                g_trial_ff = numerator / (E_l + 1e-30)
                if g_trial_ff >= 0.0:
                    g_new = g_trial_ff
                else:
                    g_new = numerator / (cav_diag + 1e-30)
                    if g_new < -1.0:
                        g_new = -1.0

            g_relax = g_old + omega * (g_new - g_old)
            if g_relax < -1.0:
                g_relax = -1.0

            g[i, j] = g_relax

            delta = g_relax - g_old
            if delta < 0.0:
                delta = -delta
            if delta > max_delta:
                max_delta = delta

    for i in range(N_Z):
        g[i, 0] = g[i, N_phi - 2]
        g[i, N_phi - 1] = g[i, 1]

    for j in range(N_phi):
        g[0, j] = 0.0
        g[N_Z - 1, j] = 0.0

    return max_delta


# ----------------------------------------------------------------------------
# PDE residual (same discretization and switch as the sweep)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _ps_pde_residual(g, H, A, B, C, D, E, d_phi, N_Z, N_phi):
    """
    Maximum absolute residual of the discrete Ausas eq. (13) over
    interior nodes, measured with the SAME switch F(g) that the sweep
    uses:
        res_ij = (A·P_jp + B·P_jm + C·P_ip + D·P_im − E·P_ij)
                 − d_phi·(h_ij·θ_ij − h_jm·θ_jm)
    where P = max(g, 0) and θ is recovered from g.

    A genuine fixed point of the sweep has res_ij = 0 at every node;
    small `max|Δg|` with large `max|res|` means the iteration has
    stalled / is cycling and has NOT converged to a true solution.
    """
    max_res = 0.0

    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jp = j + 1 if j + 1 < N_phi - 1 else 1
            jm = j - 1 if j - 1 >= 1 else N_phi - 2

            g_ij = g[i, j]
            if g_ij >= 0.0:
                P_ij = g_ij
                theta_ij = 1.0
            else:
                P_ij = 0.0
                theta_ij = 1.0 + g_ij
                if theta_ij < 0.0:
                    theta_ij = 0.0

            g_njp = g[i, jp]
            g_njm = g[i, jm]
            g_nip = g[i + 1, j]
            g_nim = g[i - 1, j]

            P_jp = g_njp if g_njp > 0.0 else 0.0
            P_jm = g_njm if g_njm > 0.0 else 0.0
            P_ip = g_nip if g_nip > 0.0 else 0.0
            P_im = g_nim if g_nim > 0.0 else 0.0

            if g_njm >= 0.0:
                theta_jm = 1.0
            else:
                theta_jm = 1.0 + g_njm
                if theta_jm < 0.0:
                    theta_jm = 0.0

            lhs = (
                A[i, j] * P_jp + B[i, j] * P_jm
                + C[i, j] * P_ip + D[i, j] * P_im
                - E[i, j] * P_ij
            )
            rhs = d_phi * (H[i, j] * theta_ij - H[i, jm] * theta_jm)

            res = lhs - rhs
            if res < 0.0:
                res = -res
            if res > max_res:
                max_res = res

    return max_res


# ----------------------------------------------------------------------------
# Public driver
# ----------------------------------------------------------------------------
def solve_payvar_salant_cpu(
    H, d_phi, d_Z, R, L,
    omega=1.0,
    tol=1e-6,
    max_iter=50000,
    check_every=200,
    hs_warmup_iter=50000,
    hs_warmup_tol=1e-5,
    hs_warmup_omega=None,
    pin_active_set=True,
    max_outer_active_set=10,
    cav_threshold=1e-10,
    g_init=None,
    verbose=False,
):
    """
    Steady-state mass-conserving JFO cavitation for a journal bearing
    via the Payvar-Salant / Elrod unified-variable approach.

    Strategy
    --------
    The discrete Ausas / unified system has TWO steady fixed points for
    typical journal-bearing parameters with flooded Z boundaries:

      (a) the physical Reynolds solution — full-film lobe with P > 0
          in the converging region, cavitated ribbon with P = 0 and
          θ < 1 in the diverging region;
      (b) the trivial "constant mass flux" solution, θ·h ≈ const and
          P ≈ 0 everywhere — also a true fixed point of the
          discretisation, but not the physical one we want.

    A plain `g = 0` cold start with nonlinear SOR drifts toward (b),
    and even an HS warmup is not enough: once the sweep is free to
    flip cells between full-film and cavitation, the cavitation ribbon
    slowly eats into the full-film lobe on every iteration. This is
    the same instability that afflicts stationary Ausas.

    The standard Elrod-Adams fix is to FREEZE THE ACTIVE SET from the
    HS warmup: HS clamps P ≥ 0 at every sweep, so a point where HS
    reports P ≈ 0 IS on the cavitated side of the rupture boundary
    (to within one grid cell). We classify every cell as full-film or
    cavitation based on HS, then run a LINEAR inner sweep in which
    full-film cells always use the Poisson diagonal E and cavitation
    cells always use the Couette diagonal d_phi·h_{i,j}. That inner
    sweep has a unique fixed point (it is a linear system), does not
    drift, and converges cleanly.

    If a physical rupture boundary is found that differs from HS by
    more than one cell, the active set is updated and the inner sweep
    is re-run. This outer loop normally converges in 2-4 active-set
    iterations.

    For an ε-continuation sweep, pass `g_init` from the previous ε;
    the HS warmup is then skipped but the initial active set is
    re-derived from `g_init` (cav where g_init < 0).

    Parameters
    ----------
    H : (N_Z, N_phi) float64
        Dimensionless gap. Ghost-packed internally.
    d_phi, d_Z : float
        Grid spacing.
    R, L : float
        Bearing radius and length (m).
    omega : float
        SOR relaxation parameter for the inner sweep. With a frozen
        active set the inner problem is linear, and ω ∈ [1.0, 1.5]
        is safe. Default 1.0.
    tol : float
        Convergence tolerance on max|Δg| for the inner sweep.
    max_iter : int
        Maximum inner sweeps per outer active-set iteration.
    check_every : int
        Interval between diagnostic prints inside the inner loop.
    hs_warmup_iter : int
        Maximum HS warmup sweeps. Set to 0 to skip the warmup entirely
        (in which case you must supply `g_init`).
    hs_warmup_tol : float
        HS warmup convergence tolerance on max|ΔP|.
    hs_warmup_omega : float or None
        SOR ω for the linear HS warmup. If None (default), computed
        automatically via ``compute_auto_omega`` — same formula as the
        main HS solver, with cap=1.97. This is critical for large
        grids where ω=1.7 is too slow to converge.
    pin_active_set : bool
        If True (default), freeze the cavitation mask inside each
        inner sweep. If False, fall back to the original nonlinear
        Elrod dispatch (for experimentation only — it drifts).
    max_outer_active_set : int
        Maximum outer active-set iterations (ignored when
        pin_active_set is False or max_outer_active_set == 1).
    cav_threshold : float
        Pressure threshold for classifying a cell as cavitated after
        the HS warmup (cav_mask_ij = [P_hs_ij < cav_threshold]).
    g_init : (N_Z, N_phi) float64 or None
        Warm start for g (e.g. for ε-continuation). Bypasses HS warmup
        when provided. The initial active set is derived from
        g_init < 0.
    verbose : bool
        Print {update, pde, maxP, cav_frac} each outer iteration.

    Returns
    -------
    P : (N_Z, N_phi) float64 — max(g, 0)
    theta : (N_Z, N_phi) float64 — 1 if g ≥ 0 else 1 + g, clipped to [0, 1]
    residual : float — final max|Δg|
    n_iter : int — total sweeps (HS warmup + all inner PS sweeps)
    """
    N_Z, N_phi = H.shape

    # Defensive ghost packing
    H = np.ascontiguousarray(H, dtype=np.float64).copy()
    H[:, 0] = H[:, N_phi - 2]
    H[:, N_phi - 1] = H[:, 1]

    A, B, C, D, E = _build_coefficients(H, d_phi, d_Z, R, L)

    n_iter_total = 0

    # ------------------------------------------------------------------
    # Initial g: caller-supplied warm start, HS warmup, or zero
    # ------------------------------------------------------------------
    if g_init is not None:
        g = np.ascontiguousarray(g_init, dtype=np.float64).copy()
        g[:, 0] = g[:, N_phi - 2]
        g[:, N_phi - 1] = g[:, 1]
        g[0, :] = 0.0
        g[-1, :] = 0.0
        if verbose:
            print("  [PS] warm start from g_init (HS warmup skipped)")
    elif hs_warmup_iter > 0:
        # Auto-omega for HS warmup — same formula as the main HS solver
        if hs_warmup_omega is None:
            from reynolds_solver.utils import compute_auto_omega
            hs_warmup_omega = compute_auto_omega(N_phi, N_Z, R, L, cap=1.97)

        F_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                jm = j - 1 if j - 1 >= 1 else N_phi - 2
                F_hs[i, j] = d_phi * (H[i, j] - H[i, jm])

        P_hs = np.zeros((N_Z, N_phi), dtype=np.float64)
        hs_res = 1.0
        for k in range(hs_warmup_iter):
            hs_res = _hs_sor_sweep(
                P_hs, A, B, C, D, E, F_hs,
                hs_warmup_omega, N_Z, N_phi,
            )
            n_iter_total += 1
            if hs_res < hs_warmup_tol and k > 5:
                break
        if verbose:
            print(
                f"  [PS] HS warmup done: iter={n_iter_total}, "
                f"res={hs_res:.3e}, maxP={P_hs.max():.4e}, "
                f"ω_hs={hs_warmup_omega:.4f}"
            )
        if hs_res > hs_warmup_tol:
            import warnings
            warnings.warn(
                f"PS HS warmup did not converge: res={hs_res:.2e} > "
                f"tol={hs_warmup_tol:.2e} after {hs_warmup_iter} iters "
                f"(omega={hs_warmup_omega:.4f}). Solution may be "
                f"unreliable. Consider increasing hs_warmup_iter.",
                stacklevel=2,
            )

        g = P_hs
        g[:, 0] = g[:, N_phi - 2]
        g[:, N_phi - 1] = g[:, 1]
        g[0, :] = 0.0
        g[-1, :] = 0.0
    else:
        g = np.zeros((N_Z, N_phi), dtype=np.float64)

    # ------------------------------------------------------------------
    # Cavitation mask from the initial state (used only when pinned)
    # ------------------------------------------------------------------
    cav_mask = (g < cav_threshold).astype(np.int32)
    # Force Z boundaries to full-film (g = 0 there is a Dirichlet, not
    # a cavitated cell; classifying those as cav would immediately get
    # re-solved into negative g).
    cav_mask[0, :] = 0
    cav_mask[-1, :] = 0
    # Ghost columns track the physical seam
    cav_mask[:, 0] = cav_mask[:, N_phi - 2]
    cav_mask[:, N_phi - 1] = cav_mask[:, 1]

    # Seed g = 0 inside the initial cavitation set (start of each inner
    # sweep from a clean mass-content state, not from P_hs clamping noise).
    if pin_active_set:
        for i in range(N_Z):
            for j in range(N_phi):
                if cav_mask[i, j] != 0:
                    g[i, j] = 0.0

    pinned = 1 if pin_active_set else 0
    n_outer = max_outer_active_set if pin_active_set else 1

    if verbose:
        cav0 = int(cav_mask[1:-1, 1:-1].sum())
        tot0 = (N_Z - 2) * (N_phi - 2)
        print(
            f"  [PS] N_Z={N_Z}, N_phi={N_phi}, ω={omega}, "
            f"pinned={pinned}, initial cav fraction={cav0}/{tot0}"
        )
        print(
            f"  {'out':>3s}  {'in-iters':>9s}  {'update':>11s}  "
            f"{'pde':>11s}  {'maxP':>10s}  {'cav_frac':>9s}  {'flips':>6s}"
        )

    residual = 1.0
    for outer in range(n_outer):
        # ---- Inner SOR with current cav_mask (or nonlinear dispatch) ----
        inner_k = 0
        for k in range(max_iter):
            residual = _ps_sor_sweep(
                g, H, A, B, C, D, E,
                cav_mask, pinned,
                d_phi, omega, N_Z, N_phi,
            )
            n_iter_total += 1
            inner_k = k + 1
            if residual < tol and k > 5:
                break

        # ---- Update the active set and check for drift ----
        if pinned == 0:
            break  # nonlinear dispatch has no outer loop

        new_cav = np.where(
            g >= 0.0,
            np.zeros_like(cav_mask),
            np.ones_like(cav_mask),
        ).astype(np.int32)
        new_cav[0, :] = 0
        new_cav[-1, :] = 0
        new_cav[:, 0] = new_cav[:, N_phi - 2]
        new_cav[:, N_phi - 1] = new_cav[:, 1]

        flips = int(
            np.sum(new_cav[1:-1, 1:-1] != cav_mask[1:-1, 1:-1])
        )

        if verbose:
            pde_res = _ps_pde_residual(
                g, H, A, B, C, D, E, d_phi, N_Z, N_phi
            )
            maxP = float(np.maximum(g, 0.0).max())
            cav_frac = float(np.mean(g[1:-1, 1:-1] < 0.0))
            print(
                f"  {outer:>3d}  {inner_k:>9d}  {residual:>11.3e}  "
                f"{pde_res:>11.3e}  {maxP:>10.4e}  "
                f"{cav_frac:>9.3f}  {flips:>6d}"
            )

        if flips == 0:
            break  # active set stable — converged
        cav_mask = new_cav

    # ------------------------------------------------------------------
    # Recover P and θ from g
    # ------------------------------------------------------------------
    P = np.where(g >= 0.0, g, 0.0)
    theta = np.where(g >= 0.0, 1.0, 1.0 + g)
    np.clip(theta, 0.0, 1.0, out=theta)

    return P, theta, residual, n_iter_total
