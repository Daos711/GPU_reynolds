"""
benchmark_squeeze.py

Ausas et al. 2009, Section 3 — Oscillatory squeeze flow between parallel
plates. 1D (no Couette, α=0) test case with an exact analytical
rupture-phase solution. Verifies the FULL Ausas formulas (eq. 12, 17, 18)
WITH the temporal term ∂(θh)/∂τ — independent of the stationary-reduction
problem seen in the journal-bearing solver.

Geometry
    Ω = [0, 1], gap H(t) = 0.125·cos(4πt) + 0.375   (uniform in x)
    α = 0  (no Couette / shaft-rotation term)

Boundary / initial
    p(0, t) = p(1, t) = p₀ = 0.025   (positive — not zero)
    θ(x, t=0) = 1                     (full film)

Exact solution (rupture phase, after t_rup ≈ 0.250079):
    Σ(t) = 1 − √(p₀·H³(t)/H'(t)),   H'(t) = −0.5π·sin(4πt)
    Left edge by symmetry: 1 − Σ(t).

Discretization (α = 0, h uniform in x, β = 2·Δx²/Δt, s = H(t_n)³).
The factor of 2 in β matches the paper's FV/convention — the effective
PDE is 2·∂(θh)/∂τ = ∂[h³·∂P/∂x]/∂x, consistent with the exact rupture
formula Σ(t) = 1 − √(p₀·H³/H').

    β·(c^n − c^{n-1}) = s·(P_{i+1} − 2P_i + P_{i-1})

Branch 1 (full-film, θ = 1, c^n = H_n):
    P_trial = [s·(P_{i+1} + P_{i-1}) − β·(H_n − c_prev_i)] / (2·s)

Branch 2 (cavitation, P given, solve for θ, c^n = θ·H_n):
    Θ_trial = [β·c_prev_i + s·(P_{i+1} − 2·P_cur + P_{i-1})] / (β·H_n)

Solver strategy (per time step)
    • If the previous step was fully full-film, try a direct Thomas
      tridiagonal solve of the pure 1D-Poisson Branch-1 subsystem.
      Symmetric and exact in O(N). If the resulting min(P) ≥ 0 the
      active set is unchanged and we accept the solution immediately.
    • Otherwise fall back to symmetric Gauss-Seidel (forward + backward
      sweep) with Ausas Branch-1 / Branch-2 complementarity logic.
      SGS is symmetric by construction, which keeps the numerical
      solution symmetric around x = 0.5.

Run:
    python -m reynolds_solver.benchmark_squeeze
"""
import numpy as np
from numba import njit


# ----------------------------------------------------------------------------
# Exact solution
# ----------------------------------------------------------------------------
def H_gap(t):
    """Gap H(t) = 0.125·cos(4πt) + 0.375   (eq. 19)"""
    return 0.125 * np.cos(4.0 * np.pi * t) + 0.375


def H_dot(t):
    """dH/dt = −0.5π·sin(4πt)"""
    return -0.5 * np.pi * np.sin(4.0 * np.pi * t)


def sigma_exact(t, p0):
    """
    Right edge of the cavitation zone Σ(t), eq. (21).
    Valid only on the rupture phase (H' > 0). Returns NaN otherwise.
    """
    Hv = H_gap(t)
    Hd = H_dot(t)
    if Hd <= 0.0:
        return np.nan
    arg = p0 * Hv * Hv * Hv / Hd
    if arg < 0.0:
        return np.nan
    val = 1.0 - np.sqrt(arg)
    if val < 0.5 or val > 1.0:
        return np.nan
    return val


# ----------------------------------------------------------------------------
# Inner solvers (Numba)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _thomas_full_film(P, c_prev, H_n, s, beta, p0, N1):
    """
    Direct tridiagonal (Thomas) solve of the full-film Branch-1 subsystem:

        s·(P_{i+1} - 2P_i + P_{i-1}) = β·(H_n - c_prev_i),   i = 1..N-2
        P[0] = P[N-1] = p0.

    This is a 1D Poisson with Dirichlet BC and a known RHS, which Ausas'
    complementarity cannot split (no cavitation anywhere). Thomas gives
    the exact (up to roundoff) symmetric solution in O(N), eliminating
    the slow GS relaxation for the pre-rupture phase of the squeeze
    benchmark. Returns the resulting min(P) so the caller can detect an
    implicit cavitation event (min P < 0 ⇒ active set changed, iterate).

    P is overwritten on interior nodes; boundaries are re-set to p0.
    """
    n_int = N1 - 2  # interior nodes i = 1..N-2
    if n_int <= 0:
        return P[0]

    # Tridiagonal coefficients. a_i·P_{i-1} + b_i·P_i + c_i·P_{i+1} = d_i
    # Here a = s, b = -2s, c = s, d = β·(H_n - c_prev_i).
    # Boundary absorption: d_1' = d_1 - s·p0,  d_{N-2}' = d_{N-2} - s·p0.

    # Allocate local work arrays (stack-allocated by Numba)
    cprime = np.empty(n_int, dtype=np.float64)
    dprime = np.empty(n_int, dtype=np.float64)

    a_val = s
    b_val = -2.0 * s
    c_val = s

    # First row (interior index 0 ≡ global i=1): absorb P[0]=p0
    d0 = beta * (H_n - c_prev[1]) - a_val * p0
    cprime[0] = c_val / b_val
    dprime[0] = d0 / b_val

    # Forward sweep
    for k in range(1, n_int):
        i = k + 1  # global index
        if k == n_int - 1:
            # Last row: absorb P[N-1]=p0 into RHS
            dk = beta * (H_n - c_prev[i]) - c_val * p0
        else:
            dk = beta * (H_n - c_prev[i])
        denom = b_val - a_val * cprime[k - 1]
        cprime[k] = c_val / denom
        dprime[k] = (dk - a_val * dprime[k - 1]) / denom

    # Back substitution
    P[N1 - 2] = dprime[n_int - 1]
    for k in range(n_int - 2, -1, -1):
        i = k + 1
        P[i] = dprime[k] - cprime[k] * P[i + 1]

    P[0] = p0
    P[N1 - 1] = p0

    min_P = P[0]
    for i in range(N1):
        if P[i] < min_P:
            min_P = P[i]
    return min_P


@njit(cache=True)
def _squeeze_sgs_sweep(P, theta, c_prev, H_n, s, beta,
                      omega_p, omega_theta, N1):
    """
    One SYMMETRIC Gauss-Seidel sweep (forward + backward) of Ausas
    eq. (17)-(18) with α = 0 on a 1D uniform-h domain. Branch 1 →
    Branch 2 per node (Table 1). Symmetric by construction → preserves
    the symmetry of the problem around x = 0.5.

    Returns (max|ΔP|, max|Δθ|) over interior nodes of this sweep pair.
    """
    max_dP = 0.0
    max_dth = 0.0
    two_s = 2.0 * s
    beta_H = beta * H_n

    # ---- Forward sweep ----
    for i in range(1, N1 - 1):
        P_old = P[i]
        th_old = theta[i]
        cp_i = c_prev[i]
        P_cur = P_old
        th_cur = th_old

        if P_old > 0.0 or th_old >= 1.0 - 1e-12:
            F_time = beta * (H_n - cp_i)
            stencil_sum = s * (P[i + 1] + P[i - 1])
            P_trial = (stencil_sum - F_time) / two_s
            P_relax = omega_p * P_trial + (1.0 - omega_p) * P_old
            if P_relax >= 0.0:
                P_cur = P_relax
                th_cur = 1.0
            else:
                P_cur = 0.0

        if P_cur <= 0.0 or th_cur < 1.0 - 1e-12:
            stencil_signed = s * (P[i + 1] - 2.0 * P_cur + P[i - 1])
            Theta_trial = (beta * cp_i + stencil_signed) / beta_H
            th_relax = omega_theta * Theta_trial + (1.0 - omega_theta) * th_old
            if th_relax < 1.0:
                if th_relax < 0.0:
                    th_relax = 0.0
                th_cur = th_relax
                P_cur = 0.0
            else:
                th_cur = 1.0

        dP = P_cur - P_old
        if dP < 0.0:
            dP = -dP
        dth = th_cur - th_old
        if dth < 0.0:
            dth = -dth
        if dP > max_dP:
            max_dP = dP
        if dth > max_dth:
            max_dth = dth

        P[i] = P_cur
        theta[i] = th_cur

    # ---- Backward sweep ----
    for i in range(N1 - 2, 0, -1):
        P_old = P[i]
        th_old = theta[i]
        cp_i = c_prev[i]
        P_cur = P_old
        th_cur = th_old

        if P_old > 0.0 or th_old >= 1.0 - 1e-12:
            F_time = beta * (H_n - cp_i)
            stencil_sum = s * (P[i + 1] + P[i - 1])
            P_trial = (stencil_sum - F_time) / two_s
            P_relax = omega_p * P_trial + (1.0 - omega_p) * P_old
            if P_relax >= 0.0:
                P_cur = P_relax
                th_cur = 1.0
            else:
                P_cur = 0.0

        if P_cur <= 0.0 or th_cur < 1.0 - 1e-12:
            stencil_signed = s * (P[i + 1] - 2.0 * P_cur + P[i - 1])
            Theta_trial = (beta * cp_i + stencil_signed) / beta_H
            th_relax = omega_theta * Theta_trial + (1.0 - omega_theta) * th_old
            if th_relax < 1.0:
                if th_relax < 0.0:
                    th_relax = 0.0
                th_cur = th_relax
                P_cur = 0.0
            else:
                th_cur = 1.0

        dP = P_cur - P_old
        if dP < 0.0:
            dP = -dP
        dth = th_cur - th_old
        if dth < 0.0:
            dth = -dth
        if dP > max_dP:
            max_dP = dP
        if dth > max_dth:
            max_dth = dth

        P[i] = P_cur
        theta[i] = th_cur

    return max_dP, max_dth


# ----------------------------------------------------------------------------
# Time marching
# ----------------------------------------------------------------------------
def run_benchmark(
    N1=450,
    dt=6.6e-4,
    T_end=0.5,
    p0=0.025,
    omega_p=1.0,
    omega_theta=1.0,
    max_relax=4000,
    inner_tol=1e-11,
    log_every=50,
    verbose=True,
):
    """
    March one full period of the squeeze benchmark and return histories.

    Parameters
    ----------
    N1 : int — number of nodes along x₁ (default 450, per paper).
    dt : float — time step (default 6.6e-4, per paper).
    T_end : float — final time (default 0.5 = one period of H).
    p0 : float — Dirichlet pressure at x=0, x=1 (default 0.025).
    omega_p, omega_theta : float — relaxation factors (paper uses 1.0).
    max_relax, inner_tol : inner GS convergence controls.
    log_every : int — print a line every `log_every` time steps.
    verbose : bool.
    """
    dx = 1.0 / (N1 - 1)
    x = np.linspace(0.0, 1.0, N1)
    beta = 2.0 * dx * dx / dt

    # Initial state: full film, P = p₀, θ = 1
    P = np.full(N1, p0, dtype=np.float64)
    theta = np.ones(N1, dtype=np.float64)

    # Mass content from previous time step. At t=0, θ=1 and H=H(0)=0.5.
    H_prev = H_gap(0.0)
    c_prev = theta * H_prev

    NT = int(np.round(T_end / dt))

    if verbose:
        print(f"# Squeeze benchmark (Ausas 2009, Section 3)")
        print(f"# N1={N1}, dx={dx:.4e}, dt={dt:.4e}, NT={NT}")
        print(f"# β = 2·dx²/dt = {beta:.4e}")
        print(f"# p₀={p0}, ω_p={omega_p}, ω_θ={omega_theta}")
        print(f"# max_relax={max_relax}, inner_tol={inner_tol:.1e}")
        print()
        print(f"  {'step':>5s}  {'t':>7s}  {'H':>7s}  "
              f"{'solver':>8s}  {'maxP':>10s}  {'minθ':>6s}  "
              f"{'cav':>4s}  {'Σ_num':>7s}  {'Σ_ex':>7s}  {'sym':>9s}")

    times = np.zeros(NT)
    Sigma_num = np.full(NT, np.nan)
    Sigma_ex = np.full(NT, np.nan)
    maxP_hist = np.zeros(NT)
    minP_hist = np.zeros(NT)
    min_theta_hist = np.zeros(NT)
    cav_nodes_hist = np.zeros(NT, dtype=np.int64)
    inner_iters_hist = np.zeros(NT, dtype=np.int64)
    sym_err_hist = np.zeros(NT)
    mass_hist = np.zeros(NT)

    prev_cav = 0
    for n_step in range(1, NT + 1):
        t = n_step * dt
        H_n = H_gap(t)
        s = H_n ** 3

        # Re-enforce Dirichlet BC (θ=1 at boundaries by complementarity)
        P[0] = p0
        P[-1] = p0
        theta[0] = 1.0
        theta[-1] = 1.0

        # ----- Solve this time step --------------------------------------
        # If the previous step was fully full-film (prev_cav == 0), try a
        # direct Thomas tridiagonal solve for Branch 1 (1D Poisson with
        # Dirichlet BC). If the resulting min(P) >= 0, the active set is
        # still "all full-film" and Thomas is the exact solution for this
        # step — O(N), symmetric, no inner iteration.
        #
        # If min(P) < 0 after Thomas, or the previous step already had
        # cavitation, fall into the iterative symmetric Gauss-Seidel (SGS)
        # relaxation with complementarity Branch-1 / Branch-2 logic.
        inner_k = 0
        used_thomas = False
        if prev_cav == 0:
            # theta is all 1.0 here; Thomas acts purely on P.
            min_P = _thomas_full_film(P, c_prev, H_n, s, beta, p0, N1)
            if min_P >= 0.0:
                # Active set unchanged — accept the direct solve as-is.
                used_thomas = True
                # Ensure theta = 1 everywhere (it already is in pre-rupture).
                theta[:] = 1.0

        if not used_thomas:
            # Iterative SGS relaxation with complementarity.
            for k in range(max_relax):
                dP, dth = _squeeze_sgs_sweep(
                    P, theta, c_prev, H_n, s, beta,
                    omega_p, omega_theta, N1,
                )
                inner_k = k + 1
                if (dP if dP > dth else dth) < inner_tol:
                    break

        # Advance mass content to the new time step
        c_prev = theta * H_n

        # Diagnostics
        idx = n_step - 1
        times[idx] = t

        cav_mask = theta < 1.0 - 1e-8
        n_cav = int(np.sum(cav_mask))
        if n_cav > 0:
            cav_idxs = np.where(cav_mask)[0]
            Sigma_num[idx] = x[cav_idxs[-1]]
        Sigma_ex[idx] = sigma_exact(t, p0)

        maxP_hist[idx] = float(P.max())
        minP_hist[idx] = float(P.min())
        min_theta_hist[idx] = float(theta.min())
        cav_nodes_hist[idx] = n_cav
        inner_iters_hist[idx] = inner_k
        sym_P = float(np.max(np.abs(P - P[::-1])))
        sym_th = float(np.max(np.abs(theta - theta[::-1])))
        sym_err_hist[idx] = max(sym_P, sym_th)
        mass_hist[idx] = float(np.sum(theta * H_n) * dx)

        nucleated = (prev_cav == 0) and (n_cav > 0)
        reformed = (prev_cav > 0) and (n_cav == 0)
        if verbose and (n_step % log_every == 0 or nucleated or reformed):
            sig_n = Sigma_num[idx]
            sig_e = Sigma_ex[idx]
            sig_n_s = f"{sig_n:.4f}" if not np.isnan(sig_n) else "    -  "
            sig_e_s = f"{sig_e:.4f}" if not np.isnan(sig_e) else "    -  "
            tag = ""
            if nucleated:
                tag = "  <<< RUPTURE"
            if reformed:
                tag = "  <<< REFORMED"
            solver_tag = "Thomas" if used_thomas else f"SGS{inner_k:>4d}"
            print(f"  {n_step:>5d}  {t:>7.4f}  {H_n:>7.4f}  "
                  f"{solver_tag:>8s}  {P.max():>10.3e}  {theta.min():>6.4f}  "
                  f"{n_cav:>4d}  {sig_n_s:>7s}  {sig_e_s:>7s}  "
                  f"{sym_err_hist[idx]:>9.1e}{tag}")
        prev_cav = n_cav

    return dict(
        x=x, N1=N1, dt=dt, p0=p0, beta=beta,
        times=times,
        Sigma_num=Sigma_num,
        Sigma_exact=Sigma_ex,
        maxP=maxP_hist, minP=minP_hist,
        min_theta=min_theta_hist,
        cav_nodes=cav_nodes_hist,
        inner_iters=inner_iters_hist,
        sym_err=sym_err_hist,
        mass=mass_hist,
        P=P, theta=theta,
    )


# ----------------------------------------------------------------------------
# Success evaluation
# ----------------------------------------------------------------------------
def evaluate_success(result):
    """Check TZ success criteria and print a verdict block."""
    x = result["x"]
    t = result["times"]
    Sigma_num = result["Sigma_num"]
    Sigma_ex = result["Sigma_exact"]
    maxP = result["maxP"]
    minP = result["minP"]
    min_theta = result["min_theta"]
    sym_err = result["sym_err"]
    cav_nodes = result["cav_nodes"]
    N1 = result["N1"]
    dx = 1.0 / (N1 - 1)

    print()
    print("=" * 72)
    print("  Acceptance check (from the TZ)")
    print("=" * 72)

    # (1) Rupture nucleation at t ≈ 0.25
    first_cav = np.where(cav_nodes > 0)[0]
    if first_cav.size > 0:
        t_rup_num = float(t[first_cav[0]])
        rupture_ok = abs(t_rup_num - 0.25) < 0.01
        print(f"  [1] Rupture nucleated at t_rup_num = {t_rup_num:.5f} "
              f"(theory ≈ 0.25008)  →  {'OK' if rupture_ok else 'FAIL'}")
    else:
        print(f"  [1] NO cavitation detected at any time step  →  FAIL")
        rupture_ok = False

    # (2) Σ tracking on rupture phase (t ∈ (t_rup, ~0.315))
    mask = (
        (t > 0.251)
        & (t < 0.314)
        & ~np.isnan(Sigma_num)
        & ~np.isnan(Sigma_ex)
    )
    if mask.any():
        err = np.abs(Sigma_num[mask] - Sigma_ex[mask])
        max_err = float(err.max())
        mean_err = float(err.mean())
        # Staircase resolution is one cell; allow up to 3·dx for rounding.
        sigma_ok = max_err < 3.0 * dx
        print(f"  [2] Σ tracking on (0.251, 0.314): max err = {max_err:.3e}, "
              f"mean = {mean_err:.3e} (staircase ~ {dx:.3e})  →  "
              f"{'OK' if sigma_ok else 'FAIL'}")
    else:
        print(f"  [2] No samples in rupture window → SKIPPED")
        sigma_ok = False

    # (3) Reformation: cavitation should start shrinking after t ≈ 0.315
    #     Compare cav_nodes.max() in [0.25, 0.35] vs in [0.35, 0.45].
    peak_mask = (t > 0.25) & (t < 0.35)
    late_mask = (t > 0.35) & (t < 0.50)
    if peak_mask.any() and late_mask.any():
        peak = int(cav_nodes[peak_mask].max())
        late = int(cav_nodes[late_mask].max())
        reform_ok = late < peak or peak == 0
        print(f"  [3] Reformation: peak cav_nodes = {peak} → "
              f"late cav_nodes = {late}  →  "
              f"{'OK' if reform_ok else 'FAIL'}")
    else:
        reform_ok = False
        print(f"  [3] Reformation window empty → SKIPPED")

    # (4) Physical invariants P ≥ 0, 0 ≤ θ ≤ 1
    min_P_global = float(minP.min())
    min_theta_global = float(min_theta.min())
    inv_ok = (min_P_global >= -1e-12) and (min_theta_global >= -1e-12)
    print(f"  [4] Invariants: min(P) = {min_P_global:.3e}, "
          f"min(θ) = {min_theta_global:.4f}  →  "
          f"{'OK' if inv_ok else 'FAIL'}")

    # (5) Symmetry around x = 0.5
    max_sym = float(sym_err.max())
    sym_ok = max_sym < 1e-10
    print(f"  [5] max symmetry error: {max_sym:.3e}  →  "
          f"{'OK' if sym_ok else 'FAIL'}")

    # Mass conservation residual (for info, not a hard criterion)
    mass = result["mass"]
    mass_drift = float(mass.max() - mass.min())
    print(f"  [i] mass drift over run: {mass_drift:.3e} "
          f"(info only)")

    # Inner iterations statistics (only for time steps that actually
    # invoked the SGS relaxation; pre-rupture steps use Thomas → 0).
    it = result["inner_iters"]
    sgs_steps = it[it > 0]
    thomas_steps = int(np.sum(it == 0))
    if sgs_steps.size > 0:
        print(f"  [i] SGS iters: max = {int(sgs_steps.max())}, "
              f"mean = {float(sgs_steps.mean()):.1f}  "
              f"(on {sgs_steps.size} steps; Thomas used on {thomas_steps})")
    else:
        print(f"  [i] Thomas used on all {thomas_steps} steps (no SGS needed)")

    passed = rupture_ok and sigma_ok and inv_ok and sym_ok
    print()
    print("  " + ("OVERALL: PASS" if passed else "OVERALL: FAIL / PARTIAL"))
    print("=" * 72)
    return passed


# ----------------------------------------------------------------------------
# Optional plotting
# ----------------------------------------------------------------------------
def plot_sigma(result, out_path="squeeze_sigma.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [plot] matplotlib unavailable: {e}")
        return

    t = result["times"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(t, result["Sigma_exact"], "k-", lw=1.6, label="Σ exact (rupture)")
    ax.plot(t, result["Sigma_num"], "r-", lw=1.0,
            drawstyle="steps-post", label="Σ numerical")
    ax.axvline(0.25, color="gray", ls=":", lw=1.0, label="t_rup ≈ 0.25")
    ax.set_xlabel("time t")
    ax.set_ylabel("Σ(t)")
    ax.set_title("Ausas 2009 squeeze benchmark — cavitation edge")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot] saved to: {out_path}")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  Ausas 2009 squeeze benchmark (Section 3)")
    print("=" * 72)
    print()

    result = run_benchmark()

    evaluate_success(result)

    plot_sigma(result)


if __name__ == "__main__":
    main()
