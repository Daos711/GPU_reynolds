"""
Stage I solver-side tests (Contract B).

Covers:
  S1. masked payvar_salant + piezoviscosity path:
      (a) mask=None, alpha_pv=None → old regression;
      (b) mask!=None, alpha_pv=None → old regression;
      (c) mask!=None, alpha_pv!=None → new working path;
      (d) invalid mask/g_bc → loud failure.

  S2. thermal helper layer:
      - Walther fit roundtrip;
      - global_relax τ→0 matches static target;
      - repeated relax converges to static target.

Run:
    python -m reynolds_solver.tests.test_stage_i_solver
"""
import sys
import math

import numpy as np


def _ok(name, passed, detail=""):
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {name}")
    if detail:
        print(f"         {detail}")
    return passed


def _gap(N_phi=50, N_Z=30, epsilon=0.4):
    phi = np.linspace(0.0, 2.0 * np.pi, N_phi)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, _ = np.meshgrid(phi, Z)
    H = 1.0 + epsilon * np.cos(Phi)
    return H, float(phi[1] - phi[0]), float(Z[1] - Z[0])


R, L = 0.035, 0.056


# ===================================================================
# S1 tests
# ===================================================================
def test_regression_none_mask_none_pv():
    """(a) mask=None, alpha_pv=None via solve_reynolds → same as before."""
    print("\n=== S1a: regression mask=None alpha_pv=None ===")
    from reynolds_solver import solve_reynolds
    H, dp, dz = _gap()
    P, th, res, n = solve_reynolds(
        H, dp, dz, R, L,
        cavitation="payvar_salant", tol=1e-7, max_iter=5000,
    )
    return _ok(
        "mask=None pv=None works",
        float(P.max()) > 0 and n > 0,
        f"maxP={P.max():.3e}, n={n}",
    )


def test_regression_mask_no_pv():
    """(b) mask!=None, alpha_pv=None via solve_reynolds."""
    print("\n=== S1b: regression mask!=None alpha_pv=None ===")
    from reynolds_solver import solve_reynolds
    H, dp, dz = _gap()
    N_Z, N_phi = H.shape
    mask = np.zeros((N_Z, N_phi), dtype=bool)
    mask[N_Z // 2, N_phi // 4] = True
    g_bc = 1e-3
    P, th, res, n = solve_reynolds(
        H, dp, dz, R, L,
        cavitation="payvar_salant", tol=1e-7, max_iter=5000,
        dirichlet_mask=mask, g_bc=g_bc,
    )
    ok_pin = abs(P[N_Z // 2, N_phi // 4] - g_bc) < 1e-12
    return _ok(
        "mask+no-PV: pin holds, solve works",
        ok_pin and float(P.max()) > 0,
        f"P[pin]={P[N_Z // 2, N_phi // 4]:.3e}, maxP={P.max():.3e}",
    )


def test_masked_payvar_with_pv_smoke():
    """(c) mask!=None, alpha_pv!=None → new working path (smoke)."""
    print("\n=== S1c: masked payvar + PV smoke ===")
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        return _ok(f"[SKIP] cupy not available: {exc}", True)
    from reynolds_solver import solve_reynolds
    H, dp, dz = _gap(N_phi=60, N_Z=30, epsilon=0.5)
    N_Z, N_phi = H.shape
    mask = np.zeros((N_Z, N_phi), dtype=bool)
    mask[N_Z // 2, N_phi // 3] = True
    g_bc = 5e-3
    P, th, res, n = solve_reynolds(
        H, dp, dz, R, L,
        cavitation="payvar_salant",
        dirichlet_mask=mask, g_bc=g_bc,
        alpha_pv=2.2e-8, p_scale=1e7,
        tol=1e-6, max_iter=10000,
    )
    ok_pin = abs(P[N_Z // 2, N_phi // 3] - g_bc) < 1e-10
    ok_phys = float(P.min()) >= -1e-12 and float(th.min()) >= -1e-12
    ok_finite = bool(np.all(np.isfinite(P)) and np.all(np.isfinite(th)))
    return _ok(
        "PV+mask smoke: pin holds, P>=0, theta in [0,1]",
        ok_pin and ok_phys and ok_finite,
        f"P[pin]={P[N_Z // 2, N_phi // 3]:.3e}, "
        f"maxP={P.max():.3e}, P.min={P.min():.2e}, "
        f"theta=[{th.min():.4f}, {th.max():.4f}]",
    )


def test_invalid_mask_shape_raises():
    """(d) shape mismatch → ValueError."""
    print("\n=== S1d: invalid mask shape ===")
    from reynolds_solver import solve_reynolds
    H, dp, dz = _gap(N_phi=30, N_Z=20)
    try:
        solve_reynolds(
            H, dp, dz, R, L,
            cavitation="payvar_salant", max_iter=10,
            dirichlet_mask=np.zeros((5, 5), dtype=bool), g_bc=0.0,
        )
        return _ok("shape mismatch should raise", False)
    except ValueError:
        return _ok("shape mismatch raises ValueError", True)


def test_invalid_gbc_raises():
    """(d) g_bc=None with mask → ValueError."""
    print("\n=== S1d: invalid g_bc (None) ===")
    from reynolds_solver import solve_reynolds
    H, dp, dz = _gap(N_phi=30, N_Z=20)
    try:
        solve_reynolds(
            H, dp, dz, R, L,
            cavitation="payvar_salant", max_iter=10,
            dirichlet_mask=np.zeros_like(H, dtype=bool), g_bc=None,
        )
        return _ok("g_bc=None with mask should raise", False)
    except ValueError:
        return _ok("g_bc=None with mask raises ValueError", True)


# ===================================================================
# S2 tests
# ===================================================================
def test_walther_fit_roundtrip():
    """fit_walther_two_point roundtrips through mu_at_T_C at both points."""
    print("\n=== S2: Walther fit roundtrip ===")
    from reynolds_solver.thermal import (
        fit_walther_two_point,
        mu_at_T_C,
    )
    from dataclasses import replace

    T1, nu1 = 40.0, 68.0    # SAE 20W-50 ish
    T2, nu2 = 100.0, 11.0
    model = fit_walther_two_point(T1, nu1, T2, nu2)
    model = replace(model, rho_kg_m3=860.0)

    mu1 = mu_at_T_C(T1, model)
    mu2 = mu_at_T_C(T2, model)

    nu1_rt = mu1 / model.rho_kg_m3 * 1e6
    nu2_rt = mu2 / model.rho_kg_m3 * 1e6

    err1 = abs(nu1_rt - nu1) / nu1
    err2 = abs(nu2_rt - nu2) / nu2
    return _ok(
        "Walther roundtrip within 1e-10",
        err1 < 1e-10 and err2 < 1e-10,
        f"err1={err1:.2e}, err2={err2:.2e}, A={model.A_w:.4f}, B={model.B_w:.4f}",
    )


def test_global_relax_tau_to_zero_matches_static():
    """τ → 0 (dt >> τ): relax step equals static target exactly."""
    print("\n=== S2: relax τ→0 matches static ===")
    from reynolds_solver.thermal import (
        global_static_target_C,
        global_relax_step_C,
    )
    T_in = 80.0
    P_loss = 500.0
    mdot = 0.1
    cp = 2000.0
    gamma = 0.5
    T_target = global_static_target_C(T_in, P_loss, mdot, cp, gamma)

    # dt >> tau → alpha clamped to 1 → T_new = T_target
    T_new = global_relax_step_C(60.0, T_target, dt_s=100.0, tau_th_s=0.01)
    return _ok(
        "τ→0: relax matches static",
        abs(T_new - T_target) < 1e-12,
        f"T_target={T_target:.4f}, T_relax={T_new:.4f}",
    )


def test_repeated_relax_converges_to_static_target():
    """Repeated relax steps converge to the static target."""
    print("\n=== S2: repeated relax → static ===")
    from reynolds_solver.thermal import (
        global_static_target_C,
        global_relax_step_C,
    )
    T_in = 80.0
    P_loss = 800.0
    mdot = 0.08
    cp = 2000.0
    gamma = 0.5
    T_target = global_static_target_C(T_in, P_loss, mdot, cp, gamma)

    T = T_in
    dt = 0.05
    tau = 1.0
    for _ in range(200):
        T = global_relax_step_C(T, T_target, dt, tau)

    err = abs(T - T_target)
    return _ok(
        "200 steps converge within 1e-6",
        err < 1e-6,
        f"T_final={T:.6f}, T_target={T_target:.6f}, err={err:.2e}",
    )


# ===================================================================
# main
# ===================================================================
def main():
    ok = True
    # S1
    ok = test_regression_none_mask_none_pv() and ok
    ok = test_regression_mask_no_pv() and ok
    ok = test_masked_payvar_with_pv_smoke() and ok
    ok = test_invalid_mask_shape_raises() and ok
    ok = test_invalid_gbc_raises() and ok
    # S2
    ok = test_walther_fit_roundtrip() and ok
    ok = test_global_relax_tau_to_zero_matches_static() and ok
    ok = test_repeated_relax_converges_to_static_target() and ok

    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
