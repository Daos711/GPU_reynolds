"""
Tests for piezoviscous (Roelands / Barus) wrapper around the dynamic
Ausas/JFO one-step GPU solver — `solve_ausas_unsteady_pv_one_step_gpu`.

Acceptance checks (ТЗ-2 §8):
    1. `pv_model='off'` → bit-for-bit identical to baseline.
    2. `roelands, alpha=0` and `barus, alpha=0` → bypass to baseline.
    3. PV affects ONLY pressure-flow coefficients
       (uniform mu_ratio = k scales A,B,C,D by 1/k; E recomputed).
    4. Roelands formula sanity (p=0 → 1; matches Barus at p << p0).
    5. Cavitation invariants: P >= 0, 0 <= theta <= 1, mu_ratio finite,
       mu_ratio == 1 in p=0 cells.
    6. Warm-start reduces outer iters.

Run:
    python -m reynolds_solver.tests.test_pv_ausas_dynamic
"""
import sys

import numpy as np


R = 0.035
L = 0.056
ETA = 0.01105
C_CLEAR = 50e-6
N_RPM = 2980
OMEGA_SHAFT = 2 * np.pi * N_RPM / 60.0
U_SHAFT = OMEGA_SHAFT * (R - C_CLEAR)
P_SCALE = (6.0 * ETA * U_SHAFT * R) / (C_CLEAR ** 2)


def _has_cupy():
    """True iff cupy imports AND a tiny CUDA allocation succeeds."""
    try:
        import cupy as cp
        cp.zeros(1, dtype=cp.float64)
        return True
    except Exception:
        return False


def _make_gap(N_Z, N_phi, eps):
    phi_1D = np.linspace(0.0, 2.0 * np.pi, N_phi)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + eps * np.cos(Phi)
    H[:, 0] = H[:, N_phi - 2]
    H[:, N_phi - 1] = H[:, 1]
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z


def _make_state(N_Z, N_phi):
    phi_1D = np.linspace(0.0, 2.0 * np.pi, N_phi)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi_1D, Z)
    P = np.maximum(np.sin(Phi) * (1.0 - Zm ** 2), 0.0)
    P[0, :] = 0.0
    P[-1, :] = 0.0
    P[:, 0] = P[:, N_phi - 2]
    P[:, N_phi - 1] = P[:, 1]
    theta = np.ones_like(P)
    cav = (P <= 0.0) & (Zm > -0.8) & (Zm < 0.8)
    theta[cav] = 0.5 + 0.3 * np.cos(Phi[cav])
    theta = np.clip(theta, 0.0, 1.0)
    theta[0, :] = 1.0
    theta[-1, :] = 1.0
    theta[:, 0] = theta[:, N_phi - 2]
    theta[:, N_phi - 1] = theta[:, 1]
    return P, theta


def _baseline_kwargs(N_Z, N_phi):
    H_prev, d_phi, d_Z = _make_gap(N_Z, N_phi, 0.55)
    H_curr, _, _ = _make_gap(N_Z, N_phi, 0.60)
    P_prev, theta_prev = _make_state(N_Z, N_phi)
    return dict(
        H_curr=H_curr, H_prev=H_prev,
        P_prev=P_prev, theta_prev=theta_prev,
        dt=0.05, d_phi=d_phi, d_Z=d_Z, R=R, L=L,
        alpha=1.0, omega_p=1.0, omega_theta=1.0,
        tol=1e-8, max_inner=400,
        p_bc_z0=0.0, p_bc_zL=0.0,
        theta_bc_z0=1.0, theta_bc_zL=1.0,
        periodic_phi=True, periodic_z=False,
        check_every=20, scheme="jacobi", verbose=False,
    )


def _report(name, ok, details=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {details}" if details else ""))
    return ok


# ---------------------------------------------------------------------------
# Test 1: pv_model='off' bypass is bit-for-bit identical to baseline.
# ---------------------------------------------------------------------------
def test_off_regression():
    print("\n=== Test 1: pv_model='off' baseline regression ===")
    if not _has_cupy():
        print("  [SKIP] cupy not available")
        return True
    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        ausas_unsteady_one_step_gpu,
    )
    from reynolds_solver.piezoviscous.solver_pv_ausas_dynamic import (
        solve_ausas_unsteady_pv_one_step_gpu,
    )

    kwargs = _baseline_kwargs(40, 80)
    base = ausas_unsteady_one_step_gpu(**kwargs)
    pv = solve_ausas_unsteady_pv_one_step_gpu(
        pv_model="off", **kwargs,
    )

    err_P = float(np.max(np.abs(base["P"] - pv["P"])))
    err_th = float(np.max(np.abs(base["theta"] - pv["theta"])))
    ok = err_P == 0.0 and err_th == 0.0
    ok &= pv["pv_model"] == "off"
    ok &= pv["pv_outer_iters"] == 0
    ok &= pv["pv_converged"] is True
    ok &= pv["pv_mu_ratio_max"] == 1.0
    return _report(
        "off bypass identical to baseline",
        ok,
        f"max|dP|={err_P:.2e}, max|dtheta|={err_th:.2e}, "
        f"outer={pv['pv_outer_iters']}",
    )


# ---------------------------------------------------------------------------
# Test 2: alpha=0 with both Roelands and Barus bypasses to baseline.
# ---------------------------------------------------------------------------
def test_alpha_zero_regression():
    print("\n=== Test 2: alpha=0 regression for roelands & barus ===")
    if not _has_cupy():
        print("  [SKIP] cupy not available")
        return True
    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        ausas_unsteady_one_step_gpu,
    )
    from reynolds_solver.piezoviscous.solver_pv_ausas_dynamic import (
        solve_ausas_unsteady_pv_one_step_gpu,
    )

    kwargs = _baseline_kwargs(32, 64)
    base = ausas_unsteady_one_step_gpu(**kwargs)

    all_ok = True
    for model in ("roelands", "barus"):
        pv = solve_ausas_unsteady_pv_one_step_gpu(
            pv_model=model, alpha_pv=0.0, p_scale=P_SCALE,
            **kwargs,
        )
        err_P = float(np.max(np.abs(base["P"] - pv["P"])))
        err_th = float(np.max(np.abs(base["theta"] - pv["theta"])))
        mu_max = pv["pv_mu_ratio_max"]
        ok = (err_P == 0.0 and err_th == 0.0
              and pv["pv_outer_iters"] == 0
              and pv["pv_converged"] is True
              and mu_max == 1.0)
        all_ok &= _report(
            f"{model}, alpha=0 → baseline",
            ok,
            f"max|dP|={err_P:.2e}, mu_max={mu_max:.6f}, "
            f"outer={pv['pv_outer_iters']}",
        )
    return all_ok


# ---------------------------------------------------------------------------
# Test 3: PV modifies only pressure-flow coefficients.
# Uniform mu_ratio = k scales A,B,C,D by 1/k (E recomputed).
# ---------------------------------------------------------------------------
def test_coefficient_injection():
    print("\n=== Test 3: PV affects only pressure-flow coefficients ===")
    if not _has_cupy():
        print("  [SKIP] cupy not available")
        return True
    import cupy as cp
    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        _build_coefficients_gpu,
    )
    from reynolds_solver.piezoviscous.solver_piezoviscous import (
        _apply_piezoviscosity,
    )

    N_Z, N_phi = 16, 32
    H, d_phi, d_Z = _make_gap(N_Z, N_phi, 0.6)
    H_gpu = cp.asarray(H)

    A0, B0, C0, D0, E0 = _build_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)

    # Identity mu_ratio == 1 must leave coefficients unchanged.
    mu1 = cp.ones((N_Z, N_phi), dtype=cp.float64)
    A1 = A0.copy(); B1 = B0.copy(); C1 = C0.copy(); D1 = D0.copy()
    E1 = _apply_piezoviscosity(A1, B1, C1, D1, E0, mu1, N_Z, N_phi)
    err_id = (
        float(cp.max(cp.abs(A1 - A0)))
        + float(cp.max(cp.abs(B1 - B0)))
        + float(cp.max(cp.abs(C1 - C0)))
        + float(cp.max(cp.abs(D1 - D0)))
        + float(cp.max(cp.abs(E1 - E0)))
    )
    ok_id = err_id == 0.0

    # Uniform mu_ratio = 2 must scale conductances by 1/2.
    k = 2.0
    muk = cp.full((N_Z, N_phi), k, dtype=cp.float64)
    A2 = A0.copy(); B2 = B0.copy(); C2 = C0.copy(); D2 = D0.copy()
    E2 = _apply_piezoviscosity(A2, B2, C2, D2, E0, muk, N_Z, N_phi)
    err_k = (
        float(cp.max(cp.abs(A2 - A0 / k)))
        + float(cp.max(cp.abs(B2 - B0 / k)))
        + float(cp.max(cp.abs(C2 - C0 / k)))
        + float(cp.max(cp.abs(D2 - D0 / k)))
        + float(cp.max(cp.abs(E2 - (A0 + B0 + C0 + D0) / k)))
    )
    ok_k = err_k < 1e-14

    return (
        _report("mu_ratio = 1 leaves coefficients unchanged", ok_id,
                f"err={err_id:.2e}")
        and _report("uniform mu_ratio = 2 scales conductances by 1/2", ok_k,
                    f"err={err_k:.2e}")
    )


# ---------------------------------------------------------------------------
# Test 4: Roelands / Barus formula reuses solver_piezoviscous helper and
#         has the expected limits.
# ---------------------------------------------------------------------------
def test_formula_limits():
    print("\n=== Test 4: Roelands / Barus formula limits ===")
    if not _has_cupy():
        print("  [SKIP] cupy not available")
        return True
    import cupy as cp
    from reynolds_solver.piezoviscous.solver_piezoviscous import (
        _compute_mu_ratio_gpu,
    )

    alpha = 18e-9
    p0 = 1.98e8
    z = 0.6

    # p = 0 → mu_ratio == 1
    P_zero = cp.zeros((4, 4), dtype=cp.float64)
    mu0, _ = _compute_mu_ratio_gpu(P_zero, alpha, 1.0, p0, z)
    ok_zero = float(cp.max(cp.abs(mu0 - 1.0))) < 1e-14

    # Barus path: z=1 reduces to mu = exp(alpha*p) exactly.
    p_test = 5e6
    P_test = cp.full((3, 3), p_test, dtype=cp.float64)
    mu_b, _ = _compute_mu_ratio_gpu(P_test, alpha, 1.0, p0, 1.0)
    err_b = abs(float(cp.mean(mu_b)) - np.exp(alpha * p_test))
    ok_barus = err_b < 1e-12

    # Roelands at p << p0 ≈ Barus.
    p_low = 1e6
    P_low = cp.full((3, 3), p_low, dtype=cp.float64)
    mu_r, _ = _compute_mu_ratio_gpu(P_low, alpha, 1.0, p0, z)
    err_low = abs(float(cp.mean(mu_r)) - np.exp(alpha * p_low)) / np.exp(
        alpha * p_low
    )
    ok_low = err_low < 0.01

    # Monotonicity in p.
    P_seq = cp.asarray(
        np.linspace(0.0, 5 * p0, 12).reshape(3, 4), dtype=cp.float64
    )
    mu_seq, _ = _compute_mu_ratio_gpu(P_seq, alpha, 1.0, p0, z)
    flat = cp.asnumpy(mu_seq).ravel()
    flat_p = cp.asnumpy(P_seq).ravel()
    order = np.argsort(flat_p)
    ok_mono = np.all(np.diff(flat[order]) >= -1e-15)

    all_ok = True
    all_ok &= _report("p=0 → mu_ratio = 1 (exact)", ok_zero)
    all_ok &= _report("z=1 reduces to Barus (exact)", ok_barus,
                      f"err={err_b:.2e}")
    all_ok &= _report("Roelands ≈ Barus at p << p0 (<1%)", ok_low,
                      f"err={err_low:.2e}")
    all_ok &= _report("mu_ratio monotone in p", ok_mono)
    return all_ok


# ---------------------------------------------------------------------------
# Test 5: Cavitation invariants on a representative case.
# ---------------------------------------------------------------------------
def test_cavitation_invariants():
    print("\n=== Test 5: cavitation / sign invariants ===")
    if not _has_cupy():
        print("  [SKIP] cupy not available")
        return True
    from reynolds_solver.piezoviscous.solver_pv_ausas_dynamic import (
        solve_ausas_unsteady_pv_one_step_gpu,
    )

    kwargs = _baseline_kwargs(32, 64)
    res = solve_ausas_unsteady_pv_one_step_gpu(
        pv_model="roelands",
        alpha_GPa_inv=15.0, p_scale=P_SCALE,
        pv_max_outer=8, pv_tol=1e-3, pv_relax=0.5,
        return_mu_ratio=True,
        **kwargs,
    )
    P = res["P"]; theta = res["theta"]; mu = res["pv_mu_ratio_field"]
    ok_finite = (
        np.all(np.isfinite(P)) and np.all(np.isfinite(theta))
        and np.all(np.isfinite(mu))
    )
    ok_p = float(P.min()) >= -1e-10
    ok_theta = (theta.min() >= -1e-8) and (theta.max() <= 1.0 + 1e-8)
    ok_mu_ge1 = float(mu.min()) >= 1.0 - 1e-10
    # In cells with p=0 (cavitated full-zero), mu should be 1.
    cav = P < 1e-12
    if cav.any():
        ok_mu_cav = float(mu[cav].max()) <= 1.0 + 1e-10
    else:
        ok_mu_cav = True
    all_ok = True
    all_ok &= _report("all fields finite", ok_finite)
    all_ok &= _report("P >= 0", ok_p, f"min P = {P.min():.2e}")
    all_ok &= _report("0 <= theta <= 1", ok_theta,
                      f"theta in [{theta.min():.2e}, {theta.max():.4f}]")
    all_ok &= _report("mu_ratio >= 1", ok_mu_ge1,
                      f"min mu = {mu.min():.6f}")
    all_ok &= _report("mu_ratio == 1 in p=0 cells", ok_mu_cav)
    return all_ok


# ---------------------------------------------------------------------------
# Test 6: Warm-start reduces outer iterations vs cold start.
# ---------------------------------------------------------------------------
def test_warm_start():
    print("\n=== Test 6: warm-start reduces outer iters ===")
    if not _has_cupy():
        print("  [SKIP] cupy not available")
        return True
    from reynolds_solver.piezoviscous.solver_pv_ausas_dynamic import (
        solve_ausas_unsteady_pv_one_step_gpu,
    )

    kwargs = _baseline_kwargs(32, 64)
    common = dict(
        pv_model="roelands", alpha_GPa_inv=15.0, p_scale=P_SCALE,
        pv_max_outer=12, pv_tol=1e-3, pv_relax=0.5,
        return_mu_ratio=True,
    )

    cold = solve_ausas_unsteady_pv_one_step_gpu(**common, **kwargs)
    warm = solve_ausas_unsteady_pv_one_step_gpu(
        pv_initial_mu_ratio=cold["pv_mu_ratio_field"],
        **common, **kwargs,
    )

    err_P = float(np.max(np.abs(cold["P"] - warm["P"])))
    ok_iters = warm["pv_outer_iters"] <= cold["pv_outer_iters"]
    ok_match = err_P / (np.max(np.abs(cold["P"])) + 1e-30) < 1e-3
    ok = ok_iters and ok_match
    return _report(
        f"warm <= cold (cold={cold['pv_outer_iters']}, "
        f"warm={warm['pv_outer_iters']})",
        ok,
        f"max|dP|/max|P|={err_P/(np.max(np.abs(cold['P']))+1e-30):.2e}",
    )


# ---------------------------------------------------------------------------
# Test 7: alpha_GPa_inv → alpha_pv conversion.
# ---------------------------------------------------------------------------
def test_alpha_unit_conversion():
    print("\n=== Test 7: alpha_GPa_inv == alpha_pv * 1e9 ===")
    if not _has_cupy():
        print("  [SKIP] cupy not available")
        return True
    from reynolds_solver.piezoviscous.solver_pv_ausas_dynamic import (
        solve_ausas_unsteady_pv_one_step_gpu,
    )
    kwargs = _baseline_kwargs(24, 48)
    common = dict(pv_model="roelands", p_scale=P_SCALE,
                  pv_max_outer=6, pv_relax=0.5, return_mu_ratio=False)
    a = solve_ausas_unsteady_pv_one_step_gpu(
        alpha_pv=15e-9, **common, **kwargs,
    )
    b = solve_ausas_unsteady_pv_one_step_gpu(
        alpha_GPa_inv=15.0, **common, **kwargs,
    )
    err = float(np.max(np.abs(a["P"] - b["P"])))
    return _report(
        "alpha_pv=15e-9 == alpha_GPa_inv=15.0",
        err < 1e-12,
        f"max|dP|={err:.2e}",
    )


def main():
    print("=" * 60)
    print("  Piezoviscous + Ausas/JFO dynamic one-step tests")
    print("=" * 60)
    results = [
        test_off_regression(),
        test_alpha_zero_regression(),
        test_coefficient_injection(),
        test_formula_limits(),
        test_cavitation_invariants(),
        test_warm_start(),
        test_alpha_unit_conversion(),
    ]
    print("\n" + "=" * 60)
    if all(results):
        print("  ALL PV-AUSAS TESTS PASSED")
        sys.exit(0)
    else:
        print("  SOME PV-AUSAS TESTS FAILED")
        for i, r in enumerate(results, 1):
            if not r:
                print(f"    Test {i} FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
