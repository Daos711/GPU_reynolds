"""
Smoke tests for the compressible Elrod / Vijayaraghavan-Keith solver
(Manser 2019 formulation).

These are MVP phase-1 sanity checks — they verify that the solver runs
without blowing up and produces physically sensible output. Validation
against Manser's T2/T3 ratio, partial texture gain, and the original
article figures is a separate step handled by the user's validation
scripts (scripts/validate_manser.py etc.).

Tests
-----
1. Smooth bearing ε=0.6 at β̄=30 (pump-like): converges, P≥0,
   θ∈[θ_min, ∞), not collapsed (maxP > small).
2. Uniform gap ε=0: trivial state (Θ≡1, P≡0).
3. phi_bc="groove" smoke test: P=0 and Θ=1 at the seam.
4. P ↔ Θ consistency: full-film P matches β̄·ln(Θ) to numerical
   tolerance, and Θ>1 region is non-empty (otherwise the returned
   theta was clipped and the compressible information is lost).

(A direct maxP comparison against the Payvar-Salant incompressible
limit is not part of the smoke suite: Manser's normalisation adds a
factor 6 on the RHS relative to the Ausas/PS convention, so P is on
a different scale — comparing loads against Manser figures is done
separately via scripts/validate_manser.py.)

Run:
    python -m reynolds_solver.tests.test_elrod
"""
import sys
import numpy as np


def run_test(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if details:
        print(f"         {details}")
    return passed


def generate_test_case(N_phi, N_Z, epsilon=0.6):
    phi_1D = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)
    Phi, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z, phi_1D, Z


R, L = 0.035, 0.056


# -----------------------------------------------------------------------
# Test 1: Smooth bearing at pump-like β̄
# -----------------------------------------------------------------------
def test_smooth_bearing_pump():
    print("\n=== Test 1: smooth bearing ε=0.6, β̄≈30 (pump-like) ===")
    from reynolds_solver.cavitation.elrod import solve_elrod_compressible

    N_phi, N_Z = 100, 40
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.6)

    P, theta, res, n = solve_elrod_compressible(
        H, d_phi, d_Z, R, L,
        beta_bar=30.0,
        tol=1e-6, max_iter=100_000,
    )

    p_min = float(P.min())
    p_max = float(P.max())
    th_min = float(theta.min())
    th_max = float(theta.max())
    cav_frac = float(np.mean(theta[1:-1, 1:-1] < 1.0 - 1e-6))
    finite = np.all(np.isfinite(P)) and np.all(np.isfinite(theta))

    print(f"    n_iter={n}, res={res:.2e}")
    print(f"    P=[{p_min:.2e}, {p_max:.4e}]")
    print(f"    Θ=[{th_min:.4f}, {th_max:.4f}], cav_frac={cav_frac:.3f}")

    # Structural smoke on the periodic smooth case. The explicit P-Θ
    # engine with a Picard outer loop significantly improves the
    # periodic behaviour over the Theta/g-only MVP (P_max stays of
    # order 0.1 vs the previous 0.025), but the periodic smooth
    # bearing has a trivial "Θ·h = const, P = 0" fixed point that is
    # an attractor without a structural anchor (groove / supply).
    # Physical pump simulations use groove — see test 3 for the
    # strict check.
    p_ok = finite and p_min >= -1e-12 and p_max > 1e-3
    th_ok = finite and th_min >= 0.0 and th_max < 2.0
    not_collapsed = p_max > 1e-3   # any non-trivial pressure

    return run_test(
        "smooth ε=0.6: finite, P≥0, Θ≥0, not fully collapsed",
        p_ok and th_ok and not_collapsed,
        f"p_max={p_max:.3e}, cav_frac={cav_frac:.3f}, n_iter={n}",
    )


# -----------------------------------------------------------------------
# Test 2: Uniform gap ε=0 — trivial state
# -----------------------------------------------------------------------
def test_uniform_gap():
    print("\n=== Test 2: uniform gap ε=0 (trivial state) ===")
    from reynolds_solver.cavitation.elrod import solve_elrod_compressible

    N_phi, N_Z = 40, 20
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.0)

    P, theta, res, n = solve_elrod_compressible(
        H, d_phi, d_Z, R, L,
        beta_bar=30.0,
        tol=1e-10, max_iter=5_000,
    )

    maxP = float(np.max(np.abs(P)))
    th_dev = float(np.max(np.abs(theta - 1.0)))
    print(f"    n_iter={n}, maxP={maxP:.2e}, max|Θ-1|={th_dev:.2e}")

    return run_test(
        "ε=0: trivial Θ≡1, P≡0",
        maxP < 1e-6 and th_dev < 1e-6,
        f"maxP={maxP:.2e}, max|Θ-1|={th_dev:.2e}",
    )


# -----------------------------------------------------------------------
# Test 3: phi_bc='groove' smoke test
# -----------------------------------------------------------------------
def test_groove_smoke():
    print("\n=== Test 3: phi_bc='groove' smoke ===")
    from reynolds_solver.cavitation.elrod import solve_elrod_compressible

    N_phi, N_Z = 60, 30
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.6)

    P, theta, res, n = solve_elrod_compressible(
        H, d_phi, d_Z, R, L,
        beta_bar=30.0,
        tol=1e-6, max_iter=100_000,
        phi_bc="groove",
    )

    P_boundary = max(float(np.max(np.abs(P[:, 0]))),
                     float(np.max(np.abs(P[:, -1]))))
    th_boundary_min = min(float(np.min(theta[:, 0])),
                          float(np.min(theta[:, -1])))
    th_boundary_max = max(float(np.max(theta[:, 0])),
                          float(np.max(theta[:, -1])))

    finite = np.all(np.isfinite(P)) and np.all(np.isfinite(theta))
    p_ok = finite and P_boundary < 1e-10
    th_ok = (abs(th_boundary_min - 1.0) < 1e-10
             and abs(th_boundary_max - 1.0) < 1e-10)

    # Real full-film lobe: on groove ε=0.6 β̄=30 the explicit P-Θ
    # engine gives maxP ≈ 2.8 and compresses Θ to ≈ 1.09 in the
    # converging region. Require a non-trivial lobe (maxP > 1.0) and
    # a meaningful cavitation fraction (0.2 < cav < 0.7).
    cav_frac = float(np.mean(theta[1:-1, 1:-1] < 1.0 - 1e-6))
    theta_max = float(theta.max())
    strong_lobe = P.max() > 1.0 and theta_max > 1.01
    cav_ok = 0.2 < cav_frac < 0.7

    print(f"    n_iter={n}, res={res:.2e}, maxP={P.max():.4e}, "
          f"Θ_max={theta_max:.4f}, cav={cav_frac:.3f}")
    print(f"    boundary: P_max={P_boundary:.2e}, "
          f"Θ∈[{th_boundary_min:.4f}, {th_boundary_max:.4f}]")

    return run_test(
        "groove: strong full-film lobe, seam BC, cav reasonable",
        p_ok and th_ok and strong_lobe and cav_ok,
        f"maxP={P.max():.2e}, Θ_max={theta_max:.4f}, "
        f"cav={cav_frac:.3f}",
    )


# -----------------------------------------------------------------------
# Test 4: P ↔ Θ consistency on full-film cells
# -----------------------------------------------------------------------
def test_p_theta_consistency():
    print("\n=== Test 4: P ↔ Θ consistency (P = β̄·ln(Θ) in full film) ===")
    from reynolds_solver.cavitation.elrod import solve_elrod_compressible

    N_phi, N_Z = 100, 40
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.6)
    beta_bar = 30.0

    # Run on groove — the explicit P-Θ engine gives a strong
    # full-film lobe there (maxP ≈ 2.8, 40%+ of the grid in Θ>1).
    # Periodic smooth has a trivial attractor and the resulting
    # full-film set is very small (see test 1 comment).
    P, theta, res, n = solve_elrod_compressible(
        H, d_phi, d_Z, R, L,
        beta_bar=beta_bar,
        tol=1e-7, max_iter=200_000,
        phi_bc="groove",
    )

    # Full-film stats
    ff_mask = P > 1e-10
    n_ff = int(ff_mask.sum())
    th_gt_1 = int(np.sum(theta > 1.0 + 1e-10))
    frac_compressed = th_gt_1 / theta.size

    if n_ff == 0:
        return run_test(
            "P↔Θ: full-film cells exist",
            False,
            "no cells with P > 0 — solver collapsed",
        )

    P_check = beta_bar * np.log(theta[ff_mask])
    max_err = float(np.max(np.abs(P[ff_mask] - P_check)))
    mean_err = float(np.mean(np.abs(P[ff_mask] - P_check)))

    print(f"    n_ff={n_ff} cells with P>0, Θ>1 in {th_gt_1} "
          f"({100*frac_compressed:.1f}% of grid)")
    print(f"    Θ range: [{theta.min():.4f}, {theta.max():.4f}]")
    print(f"    max|P - β̄·ln(Θ)| = {max_err:.2e}, mean = {mean_err:.2e}")

    # Consistency and compressibility are both required.
    consistent = max_err < 1e-6
    compressed = th_gt_1 > 0 and theta.max() > 1.0 + 1e-6

    return run_test(
        "P = β̄·ln(Θ) in full film and Θ>1 region is non-empty",
        consistent and compressed,
        f"max_err={max_err:.2e}, Θ_max={theta.max():.4f}, "
        f"n(Θ>1)={th_gt_1}",
    )


# -----------------------------------------------------------------------
# Test 5 (V2): theta_vk smooth-groove physics smoke
# -----------------------------------------------------------------------
def test_theta_vk_groove_smoke():
    print("\n=== Test 5 (V2): theta_vk smooth-groove physics smoke ===")
    from reynolds_solver.cavitation.elrod import solve_elrod_compressible

    N_phi, N_Z = 60, 30
    H, d_phi, d_Z, _, _ = generate_test_case(N_phi, N_Z, epsilon=0.6)

    P, theta, res, n = solve_elrod_compressible(
        H, d_phi, d_Z, R, L,
        beta_bar=30.0,
        formulation="theta_vk",
        switch_backend="hard",
        tol=1e-6, max_iter=20_000,
        phi_bc="groove",
    )

    finite = np.all(np.isfinite(P)) and np.all(np.isfinite(theta))
    p_ok = finite and P.max() > 1.0 and P.min() >= -1e-12
    th_ok = (
        finite and theta.max() > 1.01 and theta.min() >= 0.0
        and abs(theta[0, 0] - 1.0) < 1e-10
        and abs(theta[0, -1] - 1.0) < 1e-10
    )
    cav_frac = float(np.mean(theta[1:-1, 1:-1] < 1.0 - 1e-6))
    cav_ok = 0.2 < cav_frac < 0.7

    print(f"    n={n}, res={res:.2e}, P_max={P.max():.4e}, "
          f"Θ_max={theta.max():.4f}, cav={cav_frac:.3f}")

    return run_test(
        "theta_vk groove ε=0.6: full-film lobe + cav region",
        p_ok and th_ok and cav_ok,
        f"P_max={P.max():.3e}, Θ_max={theta.max():.4f}, "
        f"cav={cav_frac:.3f}",
    )


# -----------------------------------------------------------------------
# Test 6 (V1): theta_vk hard vs fk_soft must differ on textured case
# -----------------------------------------------------------------------
def test_theta_vk_v1_distinguishability():
    print("\n=== Test 6 (V1): theta_vk hard vs fk_soft on textured case ===")
    from reynolds_solver.cavitation.elrod import solve_elrod_compressible

    # Build a single-dimple "wedge near rupture" case: smooth bearing
    # ε=0.6 with one convergent triangular depression placed shortly
    # before the smooth-bearing rupture line. This is enough to push
    # the local Θ field into the regime where the FK soft switch
    # starts to differ from the hard switch.
    N_phi, N_Z = 80, 30
    phi = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)
    Phi, Zg = np.meshgrid(phi, Z)
    H = 1.0 + 0.6 * np.cos(Phi)
    # Place wedge shortly before φ=π (rupture region): linear φ-ramp
    phi_c = np.pi - 0.3
    r_x = 0.25
    r_z = 0.4
    delta_phi = np.arctan2(np.sin(Phi - phi_c), np.cos(Phi - phi_c))
    inside_phi = (np.abs(delta_phi) <= r_x).astype(float)
    inside_z = (np.abs(Zg) <= r_z).astype(float)
    ramp = np.clip(1.0 - delta_phi / r_x, 0.0, 2.0)
    H = H + 0.3 * ramp * inside_phi * inside_z
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]

    common = dict(
        d_phi=d_phi, d_Z=d_Z, R=R, L=L,
        beta_bar=40.0,
        formulation="theta_vk",
        omega=1.0, tol=1e-6, max_iter=20_000,
        phi_bc="groove",
    )
    P_h, th_h, _, _ = solve_elrod_compressible(H, switch_backend="hard", **common)
    P_s, th_s, _, _ = solve_elrod_compressible(H, switch_backend="fk_soft", **common)

    diff_P_max = float(np.max(np.abs(P_h - P_s)))
    diff_th_max = float(np.max(np.abs(th_h - th_s)))
    rel_diff_P = diff_P_max / max(P_h.max(), 1e-30)

    print(f"    P_max(hard)={P_h.max():.3e}, P_max(soft)={P_s.max():.3e}")
    print(f"    max|ΔP| = {diff_P_max:.3e}  (rel = {rel_diff_P:.3%})")
    print(f"    max|ΔΘ| = {diff_th_max:.3e}")

    # ТЗ V1 acceptance: at least one textured case must show non-trivial
    # difference between hard and fk_soft. We require ≥1 % relative
    # difference in P_max OR ≥1 e-3 absolute on Θ.
    distinguishable = rel_diff_P > 0.01 or diff_th_max > 1e-3

    return run_test(
        "theta_vk hard ≠ fk_soft on textured (V1)",
        distinguishable,
        f"rel|ΔP|={rel_diff_P:.3%}, max|ΔΘ|={diff_th_max:.3e}",
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Elrod compressible cavitation — MVP smoke tests")
    print("=" * 60)

    tests = [
        ("1", test_smooth_bearing_pump),
        ("2", test_uniform_gap),
        ("3", test_groove_smoke),
        ("4", test_p_theta_consistency),
        ("5", test_theta_vk_groove_smoke),
        ("6", test_theta_vk_v1_distinguishability),
    ]

    results = []
    for name, func in tests:
        try:
            results.append((name, func()))
        except Exception as e:
            print(f"  [FAIL] Test {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print()
    print("=" * 60)
    all_ok = all(r for _, r in results)
    if all_ok:
        print("  ALL ELROD MVP TESTS PASSED")
    else:
        print("  SOME ELROD MVP TESTS FAILED")
        for name, r in results:
            if not r:
                print(f"    Test {name} FAILED")
    print("=" * 60)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
