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
   θ∈[θ_min, 1], not collapsed (maxP > small).
2. Uniform gap ε=0: trivial state (Θ≡1, P≡0).
3. phi_bc="groove" smoke test: P=0 and Θ=1 at the seam.

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

    p_ok = finite and p_min >= -1e-12 and p_max > 1e-3
    th_ok = finite and th_min >= 0.0 and th_max <= 1.0 + 1e-12
    cav_ok = 0.0 < cav_frac < 0.9

    return run_test(
        "smooth ε=0.6: finite, P≥0, Θ∈[0,1], cav reasonable",
        p_ok and th_ok and cav_ok,
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

    print(f"    n_iter={n}, maxP={P.max():.4e}")
    print(f"    boundary: P_max={P_boundary:.2e}, "
          f"Θ∈[{th_boundary_min:.4f}, {th_boundary_max:.4f}]")

    return run_test(
        "groove: P=0 & Θ=1 at seam, finite solution",
        p_ok and th_ok,
        f"P_bnd={P_boundary:.2e}, Θ_bnd=[{th_boundary_min:.4f}, "
        f"{th_boundary_max:.4f}]",
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
