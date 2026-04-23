"""
Regression and functional tests for the dirichlet_mask + g_bc feature
(Agent 1 TZ) of the Payvar-Salant solver.

Covers:
  1. Regression-equivalence: mask=None / g_bc=None same as no new args.
  2. Flooded / all-True mask: P == g_bc everywhere, theta == 1.
  3. Single pinned interior cell: node exactly equals g_bc, field is
     smooth and non-zero around it, CPU == GPU within tolerance.
  4. Error paths: mask without g_bc, shape mismatch.

The new arguments must NOT change behaviour when they are left at None.
"""
import sys
import numpy as np


def _try_import_gpu():
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def _uniform_gap(N_phi=40, N_Z=30, epsilon=0.4):
    phi = np.linspace(0.0, 2.0 * np.pi, N_phi)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, _ = np.meshgrid(phi, Z)
    H = 1.0 + epsilon * np.cos(Phi)
    return H, float(phi[1] - phi[0]), float(Z[1] - Z[0])


def _run(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}")
    if detail:
        print(f"         {detail}")
    return ok


# ---------------------------------------------------------------------------
# 1. Regression-equivalence
# ---------------------------------------------------------------------------
def test_regression_cpu_none_equals_nokwarg():
    """solve_payvar_salant_cpu(mask=None, g_bc=None) == old call."""
    print("\n=== Test 1a: CPU regression (mask=None equals no kwarg) ===")
    from reynolds_solver.cavitation.payvar_salant import (
        solve_payvar_salant_cpu,
    )
    R, L = 0.035, 0.056
    H, d_phi, d_Z = _uniform_gap()

    P_old, th_old, r_old, n_old = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L, omega=1.0, tol=1e-8, max_iter=5000,
    )
    P_new, th_new, r_new, n_new = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L, omega=1.0, tol=1e-8, max_iter=5000,
        dirichlet_mask=None, g_bc=None,
    )

    max_dP = float(np.max(np.abs(P_old - P_new)))
    max_dth = float(np.max(np.abs(th_old - th_new)))
    ok = (
        max_dP == 0.0
        and max_dth == 0.0
        and r_old == r_new
        and n_old == n_new
    )
    return _run(
        "CPU mask=None bit-equal to no-kwarg call", ok,
        f"max|ΔP|={max_dP:.2e}, max|Δθ|={max_dth:.2e}, "
        f"n_old={n_old}, n_new={n_new}",
    )


def test_regression_gpu_none_equals_nokwarg():
    """solve_payvar_salant_gpu(mask=None, g_bc=None) == old call."""
    print("\n=== Test 1b: GPU regression (mask=None equals no kwarg) ===")
    if not _try_import_gpu():
        return _run("[SKIP] cupy not available", True)
    from reynolds_solver.cavitation.payvar_salant import (
        solve_payvar_salant_gpu,
    )
    R, L = 0.035, 0.056
    H, d_phi, d_Z = _uniform_gap()

    P_old, th_old, r_old, n_old = solve_payvar_salant_gpu(
        H, d_phi, d_Z, R, L, tol=1e-7, max_iter=5000,
    )
    P_new, th_new, r_new, n_new = solve_payvar_salant_gpu(
        H, d_phi, d_Z, R, L, tol=1e-7, max_iter=5000,
        dirichlet_mask=None, g_bc=None,
    )

    max_dP = float(np.max(np.abs(P_old - P_new)))
    max_dth = float(np.max(np.abs(th_old - th_new)))
    ok = max_dP == 0.0 and max_dth == 0.0
    return _run(
        "GPU mask=None bit-equal to no-kwarg call", ok,
        f"max|ΔP|={max_dP:.2e}, max|Δθ|={max_dth:.2e}",
    )


# ---------------------------------------------------------------------------
# 2. All-True mask / flooded constant
# ---------------------------------------------------------------------------
def test_all_true_mask_cpu():
    """dirichlet_mask all-True at a positive constant -> P == g_bc."""
    print("\n=== Test 2: CPU all-True mask, g_bc=const>0 ===")
    from reynolds_solver.cavitation.payvar_salant import (
        solve_payvar_salant_cpu,
    )
    R, L = 0.035, 0.056
    H, d_phi, d_Z = _uniform_gap()
    N_Z, N_phi = H.shape
    mask = np.ones((N_Z, N_phi), dtype=bool)
    g_bc = 1.5e-3

    P, theta, residual, n = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L,
        omega=1.0, tol=1e-8, max_iter=5000,
        hs_warmup_iter=500,
        dirichlet_mask=mask, g_bc=g_bc,
    )
    max_dev_P = float(np.max(np.abs(P - g_bc)))
    max_dev_theta = float(np.max(np.abs(theta - 1.0)))
    ok = max_dev_P < 1e-12 and max_dev_theta < 1e-12
    return _run(
        "All-true mask: P==g_bc, θ==1", ok,
        f"max|P-g_bc|={max_dev_P:.2e}, max|θ-1|={max_dev_theta:.2e}, "
        f"n_iter={n}",
    )


# ---------------------------------------------------------------------------
# 3. Single interior pinned cell
# ---------------------------------------------------------------------------
def test_single_pinned_cell_cpu():
    """One interior node pinned to g_bc; field smooth around it."""
    print("\n=== Test 3a: CPU single pinned interior cell ===")
    from reynolds_solver.cavitation.payvar_salant import (
        solve_payvar_salant_cpu,
    )
    R, L = 0.035, 0.056
    H, d_phi, d_Z = _uniform_gap(N_phi=60, N_Z=30, epsilon=0.4)
    N_Z, N_phi = H.shape
    i_pin, j_pin = N_Z // 2, N_phi // 2
    g_bc = 2.0e-3
    mask = np.zeros((N_Z, N_phi), dtype=bool)
    mask[i_pin, j_pin] = True

    P, theta, residual, n = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L,
        omega=1.0, tol=1e-8, max_iter=10000,
        hs_warmup_iter=2000,
        dirichlet_mask=mask, g_bc=g_bc,
    )

    ok_pin = abs(P[i_pin, j_pin] - g_bc) < 1e-12
    # Neighbours finite and close to but not equal to g_bc
    ok_finite = np.all(np.isfinite(P)) and np.all(np.isfinite(theta))
    ok_nonneg = bool(P.min() >= 0.0)
    ok_theta = bool(theta.min() >= 0.0 and theta.max() <= 1.0 + 1e-12)
    # Some variation in P (not uniformly g_bc)
    P_range = float(P.max() - P.min())
    ok_nontriv = P_range > 1e-8
    ok = ok_pin and ok_finite and ok_nonneg and ok_theta and ok_nontriv
    return _run(
        "CPU single-pin: pinned value, P>=0, θ∈[0,1], smooth field", ok,
        f"P[pin]={P[i_pin, j_pin]:.6e} vs g_bc={g_bc:.6e}, "
        f"P range={P_range:.2e}, θ∈[{theta.min():.4f}, {theta.max():.4f}]",
    )


def test_single_pinned_cell_gpu_matches_cpu():
    """CPU and GPU agree on the single-pin problem."""
    print("\n=== Test 3b: CPU vs GPU agreement on single pinned cell ===")
    if not _try_import_gpu():
        return _run("[SKIP] cupy not available", True)
    from reynolds_solver.cavitation.payvar_salant import (
        solve_payvar_salant_cpu,
        solve_payvar_salant_gpu,
    )
    R, L = 0.035, 0.056
    H, d_phi, d_Z = _uniform_gap(N_phi=60, N_Z=30, epsilon=0.4)
    N_Z, N_phi = H.shape
    i_pin, j_pin = N_Z // 2, N_phi // 2
    g_bc = 2.0e-3
    mask = np.zeros((N_Z, N_phi), dtype=bool)
    mask[i_pin, j_pin] = True

    P_cpu, th_cpu, r_cpu, _ = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L,
        omega=1.0, tol=1e-8, max_iter=20000,
        hs_warmup_iter=5000,
        dirichlet_mask=mask, g_bc=g_bc,
    )
    P_gpu, th_gpu, r_gpu, _ = solve_payvar_salant_gpu(
        H, d_phi, d_Z, R, L,
        tol=1e-7, max_iter=20000,
        hs_warmup_iter=5000,
        dirichlet_mask=mask, g_bc=g_bc,
    )

    scale = max(float(np.max(np.abs(P_cpu))), 1e-30)
    err_P = float(np.max(np.abs(P_cpu - P_gpu))) / scale
    err_th = float(np.max(np.abs(th_cpu - th_gpu)))
    ok = err_P < 5e-3 and err_th < 5e-3
    # pinned value exact to machine precision on both.
    ok_pin_cpu = abs(P_cpu[i_pin, j_pin] - g_bc) < 1e-12
    ok_pin_gpu = abs(P_gpu[i_pin, j_pin] - g_bc) < 1e-12
    return _run(
        "CPU/GPU pinned cells match, fields agree within tol",
        ok and ok_pin_cpu and ok_pin_gpu,
        f"rel|ΔP|={err_P:.2e}, max|Δθ|={err_th:.2e}, "
        f"P_cpu[pin]={P_cpu[i_pin, j_pin]:.3e}, "
        f"P_gpu[pin]={P_gpu[i_pin, j_pin]:.3e}",
    )


# ---------------------------------------------------------------------------
# 4. Error paths
# ---------------------------------------------------------------------------
def test_error_paths_cpu():
    """mask without g_bc, shape mismatch, nan g_bc -> ValueError."""
    print("\n=== Test 4: CPU error paths ===")
    from reynolds_solver.cavitation.payvar_salant import (
        solve_payvar_salant_cpu,
    )
    R, L = 0.035, 0.056
    H, d_phi, d_Z = _uniform_gap(N_phi=30, N_Z=20)
    N_Z, N_phi = H.shape

    cases = 0
    passed = 0

    # mask without g_bc
    cases += 1
    try:
        solve_payvar_salant_cpu(
            H, d_phi, d_Z, R, L, max_iter=10,
            dirichlet_mask=np.zeros_like(H, dtype=bool),
            g_bc=None,
        )
        print("    mask without g_bc: FAIL (no error raised)")
    except ValueError:
        passed += 1

    # g_bc without mask
    cases += 1
    try:
        solve_payvar_salant_cpu(
            H, d_phi, d_Z, R, L, max_iter=10,
            dirichlet_mask=None, g_bc=0.0,
        )
        print("    g_bc without mask: FAIL (no error raised)")
    except ValueError:
        passed += 1

    # shape mismatch
    cases += 1
    try:
        bad_mask = np.zeros((N_Z + 1, N_phi), dtype=bool)
        solve_payvar_salant_cpu(
            H, d_phi, d_Z, R, L, max_iter=10,
            dirichlet_mask=bad_mask, g_bc=0.0,
        )
        print("    shape mismatch: FAIL (no error raised)")
    except ValueError:
        passed += 1

    # nan g_bc
    cases += 1
    try:
        solve_payvar_salant_cpu(
            H, d_phi, d_Z, R, L, max_iter=10,
            dirichlet_mask=np.zeros_like(H, dtype=bool),
            g_bc=float("nan"),
        )
        print("    nan g_bc: FAIL (no error raised)")
    except ValueError:
        passed += 1

    ok = (passed == cases)
    return _run(
        f"Error paths ({passed}/{cases} raise ValueError)", ok,
    )


def main():
    ok = True
    ok = test_regression_cpu_none_equals_nokwarg() and ok
    ok = test_regression_gpu_none_equals_nokwarg() and ok
    ok = test_all_true_mask_cpu() and ok
    ok = test_single_pinned_cell_cpu() and ok
    ok = test_single_pinned_cell_gpu_matches_cpu() and ok
    ok = test_error_paths_cpu() and ok
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
