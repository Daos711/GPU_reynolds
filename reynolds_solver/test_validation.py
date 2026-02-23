"""
Validation of all solver methods: CPU (Numba), GPU SOR, AMG (PyAMG).

Checks:
  1. Static: AMG vs CPU
  2. Dynamic: AMG vs CPU (xprime, yprime != 0)
  3. SOR vs AMG cross-validation
  4. Large grid (1000x1000) -- AMG
  5. Backward compatibility

Run:
    python -m reynolds_solver.test_validation
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# -----------------------------------------------------------------------
# CPU reference: static solver
# -----------------------------------------------------------------------
@njit
def solve_reynolds_cpu(H, d_phi, d_Z, R, L, omega=1.5, tol=1e-5, max_iter=20000):
    N_Z, N_phi = H.shape
    P = np.zeros((N_Z, N_phi))

    H_i_plus_half = 0.5 * (H[:, :-1] + H[:, 1:])
    H_i_minus_half = np.hstack((H_i_plus_half[:, -1:], H_i_plus_half[:, :-1]))
    H_j_plus_half = 0.5 * (H[:-1, :] + H[1:, :])
    H_j_minus_half = np.vstack((H_j_plus_half[-1:, :], H_j_plus_half[:-1, :]))

    D_over_L = 2 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    A = H_i_plus_half ** 3
    B = H_i_minus_half ** 3
    C = alpha_sq * H_j_plus_half ** 3
    D_coef = alpha_sq * H_j_minus_half ** 3

    A_full = np.zeros((N_Z, N_phi))
    B_full = np.zeros((N_Z, N_phi))
    C_full = np.zeros((N_Z, N_phi))
    D_full = np.zeros((N_Z, N_phi))

    A_full[:, :-1] = A
    A_full[:, -1] = A[:, 0]
    B_full[:, 1:] = B
    B_full[:, 0] = B[:, -1]
    C_full[:-1, :] = C
    C_full[-1, :] = C[0, :]
    D_full[1:, :] = D_coef
    D_full[0, :] = D_coef[-1, :]

    E = A_full + B_full + C_full + D_full

    F = d_phi * (H_i_plus_half - H_i_minus_half)
    F_full = np.zeros((N_Z, N_phi))
    F_full[:, :-1] = F
    F_full[:, -1] = F[:, 0]

    delta = 1.0
    iteration = 0
    while delta > tol and iteration < max_iter:
        delta = 0.0
        norm_P = 0.0
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                P_old_ij = P[i, j]
                P_new = (
                    A_full[i, j] * P[i, (j + 1) % N_phi]
                    + B_full[i, j] * P[i, (j - 1) % N_phi]
                    + C_full[i, j] * P[i + 1, j]
                    + D_full[i, j] * P[i - 1, j]
                    - F_full[i, j]
                ) / E[i, j]
                P_new = max(P_new, 0.0)
                P[i, j] = P_old_ij + omega * (P_new - P_old_ij)
                delta += abs(P[i, j] - P_old_ij)
                norm_P += abs(P[i, j])
        P[:, 0] = P[:, -2]
        P[:, -1] = P[:, 1]
        P[0, :] = 0.0
        P[-1, :] = 0.0
        delta /= norm_P + 1e-8
        iteration += 1
    return P, delta, iteration


# -----------------------------------------------------------------------
# CPU reference: dynamic solver
# -----------------------------------------------------------------------
@njit
def solve_reynolds_cpu_dynamic(H, d_phi, d_Z, R, L,
                                xprime=0.0, yprime=0.0, beta=2.0,
                                omega=1.5, tol=1e-5, max_iter=20000):
    N_Z, N_phi = H.shape
    P = np.zeros((N_Z, N_phi))

    H_i_plus_half = 0.5 * (H[:, :-1] + H[:, 1:])
    H_i_minus_half = np.hstack((H_i_plus_half[:, -1:], H_i_plus_half[:, :-1]))
    H_j_plus_half = 0.5 * (H[:-1, :] + H[1:, :])
    H_j_minus_half = np.vstack((H_j_plus_half[-1:, :], H_j_plus_half[:-1, :]))

    D_over_L = 2 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    A = H_i_plus_half ** 3
    B = H_i_minus_half ** 3
    C = alpha_sq * H_j_plus_half ** 3
    D_coef = alpha_sq * H_j_minus_half ** 3

    A_full = np.zeros((N_Z, N_phi))
    B_full = np.zeros((N_Z, N_phi))
    C_full = np.zeros((N_Z, N_phi))
    D_full = np.zeros((N_Z, N_phi))

    A_full[:, :-1] = A
    A_full[:, -1] = A[:, 0]
    B_full[:, 1:] = B
    B_full[:, 0] = B[:, -1]
    C_full[:-1, :] = C
    C_full[-1, :] = C[0, :]
    D_full[1:, :] = D_coef
    D_full[0, :] = D_coef[-1, :]

    E = A_full + B_full + C_full + D_full

    static_part = d_phi * (H_i_plus_half - H_i_minus_half)
    F_full = np.zeros((N_Z, N_phi))
    F_full[:, :-1] = static_part
    F_full[:, -1] = static_part[:, 0]

    for j in range(N_phi):
        phi_local = j * d_phi
        phi_global = phi_local + np.pi / 4.0
        sin_phi_global = np.sin(phi_global)
        cos_phi_global = np.cos(phi_global)
        dyn_term = beta * (xprime * sin_phi_global + yprime * cos_phi_global)
        for i in range(N_Z):
            F_full[i, j] += dyn_term

    delta = 1.0
    iteration = 0
    while delta > tol and iteration < max_iter:
        delta = 0.0
        norm_P = 0.0
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                P_old_ij = P[i, j]
                P_new = (
                    A_full[i, j] * P[i, (j + 1) % N_phi]
                    + B_full[i, j] * P[i, (j - 1) % N_phi]
                    + C_full[i, j] * P[i + 1, j]
                    + D_full[i, j] * P[i - 1, j]
                    - F_full[i, j]
                ) / E[i, j]
                if P_new < 0.0:
                    P_new = 0.0
                P[i, j] = P_old_ij + omega * (P_new - P_old_ij)
                delta += abs(P[i, j] - P_old_ij)
                norm_P += abs(P[i, j])
        P[:, 0] = P[:, -2]
        P[:, -1] = P[:, 1]
        P[0, :] = 0.0
        P[-1, :] = 0.0
        delta /= norm_P + 1e-12
        iteration += 1
    return P, delta, iteration


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def compute_loads(P, phi_1D, Z):
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    cos_phi = np.cos(Phi_mesh)
    sin_phi = np.sin(Phi_mesh)
    F_r = np.trapezoid(np.trapezoid(P * cos_phi, phi_1D, axis=1), Z)
    F_t = np.trapezoid(np.trapezoid(P * sin_phi, phi_1D, axis=1), Z)
    F_total = np.sqrt(F_r**2 + F_t**2)
    return F_r, F_t, F_total


def run_test(test_name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}")
    if details:
        print(f"         {details}")
    return passed


def generate_test_case(N, epsilon=0.6):
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z, phi_1D, Z


# -----------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------
def save_pressure_comparison_3d(P_cpu, P_gpu, phi_1D, Z, title_prefix, filename):
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z)
    fig = plt.figure(figsize=(22, 6))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(Phi_mesh, Z_mesh, P_cpu, cmap="plasma", rcount=100, ccount=100)
    ax1.set_xlabel("phi, rad")
    ax1.set_ylabel("Z")
    ax1.set_zlabel("P")
    ax1.set_title(f"{title_prefix} -- CPU (Numba)")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(Phi_mesh, Z_mesh, P_gpu, cmap="plasma", rcount=100, ccount=100)
    ax2.set_xlabel("phi, rad")
    ax2.set_ylabel("Z")
    ax2.set_zlabel("P")
    ax2.set_title(f"{title_prefix} -- AMG")

    diff = np.abs(P_cpu - P_gpu)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(Phi_mesh, Z_mesh, diff, cmap="hot", rcount=100, ccount=100)
    ax3.set_xlabel("phi, rad")
    ax3.set_ylabel("Z")
    ax3.set_zlabel("|P_cpu - P_amg|")
    ax3.set_title(f"{title_prefix} -- |Difference|")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {path}")


def save_pressure_slice(P_cpu, P_gpu, phi_1D, Z, Z_val, title_prefix, filename):
    Z_idx = np.argmin(np.abs(Z - Z_val))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(phi_1D, P_cpu[Z_idx, :], "b-", lw=1.5, label="CPU (Numba)")
    ax1.plot(phi_1D, P_gpu[Z_idx, :], "r--", lw=1.5, label="AMG")
    ax1.set_xlabel("phi, rad")
    ax1.set_ylabel("P")
    ax1.set_title(f"{title_prefix} -- P(phi) at Z = {Z_val}")
    ax1.legend()
    ax1.grid(True)

    diff = P_cpu[Z_idx, :] - P_gpu[Z_idx, :]
    ax2.plot(phi_1D, diff, "k-", lw=1.0)
    ax2.set_xlabel("phi, rad")
    ax2.set_ylabel("P_cpu - P_amg")
    ax2.set_title("Difference")
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {path}")


def save_error_heatmap(P_cpu, P_gpu, phi_1D, Z, title_prefix, filename):
    diff = np.abs(P_cpu - P_gpu)
    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(phi_1D, Z, diff, cmap="hot", shading="auto")
    fig.colorbar(c, ax=ax, label="|P_cpu - P_amg|")
    ax.set_xlabel("phi, rad")
    ax.set_ylabel("Z")
    ax.set_title(f"{title_prefix} -- Error heatmap")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {path}")


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
def test_static_amg_vs_cpu():
    """Test 1: AMG vs CPU (static)."""
    print("\n=== Test 1: Static -- AMG vs CPU ===")

    from reynolds_solver.api import solve_reynolds

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 500
    tol = 1e-5
    max_iter = 50000

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    # CPU
    print("  Solving on CPU (Numba)...")
    t0 = time.perf_counter()
    P_cpu, delta_cpu, iter_cpu = solve_reynolds_cpu(H, d_phi, d_Z, R, L, 1.5, tol, max_iter)
    t_cpu = time.perf_counter() - t0
    print(f"  CPU: {iter_cpu} iters, delta = {delta_cpu:.2e}, time = {t_cpu:.2f} s")

    # AMG
    print("  Solving with AMG...")
    t0 = time.perf_counter()
    P_amg, delta_amg, iter_amg = solve_reynolds(
        H, d_phi, d_Z, R, L, method="amg",
        amg_tol=1e-8, amg_maxiter=200, max_cav_iter=20,
    )
    t_amg = time.perf_counter() - t0
    speedup = t_cpu / t_amg if t_amg > 0 else float("inf")
    print(f"  AMG: {iter_amg} outer iters, delta = {delta_amg:.2e}, time = {t_amg:.2f} s")
    print(f"  Speedup vs CPU: {speedup:.1f}x")

    all_passed = True

    # Pressure field
    P_max = np.max(P_cpu)
    if P_max > 0:
        max_err = np.max(np.abs(P_cpu - P_amg)) / P_max
        mean_err = np.mean(np.abs(P_cpu - P_amg)) / P_max
    else:
        max_err = np.max(np.abs(P_cpu - P_amg))
        mean_err = np.mean(np.abs(P_cpu - P_amg))

    passed = max_err < 1e-3
    all_passed &= run_test(
        "Pressure field: max|P_cpu - P_amg| / max(P_cpu) < 1e-3",
        passed,
        f"max_err = {max_err:.2e}, mean_err = {mean_err:.2e}",
    )

    # Integral loads
    _, _, F_cpu = compute_loads(P_cpu, phi_1D, Z)
    _, _, F_amg = compute_loads(P_amg, phi_1D, Z)
    load_err = abs(F_cpu - F_amg) / F_cpu if F_cpu > 0 else abs(F_cpu - F_amg)

    passed = load_err < 1e-3
    all_passed &= run_test(
        "Integral load: |F_cpu - F_amg| / F_cpu < 1e-3",
        passed,
        f"F_cpu = {F_cpu:.6f}, F_amg = {F_amg:.6f}, err = {load_err:.2e}",
    )

    # Plots
    print("  Saving comparison plots...")
    save_pressure_comparison_3d(P_cpu, P_amg, phi_1D, Z, "Static AMG", "static_amg_pressure_3d.png")
    save_pressure_slice(P_cpu, P_amg, phi_1D, Z, 0.0, "Static AMG", "static_amg_slice_Z0.png")
    save_error_heatmap(P_cpu, P_amg, phi_1D, Z, "Static AMG", "static_amg_error_heatmap.png")

    return all_passed


def test_dynamic_amg_vs_cpu():
    """Test 2: AMG vs CPU (dynamic)."""
    print("\n=== Test 2: Dynamic -- AMG vs CPU ===")

    from reynolds_solver.api import solve_reynolds

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 500
    tol = 1e-5
    max_iter = 50000
    xprime = 0.001
    yprime = 0.001
    beta = 2.0

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    # CPU
    print("  Solving on CPU (Numba)...")
    t0 = time.perf_counter()
    P_cpu, delta_cpu, iter_cpu = solve_reynolds_cpu_dynamic(
        H, d_phi, d_Z, R, L,
        xprime=xprime, yprime=yprime, beta=beta,
        omega=1.5, tol=tol, max_iter=max_iter,
    )
    t_cpu = time.perf_counter() - t0
    print(f"  CPU: {iter_cpu} iters, delta = {delta_cpu:.2e}, time = {t_cpu:.2f} s")

    # AMG
    print("  Solving with AMG...")
    t0 = time.perf_counter()
    P_amg, delta_amg, iter_amg = solve_reynolds(
        H, d_phi, d_Z, R, L, method="amg",
        xprime=xprime, yprime=yprime, beta=beta,
        amg_tol=1e-8, amg_maxiter=200, max_cav_iter=20,
    )
    t_amg = time.perf_counter() - t0
    speedup = t_cpu / t_amg if t_amg > 0 else float("inf")
    print(f"  AMG: {iter_amg} outer iters, delta = {delta_amg:.2e}, time = {t_amg:.2f} s")
    print(f"  Speedup vs CPU: {speedup:.1f}x")

    all_passed = True

    P_max = np.max(P_cpu)
    if P_max > 0:
        max_err = np.max(np.abs(P_cpu - P_amg)) / P_max
        mean_err = np.mean(np.abs(P_cpu - P_amg)) / P_max
    else:
        max_err = np.max(np.abs(P_cpu - P_amg))
        mean_err = np.mean(np.abs(P_cpu - P_amg))

    passed = max_err < 1e-3
    all_passed &= run_test(
        "Pressure field (dynamic): max|P_cpu - P_amg| / max(P_cpu) < 1e-3",
        passed,
        f"max_err = {max_err:.2e}, mean_err = {mean_err:.2e}",
    )

    _, _, F_cpu = compute_loads(P_cpu, phi_1D, Z)
    _, _, F_amg = compute_loads(P_amg, phi_1D, Z)
    load_err = abs(F_cpu - F_amg) / F_cpu if F_cpu > 0 else abs(F_cpu - F_amg)

    passed = load_err < 1e-3
    all_passed &= run_test(
        "Integral load (dynamic): |F_cpu - F_amg| / F_cpu < 1e-3",
        passed,
        f"F_cpu = {F_cpu:.6f}, F_amg = {F_amg:.6f}, err = {load_err:.2e}",
    )

    print("  Saving comparison plots...")
    save_pressure_comparison_3d(P_cpu, P_amg, phi_1D, Z, "Dynamic AMG", "dynamic_amg_pressure_3d.png")
    save_pressure_slice(P_cpu, P_amg, phi_1D, Z, 0.0, "Dynamic AMG", "dynamic_amg_slice_Z0.png")
    save_error_heatmap(P_cpu, P_amg, phi_1D, Z, "Dynamic AMG", "dynamic_amg_error_heatmap.png")

    return all_passed


def test_sor_vs_amg():
    """Test 3: SOR vs AMG cross-validation."""
    print("\n=== Test 3: SOR vs AMG cross-validation ===")

    from reynolds_solver.api import solve_reynolds
    import cupy as cp

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 500
    tol = 1e-5

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    # SOR
    print("  Solving with SOR...")
    t0 = time.perf_counter()
    P_sor, delta_sor, iter_sor = solve_reynolds(H, d_phi, d_Z, R, L, method="sor", tol=tol)
    cp.cuda.Device(0).synchronize()
    t_sor = time.perf_counter() - t0
    print(f"  SOR: {iter_sor} iters, delta = {delta_sor:.2e}, time = {t_sor:.2f} s")

    # AMG
    print("  Solving with AMG...")
    t0 = time.perf_counter()
    P_amg, delta_amg, iter_amg = solve_reynolds(
        H, d_phi, d_Z, R, L, method="amg",
        amg_tol=1e-8, amg_maxiter=200,
    )
    t_amg = time.perf_counter() - t0
    print(f"  AMG: {iter_amg} outer iters, delta = {delta_amg:.2e}, time = {t_amg:.2f} s")

    all_passed = True

    P_max = np.max(P_sor)
    if P_max > 0:
        max_err = np.max(np.abs(P_sor - P_amg)) / P_max
    else:
        max_err = np.max(np.abs(P_sor - P_amg))

    passed = max_err < 5e-3
    all_passed &= run_test(
        "SOR vs AMG: max|P_sor - P_amg| / max(P_sor) < 5e-3",
        passed,
        f"max_err = {max_err:.2e}",
    )

    _, _, F_sor = compute_loads(P_sor, phi_1D, Z)
    _, _, F_amg = compute_loads(P_amg, phi_1D, Z)
    load_err = abs(F_sor - F_amg) / F_sor if F_sor > 0 else abs(F_sor - F_amg)

    passed = load_err < 1e-3
    all_passed &= run_test(
        "SOR vs AMG load: |F_sor - F_amg| / F_sor < 1e-3",
        passed,
        f"F_sor = {F_sor:.6f}, F_amg = {F_amg:.6f}, err = {load_err:.2e}",
    )

    # Timing comparison
    if t_amg > 0:
        ratio = t_sor / t_amg
        passed = ratio > 0.3  # AMG should be at least comparable to SOR
        all_passed &= run_test(
            "AMG competitive with SOR (ratio > 0.3x)",
            passed,
            f"SOR = {t_sor:.3f}s, AMG = {t_amg:.3f}s, ratio = {ratio:.1f}x",
        )

    return all_passed


def test_large_grid_amg():
    """Test 4: Large grid (1000x1000) -- AMG."""
    print("\n=== Test 4: Large grid 1000x1000 -- AMG ===")

    from reynolds_solver.api import solve_reynolds
    import cupy as cp

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 1000

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    print("  Solving 1000x1000 with AMG...")
    t0 = time.perf_counter()
    P_amg, delta_amg, iter_amg = solve_reynolds(
        H, d_phi, d_Z, R, L, method="amg",
        amg_tol=1e-8, amg_maxiter=200, max_cav_iter=20,
    )
    t_amg = time.perf_counter() - t0
    print(f"  AMG: {iter_amg} outer iters, delta = {delta_amg:.2e}, time = {t_amg:.2f} s")

    all_passed = True

    # Check non-trivial solution
    P_max = np.max(P_amg)
    passed = P_max > 0
    all_passed &= run_test(
        "Non-trivial solution: max(P) > 0",
        passed,
        f"max(P) = {P_max:.6f}",
    )

    # Check timing < 10 seconds
    passed = t_amg < 10.0
    all_passed &= run_test(
        "AMG 1000x1000 < 10 seconds",
        passed,
        f"time = {t_amg:.3f} s",
    )

    # Also run SOR for comparison
    print("  Solving 1000x1000 with SOR...")
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    P_sor, delta_sor, iter_sor = solve_reynolds(
        H, d_phi, d_Z, R, L, method="sor", tol=1e-5, max_iter=50000,
    )
    cp.cuda.Device(0).synchronize()
    t_sor = time.perf_counter() - t0
    print(f"  SOR: {iter_sor} iters, delta = {delta_sor:.2e}, time = {t_sor:.2f} s")

    # Cross-validate
    P_max_sor = np.max(P_sor)
    if P_max_sor > 0:
        max_err = np.max(np.abs(P_sor - P_amg)) / P_max_sor
    else:
        max_err = np.max(np.abs(P_sor - P_amg))

    passed = max_err < 5e-3
    all_passed &= run_test(
        "SOR vs AMG 1000x1000: max_err < 5e-3",
        passed,
        f"max_err = {max_err:.2e}",
    )

    return all_passed


def test_backward_compatibility():
    """Test 5: Old API names still work."""
    print("\n=== Test 5: Backward compatibility ===")

    all_passed = True

    try:
        from reynolds_solver import solve_reynolds_gpu, solve_reynolds_gpu_dynamic, ReynoldsSolverGPU
        passed = True
    except ImportError as e:
        passed = False
    all_passed &= run_test("Import old names", passed)

    # Quick solve
    H, d_phi, d_Z, _, _ = generate_test_case(50)
    R, L = 0.035, 0.056

    try:
        P, delta, n_iter = solve_reynolds_gpu(H, d_phi, d_Z, R, L, tol=0.1, max_iter=100)
        passed = P.shape == H.shape
    except Exception as e:
        passed = False
    all_passed &= run_test("solve_reynolds_gpu works", passed)

    try:
        P, delta, n_iter = solve_reynolds_gpu_dynamic(
            H, d_phi, d_Z, R, L, xprime=0.001, yprime=0.001, tol=0.1, max_iter=100,
        )
        passed = P.shape == H.shape
    except Exception as e:
        passed = False
    all_passed &= run_test("solve_reynolds_gpu_dynamic works", passed)

    # Test method="krylov" backward compat alias
    try:
        from reynolds_solver.api import solve_reynolds
        P, delta, n_iter = solve_reynolds(
            H, d_phi, d_Z, R, L, method="krylov",
            krylov_tol=0.1, krylov_maxiter=5, max_cav_iter=2,
        )
        passed = P.shape == H.shape
    except Exception as e:
        passed = False
    all_passed &= run_test("method='krylov' alias works", passed)

    # Test method="direct"
    try:
        P, delta, n_iter = solve_reynolds(
            H, d_phi, d_Z, R, L, method="direct", max_cav_iter=5,
        )
        passed = P.shape == H.shape and np.max(P) > 0
    except Exception as e:
        passed = False
    all_passed &= run_test("method='direct' works", passed)

    return all_passed


def main():
    print("=" * 60)
    print("  reynolds_solver validation (CPU / SOR / AMG)")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\nPlots will be saved to: {RESULTS_DIR}")

    # Numba warmup
    print("\nWarming up Numba JIT...")
    H_w = 1.0 + 0.6 * np.cos(np.linspace(0, 2*np.pi, 50)[np.newaxis, :] * np.ones((50, 1)))
    solve_reynolds_cpu(H_w, 0.1, 0.1, 0.035, 0.056, 1.5, 0.1, 10)
    solve_reynolds_cpu_dynamic(H_w, 0.1, 0.1, 0.035, 0.056,
                                xprime=0.001, yprime=0.001, beta=2.0,
                                omega=1.5, tol=0.1, max_iter=10)
    print("Numba JIT warmed up.")

    results = []
    results.append(test_static_amg_vs_cpu())
    results.append(test_dynamic_amg_vs_cpu())
    results.append(test_sor_vs_amg())
    results.append(test_large_grid_amg())
    results.append(test_backward_compatibility())

    print("\n" + "=" * 60)
    all_ok = all(results)
    if all_ok:
        print("  ALL TESTS PASSED")
    else:
        n_passed = sum(results)
        n_total = len(results)
        print(f"  {n_passed}/{n_total} TEST GROUPS PASSED")
    print("=" * 60)

    print(f"\nResults saved to: {RESULTS_DIR}/")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
