"""
Validation of GPU solver: comparison of GPU (CuPy Red-Black SOR) vs CPU (Numba).

Checks:
  1. Static solver  -- pressure field + integral loads + timing
  2. Dynamic solver -- same with xprime, yprime != 0
  3. Multi-grid benchmark -- 250x250, 500x500, 1000x1000

Saves comparison plots and benchmark chart to results/ directory.

Run:
    python -m gpu_reynolds.test_validation
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


# ───────────────────────────────────────────────────────────────────────────
# CPU reference: static solver
# ───────────────────────────────────────────────────────────────────────────
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


# ───────────────────────────────────────────────────────────────────────────
# CPU reference: dynamic solver
# ───────────────────────────────────────────────────────────────────────────
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


# ───────────────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────────────
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
    """Generate H, d_phi, d_Z, phi_1D, Z for NxN grid."""
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z, phi_1D, Z


# ───────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ───────────────────────────────────────────────────────────────────────────
def save_pressure_comparison_3d(P_cpu, P_gpu, phi_1D, Z, title_prefix, filename):
    """3D surface plots: CPU, GPU, |difference|."""
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
    ax2.set_title(f"{title_prefix} -- GPU (CuPy)")

    diff = np.abs(P_cpu - P_gpu)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_surface(Phi_mesh, Z_mesh, diff, cmap="hot", rcount=100, ccount=100)
    ax3.set_xlabel("phi, rad")
    ax3.set_ylabel("Z")
    ax3.set_zlabel("|P_cpu - P_gpu|")
    ax3.set_title(f"{title_prefix} -- |Difference|")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {path}")


def save_pressure_slice(P_cpu, P_gpu, phi_1D, Z, Z_val, title_prefix, filename):
    """1D slice P(phi) at given Z value: CPU vs GPU + difference."""
    Z_idx = np.argmin(np.abs(Z - Z_val))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(phi_1D, P_cpu[Z_idx, :], "b-", lw=1.5, label="CPU (Numba)")
    ax1.plot(phi_1D, P_gpu[Z_idx, :], "r--", lw=1.5, label="GPU (CuPy)")
    ax1.set_xlabel("phi, rad")
    ax1.set_ylabel("P")
    ax1.set_title(f"{title_prefix} -- P(phi) at Z = {Z_val}")
    ax1.legend()
    ax1.grid(True)

    diff = P_cpu[Z_idx, :] - P_gpu[Z_idx, :]
    ax2.plot(phi_1D, diff, "k-", lw=1.0)
    ax2.set_xlabel("phi, rad")
    ax2.set_ylabel("P_cpu - P_gpu")
    ax2.set_title("Difference")
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {path}")


def save_error_heatmap(P_cpu, P_gpu, phi_1D, Z, title_prefix, filename):
    """2D heatmap of |P_cpu - P_gpu|."""
    diff = np.abs(P_cpu - P_gpu)

    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(phi_1D, Z, diff, cmap="hot", shading="auto")
    fig.colorbar(c, ax=ax, label="|P_cpu - P_gpu|")
    ax.set_xlabel("phi, rad")
    ax.set_ylabel("Z")
    ax.set_title(f"{title_prefix} -- Error heatmap")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {path}")


def save_benchmark_chart(grid_labels, cpu_times, gpu_times, speedups, filename):
    """Bar chart: CPU vs GPU time + speedup annotations."""
    x = np.arange(len(grid_labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    bars_cpu = ax1.bar(x - width/2, cpu_times, width, label="CPU (Numba)", color="#4477AA")
    bars_gpu = ax1.bar(x + width/2, gpu_times, width, label="GPU (CuPy)", color="#EE6677")

    ax1.set_xlabel("Grid size")
    ax1.set_ylabel("Time, s")
    ax1.set_title("CPU vs GPU solver performance")
    ax1.set_xticks(x)
    ax1.set_xticklabels(grid_labels)
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, axis="y", alpha=0.3)

    # Annotate speedup
    for i, sp in enumerate(speedups):
        y_max = max(cpu_times[i], gpu_times[i])
        ax1.annotate(
            f"{sp:.0f}x",
            xy=(x[i], y_max),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center", fontsize=14, fontweight="bold", color="#228833",
        )

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  -> Saved: {path}")


# ───────────────────────────────────────────────────────────────────────────
# Tests
# ───────────────────────────────────────────────────────────────────────────
def test_static_solver():
    print("\n=== Test 1: Static solver ===")

    from gpu_reynolds.solver import solve_reynolds_gpu
    import cupy as cp

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 500
    omega_sor = 1.5
    tol = 1e-5
    max_iter = 50000

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    # --- CPU ---
    print("  Solving on CPU (Numba)...")
    t0 = time.perf_counter()
    P_cpu, delta_cpu, iter_cpu = solve_reynolds_cpu(
        H, d_phi, d_Z, R, L, omega_sor, tol, max_iter
    )
    t_cpu = time.perf_counter() - t0
    print(f"  CPU: {iter_cpu} iters, delta = {delta_cpu:.2e}, time = {t_cpu:.2f} s")

    # --- GPU ---
    print("  Solving on GPU (CuPy)...")
    # warmup GPU (kernel compilation)
    solve_reynolds_gpu(H, d_phi, d_Z, R, L, omega_sor, tol=0.1, max_iter=10, check_every=5)
    cp.cuda.Device(0).synchronize()

    t0 = time.perf_counter()
    P_gpu, delta_gpu, iter_gpu = solve_reynolds_gpu(
        H, d_phi, d_Z, R, L, omega_sor, tol, max_iter
    )
    cp.cuda.Device(0).synchronize()
    t_gpu = time.perf_counter() - t0
    speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
    print(f"  GPU: {iter_gpu} iters, delta = {delta_gpu:.2e}, time = {t_gpu:.2f} s")
    print(f"  Speedup: {speedup:.1f}x")

    all_passed = True

    # Test 1a: pressure field
    P_max = np.max(P_cpu)
    if P_max > 0:
        max_err = np.max(np.abs(P_cpu - P_gpu)) / P_max
        mean_err = np.mean(np.abs(P_cpu - P_gpu)) / P_max
    else:
        max_err = np.max(np.abs(P_cpu - P_gpu))
        mean_err = np.mean(np.abs(P_cpu - P_gpu))

    passed = max_err < 2e-3
    all_passed &= run_test(
        "Pressure field: max|P_cpu - P_gpu| / max(P_cpu) < 2e-3",
        passed,
        f"max_err = {max_err:.2e}, mean_err = {mean_err:.2e}"
    )

    # Test 1b: integral loads
    _, _, F_cpu = compute_loads(P_cpu, phi_1D, Z)
    _, _, F_gpu = compute_loads(P_gpu, phi_1D, Z)
    load_err = abs(F_cpu - F_gpu) / F_cpu if F_cpu > 0 else abs(F_cpu - F_gpu)

    passed = load_err < 1e-3
    all_passed &= run_test(
        "Integral load: |F_cpu - F_gpu| / F_cpu < 1e-3",
        passed,
        f"F_cpu = {F_cpu:.6f}, F_gpu = {F_gpu:.6f}, err = {load_err:.2e}"
    )

    # --- Save plots ---
    print("  Saving comparison plots...")
    save_pressure_comparison_3d(P_cpu, P_gpu, phi_1D, Z, "Static", "static_pressure_3d.png")
    save_pressure_slice(P_cpu, P_gpu, phi_1D, Z, 0.0, "Static", "static_pressure_slice_Z0.png")
    save_error_heatmap(P_cpu, P_gpu, phi_1D, Z, "Static", "static_error_heatmap.png")

    return all_passed


def test_dynamic_solver():
    print("\n=== Test 2: Dynamic solver (xprime=0.001, yprime=0.001) ===")

    from gpu_reynolds.solver_dynamic import solve_reynolds_gpu_dynamic
    import cupy as cp

    R = 0.035
    L = 0.056
    epsilon = 0.6
    N = 500
    omega_sor = 1.5
    tol = 1e-5
    max_iter = 50000
    xprime = 0.001
    yprime = 0.001
    beta = 2.0

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

    # --- CPU ---
    print("  Solving on CPU (Numba)...")
    t0 = time.perf_counter()
    P_cpu, delta_cpu, iter_cpu = solve_reynolds_cpu_dynamic(
        H, d_phi, d_Z, R, L,
        xprime=xprime, yprime=yprime, beta=beta,
        omega=omega_sor, tol=tol, max_iter=max_iter
    )
    t_cpu = time.perf_counter() - t0
    print(f"  CPU: {iter_cpu} iters, delta = {delta_cpu:.2e}, time = {t_cpu:.2f} s")

    # --- GPU ---
    print("  Solving on GPU (CuPy)...")
    # warmup
    solve_reynolds_gpu_dynamic(H, d_phi, d_Z, R, L,
                                xprime=xprime, yprime=yprime, beta=beta,
                                omega=omega_sor, tol=0.1, max_iter=10, check_every=5)
    cp.cuda.Device(0).synchronize()

    t0 = time.perf_counter()
    P_gpu, delta_gpu, iter_gpu = solve_reynolds_gpu_dynamic(
        H, d_phi, d_Z, R, L,
        xprime=xprime, yprime=yprime, beta=beta,
        omega=omega_sor, tol=tol, max_iter=max_iter
    )
    cp.cuda.Device(0).synchronize()
    t_gpu = time.perf_counter() - t0
    speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
    print(f"  GPU: {iter_gpu} iters, delta = {delta_gpu:.2e}, time = {t_gpu:.2f} s")
    print(f"  Speedup: {speedup:.1f}x")

    all_passed = True

    # Test 2a: pressure field
    P_max = np.max(P_cpu)
    if P_max > 0:
        max_err = np.max(np.abs(P_cpu - P_gpu)) / P_max
        mean_err = np.mean(np.abs(P_cpu - P_gpu)) / P_max
    else:
        max_err = np.max(np.abs(P_cpu - P_gpu))
        mean_err = np.mean(np.abs(P_cpu - P_gpu))

    passed = max_err < 2e-3
    all_passed &= run_test(
        "Pressure field (dynamic): max|P_cpu - P_gpu| / max(P_cpu) < 2e-3",
        passed,
        f"max_err = {max_err:.2e}, mean_err = {mean_err:.2e}"
    )

    # Test 2b: integral loads
    _, _, F_cpu = compute_loads(P_cpu, phi_1D, Z)
    _, _, F_gpu = compute_loads(P_gpu, phi_1D, Z)
    load_err = abs(F_cpu - F_gpu) / F_cpu if F_cpu > 0 else abs(F_cpu - F_gpu)

    passed = load_err < 1e-3
    all_passed &= run_test(
        "Integral load (dynamic): |F_cpu - F_gpu| / F_cpu < 1e-3",
        passed,
        f"F_cpu = {F_cpu:.6f}, F_gpu = {F_gpu:.6f}, err = {load_err:.2e}"
    )

    # --- Save plots ---
    print("  Saving comparison plots...")
    save_pressure_comparison_3d(P_cpu, P_gpu, phi_1D, Z, "Dynamic", "dynamic_pressure_3d.png")
    save_pressure_slice(P_cpu, P_gpu, phi_1D, Z, 0.0, "Dynamic", "dynamic_pressure_slice_Z0.png")
    save_error_heatmap(P_cpu, P_gpu, phi_1D, Z, "Dynamic", "dynamic_error_heatmap.png")

    return all_passed


def run_benchmark():
    """Benchmark on 250x250, 500x500, 1000x1000 grids."""
    print("\n" + "=" * 60)
    print("  Benchmark: CPU vs GPU on multiple grid sizes")
    print("=" * 60)

    from gpu_reynolds.solver import solve_reynolds_gpu
    import cupy as cp

    R = 0.035
    L = 0.056
    epsilon = 0.6
    omega_sor = 1.5
    tol = 1e-5
    max_iter = 50000

    grids = [250, 500, 1000]
    grid_labels = []
    cpu_times = []
    gpu_times = []
    speedups = []

    # GPU warmup
    H_w, dp_w, dz_w, _, _ = generate_test_case(50, epsilon)
    solve_reynolds_gpu(H_w, dp_w, dz_w, R, L, omega_sor, tol=0.1, max_iter=10, check_every=5)
    cp.cuda.Device(0).synchronize()

    header = f"{'Grid':<12} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<10} {'CPU iters':<12} {'GPU iters':<12}"
    print(f"\n{header}")
    print("-" * len(header))

    for N in grids:
        H, d_phi, d_Z, _, _ = generate_test_case(N, epsilon)
        label = f"{N}x{N}"
        grid_labels.append(label)

        # CPU
        t0 = time.perf_counter()
        _, delta_cpu, iter_cpu = solve_reynolds_cpu(H, d_phi, d_Z, R, L, omega_sor, tol, max_iter)
        t_cpu = time.perf_counter() - t0

        # GPU
        cp.cuda.Device(0).synchronize()
        t0 = time.perf_counter()
        _, delta_gpu, iter_gpu = solve_reynolds_gpu(H, d_phi, d_Z, R, L, omega_sor, tol, max_iter)
        cp.cuda.Device(0).synchronize()
        t_gpu = time.perf_counter() - t0

        sp = t_cpu / t_gpu if t_gpu > 0 else float("inf")
        cpu_times.append(t_cpu)
        gpu_times.append(t_gpu)
        speedups.append(sp)

        print(f"{label:<12} {t_cpu:<12.2f} {t_gpu:<12.2f} {sp:<10.1f}x {iter_cpu:<12} {iter_gpu:<12}")

    # Save chart
    save_benchmark_chart(grid_labels, cpu_times, gpu_times, speedups, "benchmark_cpu_vs_gpu.png")

    return grid_labels, cpu_times, gpu_times, speedups


def main():
    print("=" * 60)
    print("  GPU Reynolds solver validation")
    print("=" * 60)

    # Create results directory
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
    results.append(test_static_solver())
    results.append(test_dynamic_solver())

    # Multi-grid benchmark
    run_benchmark()

    print("\n" + "=" * 60)
    all_ok = all(results)
    if all_ok:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 60)

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print("  static_pressure_3d.png         -- 3D: CPU vs GPU vs |diff|")
    print("  static_pressure_slice_Z0.png   -- 1D: P(phi) at Z=0")
    print("  static_error_heatmap.png       -- 2D: error heatmap")
    print("  dynamic_pressure_3d.png        -- 3D: CPU vs GPU vs |diff|")
    print("  dynamic_pressure_slice_Z0.png  -- 1D: P(phi) at Z=0")
    print("  dynamic_error_heatmap.png      -- 2D: error heatmap")
    print("  benchmark_cpu_vs_gpu.png       -- Bar chart: timing + speedup")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
