"""
Benchmark: CPU (Numba) vs GPU (CuPy Red-Black SOR) on different grid sizes.

Run:
    python -m reynolds_solver.benchmark
"""

import time
import numpy as np
from numba import njit


# -----------------------------------------------------------------------
# CPU solver (Numba) -- reference implementation
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


def generate_test_H(N_Z, N_phi, epsilon=0.6):
    """Generate test gap H = 1 + epsilon * cos(phi)."""
    phi = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)
    Phi_mesh, _ = np.meshgrid(phi, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi[1] - phi[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z


def run_benchmark():
    R = 0.035
    L = 0.056
    epsilon = 0.6
    omega_sor = 1.5
    tol = 1e-5
    max_iter = 50000

    grids = [(250, 250), (500, 500), (1000, 1000)]
    n_runs = 3

    # Numba JIT warmup
    print("Warming up Numba JIT...")
    H_warmup, dp_w, dz_w = generate_test_H(50, 50, epsilon)
    solve_reynolds_cpu(H_warmup, dp_w, dz_w, R, L, omega_sor, tol=0.1, max_iter=10)
    print("Numba JIT warmed up.\n")

    # Import GPU solver
    try:
        import cupy as cp
        from reynolds_solver.solver import solve_reynolds_gpu

        # GPU warmup (kernel compilation + allocations)
        print("Warming up GPU...")
        H_warmup_gpu, dp_w, dz_w = generate_test_H(50, 50, epsilon)
        solve_reynolds_gpu(H_warmup_gpu, dp_w, dz_w, R, L, omega_sor, tol=0.1, max_iter=10)
        cp.cuda.Device(0).synchronize()
        print("GPU warmed up.\n")
        gpu_available = True
    except Exception as e:
        print(f"GPU unavailable: {e}\n")
        gpu_available = False

    header = f"{'Grid':<12} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<10} {'Iters':<10} {'CPU delta':<14} {'GPU delta':<14}"
    print(header)
    print("-" * len(header))

    for N_Z, N_phi in grids:
        H, d_phi, d_Z = generate_test_H(N_Z, N_phi, epsilon)

        # --- CPU ---
        cpu_times = []
        cpu_delta = 0.0
        cpu_iters = 0
        for run in range(n_runs):
            t0 = time.perf_counter()
            P_cpu, cpu_delta, cpu_iters = solve_reynolds_cpu(
                H, d_phi, d_Z, R, L, omega_sor, tol, max_iter
            )
            t1 = time.perf_counter()
            cpu_times.append(t1 - t0)
        cpu_avg = np.mean(cpu_times)

        # --- GPU ---
        if gpu_available:
            gpu_times = []
            gpu_delta = 0.0
            gpu_iters = 0
            for run in range(n_runs):
                cp.cuda.Device(0).synchronize()
                t0 = time.perf_counter()
                P_gpu, gpu_delta, gpu_iters = solve_reynolds_gpu(
                    H, d_phi, d_Z, R, L, omega_sor, tol, max_iter
                )
                cp.cuda.Device(0).synchronize()
                t1 = time.perf_counter()
                gpu_times.append(t1 - t0)
            gpu_avg = np.mean(gpu_times)
            speedup = cpu_avg / gpu_avg
            gpu_str = f"{gpu_avg:.3f}"
            speedup_str = f"{speedup:.1f}x"
            gpu_delta_str = f"{gpu_delta:.2e}"
        else:
            gpu_str = "N/A"
            speedup_str = "N/A"
            gpu_delta_str = "N/A"
            gpu_iters = "-"

        grid_str = f"{N_Z}x{N_phi}"
        print(
            f"{grid_str:<12} {cpu_avg:<12.3f} {gpu_str:<12} {speedup_str:<10} "
            f"{cpu_iters:<10} {cpu_delta:<14.2e} {gpu_delta_str:<14}"
        )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    run_benchmark()
