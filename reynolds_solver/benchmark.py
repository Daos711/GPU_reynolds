"""
Benchmark: CPU (Numba) vs GPU SOR vs AMG (PyAMG) on multiple grid sizes.

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

    # Import GPU solvers
    try:
        import cupy as cp
        from reynolds_solver.linear_solvers.gpu_sor import solve_reynolds_sor
        from reynolds_solver.api import solve_reynolds

        # GPU warmup
        print("Warming up GPU...")
        H_warmup_gpu, dp_w, dz_w = generate_test_H(50, 50, epsilon)
        solve_reynolds_sor(H_warmup_gpu, dp_w, dz_w, R, L, omega_sor, tol=0.1, max_iter=10)
        cp.cuda.Device(0).synchronize()
        print("GPU warmed up.\n")

        # AMG warmup
        print("Warming up AMG...")
        solve_reynolds(H_warmup_gpu, dp_w, dz_w, R, L, method="amg",
                       amg_tol=0.1, amg_maxiter=5, max_cav_iter=2)
        print("AMG warmed up.\n")

        gpu_available = True
    except Exception as e:
        print(f"GPU unavailable: {e}\n")
        gpu_available = False

    header = (
        f"{'Grid':<12} {'CPU (s)':<12} {'SOR (s)':<12} {'AMG (s)':<12} "
        f"{'SOR iters':<12} {'AMG iters':<14} {'Best speedup':<14}"
    )
    print(header)
    print("-" * len(header))

    skip_cpu_threshold = 500  # Skip CPU for grids larger than this

    for N_Z, N_phi in grids:
        H, d_phi, d_Z = generate_test_H(N_Z, N_phi, epsilon)
        grid_str = f"{N_Z}x{N_phi}"

        # --- CPU ---
        if N_Z <= skip_cpu_threshold:
            cpu_times = []
            for run in range(n_runs):
                t0 = time.perf_counter()
                P_cpu, cpu_delta, cpu_iters = solve_reynolds_cpu(
                    H, d_phi, d_Z, R, L, omega_sor, tol, max_iter
                )
                t1 = time.perf_counter()
                cpu_times.append(t1 - t0)
            cpu_avg = np.mean(cpu_times)
            cpu_str = f"{cpu_avg:.2f}"
        else:
            cpu_avg = None
            cpu_str = "(skip)"

        # --- GPU SOR ---
        if gpu_available:
            import cupy as cp
            from reynolds_solver.linear_solvers.gpu_sor import solve_reynolds_sor

            sor_times = []
            sor_iters = 0
            for run in range(n_runs):
                cp.cuda.Device(0).synchronize()
                t0 = time.perf_counter()
                _, sor_delta, sor_iters = solve_reynolds_sor(
                    H, d_phi, d_Z, R, L, omega_sor, tol, max_iter
                )
                cp.cuda.Device(0).synchronize()
                t1 = time.perf_counter()
                sor_times.append(t1 - t0)
            sor_avg = np.mean(sor_times)
            sor_str = f"{sor_avg:.3f}"
        else:
            sor_avg = None
            sor_str = "N/A"
            sor_iters = "-"

        # --- AMG ---
        if gpu_available:
            from reynolds_solver.api import solve_reynolds

            amg_times = []
            amg_iters = 0
            for run in range(n_runs):
                t0 = time.perf_counter()
                _, amg_delta, amg_iters = solve_reynolds(
                    H, d_phi, d_Z, R, L, method="amg", tol=tol,
                    amg_tol=1e-8, amg_maxiter=200, max_cav_iter=20,
                )
                t1 = time.perf_counter()
                amg_times.append(t1 - t0)
            amg_avg = np.mean(amg_times)
            amg_str = f"{amg_avg:.3f}"
        else:
            amg_avg = None
            amg_str = "N/A"
            amg_iters = "-"

        # Best speedup
        if cpu_avg is not None and amg_avg is not None and amg_avg > 0:
            best_speedup = cpu_avg / amg_avg
            speedup_str = f"{best_speedup:.0f}x"
        elif cpu_avg is not None and sor_avg is not None and sor_avg > 0:
            best_speedup = cpu_avg / sor_avg
            speedup_str = f"{best_speedup:.0f}x"
        elif sor_avg is not None and amg_avg is not None:
            ratio = sor_avg / amg_avg if amg_avg > 0 else 0
            speedup_str = f"SOR/AMG={ratio:.1f}x"
        else:
            speedup_str = "N/A"

        print(
            f"{grid_str:<12} {cpu_str:<12} {sor_str:<12} {amg_str:<12} "
            f"{str(sor_iters):<12} {str(amg_iters):<14} {speedup_str:<14}"
        )

    print("\nBenchmark complete.")


if __name__ == "__main__":
    run_benchmark()
