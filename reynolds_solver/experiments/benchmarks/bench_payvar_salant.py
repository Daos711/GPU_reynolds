"""
CPU vs GPU speedup benchmark for the Payvar-Salant JFO solver.

Measures wall-clock time for both solve paths across several grid
sizes, prints a summary table with per-phase breakdown for the largest
grid, and saves results to CSV.

Run:
    python -m reynolds_solver.experiments.benchmarks.bench_payvar_salant
"""
import os
import time
import numpy as np


def generate_test_case(N_phi, N_Z, epsilon=0.6):
    phi_1D = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)
    Phi, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z


def bench_one(N_Z, N_phi, R, L, epsilon, tol, max_iter, n_runs):
    """Return dict with timing and solution statistics for one grid."""
    from reynolds_solver.cavitation.payvar_salant import (
        solve_payvar_salant_cpu,
    )

    H, d_phi, d_Z = generate_test_case(N_phi, N_Z, epsilon)

    # --- CPU ---
    cpu_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        P_cpu, th_cpu, res_cpu, n_cpu = solve_payvar_salant_cpu(
            H, d_phi, d_Z, R, L, tol=tol, max_iter=max_iter,
        )
        cpu_times.append(time.perf_counter() - t0)
    cpu_median = float(np.median(cpu_times))
    maxP_cpu = float(P_cpu.max())
    cav_cpu = float(np.mean(th_cpu[1:-1, 1:-1] < 1.0 - 1e-6))

    # --- GPU ---
    try:
        from reynolds_solver.cavitation.payvar_salant import (
            solve_payvar_salant_gpu,
        )
        import cupy as cp

        # Warmup (compile kernels)
        _ = solve_payvar_salant_gpu(
            H, d_phi, d_Z, R, L, tol=tol, max_iter=max_iter,
        )
        cp.cuda.Device().synchronize()

        gpu_times = []
        for _ in range(n_runs):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            P_gpu, th_gpu, res_gpu, n_gpu = solve_payvar_salant_gpu(
                H, d_phi, d_Z, R, L, tol=tol, max_iter=max_iter,
            )
            cp.cuda.Device().synchronize()
            gpu_times.append(time.perf_counter() - t0)
        gpu_median = float(np.median(gpu_times))
        maxP_gpu = float(P_gpu.max())
        cav_gpu = float(np.mean(th_gpu[1:-1, 1:-1] < 1.0 - 1e-6))
        gpu_available = True
    except ImportError:
        gpu_median = float("nan")
        maxP_gpu = float("nan")
        cav_gpu = float("nan")
        n_gpu = 0
        gpu_available = False

    speedup = cpu_median / gpu_median if gpu_available else float("nan")

    return dict(
        N_Z=N_Z, N_phi=N_phi,
        cpu_time=cpu_median, gpu_time=gpu_median, speedup=speedup,
        n_iter_cpu=n_cpu, n_iter_gpu=n_gpu if gpu_available else 0,
        maxP_cpu=maxP_cpu, maxP_gpu=maxP_gpu,
        cav_cpu=cav_cpu, cav_gpu=cav_gpu,
    )


def main():
    R, L = 0.035, 0.056
    epsilon = 0.6
    tol = 1e-6
    max_iter = 100000
    n_runs = 3

    grids = [
        (40, 100),
        (80, 200),
        (200, 500),
        (400, 1000),
        (800, 2000),
    ]

    print("=" * 78)
    print(f"  Payvar-Salant CPU vs GPU speedup  (ε={epsilon}, tol={tol})")
    print("=" * 78)
    print(
        f"  {'Grid (NZ×Nφ)':>14s}  {'CPU (s)':>10s}  {'GPU (s)':>10s}  "
        f"{'Speedup':>8s}  {'Iters':>7s}  {'maxP_cpu':>10s}  "
        f"{'cav_cpu':>8s}"
    )

    results = []
    for N_Z, N_phi in grids:
        print(f"  Running {N_Z}×{N_phi}...", end="", flush=True)
        try:
            r = bench_one(N_Z, N_phi, R, L, epsilon, tol, max_iter, n_runs)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        results.append(r)
        print(
            f"\r  {N_Z:>4d} × {N_phi:<5d}  "
            f"{r['cpu_time']:>10.3f}  {r['gpu_time']:>10.3f}  "
            f"{r['speedup']:>7.1f}x  "
            f"{r['n_iter_cpu']:>7d}  {r['maxP_cpu']:>10.4e}  "
            f"{r['cav_cpu']:>8.3f}"
        )

    print("=" * 78)

    # Save CSV
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "results", "benchmarks"
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "payvar_salant_speedup.csv")
    with open(csv_path, "w") as f:
        cols = [
            "N_Z", "N_phi", "cpu_time", "gpu_time", "speedup",
            "n_iter_cpu", "n_iter_gpu",
            "maxP_cpu", "maxP_gpu", "cav_cpu", "cav_gpu",
        ]
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\n  CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
