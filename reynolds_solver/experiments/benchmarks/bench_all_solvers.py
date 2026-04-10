"""
Compare all working Reynolds solver modes on a single bearing case.

Compares wall-clock time and GPU speedup for:
  - Half-Sommerfeld (GPU)
  - Payvar-Salant steady JFO (CPU + GPU)

Run:
    python -m reynolds_solver.experiments.benchmarks.bench_all_solvers
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
    return H, d_phi, d_Z, phi_1D, Z


def main():
    R, L = 0.035, 0.056
    epsilon = 0.6
    tol = 1e-6
    N_Z, N_phi = 400, 1000
    n_runs = 3

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N_phi, N_Z, epsilon)

    print("=" * 72)
    print(
        f"  All-solver comparison  "
        f"({N_Z}×{N_phi} grid, ε={epsilon}, tol={tol})"
    )
    print("=" * 72)
    print(
        f"  {'Solver':<30s}  {'Time (s)':>10s}  {'maxP':>10s}  "
        f"{'cav_frac':>9s}  {'Iters':>7s}"
    )

    results = []

    # --- Half-Sommerfeld (GPU) ---
    try:
        from reynolds_solver import solve_reynolds
        import cupy as cp

        # Warmup
        _ = solve_reynolds(H, d_phi, d_Z, R, L)
        cp.cuda.Device().synchronize()

        times = []
        for _ in range(n_runs):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            P_hs, delta_hs, n_hs = solve_reynolds(
                H, d_phi, d_Z, R, L, tol=tol,
            )
            cp.cuda.Device().synchronize()
            times.append(time.perf_counter() - t0)
        t_hs = float(np.median(times))
        print(
            f"  {'HS (half_sommerfeld, GPU)':<30s}  {t_hs:>10.3f}  "
            f"{P_hs.max():>10.4e}  {'n/a':>9s}  {n_hs:>7d}"
        )
        results.append(("HS_GPU", t_hs, float(P_hs.max()), None, n_hs))
    except Exception as e:
        print(f"  HS GPU: FAILED ({e})")

    # --- Payvar-Salant CPU ---
    try:
        from reynolds_solver.cavitation.payvar_salant import (
            solve_payvar_salant_cpu,
        )

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            P_ps_c, th_ps_c, res_c, n_c = solve_payvar_salant_cpu(
                H, d_phi, d_Z, R, L, tol=tol, max_iter=100000,
            )
            times.append(time.perf_counter() - t0)
        t_ps_cpu = float(np.median(times))
        cav_c = float(np.mean(th_ps_c[1:-1, 1:-1] < 1.0 - 1e-6))
        print(
            f"  {'PS JFO (CPU)':<30s}  {t_ps_cpu:>10.3f}  "
            f"{P_ps_c.max():>10.4e}  {cav_c:>9.3f}  {n_c:>7d}"
        )
        results.append(("PS_CPU", t_ps_cpu, float(P_ps_c.max()), cav_c, n_c))
    except Exception as e:
        print(f"  PS CPU: FAILED ({e})")

    # --- Payvar-Salant GPU ---
    try:
        from reynolds_solver.cavitation.payvar_salant import (
            solve_payvar_salant_gpu,
        )
        import cupy as cp

        # Warmup
        _ = solve_payvar_salant_gpu(H, d_phi, d_Z, R, L, tol=tol)
        cp.cuda.Device().synchronize()

        times = []
        for _ in range(n_runs):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            P_ps_g, th_ps_g, res_g, n_g = solve_payvar_salant_gpu(
                H, d_phi, d_Z, R, L, tol=tol, max_iter=100000,
            )
            cp.cuda.Device().synchronize()
            times.append(time.perf_counter() - t0)
        t_ps_gpu = float(np.median(times))
        cav_g = float(np.mean(th_ps_g[1:-1, 1:-1] < 1.0 - 1e-6))
        print(
            f"  {'PS JFO (GPU)':<30s}  {t_ps_gpu:>10.3f}  "
            f"{P_ps_g.max():>10.4e}  {cav_g:>9.3f}  {n_g:>7d}"
        )
        results.append(("PS_GPU", t_ps_gpu, float(P_ps_g.max()), cav_g, n_g))
    except Exception as e:
        print(f"  PS GPU: FAILED ({e})")

    print("=" * 72)

    # Speedup summary
    t_map = {r[0]: r[1] for r in results}
    if "PS_CPU" in t_map and "PS_GPU" in t_map:
        print(
            f"\n  PS speedup (CPU→GPU): "
            f"{t_map['PS_CPU']:.1f}s → {t_map['PS_GPU']:.1f}s = "
            f"{t_map['PS_CPU'] / t_map['PS_GPU']:.1f}x"
        )

    # CSV
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "results", "benchmarks"
    ))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "all_solvers_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("solver,time_s,maxP,cav_frac,n_iter\n")
        for name, t, mp, cf, ni in results:
            f.write(f"{name},{t:.4f},{mp},{cf},{ni}\n")
    print(f"  CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
