"""
Phase 4.1 profiling harness for the dynamic journal-bearing solver.

Instruments the inner loop of `solve_ausas_journal_dynamic_gpu` with
cudaEvent-based timers for each logical component:

  * Forces         (WX, WY reductions)
  * Predictor      (Newmark X_k, Y_k)
  * Gap rebuild    (build_gap_inplace kernel)
  * Coeffs rebuild (build_coefficients_inplace kernel)
  * Sweep          (RB red + black kernels OR Jacobi + swap)
  * BC             (phi + z BC kernels)
  * Residual       (norms + host sync when measured)
  * Snapshot       (P_new <- P_old / theta_new <- theta_old, RB only)

Because the production code does NOT want to carry timing overhead
inside the hot loop, this harness runs a SEPARATE, mirror-copy inner
loop with identical semantics but with timers injected. Both code
paths are kept in sync — any change to the production solver must
be reflected here.

Usage:
    python -m reynolds_solver.cavitation.ausas.profiling \\
        [--N1 100 --N2 12 --dt 2e-3 --NT 1500]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional


def _fmt_s(t_s: float) -> str:
    if t_s < 1e-3:
        return f"{t_s*1e6:7.1f} us"
    if t_s < 1.0:
        return f"{t_s*1e3:7.2f} ms"
    return f"{t_s:7.3f} s"


@dataclass
class ProfileReport:
    wall_time_s: float
    total_inner_iters: int
    per_step_iters_mean: float
    per_step_iters_max: int
    breakdown_s: dict
    breakdown_pct: dict
    settings: dict


def run_profiled(
    N1: int = 100,
    N2: int = 12,
    dt: float = 2e-3,
    NT: int = 1500,
    p_a: float = 0.0075,
    B_width: float = 0.1,
    mass_M: float = 1e-6,
    X0: float = 0.5,
    Y0: float = 0.5,
    U0: float = 0.0,
    V0: float = 0.0,
    alpha: float = 1.0,
    omega_p: float = 1.0,
    omega_theta: float = 1.0,
    tol_inner: float = 1e-6,
    max_inner: int = 5000,
    check_every: int = 10,
    verbose: bool = False,
) -> ProfileReport:
    """
    Runs the full journal-bearing benchmark but accumulates per-component
    wall-time using CUDA events (which measure GPU kernel time after
    implicit sync). Reports a breakdown and a total.
    """
    # Lazy imports so the module is importable on CPU-only machines.
    import numpy as np
    import cupy as cp

    from reynolds_solver.cavitation.ausas.kernels_dynamic import (
        get_unsteady_ausas_rb_kernel,
        get_unsteady_ausas_bc_phi_kernel,
        get_unsteady_ausas_bc_z_kernel,
        get_build_coefficients_kernel,
        get_build_gap_kernel,
    )
    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        _apply_bc_python, _launch_bc, _launch_configs, _build_gap_inplace,
    )
    from reynolds_solver.cavitation.ausas.benchmark_dynamic_journal import (
        journal_load,
    )

    # -- Layout / constants ------------------------------------------------
    N_phi = N1 + 2
    N_Z = N2 + 2
    d_phi = 1.0 / N1
    d_Z = B_width / N2
    R = 0.5
    L = 1.0
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2
    dt_sq_half = (dt * dt) / (2.0 * mass_M)
    dt_over_M = dt / mass_M
    d_phi_d_Z = d_phi * d_Z
    shape = (N_Z, N_phi)

    # -- Buffers -----------------------------------------------------------
    k_arr = cp.arange(N_phi, dtype=cp.float64) - 1.0
    phi_vec = k_arr * d_phi
    cos_phi_1d = cp.cos(2.0 * np.pi * phi_vec)
    sin_phi_1d = cp.sin(2.0 * np.pi * phi_vec)
    cos_phi_interior = cos_phi_1d[1:-1]
    sin_phi_interior = sin_phi_1d[1:-1]

    H_prev = cp.empty(shape, dtype=cp.float64)
    H_curr = cp.empty(shape, dtype=cp.float64)
    C_prev = cp.empty(shape, dtype=cp.float64)
    P_old = cp.empty(shape, dtype=cp.float64)
    P_new = cp.empty(shape, dtype=cp.float64)
    theta_old = cp.empty(shape, dtype=cp.float64)
    theta_new = cp.empty(shape, dtype=cp.float64)
    A = cp.empty(shape, dtype=cp.float64)
    B = cp.empty(shape, dtype=cp.float64)
    C = cp.empty(shape, dtype=cp.float64)
    D = cp.empty(shape, dtype=cp.float64)
    E = cp.empty(shape, dtype=cp.float64)
    texture_for_kernel = cp.zeros(shape, dtype=cp.float64)

    _build_gap_inplace(H_prev, X0, Y0, cos_phi_1d, sin_phi_1d, None)
    P_old[:] = 0.0
    theta_old[:] = 1.0
    _apply_bc_python(
        P_old, theta_old, True, False,
        0.0, 0.0, 1.0, 1.0, 0.0, p_a, 1.0, 1.0,
    )
    P_new[:] = P_old
    theta_new[:] = theta_old

    # -- Kernels / launch config ------------------------------------------
    rb_kernel = get_unsteady_ausas_rb_kernel()
    bc_phi_kernel = get_unsteady_ausas_bc_phi_kernel()
    bc_z_kernel = get_unsteady_ausas_bc_z_kernel()
    build_coeffs_kernel = get_build_coefficients_kernel()
    build_gap_kernel = get_build_gap_kernel()
    block, grid, bc_block, bc_grid_phi, bc_grid_z = _launch_configs(N_Z, N_phi)
    full_grid = (
        (N_phi + block[0] - 1) // block[0],
        (N_Z + block[1] - 1) // block[1],
        1,
    )

    # -- State ------------------------------------------------------------
    X_dev = cp.array(float(X0), dtype=cp.float64)
    Y_dev = cp.array(float(Y0), dtype=cp.float64)
    U_dev = cp.array(float(U0), dtype=cp.float64)
    V_dev = cp.array(float(V0), dtype=cp.float64)
    WX_dev = cp.array(0.0, dtype=cp.float64)
    WY_dev = cp.array(0.0, dtype=cp.float64)

    # -- Timing buckets ---------------------------------------------------
    # Use a pair of events per bucket; accumulate elapsed time in ms.
    buckets = [
        "forces", "predictor", "gap", "coeffs",
        "sweep_rb", "bc", "residual", "snapshot",
    ]
    totals_ms = {k: 0.0 for k in buckets}
    starts = {k: cp.cuda.Event() for k in buckets}
    stops  = {k: cp.cuda.Event() for k in buckets}

    def tick(name: str):
        starts[name].record()

    def tock(name: str):
        stops[name].record()
        stops[name].synchronize()
        totals_ms[name] += cp.cuda.get_elapsed_time(starts[name], stops[name])

    # -- Run --------------------------------------------------------------
    n_inner_total = 0
    n_inner_max = 0
    wall_t0 = time.perf_counter()

    for n in range(1, NT + 1):
        t_n = n * dt
        WaX_n, WaY_n = journal_load(t_n)
        WaX_n = float(WaX_n); WaY_n = float(WaY_n)

        cp.multiply(theta_old, H_prev, out=C_prev)

        X_k_dev = X_dev
        Y_k_dev = Y_dev
        X_k_prev_dev = X_dev
        Y_k_prev_dev = Y_dev

        converged = False
        k_done = 0

        for k in range(max_inner):
            check_iter = (k % check_every == 0) or (k < 3)

            if check_iter:
                tick("snapshot")
                P_new[:] = P_old
                theta_new[:] = theta_old
                tock("snapshot")

            tick("forces")
            WX_dev = d_phi_d_Z * cp.sum(
                P_old[1:-1, 1:-1] * cos_phi_interior[None, :]
            )
            WY_dev = d_phi_d_Z * cp.sum(
                P_old[1:-1, 1:-1] * sin_phi_interior[None, :]
            )
            tock("forces")

            tick("predictor")
            X_k_dev = X_dev + dt * U_dev + dt_sq_half * (WX_dev + WaX_n)
            Y_k_dev = Y_dev + dt * V_dev + dt_sq_half * (WY_dev + WaY_n)
            tock("predictor")

            tick("gap")
            build_gap_kernel(
                full_grid, block,
                (H_curr, X_k_dev, Y_k_dev, cos_phi_1d, sin_phi_1d,
                 texture_for_kernel, np.int32(N_Z), np.int32(N_phi),
                 np.int32(0)),
            )
            tock("gap")

            tick("coeffs")
            build_coeffs_kernel(
                full_grid, block,
                (H_curr, A, B, C, D, E,
                 np.float64(alpha_sq),
                 np.int32(N_Z), np.int32(N_phi)),
            )
            tock("coeffs")

            tick("sweep_rb")
            for color in (0, 1):
                rb_kernel(
                    grid, block,
                    (P_old, theta_old,
                     H_curr, C_prev, A, B, C, D, E,
                     np.float64(d_phi), np.float64(d_Z),
                     np.float64(dt), np.float64(alpha),
                     np.float64(omega_p), np.float64(omega_theta),
                     np.int32(N_Z), np.int32(N_phi),
                     np.int32(1), np.int32(0),
                     np.int32(color)),
                )
                tick("bc")
                _launch_bc(
                    bc_phi_kernel, bc_z_kernel,
                    P_old, theta_old, N_Z, N_phi,
                    bc_block, bc_grid_phi, bc_grid_z,
                    True, False,
                    0.0, 0.0, 1.0, 1.0,
                    0.0, p_a, 1.0, 1.0,
                )
                tock("bc")
            tock("sweep_rb")
            k_done = k + 1

            if check_iter:
                tick("residual")
                dP_dev = cp.sqrt(cp.sum((P_old - P_new) ** 2))
                dth_dev = cp.sqrt(cp.sum((theta_old - theta_new) ** 2))
                dX_dev = cp.abs(X_k_dev - X_k_prev_dev)
                dY_dev = cp.abs(Y_k_dev - Y_k_prev_dev)
                residual = float(dP_dev + dth_dev + dX_dev + dY_dev)
                tock("residual")
                if residual < tol_inner and k > 2:
                    converged = True

            X_k_prev_dev = X_k_dev
            Y_k_prev_dev = Y_k_dev

            if converged:
                break

        n_inner_total += k_done
        n_inner_max = max(n_inner_max, k_done)

        U_dev = U_dev + dt_over_M * (WX_dev + WaX_n)
        V_dev = V_dev + dt_over_M * (WY_dev + WaY_n)
        X_dev = X_k_dev
        Y_dev = Y_k_dev
        H_prev, H_curr = H_curr, H_prev

        if verbose and (n <= 3 or n % max(NT // 10, 1) == 0):
            e = float(cp.sqrt(X_dev * X_dev + Y_dev * Y_dev))
            print(f"  step {n}/{NT}: inner={k_done}, e={e:.3f}")

    cp.cuda.Stream.null.synchronize()
    wall = time.perf_counter() - wall_t0

    # NB: the bucket totals are in ms (CUDA events). Convert to seconds.
    breakdown_s = {k: v / 1000.0 for k, v in totals_ms.items()}
    total_bucket_s = sum(breakdown_s.values())
    breakdown_pct = {
        k: (100.0 * v / wall) if wall > 0 else 0.0
        for k, v in breakdown_s.items()
    }
    return ProfileReport(
        wall_time_s=wall,
        total_inner_iters=n_inner_total,
        per_step_iters_mean=n_inner_total / NT,
        per_step_iters_max=n_inner_max,
        breakdown_s=breakdown_s,
        breakdown_pct=breakdown_pct,
        settings=dict(
            N1=N1, N2=N2, N_phi=N_phi, N_Z=N_Z, dt=dt, NT=NT,
            p_a=p_a, B_width=B_width, mass_M=mass_M,
            X0=X0, Y0=Y0, alpha=alpha,
            omega_p=omega_p, omega_theta=omega_theta,
            tol_inner=tol_inner, max_inner=max_inner,
            check_every=check_every,
        ),
    )


def print_report(rep: ProfileReport):
    print("=== Phase 4.1 profiling report ===")
    print(f"  wall time          : {_fmt_s(rep.wall_time_s)}")
    print(f"  total inner iters  : {rep.total_inner_iters}")
    print(f"  inner/step mean    : {rep.per_step_iters_mean:.1f}")
    print(f"  inner/step max     : {rep.per_step_iters_max}")
    print(f"  NT / N1 / N2       : "
          f"{rep.settings['NT']} / {rep.settings['N1']} / {rep.settings['N2']}")
    print(f"  check_every / tol  : {rep.settings['check_every']} / "
          f"{rep.settings['tol_inner']:.0e}")
    print()
    print("  component       cuda-time   (% of wall)")
    for k, v in sorted(rep.breakdown_s.items(), key=lambda kv: -kv[1]):
        pct = rep.breakdown_pct[k]
        print(f"    {k:<12} {_fmt_s(v)}   ({pct:5.1f} %)")
    total_bucket = sum(rep.breakdown_s.values())
    print(f"    {'SUM':<12} {_fmt_s(total_bucket)} "
          f"({100.0 * total_bucket / rep.wall_time_s:5.1f} %)")


def main(argv=None):
    argv = argv or sys.argv[1:]
    p = argparse.ArgumentParser(
        description="Phase 4.1 performance profiler for the journal solver."
    )
    p.add_argument("--N1", type=int, default=100)
    p.add_argument("--N2", type=int, default=12)
    p.add_argument("--dt", type=float, default=2e-3)
    p.add_argument("--NT", type=int, default=1500)
    p.add_argument("--tol-inner", type=float, default=1e-6)
    p.add_argument("--max-inner", type=int, default=5000)
    p.add_argument("--omega-p", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--json", type=str, default=None,
                   help="Write the report to this JSON file.")
    args = p.parse_args(argv)

    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return 0

    rep = run_profiled(
        N1=args.N1, N2=args.N2, dt=args.dt, NT=args.NT,
        tol_inner=args.tol_inner, max_inner=args.max_inner,
        omega_p=args.omega_p, check_every=args.check_every,
        verbose=args.verbose,
    )
    print_report(rep)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(rep), f, indent=2)
        print(f"  JSON written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
