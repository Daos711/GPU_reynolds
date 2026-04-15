"""
Dynamic squeeze-film benchmark (Ausas, Jai, Buscaglia 2008, Section 3).

Problem setup
-------------
1-D parallel-plate squeeze film on x1 in [0, 1], periodic in x2.

    Gap:            h(t) = 0.125 * cos(4 pi t) + 0.375     (period T = 0.5)
    Pressure BC:    p(0, t) = p(1, t) = p0 = 0.025
    Sliding:        alpha = 0  (pure squeeze, no Couette)
    Initial:        theta(x, 0) = 1,   p(x, 0) = p0

Rupture starts shortly after t = 0.25 (when h'(t) crosses zero into h' > 0,
meaning the plates begin separating). The analytical rupture front during
the rupture phase is

    Sigma_exact(t) = 1 - sqrt( p0 * h^3 / h' ),

valid for t in (t_rup, t_ref), where t_rup is the first time at which
Sigma_exact = 0 and t_ref is the corresponding reformation time.

Analytic rupture time (from p0 * h^3 = h', near t = 0.25)
    t_rup ~ 0.250079   (p0 = 0.025)

This script reproduces the benchmark on the unsteady Ausas GPU solver
(see solver_dynamic_gpu.solve_ausas_prescribed_h_gpu) with the repository
axis convention phi = x1 (pressure-gradient direction, Dirichlet p = p0)
and Z = x2 (trivial, periodic). Because the problem is uniform in x2 by
construction, N_Z can be kept very small (N_Z = 4 is plenty).

Run
    python -m reynolds_solver.cavitation.ausas.benchmark_squeeze_dynamic
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Analytic expressions
# ---------------------------------------------------------------------------
def h_squeeze(t):
    """Ausas 2008 squeeze gap h(t)."""
    return 0.125 * np.cos(4.0 * np.pi * t) + 0.375


def hdot_squeeze(t):
    """Time derivative dh/dt."""
    return -0.125 * 4.0 * np.pi * np.sin(4.0 * np.pi * t)


def sigma_exact(t, p0=0.025):
    """
    Analytical cavitation-front position Sigma(t) during the rupture phase.

    Returns NaN outside the rupture phase (i.e., when h' <= 0 or when the
    radicand leaves [0, 1]).
    """
    h = h_squeeze(t)
    hp = hdot_squeeze(t)
    if hp <= 0.0:
        return np.nan
    rad = p0 * h ** 3 / hp
    if rad < 0.0 or rad > 1.0:
        return np.nan
    return 1.0 - np.sqrt(rad)


def analytic_rupture_time(p0=0.025):
    """
    First root of p0 * h^3 = h' for t > 0.25.

    Uses a bracketed bisection so scipy is not required.
    """
    def f(t):
        return p0 * h_squeeze(t) ** 3 - hdot_squeeze(t)

    a, b = 0.250001, 0.2510
    fa, fb = f(a), f(b)
    if fa * fb > 0.0:
        # Fall back to a wider bracket if the narrow one fails (shouldn't
        # happen for p0 = 0.025 but keep this safe).
        a, b = 0.2500001, 0.26
        fa, fb = f(a), f(b)
    for _ in range(200):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < 1e-14 or (b - a) < 1e-14:
            return c
        if fa * fc < 0.0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Numerical rupture-front tracker (nodal)
# ---------------------------------------------------------------------------
def sigma_numerical_from_theta(theta, x1_nodes, cav_thr=1.0 - 1e-6):
    """
    Estimate the cavitation front position Sigma from a 1-D theta profile
    (averaged over x2): Sigma = rightmost x1 node where theta < 1.

    Returns NaN if no cavitation is present.
    """
    theta_1d = np.asarray(theta)
    if theta_1d.ndim == 2:
        theta_1d = theta_1d.mean(axis=0)     # average over x2 rows
    cav = theta_1d < cav_thr
    if not cav.any():
        return np.nan
    # Rightmost cavitation node; +1 so Sigma is the physical cavitation
    # boundary (left edge of the full-film recovery region).
    idx = int(np.argmax(cav[::-1]))          # from the right
    sigma_idx = len(theta_1d) - 1 - idx
    return float(x1_nodes[sigma_idx])


# ---------------------------------------------------------------------------
# Benchmark entry point
# ---------------------------------------------------------------------------
@dataclass
class SqueezeBenchmarkResult:
    t: np.ndarray
    p_max: np.ndarray
    cav_frac: np.ndarray
    n_inner: np.ndarray
    converged: np.ndarray
    P_last: np.ndarray
    theta_last: np.ndarray
    field_checkpoints: Optional[dict]
    t_rup_numerical: float
    t_rup_analytical: float
    rupture_relative_error: float
    # Bench settings (for reproducibility)
    N1: int
    N2: int
    dt: float
    p0: float


def run_squeeze_benchmark(
    N1: int = 450,
    N2: int = 4,
    dt: float = 6.6e-4,
    NT: Optional[int] = None,
    p0: float = 0.025,
    omega_p: float = 1.8,
    omega_theta: float = 1.0,
    tol_inner: float = 1e-6,
    max_inner: int = 5000,
    scheme: str = "rb",
    save_stride: Optional[int] = None,
    field_callback=None,
    state=None,
    accel=None,
    verbose: bool = False,
) -> SqueezeBenchmarkResult:
    """
    Run one full period (T = 0.5) of the Ausas 2008 squeeze benchmark on
    the unsteady GPU solver and return per-step histories + rupture-time
    statistics.

    Axis mapping
    ------------
    x1 -> phi (Dirichlet p = p0 at both ends, N_phi = N1).
    x2 -> Z   (periodic, N_Z = N2).

    N2 = 4 is enough because the solution is uniform in x2 by construction.
    """
    # Lazy import so the module is importable on CPU-only machines.
    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        solve_ausas_prescribed_h_gpu,
    )

    if NT is None:
        NT = int(round(0.5 / dt))    # one period

    # Mapping: x1 -> phi, x2 -> Z. Use a unit domain in x1 (cell size 1/N1)
    # and pick R, L so that alpha_sq = (2R/L * d_phi/d_Z)^2 reduces to the
    # physical q^2 = (d_phi/d_Z)^2.
    N_phi = N1
    N_Z = N2
    d_phi = 1.0 / N_phi
    d_Z = 1.0 / N_Z
    R = 0.5
    L = 1.0

    x1_nodes = (np.arange(N_phi) + 0.5) * d_phi   # cell-centred x1

    # Prescribed gap: uniform in x2 (all rows identical).
    def H_provider(n, t):
        h = h_squeeze(t)
        return np.full((N_Z, N_phi), h, dtype=np.float64)

    P0 = np.full((N_Z, N_phi), p0, dtype=np.float64)
    theta0 = np.ones((N_Z, N_phi), dtype=np.float64)

    result = solve_ausas_prescribed_h_gpu(
        H_provider, NT=NT, dt=dt,
        d_phi=d_phi, d_Z=d_Z, R=R, L=L,
        alpha=0.0,                     # pure squeeze
        omega_p=omega_p, omega_theta=omega_theta,
        tol_inner=tol_inner, max_inner=max_inner,
        P0=P0, theta0=theta0,
        # Dirichlet p = p0 on the x1 (phi) ends; theta clamped to 1
        # (flooded) at the ends as well.
        p_bc_phi0=p0, p_bc_phiL=p0,
        theta_bc_phi0=1.0, theta_bc_phiL=1.0,
        # Z ends: periodic, so these Dirichlet values are unused.
        p_bc_z0=p0, p_bc_zL=p0,
        theta_bc_z0=1.0, theta_bc_zL=1.0,
        periodic_phi=False, periodic_z=True,
        scheme=scheme,
        save_stride=save_stride,
        field_callback=field_callback,
        state=state,
        accel=accel,
        verbose=verbose,
    )

    # Rupture detection: first step where cav_frac > 0.
    cav_frac = result.cav_frac
    if cav_frac.max() <= 0.0:
        t_rup_num = float("nan")
    else:
        idx = int(np.argmax(cav_frac > 0.0))
        t_rup_num = float(result.t[idx])

    t_rup_ana = analytic_rupture_time(p0=p0)
    rel_err = (
        abs(t_rup_num - t_rup_ana) / t_rup_ana if np.isfinite(t_rup_num) else np.nan
    )

    return SqueezeBenchmarkResult(
        t=result.t,
        p_max=result.p_max,
        cav_frac=result.cav_frac,
        n_inner=result.n_inner,
        converged=result.converged,
        P_last=result.P_last,
        theta_last=result.theta_last,
        field_checkpoints=result.field_checkpoints,
        t_rup_numerical=t_rup_num,
        t_rup_analytical=t_rup_ana,
        rupture_relative_error=rel_err,
        N1=N1, N2=N2, dt=dt, p0=p0,
    )


# ---------------------------------------------------------------------------
# Stand-alone driver
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = argv or sys.argv[1:]
    import argparse
    parser = argparse.ArgumentParser(
        description="Ausas 2008 squeeze benchmark on the GPU unsteady solver."
    )
    parser.add_argument("--N1", type=int, default=450)
    parser.add_argument("--N2", type=int, default=4)
    parser.add_argument("--dt", type=float, default=6.6e-4)
    parser.add_argument("--NT", type=int, default=None,
                        help="Default: one full period T = 0.5.")
    parser.add_argument("--tol-inner", type=float, default=1e-6)
    parser.add_argument("--max-inner", type=int, default=5000)
    parser.add_argument("--omega-p", type=float, default=1.8,
                        help="SOR factor for the RB inner loop. omega_p ~ 2/(1+sin(pi/N1)) is optimal.")
    parser.add_argument("--scheme", choices=("rb", "jacobi"), default="rb")
    parser.add_argument("--save-stride", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    res = run_squeeze_benchmark(
        N1=args.N1, N2=args.N2, dt=args.dt, NT=args.NT,
        tol_inner=args.tol_inner, max_inner=args.max_inner,
        omega_p=args.omega_p,
        scheme=args.scheme,
        save_stride=args.save_stride,
        verbose=args.verbose,
    )

    print("\n=== Ausas 2008 squeeze benchmark ===")
    print(
        f"  N1={res.N1}, N2={res.N2}, dt={res.dt:.3e}, "
        f"NT={len(res.t)}, p0={res.p0}"
    )
    print(f"  t_rup (numerical)  = {res.t_rup_numerical:.6f}")
    print(f"  t_rup (analytical) = {res.t_rup_analytical:.6f}")
    print(f"  relative error     = {100.0*res.rupture_relative_error:.3f} %")
    print(
        f"  p_max   range over period: [{res.p_max.min():.4e}, "
        f"{res.p_max.max():.4e}]"
    )
    print(
        f"  cav_frac over period: [{res.cav_frac.min():.3f}, "
        f"{res.cav_frac.max():.3f}]"
    )
    print(
        f"  inner iters over period: min={int(res.n_inner.min())}, "
        f"max={int(res.n_inner.max())}, mean={res.n_inner.mean():.1f}"
    )
    print(f"  converged steps: {int(res.converged.sum())}/{len(res.t)}")


if __name__ == "__main__":
    main()
