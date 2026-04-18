"""
Dynamic journal-bearing benchmark (Ausas, Jai, Buscaglia 2008, Section 5).

Problem setup
-------------
Dimensionless journal bearing on a unit circumferential period, axial
width B_width = 0.1. Shaft position (X, Y) is a dynamical unknown driven
by time-dependent applied load:

    WaX(t) = 0.01 * ( exp(-400 (t - 0.25)^2)
                      + 0.95534 * exp(-400 (t - 0.5)^2) )
    WaY(t) = 0.29552e-2 * exp(-400 (t - 0.5)^2)

These are periodic bumps with period T = 1.0 (the load wraps every T).

Discretisation
--------------
    dx1 = dx2 = 5e-3   ->   200 interior cells in each direction
    N_phi = 202, N_Z = 22       (two ghost cells per axis, per the
                                 repo's stencil convention)
    dt = 1e-3, NT = 3000         (3 periods: one transient + two
                                  periodic orbits)
    alpha = 1                    (journal-bearing slide velocity
                                  scaling)
    p_a = 0.0075                 (supply pressure at z = B)
    M  = 1e-6                    (dimensionless shaft mass)
    X0 = Y0 = 0.5                (initial eccentricity = 0.707)
    U0 = V0 = 0.0

What the benchmark reports
--------------------------
* Trajectory X(t), Y(t).
* Eccentricity history e(t) = sqrt(X^2 + Y^2) and its peak.
* Applied loads WaX(t), WaY(t) vs. hydrodynamic forces WX, WY (the two
  should approximately balance at equilibrium / in the periodic orbit).
* Per-step inner-iteration count as a diagnostic of coupled-loop cost.

Run
    python -m reynolds_solver.cavitation.ausas.benchmark_dynamic_journal [--NT 3000 --verbose]
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Applied load (Ausas 2008 Section 5)
# ---------------------------------------------------------------------------
def WaX_of_t(t, period: float = 1.0):
    """Periodic circumferential load with Gaussian bumps at t/T = 0.25, 0.5."""
    tp = t % period
    return 0.01 * (
        np.exp(-400.0 * (tp - 0.25) ** 2)
        + 0.95534 * np.exp(-400.0 * (tp - 0.5) ** 2)
    )


def WaY_of_t(t, period: float = 1.0):
    """Periodic axial-cross load, single Gaussian bump at t/T = 0.5."""
    tp = t % period
    return 0.29552e-2 * np.exp(-400.0 * (tp - 0.5) ** 2)


def journal_load(t, period: float = 1.0):
    """Return (WaX, WaY) as a pair at time t (period-wrapped)."""
    return WaX_of_t(t, period), WaY_of_t(t, period)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
@dataclass
class JournalBenchmarkResult:
    t: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    U: np.ndarray
    V: np.ndarray
    WX: np.ndarray
    WY: np.ndarray
    WaX: np.ndarray
    WaY: np.ndarray
    eccentricity: np.ndarray
    p_max: np.ndarray
    h_min: np.ndarray
    cav_frac: np.ndarray
    n_inner: np.ndarray
    converged: np.ndarray
    P_last: np.ndarray
    theta_last: np.ndarray
    # Full restart state (pass to `run_journal_benchmark(state=...)`).
    final_state: object = None
    # Settings
    N1: int = 0
    N2: int = 0
    dt: float = 0.0


def run_journal_benchmark(
    N1: int = 200,
    N2: int = 20,
    dt: float = 1e-3,
    NT: int = 3000,
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
    scheme: str = "rb",
    state=None,
    accel=None,
    verbose: bool = False,
) -> JournalBenchmarkResult:
    """
    Run Section 5 of Ausas 2008 on the dynamic GPU solver. See module
    docstring for the parameter choices (defaults match the paper).
    """
    # Lazy import so the module is importable on CPU-only machines.
    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        solve_ausas_journal_dynamic_gpu,
    )

    N_phi = N1 + 2         # 2 ghost cells (periodic phi)
    N_Z = N2 + 2           # 2 ghost cells (Dirichlet z)
    d_phi = 1.0 / N1       # d_phi * N1 = 1.0 (full circumferential period)
    d_Z = B_width / N2     # d_Z * N2 = B_width

    # R, L chosen so alpha_sq = (2R/L * d_phi/d_Z)^2 reduces to q^2
    # (the Ausas-2008 ratio factor).
    R = 0.5
    L = 1.0

    result = solve_ausas_journal_dynamic_gpu(
        NT=NT, dt=dt,
        N_Z=N_Z, N_phi=N_phi,
        d_phi=d_phi, d_Z=d_Z, R=R, L=L,
        mass_M=mass_M,
        load_fn=journal_load,
        X0=X0, Y0=Y0, U0=U0, V0=V0,
        p_a=p_a, B_width=B_width,
        alpha=alpha,
        omega_p=omega_p, omega_theta=omega_theta,
        tol_inner=tol_inner, max_inner=max_inner,
        scheme=scheme,
        state=state,
        accel=accel,
        verbose=verbose,
    )

    # Evaluate the reference applied load on the same time grid.
    WaX = np.array([WaX_of_t(float(tt)) for tt in result.t])
    WaY = np.array([WaY_of_t(float(tt)) for tt in result.t])

    eccentricity = np.sqrt(result.X ** 2 + result.Y ** 2)
    # Prefer the solver's own WaX/WaY histories if populated (Phase 4.3);
    # fall back to the hand-evaluated arrays for older callers.
    WaX_out = result.WaX if result.WaX is not None else WaX
    WaY_out = result.WaY if result.WaY is not None else WaY

    return JournalBenchmarkResult(
        t=result.t,
        X=result.X, Y=result.Y,
        U=result.U, V=result.V,
        WX=result.WX, WY=result.WY,
        WaX=WaX_out, WaY=WaY_out,
        eccentricity=eccentricity,
        p_max=result.p_max,
        h_min=result.h_min,
        cav_frac=result.cav_frac,
        n_inner=result.n_inner,
        converged=result.converged,
        P_last=result.P_last,
        theta_last=result.theta_last,
        final_state=result.final_state,
        N1=N1, N2=N2, dt=dt,
    )


# ---------------------------------------------------------------------------
# Stand-alone driver
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = argv or sys.argv[1:]
    import argparse
    p = argparse.ArgumentParser(
        description="Dynamic journal-bearing benchmark on the GPU Ausas solver."
    )
    p.add_argument("--N1", type=int, default=200)
    p.add_argument("--N2", type=int, default=20)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--NT", type=int, default=3000)
    p.add_argument("--tol-inner", type=float, default=1e-6)
    p.add_argument("--max-inner", type=int, default=5000)
    p.add_argument("--omega-p", type=float, default=1.0)
    p.add_argument("--scheme", choices=("rb", "jacobi"), default="rb")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    res = run_journal_benchmark(
        N1=args.N1, N2=args.N2, dt=args.dt, NT=args.NT,
        tol_inner=args.tol_inner, max_inner=args.max_inner,
        omega_p=args.omega_p, scheme=args.scheme,
        verbose=args.verbose,
    )

    print("\n=== Dynamic journal bearing (Ausas 2008 Section 5) ===")
    print(
        f"  N1={res.N1}, N2={res.N2}, dt={res.dt:.1e}, NT={len(res.t)}"
    )
    # Summary statistics over the full run and over the last period.
    last_period = res.t > res.t[-1] - 1.0
    print(f"  eccentricity: max={res.eccentricity.max():.3f} "
          f"at t={res.t[int(np.argmax(res.eccentricity))]:.3f}")
    print(f"  e(last-period): min={res.eccentricity[last_period].min():.3f}, "
          f"max={res.eccentricity[last_period].max():.3f}")
    print(f"  p_max over run: {res.p_max.max():.3e}")
    print(f"  h_min over run: {res.h_min.min():.3e}")
    print(f"  cav_frac over run: [{res.cav_frac.min():.3f}, {res.cav_frac.max():.3f}]")
    print(
        f"  WX+WaX integral-balance over last period: "
        f"{np.mean(np.abs(res.WX[last_period] + res.WaX[last_period])):.3e}"
    )
    print(
        f"  WY+WaY integral-balance over last period: "
        f"{np.mean(np.abs(res.WY[last_period] + res.WaY[last_period])):.3e}"
    )
    print(
        f"  inner iters: min={int(res.n_inner.min())}, "
        f"max={int(res.n_inner.max())}, mean={res.n_inner.mean():.1f}"
    )
    print(f"  converged steps: {int(res.converged.sum())}/{len(res.t)}")


if __name__ == "__main__":
    main()
