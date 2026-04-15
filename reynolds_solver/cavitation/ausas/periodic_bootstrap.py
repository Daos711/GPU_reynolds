"""
Two-stage periodic-orbit bootstrap utility.

For periodic journal problems (e.g. a pump harmonic load) the first
transient cycle is the expensive part of the simulation — after one
full period the orbit is typically close to its periodic attractor.
A pragmatic speed-up is to run the transient on a COARSE grid with a
large dt, then interpolate the converged state onto the production
(fine) grid and continue from there for one or two precise periods.

This helper orchestrates such a two-stage run. The solver is NOT
modified; the utility just:

  1. runs `solve_ausas_journal_dynamic_gpu` on a coarse grid;
  2. interpolates (P, theta, H_prev) onto the fine grid by bilinear
     interpolation of the interior arrays (ghost rows/cols are
     reconstructed by the solver when the fine run starts);
  3. runs the fine-grid solver from the interpolated `state`.

Requires numpy only (no scipy); the interpolation is an in-module
bilinear implementation so the utility has no extra dependencies.

Usage:
    from reynolds_solver.cavitation.ausas.periodic_bootstrap import (
        run_periodic_bootstrap,
    )
    from reynolds_solver import AusasAccelerationOptions

    res_fine = run_periodic_bootstrap(
        # Common problem parameters
        dt_coarse=5e-3, dt_fine=1e-3,
        NT_coarse=200, NT_fine=300,
        coarse_grid=(50, 6),
        fine_grid=(100, 12),
        p_a=0.0075, B_width=0.1, mass_M=1e-6,
        X0=0.5, Y0=0.5, U0=0.0, V0=0.0,
        # accel options for each stage (optional)
        accel_coarse=AusasAccelerationOptions(adaptive_dt=True, ...),
        accel_fine=None,
    )
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _bilinear_interp_interior(
    src: np.ndarray, dst_shape: tuple,
) -> np.ndarray:
    """
    Bilinear interpolation of the INTERIOR of `src` onto a grid of
    shape `dst_shape` (also including its 2-cell ghost border).

    `src` has shape (N_Z_src, N_phi_src). Interior is src[1:-1, 1:-1].
    Returned array has shape dst_shape with interior filled from the
    interpolation; ghost rows/cols are filled by edge replication
    (the solver will rebuild them on entry).
    """
    sZ, sP = src.shape
    dZ, dP = dst_shape
    out = np.empty(dst_shape, dtype=src.dtype)

    # Interior mesh coordinates in source.
    src_int = src[1:-1, 1:-1]
    nZs, nPs = src_int.shape
    nZd, nPd = dZ - 2, dP - 2

    # Normalised coordinates: 0 -> left edge of source interior, 1 ->
    # right edge.
    # Destination interior sample positions in source index space.
    zs = (np.arange(nZd) + 0.5) * (nZs / nZd) - 0.5
    ps = (np.arange(nPd) + 0.5) * (nPs / nPd) - 0.5
    zs = np.clip(zs, 0.0, nZs - 1)
    ps = np.clip(ps, 0.0, nPs - 1)

    z0 = np.floor(zs).astype(int)
    z1 = np.clip(z0 + 1, 0, nZs - 1)
    dz = (zs - z0).reshape(-1, 1)
    p0 = np.floor(ps).astype(int)
    p1 = np.clip(p0 + 1, 0, nPs - 1)
    dp = (ps - p0).reshape(1, -1)

    i00 = src_int[np.ix_(z0, p0)]
    i01 = src_int[np.ix_(z0, p1)]
    i10 = src_int[np.ix_(z1, p0)]
    i11 = src_int[np.ix_(z1, p1)]

    interior_dst = (
        (1 - dz) * (1 - dp) * i00
        + (1 - dz) * dp * i01
        + dz * (1 - dp) * i10
        + dz * dp * i11
    )

    out[1:-1, 1:-1] = interior_dst
    # Edge replication for ghost rows/cols; the solver will rewrite
    # them on entry anyway.
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out


def run_periodic_bootstrap(
    *,
    dt_coarse: float,
    dt_fine: float,
    NT_coarse: int,
    NT_fine: int,
    coarse_grid: tuple,             # (N1_coarse, N2_coarse)
    fine_grid: tuple,                # (N1_fine, N2_fine)
    load_fn,                          # callable(t) -> (WaX, WaY)
    B_width: float = 0.1,
    p_a: float = 0.0075,
    mass_M: float = 1e-6,
    X0: float = 0.5,
    Y0: float = 0.5,
    U0: float = 0.0,
    V0: float = 0.0,
    R: float = 0.5,
    L: float = 1.0,
    alpha: float = 1.0,
    omega_p: float = 1.0,
    omega_theta: float = 1.0,
    tol_inner: float = 1e-6,
    max_inner: int = 5000,
    scheme: str = "rb",
    accel_coarse=None,
    accel_fine=None,
    verbose: bool = False,
):
    """
    Run the coarse-then-fine bootstrap. Returns the fine-grid
    `AusasTransientResult`.
    """
    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        solve_ausas_journal_dynamic_gpu,
    )
    from reynolds_solver.cavitation.ausas.state_io import AusasState

    N1_c, N2_c = coarse_grid
    N_phi_c = N1_c + 2
    N_Z_c = N2_c + 2
    d_phi_c = 1.0 / N1_c
    d_Z_c = B_width / N2_c

    N1_f, N2_f = fine_grid
    N_phi_f = N1_f + 2
    N_Z_f = N2_f + 2
    d_phi_f = 1.0 / N1_f
    d_Z_f = B_width / N2_f

    if verbose:
        print(
            f"  [bootstrap] coarse {N_Z_c}x{N_phi_c} dt={dt_coarse:.1e} "
            f"NT={NT_coarse}"
        )

    res_c = solve_ausas_journal_dynamic_gpu(
        NT=NT_coarse, dt=dt_coarse,
        N_Z=N_Z_c, N_phi=N_phi_c,
        d_phi=d_phi_c, d_Z=d_Z_c, R=R, L=L,
        mass_M=mass_M, load_fn=load_fn,
        X0=X0, Y0=Y0, U0=U0, V0=V0,
        p_a=p_a, B_width=B_width, alpha=alpha,
        omega_p=omega_p, omega_theta=omega_theta,
        tol_inner=tol_inner, max_inner=max_inner,
        scheme=scheme, accel=accel_coarse,
        verbose=verbose,
    )

    # Interpolate coarse state onto fine grid.
    state_c = res_c.final_state
    fine_shape = (N_Z_f, N_phi_f)
    P_fine = _bilinear_interp_interior(state_c.P, fine_shape)
    theta_fine = _bilinear_interp_interior(state_c.theta, fine_shape)
    H_prev_fine = _bilinear_interp_interior(state_c.H_prev, fine_shape)
    # Clamp theta to [0, 1] after interpolation.
    theta_fine = np.clip(theta_fine, 0.0, 1.0)
    P_fine = np.maximum(P_fine, 0.0)

    state_fine = AusasState(
        P=P_fine, theta=theta_fine, H_prev=H_prev_fine,
        X=state_c.X, Y=state_c.Y, U=state_c.U, V=state_c.V,
        step_index=state_c.step_index,
        time=state_c.time,
        dt_last=state_c.dt_last,
    )

    if verbose:
        print(
            f"  [bootstrap] fine  {N_Z_f}x{N_phi_f} dt={dt_fine:.1e} "
            f"NT={NT_fine}"
        )

    res_f = solve_ausas_journal_dynamic_gpu(
        NT=NT_fine, dt=dt_fine,
        N_Z=N_Z_f, N_phi=N_phi_f,
        d_phi=d_phi_f, d_Z=d_Z_f, R=R, L=L,
        mass_M=mass_M, load_fn=load_fn,
        X0=X0, Y0=Y0, U0=U0, V0=V0,     # ignored because `state` wins
        p_a=p_a, B_width=B_width, alpha=alpha,
        omega_p=omega_p, omega_theta=omega_theta,
        tol_inner=tol_inner, max_inner=max_inner,
        scheme=scheme, accel=accel_fine,
        state=state_fine,
        verbose=verbose,
    )
    return res_f
