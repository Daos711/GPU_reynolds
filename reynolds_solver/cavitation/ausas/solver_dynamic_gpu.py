"""
GPU solver for the UNSTEADY Ausas mass-conserving JFO cavitation problem.

This module exposes two entry points:

* `ausas_unsteady_one_step_gpu` — advance (P, θ) by ONE real time step
  Δt. Used by the Stage-1 unit test against the CPU reference.

* `solve_ausas_prescribed_h_gpu` — full time loop over a prescribed gap
  h(t). Allocates GPU buffers once, iterates the Jacobi kernel inside
  each step, and returns per-step scalar histories (+ optional field
  snapshots). Used by the Stage-2 squeeze benchmark.

The actual CUDA sweeps live in `kernels_dynamic.unsteady_ausas_step` and
`kernels_dynamic.unsteady_ausas_bc`.

This solver is intentionally NOT a port of the stationary Payvar–Salant
or Ausas GPU solvers. In particular it does NOT freeze an active set —
each cell may switch between full-film and cavitation on every sweep.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import cupy as cp

from reynolds_solver.cavitation.ausas.kernels_dynamic import (
    get_unsteady_ausas_kernel,
    get_unsteady_ausas_rb_kernel,
    get_unsteady_ausas_bc_phi_kernel,
    get_unsteady_ausas_bc_z_kernel,
)
from reynolds_solver.cavitation.ausas.state_io import AusasState
from reynolds_solver.cavitation.ausas.accel_options import AusasAccelerationOptions


# ===========================================================================
# Coefficient builder (average-of-cubes Poiseuille conductance)
# ===========================================================================
def _build_coefficients_gpu(H_gpu, d_phi, d_Z, R, L):
    """
    Average-of-cubes Poiseuille conductance coefficients (identical to the
    stationary Ausas GPU solver — see cavitation/ausas/solver_gpu.py).

    `H_gpu` must already carry the correct ghost rows/columns for whichever
    axis is periodic (φ, Z, both, or neither). For periodic axes the
    ghosts contain the physical seam; for Dirichlet axes the ghost values
    are irrelevant (the corresponding C/D or A/B boundary entries are
    never read by the step kernel).
    """
    N_Z, N_phi = H_gpu.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    H3 = H_gpu ** 3

    Ah = 0.5 * (H3[:, :-1] + H3[:, 1:])           # (N_Z, N_phi-1)

    A = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    B = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    A[:, :-1] = Ah
    A[:, -1] = Ah[:, 0]
    B[:, 1:] = Ah
    B[:, 0] = Ah[:, -1]

    H_jph3 = 0.5 * (H3[:-1, :] + H3[1:, :])       # (N_Z-1, N_phi)

    C = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    D = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    C[1:-1, :] = alpha_sq * H_jph3[1:, :]
    D[1:-1, :] = alpha_sq * H_jph3[:-1, :]

    E = A + B + C + D
    return A, B, C, D, E


def _pack_ghosts(arr_gpu, periodic_phi: bool, periodic_z: bool):
    """
    In-place φ/Z ghost packing for periodic axes. For Dirichlet axes the
    ghost layer is left untouched (the BC kernel will overwrite it).
    """
    N_Z, N_phi = arr_gpu.shape
    if periodic_phi:
        arr_gpu[:, 0] = arr_gpu[:, N_phi - 2]
        arr_gpu[:, N_phi - 1] = arr_gpu[:, 1]
    if periodic_z:
        arr_gpu[0, :] = arr_gpu[N_Z - 2, :]
        arr_gpu[N_Z - 1, :] = arr_gpu[1, :]


def _apply_bc_python(
    P_gpu, theta_gpu,
    periodic_phi: bool, periodic_z: bool,
    p_phi0, p_phiL, theta_phi0, theta_phiL,
    p_z0, p_zL, theta_z0, theta_zL,
):
    """
    Host-driven counterpart of the BC CUDA kernel, used to seed the initial
    iterate before the first inner sweep. Applies periodic-or-Dirichlet BC
    on both axes (same semantics as `unsteady_ausas_bc`).
    """
    N_Z, N_phi = P_gpu.shape
    if periodic_phi:
        P_gpu[:, 0] = P_gpu[:, N_phi - 2]
        P_gpu[:, N_phi - 1] = P_gpu[:, 1]
        theta_gpu[:, 0] = theta_gpu[:, N_phi - 2]
        theta_gpu[:, N_phi - 1] = theta_gpu[:, 1]
    else:
        P_gpu[:, 0] = p_phi0
        P_gpu[:, N_phi - 1] = p_phiL
        theta_gpu[:, 0] = theta_phi0
        theta_gpu[:, N_phi - 1] = theta_phiL
    if periodic_z:
        P_gpu[0, :] = P_gpu[N_Z - 2, :]
        P_gpu[N_Z - 1, :] = P_gpu[1, :]
        theta_gpu[0, :] = theta_gpu[N_Z - 2, :]
        theta_gpu[N_Z - 1, :] = theta_gpu[1, :]
    else:
        P_gpu[0, :] = p_z0
        P_gpu[-1, :] = p_zL
        theta_gpu[0, :] = theta_z0
        theta_gpu[-1, :] = theta_zL


def _launch_configs(N_Z, N_phi):
    block = (32, 8, 1)
    grid = (
        (N_phi - 2 + block[0] - 1) // block[0],
        (N_Z - 2 + block[1] - 1) // block[1],
        1,
    )
    bc_block = (256, 1, 1)
    bc_grid_phi = ((N_Z + 255) // 256, 1, 1)
    bc_grid_z = ((N_phi + 255) // 256, 1, 1)
    return block, grid, bc_block, bc_grid_phi, bc_grid_z


def _launch_bc(
    bc_phi_kernel, bc_z_kernel,
    P, theta, N_Z, N_phi,
    bc_block, bc_grid_phi, bc_grid_z,
    periodic_phi, periodic_z,
    p_phi0, p_phiL, theta_phi0, theta_phiL,
    p_z0, p_zL, theta_z0, theta_zL,
):
    bc_phi_kernel(
        bc_grid_phi, bc_block,
        (
            P, theta,
            np.int32(N_Z), np.int32(N_phi),
            np.int32(1 if periodic_phi else 0),
            np.float64(p_phi0), np.float64(p_phiL),
            np.float64(theta_phi0), np.float64(theta_phiL),
        ),
    )
    bc_z_kernel(
        bc_grid_z, bc_block,
        (
            P, theta,
            np.int32(N_Z), np.int32(N_phi),
            np.int32(1 if periodic_z else 0),
            np.float64(p_z0), np.float64(p_zL),
            np.float64(theta_z0), np.float64(theta_zL),
        ),
    )


# ===========================================================================
# Stage 1 : one-step wrapper (unchanged semantics, extended BC)
# ===========================================================================
def ausas_unsteady_one_step_gpu(
    H_curr,
    H_prev,
    P_prev,
    theta_prev,
    dt,
    d_phi,
    d_Z,
    R,
    L,
    alpha=1.0,
    omega_p=1.0,
    omega_theta=1.0,
    tol=1e-6,
    max_inner=5000,
    p_bc_z0=0.0,
    p_bc_zL=0.0,
    theta_bc_z0=1.0,
    theta_bc_zL=1.0,
    p_bc_phi0=0.0,
    p_bc_phiL=0.0,
    theta_bc_phi0=1.0,
    theta_bc_phiL=1.0,
    periodic_phi=True,
    periodic_z=False,
    check_every=50,
    verbose=False,
):
    """
    Advance (P, θ) by one real time step Δt using the unsteady Ausas
    Jacobi kernel. See module docstring for discretization details.

    Axis-BC semantics: each axis (φ, Z) is either periodic or Dirichlet.
    For a Dirichlet axis, (p_bc_<axis>0, p_bc_<axis>L) and the matching
    θ-Dirichlet values are written into the ghost row/column by the BC
    kernel after each sweep.

    Returns
    -------
    P : (N_Z, N_phi) ndarray (CPU, float64)
    theta : (N_Z, N_phi) ndarray (CPU, float64)
    residual : float — last inner residual at time step n.
    n_inner : int — Jacobi sweeps actually performed.
    """
    H_curr = np.ascontiguousarray(H_curr, dtype=np.float64)
    H_prev = np.ascontiguousarray(H_prev, dtype=np.float64)
    P_prev = np.ascontiguousarray(P_prev, dtype=np.float64)
    theta_prev = np.ascontiguousarray(theta_prev, dtype=np.float64)

    N_Z, N_phi = H_curr.shape
    if H_prev.shape != (N_Z, N_phi):
        raise ValueError("H_prev shape must match H_curr.")
    if P_prev.shape != (N_Z, N_phi):
        raise ValueError("P_prev shape must match H_curr.")
    if theta_prev.shape != (N_Z, N_phi):
        raise ValueError("theta_prev shape must match H_curr.")

    H_curr_gpu = cp.asarray(H_curr)
    H_prev_gpu = cp.asarray(H_prev)
    _pack_ghosts(H_curr_gpu, periodic_phi, periodic_z)
    _pack_ghosts(H_prev_gpu, periodic_phi, periodic_z)

    theta_prev_gpu = cp.asarray(theta_prev)
    C_prev = theta_prev_gpu * H_prev_gpu

    A, B, C, D, E = _build_coefficients_gpu(H_curr_gpu, d_phi, d_Z, R, L)

    P_old = cp.asarray(P_prev)
    theta_old = cp.asarray(theta_prev)
    cp.maximum(P_old, 0.0, out=P_old)
    cp.clip(theta_old, 0.0, 1.0, out=theta_old)

    _apply_bc_python(
        P_old, theta_old, periodic_phi, periodic_z,
        p_bc_phi0, p_bc_phiL, theta_bc_phi0, theta_bc_phiL,
        p_bc_z0, p_bc_zL, theta_bc_z0, theta_bc_zL,
    )

    P_new = P_old.copy()
    theta_new = theta_old.copy()

    kernel = get_unsteady_ausas_kernel()
    bc_phi_kernel = get_unsteady_ausas_bc_phi_kernel()
    bc_z_kernel = get_unsteady_ausas_bc_z_kernel()
    block, grid, bc_block, bc_grid_phi, bc_grid_z = _launch_configs(N_Z, N_phi)

    pphi = 1 if periodic_phi else 0
    pz = 1 if periodic_z else 0

    residual = float("inf")
    n_inner = 0
    for k in range(max_inner):
        kernel(
            grid, block,
            (
                P_old, P_new, theta_old, theta_new,
                H_curr_gpu, C_prev,
                A, B, C, D, E,
                np.float64(d_phi), np.float64(d_Z),
                np.float64(dt), np.float64(alpha),
                np.float64(omega_p), np.float64(omega_theta),
                np.int32(N_Z), np.int32(N_phi),
                np.int32(pphi), np.int32(pz),
            ),
        )
        _launch_bc(
            bc_phi_kernel, bc_z_kernel,
            P_new, theta_new, N_Z, N_phi,
            bc_block, bc_grid_phi, bc_grid_z,
            periodic_phi, periodic_z,
            p_bc_phi0, p_bc_phiL, theta_bc_phi0, theta_bc_phiL,
            p_bc_z0, p_bc_zL, theta_bc_z0, theta_bc_zL,
        )
        n_inner += 1

        if k % check_every == 0 or k < 3:
            dP = float(cp.sqrt(cp.sum((P_new - P_old) ** 2)))
            dth = float(cp.sqrt(cp.sum((theta_new - theta_old) ** 2)))
            residual = dP + dth
            if verbose:
                print(
                    f"  [Ausas-dyn-GPU] inner={k:>5d}: residual={residual:.4e}, "
                    f"dP={dP:.2e}, dth={dth:.2e}, maxP={float(cp.max(P_new)):.4e}"
                )

        P_old, P_new = P_new, P_old
        theta_old, theta_new = theta_new, theta_old

        if residual < tol and k > 2:
            if verbose:
                print(
                    f"  [Ausas-dyn-GPU] CONVERGED at inner={k}, "
                    f"residual={residual:.4e}"
                )
            break

    P_cpu = cp.asnumpy(P_old)
    theta_cpu = cp.asnumpy(theta_old)
    return P_cpu, theta_cpu, float(residual), n_inner


# ===========================================================================
# Stage 2 : full time loop (prescribed h(t))
# ===========================================================================
@dataclass
class AusasTransientResult:
    """
    Scalar histories + last field + optional field checkpoints returned by
    `solve_ausas_prescribed_h_gpu` (Stage 2) and
    `solve_ausas_journal_dynamic_gpu` (Stage 3).

    Memory: scalars are O(NT) floats (NT = number of time steps).  The
    field `P_last` / `theta_last` is a single snapshot. `field_checkpoints`
    is populated only when `save_stride` is given; keys are 1-indexed step
    numbers (+ 0 for the initial state).

    Stage-3 journal-bearing fields (X, Y, U, V, WX, WY) are None for the
    prescribed-h solver.
    """
    t: np.ndarray
    p_max: np.ndarray
    cav_frac: np.ndarray
    n_inner: np.ndarray
    converged: np.ndarray
    P_last: np.ndarray
    theta_last: np.ndarray
    field_checkpoints: Optional[dict]
    # --- Optional (only populated by the journal-bearing solver) ---
    h_min: Optional[np.ndarray] = None
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None
    U: Optional[np.ndarray] = None
    V: Optional[np.ndarray] = None
    WX: Optional[np.ndarray] = None
    WY: Optional[np.ndarray] = None
    # --- Optional applied-load histories (journal only) ---
    WaX: Optional[np.ndarray] = None
    WaY: Optional[np.ndarray] = None
    # --- Final-state snapshot ready to be passed back into the solver
    # via the `state=` kwarg for a seamless restart. ---
    final_state: Optional["AusasState"] = None


def solve_ausas_prescribed_h_gpu(
    H_provider: Callable[[int, float], np.ndarray],
    NT: int,
    dt: float,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    alpha: float = 1.0,
    omega_p: float = 1.0,
    omega_theta: float = 1.0,
    tol_inner: float = 1e-6,
    max_inner: int = 5000,
    P0=None,
    theta0=None,
    p_bc_z0: float = 0.0,
    p_bc_zL: float = 0.0,
    theta_bc_z0: float = 1.0,
    theta_bc_zL: float = 1.0,
    p_bc_phi0: float = 0.0,
    p_bc_phiL: float = 0.0,
    theta_bc_phi0: float = 1.0,
    theta_bc_phiL: float = 1.0,
    periodic_phi: bool = True,
    periodic_z: bool = False,
    save_stride: Optional[int] = None,
    field_callback: Optional[Callable[[int, float, np.ndarray, np.ndarray], None]] = None,
    check_every: int = 10,
    scheme: str = "rb",
    state: Optional[AusasState] = None,
    verbose: bool = False,
) -> AusasTransientResult:
    """
    Full time loop: for n = 1..NT advance (P, theta) under a prescribed gap
    h(t) supplied by `H_provider(n, t_n)`.

    Restart
    -------
    Passing `state=AusasState(...)` resumes from a previous run. P0,
    theta0, H_prev and (step_index, time) are taken from the state;
    H_provider is called with the GLOBAL step index `state.step_index +
    local_step` and the GLOBAL time `state.time + local_step * dt`, so
    the user's H_provider sees a continuous time axis across restarts.
    Histories also use global times. `result.final_state` is the state
    snapshot ready to be passed to a subsequent call.

    Per step
    --------
      1. fetch H_curr on GPU, pack ghosts;
      2. freeze C_prev = θ^{n-1} · h^{n-1};
      3. rebuild average-of-cubes coefficients from H_curr;
      4. Jacobi inner loop (kernel + BC) until `tol_inner` or `max_inner`
         iterations;
      5. record scalar history (p_max, cav_frac, n_inner, converged);
      6. optionally snapshot fields (every `save_stride` steps) or hand
         them to `field_callback`.

    Memory-safe: only persistent buffers are the scalar histories and a
    fixed set of (N_Z, N_phi) GPU arrays — no accumulation of per-step
    fields unless explicitly requested.

    `H_provider(n, t)` is called with the 0-based step index n = 0..NT
    (step 0 = initial state, t = 0) and the corresponding time. It must
    return a CPU array of shape (N_Z, N_phi).

    scheme : {"rb", "jacobi"}
        Inner-loop relaxation scheme.
        * "rb" (default) — Red-Black Gauss-Seidel / SOR: in-place updates,
          two colour passes per iteration, O(N) iters with omega_p close
          to the optimal SOR value. This is the fast path for the
          Stage-2 squeeze benchmark.
        * "jacobi" — pure frozen-iterate Jacobi using the Stage-1 kernel
          (separate *_old / *_new buffers). Converges as O(N^2) for
          Poisson-like problems; kept for bit-for-bit reproducibility
          against the CPU reference.

    Returns
    -------
    AusasTransientResult
    """
    if scheme not in ("rb", "jacobi"):
        raise ValueError(f"Unknown scheme {scheme!r}. Use 'rb' or 'jacobi'.")
    # --- Restart / initial gap / dimensions ---------------------------------
    # Resume-from-state takes precedence; otherwise ask H_provider for H(0).
    if state is not None:
        H_prev_cpu = np.asarray(state.H_prev, dtype=np.float64)
        t_offset = float(state.time)
        n_offset = int(state.step_index)
    else:
        H_prev_cpu = np.asarray(H_provider(0, 0.0), dtype=np.float64)
        t_offset = 0.0
        n_offset = 0
    N_Z, N_phi = H_prev_cpu.shape
    if N_Z < 3 or N_phi < 3:
        raise ValueError(
            f"Grid too small: ({N_Z}, {N_phi}) - need at least 3 on each axis."
        )

    shape = (N_Z, N_phi)

    # --- Persistent GPU buffers -------------------------------------------
    H_prev = cp.asarray(H_prev_cpu)
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

    _pack_ghosts(H_prev, periodic_phi, periodic_z)

    # Initial (P, theta) — state takes precedence over the P0/theta0 kwargs.
    if state is not None:
        P_old[:] = cp.asarray(state.P, dtype=cp.float64)
        theta_old[:] = cp.asarray(state.theta, dtype=cp.float64)
    else:
        if P0 is None:
            P_old[:] = 0.0
        else:
            P_arr = np.asarray(P0, dtype=np.float64)
            if P_arr.shape != shape:
                raise ValueError(f"P0 shape {P_arr.shape} != {shape}")
            P_old[:] = cp.asarray(P_arr)
        if theta0 is None:
            theta_old[:] = 1.0
        else:
            th_arr = np.asarray(theta0, dtype=np.float64)
            if th_arr.shape != shape:
                raise ValueError(f"theta0 shape {th_arr.shape} != {shape}")
            theta_old[:] = cp.asarray(th_arr)
    cp.maximum(P_old, 0.0, out=P_old)
    cp.clip(theta_old, 0.0, 1.0, out=theta_old)

    _apply_bc_python(
        P_old, theta_old, periodic_phi, periodic_z,
        p_bc_phi0, p_bc_phiL, theta_bc_phi0, theta_bc_phiL,
        p_bc_z0, p_bc_zL, theta_bc_z0, theta_bc_zL,
    )

    # --- Kernel / launch config -------------------------------------------
    if scheme == "rb":
        rb_kernel = get_unsteady_ausas_rb_kernel()
        jac_kernel = None
    else:
        jac_kernel = get_unsteady_ausas_kernel()
        rb_kernel = None
    bc_phi_kernel = get_unsteady_ausas_bc_phi_kernel()
    bc_z_kernel = get_unsteady_ausas_bc_z_kernel()
    block, grid, bc_block, bc_grid_phi, bc_grid_z = _launch_configs(N_Z, N_phi)

    pphi = 1 if periodic_phi else 0
    pz = 1 if periodic_z else 0

    # --- History arrays ---------------------------------------------------
    t_hist = np.empty(NT, dtype=np.float64)
    p_max_hist = np.empty(NT, dtype=np.float64)
    h_min_hist = np.empty(NT, dtype=np.float64)
    cav_hist = np.empty(NT, dtype=np.float64)
    n_inner_hist = np.empty(NT, dtype=np.int32)
    conv_hist = np.empty(NT, dtype=bool)

    checkpoints: Optional[dict] = None
    if save_stride is not None and save_stride > 0:
        checkpoints = {0: (cp.asnumpy(P_old), cp.asnumpy(theta_old))}

    # --- Time loop ---------------------------------------------------------
    for step in range(1, NT + 1):
        n = n_offset + step                     # global step index
        t_n = t_offset + step * dt              # global time

        H_curr_cpu = np.asarray(H_provider(n, t_n), dtype=np.float64)
        if H_curr_cpu.shape != shape:
            raise ValueError(
                f"H_provider returned shape {H_curr_cpu.shape}, expected {shape}"
            )
        H_curr[:] = cp.asarray(H_curr_cpu)
        _pack_ghosts(H_curr, periodic_phi, periodic_z)

        # C_prev = θ^{n-1} · h^{n-1} — frozen for the whole inner loop.
        cp.multiply(theta_old, H_prev, out=C_prev)

        A[:], B[:], C[:], D[:], E[:] = _build_coefficients_gpu(
            H_curr, d_phi, d_Z, R, L
        )

        # Seed (P_new, theta_new) with the current iterate so that ghost
        # rows on both sides are consistent before the first sweep.
        P_new[:] = P_old
        theta_new[:] = theta_old

        residual = float("inf")
        converged = False
        k_done = 0
        for k in range(max_inner):
            if scheme == "jacobi":
                jac_kernel(
                    grid, block,
                    (
                        P_old, P_new, theta_old, theta_new,
                        H_curr, C_prev,
                        A, B, C, D, E,
                        np.float64(d_phi), np.float64(d_Z),
                        np.float64(dt), np.float64(alpha),
                        np.float64(omega_p), np.float64(omega_theta),
                        np.int32(N_Z), np.int32(N_phi),
                        np.int32(pphi), np.int32(pz),
                    ),
                )
                _launch_bc(
                    bc_phi_kernel, bc_z_kernel,
                    P_new, theta_new, N_Z, N_phi,
                    bc_block, bc_grid_phi, bc_grid_z,
                    periodic_phi, periodic_z,
                    p_bc_phi0, p_bc_phiL, theta_bc_phi0, theta_bc_phiL,
                    p_bc_z0, p_bc_zL, theta_bc_z0, theta_bc_zL,
                )
            else:
                # Red-black SOR: two in-place colour passes + BC between
                # and after. Both passes write to (P_old, theta_old); the
                # (*_new) buffers are reused only for the residual check.
                for color in (0, 1):
                    rb_kernel(
                        grid, block,
                        (
                            P_old, theta_old,
                            H_curr, C_prev,
                            A, B, C, D, E,
                            np.float64(d_phi), np.float64(d_Z),
                            np.float64(dt), np.float64(alpha),
                            np.float64(omega_p), np.float64(omega_theta),
                            np.int32(N_Z), np.int32(N_phi),
                            np.int32(pphi), np.int32(pz),
                            np.int32(color),
                        ),
                    )
                    _launch_bc(
                        bc_phi_kernel, bc_z_kernel,
                        P_old, theta_old, N_Z, N_phi,
                        bc_block, bc_grid_phi, bc_grid_z,
                        periodic_phi, periodic_z,
                        p_bc_phi0, p_bc_phiL, theta_bc_phi0, theta_bc_phiL,
                        p_bc_z0, p_bc_zL, theta_bc_z0, theta_bc_zL,
                    )
            k_done = k + 1

            if k % check_every == 0 or k < 3:
                if scheme == "jacobi":
                    dP = float(cp.max(cp.abs(P_new - P_old)))
                    dth = float(cp.max(cp.abs(theta_new - theta_old)))
                else:
                    # For RB we need a dedicated "previous iterate" buffer
                    # to measure the change. Use P_new / theta_new for
                    # that purpose: save the state BEFORE the next sweep.
                    dP = float(cp.max(cp.abs(P_old - P_new)))
                    dth = float(cp.max(cp.abs(theta_old - theta_new)))
                residual = dP + dth
                if residual < tol_inner and k > 2:
                    converged = True

            if scheme == "jacobi":
                # Swap — latest iterate now sits in (P_old, theta_old).
                P_old, P_new = P_new, P_old
                theta_old, theta_new = theta_new, theta_old
            else:
                # Snapshot current iterate into (P_new, theta_new) so the
                # NEXT iteration's residual check compares to the most
                # recent state.
                P_new[:] = P_old
                theta_new[:] = theta_old

            if converged:
                break

        # Scalar history — LOCAL index (this run's 0..NT-1).
        t_hist[step - 1] = t_n
        p_max_hist[step - 1] = float(cp.max(P_old))
        h_min_hist[step - 1] = float(cp.min(H_curr[1:-1, 1:-1]))
        cav_hist[step - 1] = float(
            cp.mean((theta_old < 1.0 - 1e-6).astype(cp.float64))
        )
        n_inner_hist[step - 1] = k_done
        conv_hist[step - 1] = converged

        if verbose and (step <= 3 or step % max(NT // 20, 1) == 0):
            print(
                f"  [step {n:>6d}] t={t_n:.5f} "
                f"p_max={p_max_hist[step-1]:.4e} "
                f"cav={cav_hist[step-1]:.3f} "
                f"inner={k_done:>4d} "
                f"res={residual:.2e} "
                f"conv={'Y' if converged else 'N'}"
            )

        if checkpoints is not None and save_stride and (
            n % save_stride == 0 or step == NT
        ):
            checkpoints[n] = (cp.asnumpy(P_old), cp.asnumpy(theta_old))

        if field_callback is not None:
            field_callback(n, t_n, cp.asnumpy(P_old), cp.asnumpy(theta_old))

        # Advance H_prev <- H_curr (swap buffers so the next step overwrites
        # the stale one).
        H_prev, H_curr = H_curr, H_prev

    P_last = cp.asnumpy(P_old)
    theta_last = cp.asnumpy(theta_old)
    # H_prev (after the final swap) is H at the LAST completed step — the
    # correct seed for a subsequent restart.
    H_prev_last = cp.asnumpy(H_prev)
    final_state = AusasState(
        P=P_last,
        theta=theta_last,
        H_prev=H_prev_last,
        step_index=n_offset + NT,
        time=t_offset + NT * dt,
    )

    return AusasTransientResult(
        t=t_hist,
        p_max=p_max_hist,
        cav_frac=cav_hist,
        n_inner=n_inner_hist,
        converged=conv_hist,
        P_last=P_last,
        theta_last=theta_last,
        field_checkpoints=checkpoints,
        h_min=h_min_hist,
        final_state=final_state,
    )


# ===========================================================================
# Stage 3 : fully dynamical journal bearing (coupled to equations of motion)
# ===========================================================================
def _build_gap_inplace(H_out, X, Y, cos_phi_1d, sin_phi_1d, texture_gpu=None):
    """
    H_out[i, j] = 1 + X * cos_phi_1d[j] + Y * sin_phi_1d[j] [+ texture[i, j]]

    Broadcasts cos_phi_1d / sin_phi_1d (shape (N_phi,)) over Z rows. A
    temporary array is materialised on the GPU (~N_Z*N_phi doubles, which
    for the Stage-3 benchmark is <40 kB) and copied into H_out; the cost
    is negligible vs. kernel launches.
    """
    H_out[:] = 1.0 + X * cos_phi_1d[None, :] + Y * sin_phi_1d[None, :]
    if texture_gpu is not None:
        H_out += texture_gpu


def solve_ausas_journal_dynamic_gpu(
    NT: int,
    dt: float,
    N_Z: int,
    N_phi: int,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    mass_M: float,
    load_fn: Callable[[float], tuple],
    X0: float,
    Y0: float,
    U0: float = 0.0,
    V0: float = 0.0,
    p_a: float = 0.0,
    B_width: float = 0.1,         # informational — B = (N_Z - 2) * d_Z
    alpha: float = 1.0,
    omega_p: float = 1.0,
    omega_theta: float = 1.0,
    tol_inner: float = 1e-6,
    max_inner: int = 5000,
    P0=None,
    theta0=None,
    texture_relief=None,
    scheme: str = "rb",
    check_every: int = 10,
    save_stride: Optional[int] = None,
    field_callback: Optional[Callable[[int, float, np.ndarray, np.ndarray], None]] = None,
    state: Optional[AusasState] = None,
    accel: Optional[AusasAccelerationOptions] = None,
    verbose: bool = False,
) -> AusasTransientResult:
    """
    Fully dynamical journal-bearing solver (Ausas, Jai, Buscaglia 2008,
    Table 2 / Section 5).

    Restart
    -------
    Passing `state=AusasState(...)` resumes from a previous run: P0,
    theta0, X0, Y0, U0, V0, H_prev and (step_index, time) are taken
    from the state (overriding the matching kwargs). `load_fn` is
    called with the GLOBAL time `state.time + local_step * dt`, so
    the user's load_fn sees a continuous time axis across restarts.
    Histories use global times. `result.final_state` is the state
    snapshot ready to be passed to a subsequent call.

    Acceleration (Phase 5 Part 1 — lagged mechanics)
    ------------------------------------------------
    `accel=AusasAccelerationOptions(mech_update_every=K, ...)` switches
    the inner loop to lagged-mechanics mode: forces, Newmark predictor,
    gap H and coefficients A..E are rebuilt only every K iterations,
    while the RB/Jacobi sweep runs every iteration. Guards force an
    out-of-schedule refresh when the residual stalls, the cavitation
    fraction jumps, or the loop is approaching max_inner. Leaving
    `accel=None` (or `mech_update_every=1`) preserves the baseline
    behaviour bit-for-bit.

    Phase-4.1 performance notes
    ---------------------------
    The inner loop keeps the entire (W_X, W_Y, X_k, Y_k, U, V) pipeline on
    the GPU — no host sync per inner iteration. Host transfers are only
    taken when a residual is actually being computed (every `check_every`
    inner iterations plus the first three). Gap H and coefficients
    A, B, C, D, E are rebuilt by dedicated single-launch RawKernels
    (`build_gap_inplace`, `build_coefficients_inplace`) instead of a
    multi-op CuPy pipeline. The RB-snapshot `P_new[:] = P_old` is only
    taken when the residual check will actually read from P_new.

    None of these change the Ausas discretization — they only reduce
    kernel-launch count and GPU->host synchronisations.

    The shaft position (X, Y) is an unknown, determined together with the
    pressure / cavitation fields. Per time step n, every inner iteration:

        1. W_X = dx1*dx2 * sum(P_old * cos_phi_2d)
           W_Y = dx1*dx2 * sum(P_old * sin_phi_2d)
        2. X_k = X_{n-1} + dt*U_{n-1} + dt^2 / (2 M) * (W_X + WaX(t_n))
           Y_k = Y_{n-1} + dt*V_{n-1} + dt^2 / (2 M) * (W_Y + WaY(t_n))
        3. H_curr[i, j] = 1 + X_k*cos_phi[j] + Y_k*sin_phi[j] [+ texture]
        4. Rebuild A, B, C, D, E from H_curr.
        5. One GPU relaxation sweep (RB-SOR by default, Jacobi on request).
        6. BC sync.
        7. change = ||dP||_2 + ||dtheta||_2 + |dX| + |dY|.
        8. Swap (Jacobi) or snapshot (RB).
        9. Break if change < tol_inner.

    After convergence, the end-of-step velocities are closed up with the
    same (converged) W_X, W_Y:

        U_n = U_{n-1} + dt / M * (W_X + WaX(t_n))
        V_n = V_{n-1} + dt / M * (W_Y + WaY(t_n))

    Grid convention
    ---------------
    N_phi, N_Z are TOTAL grid sizes including one ghost cell on each end
    per axis. The physical circumference is spanned by the (N_phi - 2)
    interior cells at spacing d_phi (so the closed-period condition is
    d_phi * (N_phi - 2) = 1), and the axial width is (N_Z - 2) * d_Z.
    The ghost pack H[:, 0] = H[:, N_phi-2], H[:, -1] = H[:, 1] is exact
    when cos/sin are sampled at the shifted positions (k - 1) * d_phi
    (done automatically by _build_gap_inplace).

    Boundary conditions
    -------------------
    phi:  periodic (journal-bearing circumferential wrap).
    Z:    p = 0 at z = 0, p = p_a at z = B (feeding/supply).
          theta = 1 at both ends (flooded).

    Returns
    -------
    AusasTransientResult with X, Y, U, V, WX, WY, h_min populated.
    """
    if scheme not in ("rb", "jacobi"):
        raise ValueError(f"Unknown scheme {scheme!r}. Use 'rb' or 'jacobi'.")
    if N_phi < 3 or N_Z < 3:
        raise ValueError(f"Grid too small: ({N_Z}, {N_phi})")

    shape = (N_Z, N_phi)

    # --- Pre-compute angular arrays (shifted so col 1 is at x1 = 0,
    # col 0 is the ghost at x1 = -d_phi and col N_phi-1 at x1 = 1). This
    # makes H[:, 0] = H[:, N_phi-2] and H[:, -1] = H[:, 1] EXACT. -----
    k_arr = cp.arange(N_phi, dtype=cp.float64) - 1.0
    phi_vec = k_arr * d_phi
    cos_phi_1d = cp.cos(2.0 * np.pi * phi_vec)
    sin_phi_1d = cp.sin(2.0 * np.pi * phi_vec)

    # 2-D views for the force integral (no extra memory — just broadcasts).
    cos_phi_2d = cp.broadcast_to(cos_phi_1d[None, :], shape)
    sin_phi_2d = cp.broadcast_to(sin_phi_1d[None, :], shape)

    # --- Texture relief (optional) ---
    if texture_relief is not None:
        texture_gpu = cp.asarray(texture_relief, dtype=cp.float64)
        if texture_gpu.shape != shape:
            raise ValueError(
                f"texture_relief shape {texture_gpu.shape} != {shape}"
            )
    else:
        texture_gpu = None

    # --- Persistent GPU buffers ---
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

    # --- Initial state (state lives on device after this point) ---
    if state is not None:
        # Restart: all (X, Y, U, V, P, theta, H_prev) come from state.
        X_init = float(state.X)
        Y_init = float(state.Y)
        U_init = float(state.U)
        V_init = float(state.V)
        t_offset = float(state.time)
        n_offset = int(state.step_index)
    else:
        X_init = float(X0)
        Y_init = float(Y0)
        U_init = float(U0)
        V_init = float(V0)
        t_offset = 0.0
        n_offset = 0

    X_dev = cp.array(X_init, dtype=cp.float64)
    Y_dev = cp.array(Y_init, dtype=cp.float64)
    U_dev = cp.array(U_init, dtype=cp.float64)
    V_dev = cp.array(V_init, dtype=cp.float64)

    if state is not None:
        H_arr = np.asarray(state.H_prev, dtype=np.float64)
        if H_arr.shape != shape:
            raise ValueError(f"state.H_prev shape {H_arr.shape} != {shape}")
        H_prev[:] = cp.asarray(H_arr)
    else:
        # Initial gap from the host-scalar X0/Y0 (one-off, not in hot loop).
        _build_gap_inplace(H_prev, X_init, Y_init,
                           cos_phi_1d, sin_phi_1d, texture_gpu)
    _pack_ghosts(H_prev, periodic_phi=True, periodic_z=False)

    if state is not None:
        P_old[:] = cp.asarray(state.P, dtype=cp.float64)
        theta_old[:] = cp.asarray(state.theta, dtype=cp.float64)
    else:
        if P0 is None:
            P_old[:] = 0.0
        else:
            P_arr = np.asarray(P0, dtype=np.float64)
            if P_arr.shape != shape:
                raise ValueError(f"P0 shape {P_arr.shape} != {shape}")
            P_old[:] = cp.asarray(P_arr)
        if theta0 is None:
            theta_old[:] = 1.0
        else:
            th_arr = np.asarray(theta0, dtype=np.float64)
            if th_arr.shape != shape:
                raise ValueError(f"theta0 shape {th_arr.shape} != {shape}")
            theta_old[:] = cp.asarray(th_arr)
    cp.maximum(P_old, 0.0, out=P_old)
    cp.clip(theta_old, 0.0, 1.0, out=theta_old)

    _apply_bc_python(
        P_old, theta_old,
        periodic_phi=True, periodic_z=False,
        p_phi0=0.0, p_phiL=0.0,
        theta_phi0=1.0, theta_phiL=1.0,
        p_z0=0.0, p_zL=p_a,
        theta_z0=1.0, theta_zL=1.0,
    )
    # P_new / theta_new are only used by RB as a pre-sweep snapshot for
    # the residual check; by Jacobi as the "new iterate" target buffer.
    P_new[:] = P_old
    theta_new[:] = theta_old

    # --- Kernels / launch config ---
    if scheme == "rb":
        rb_kernel = get_unsteady_ausas_rb_kernel()
        jac_kernel = None
    else:
        jac_kernel = get_unsteady_ausas_kernel()
        rb_kernel = None
    bc_phi_kernel = get_unsteady_ausas_bc_phi_kernel()
    bc_z_kernel = get_unsteady_ausas_bc_z_kernel()
    # Phase 4.1 kernels (single-launch gap + coeffs rebuild).
    from reynolds_solver.cavitation.ausas.kernels_dynamic import (
        get_build_coefficients_kernel,
        get_build_gap_kernel,
    )
    build_coeffs_kernel = get_build_coefficients_kernel()
    build_gap_kernel = get_build_gap_kernel()

    block, grid, bc_block, bc_grid_phi, bc_grid_z = _launch_configs(N_Z, N_phi)
    # Grid covering the FULL (N_Z, N_phi) shape for the helper kernels.
    full_grid = (
        (N_phi + block[0] - 1) // block[0],
        (N_Z + block[1] - 1) // block[1],
        1,
    )

    # Helper kernel wants a valid texture pointer even when no texture is
    # set; allocate a full-size zero buffer (single ~32 kB alloc) and pass
    # has_texture = 0 so the kernel branch skips the load.
    if texture_gpu is None:
        texture_for_kernel = cp.zeros(shape, dtype=cp.float64)
        has_texture_flag = 0
    else:
        texture_for_kernel = texture_gpu
        has_texture_flag = 1

    # Precomputed Python scalars used in the inner loop (host, broadcast
    # into device expressions for free).
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2
    dt_sq_half = (dt * dt) / (2.0 * mass_M)
    dt_over_M = dt / mass_M
    d_phi_d_Z = d_phi * d_Z
    # Views of the cos/sin over the interior for the force integrals.
    cos_phi_interior = cos_phi_1d[1:-1]
    sin_phi_interior = sin_phi_1d[1:-1]

    # --- History arrays ---
    t_hist = np.empty(NT, dtype=np.float64)
    X_hist = np.empty(NT, dtype=np.float64)
    Y_hist = np.empty(NT, dtype=np.float64)
    U_hist = np.empty(NT, dtype=np.float64)
    V_hist = np.empty(NT, dtype=np.float64)
    WX_hist = np.empty(NT, dtype=np.float64)
    WY_hist = np.empty(NT, dtype=np.float64)
    WaX_hist = np.empty(NT, dtype=np.float64)
    WaY_hist = np.empty(NT, dtype=np.float64)
    p_max_hist = np.empty(NT, dtype=np.float64)
    h_min_hist = np.empty(NT, dtype=np.float64)
    cav_hist = np.empty(NT, dtype=np.float64)
    n_inner_hist = np.empty(NT, dtype=np.int32)
    conv_hist = np.empty(NT, dtype=bool)

    checkpoints = None
    if save_stride is not None and save_stride > 0:
        checkpoints = {0: (cp.asnumpy(P_old), cp.asnumpy(theta_old))}

    # --- Acceleration options (Phase 5 Part 1) -----------------------
    # None means "run the baseline code path". A provided dataclass
    # with mech_update_every == 1 also reduces to baseline (need_refresh
    # is True every iteration), but we still take the fast path so the
    # host-side guard bookkeeping is skipped.
    _accel = accel if accel is not None else AusasAccelerationOptions()
    mech_K = max(1, int(_accel.mech_update_every))
    lagged_mode_active = mech_K > 1
    mech_force_first = int(_accel.mech_force_first_iters)
    stall_enabled = bool(_accel.mech_force_if_residual_stalls)
    cav_jump_enabled = bool(_accel.mech_force_if_cav_jumps)
    stall_ratio = float(_accel.mech_residual_stall_ratio)
    cav_tol = float(_accel.mech_cav_jump_tol)

    # --- Time loop ---
    # Running values of WX, WY held on device; initialised to 0 for the
    # first step's "before any iteration" diagnostic.
    WX_dev = cp.array(0.0, dtype=cp.float64)
    WY_dev = cp.array(0.0, dtype=cp.float64)

    for step in range(1, NT + 1):
        n = n_offset + step                   # global step index
        t_n = t_offset + step * dt            # global time
        WaX_n, WaY_n = load_fn(t_n)
        WaX_n = float(WaX_n)
        WaY_n = float(WaY_n)

        # Freeze c_prev = theta^{n-1} * h^{n-1}.
        cp.multiply(theta_old, H_prev, out=C_prev)

        # Iterative predictor, all on device.
        # Start with X_k = X, Y_k = Y (no iteration yet).
        X_k_dev = X_dev
        Y_k_dev = Y_dev
        X_k_prev_dev = X_dev
        Y_k_prev_dev = Y_dev

        residual = float("inf")
        converged = False
        k_done = 0

        # --- Lagged-mechanics bookkeeping (only non-trivial when
        #     lagged_mode_active = mech_update_every > 1) ---
        last_residual = float("inf")
        residual_at_last_refresh = float("inf")
        cav_at_last_refresh = float("nan")
        cav_jump_pending = False
        did_mech_refresh = False

        for k in range(max_inner):
            check_iter = (k % check_every == 0) or (k < 3)

            # --- Decide whether mechanics (forces, predictor, H, A..E)
            #     get rebuilt this iteration. In baseline mode (K=1)
            #     this always fires and the guarded-refresh path
            #     collapses to the old code. ---
            if lagged_mode_active:
                residual_stalled = (
                    stall_enabled
                    and did_mech_refresh
                    and np.isfinite(last_residual)
                    and np.isfinite(residual_at_last_refresh)
                    and last_residual > stall_ratio * residual_at_last_refresh
                )
                need_refresh = (
                    k < mech_force_first
                    or (k % mech_K == 0)
                    or residual_stalled
                    or cav_jump_pending
                    or k >= max_inner - 3
                )
            else:
                need_refresh = True

            # Snapshot BEFORE the sweep, for RB post-sweep residual.
            if check_iter and scheme == "rb":
                P_new[:] = P_old
                theta_new[:] = theta_old

            # --- Mechanics (1)-(4): executed only on refresh iters. ---
            if need_refresh:
                # (1) Hydrodynamic forces — device 0-d reductions.
                WX_dev = d_phi_d_Z * cp.sum(P_old[1:-1, 1:-1] * cos_phi_interior[None, :])
                WY_dev = d_phi_d_Z * cp.sum(P_old[1:-1, 1:-1] * sin_phi_interior[None, :])

                # (2) Newmark predictor — device 0-d arithmetic.
                X_k_dev = X_dev + dt * U_dev + dt_sq_half * (WX_dev + WaX_n)
                Y_k_dev = Y_dev + dt * V_dev + dt_sq_half * (WY_dev + WaY_n)

                # (3) Rebuild gap via single-launch kernel.
                build_gap_kernel(
                    full_grid, block,
                    (
                        H_curr, X_k_dev, Y_k_dev,
                        cos_phi_1d, sin_phi_1d,
                        texture_for_kernel,
                        np.int32(N_Z), np.int32(N_phi),
                        np.int32(has_texture_flag),
                    ),
                )
                # (4) Rebuild coefficients A..E via single-launch kernel.
                build_coeffs_kernel(
                    full_grid, block,
                    (
                        H_curr, A, B, C, D, E,
                        np.float64(alpha_sq),
                        np.int32(N_Z), np.int32(N_phi),
                    ),
                )
                # Refresh bookkeeping (cheap host scalars).
                if lagged_mode_active:
                    residual_at_last_refresh = last_residual
                    cav_jump_pending = False
                    did_mech_refresh = True
                    if cav_jump_enabled:
                        # Capture current cav to measure future drift
                        # at check_iters. One extra sync per refresh.
                        cav_at_last_refresh = float(
                            cp.mean(
                                (theta_old < 1.0 - 1e-6).astype(cp.float64)
                            )
                        )

            # (5) + (6) One relaxation sweep + BC.
            if scheme == "jacobi":
                jac_kernel(
                    grid, block,
                    (
                        P_old, P_new, theta_old, theta_new,
                        H_curr, C_prev, A, B, C, D, E,
                        np.float64(d_phi), np.float64(d_Z),
                        np.float64(dt), np.float64(alpha),
                        np.float64(omega_p), np.float64(omega_theta),
                        np.int32(N_Z), np.int32(N_phi),
                        np.int32(1), np.int32(0),
                    ),
                )
                _launch_bc(
                    bc_phi_kernel, bc_z_kernel,
                    P_new, theta_new, N_Z, N_phi,
                    bc_block, bc_grid_phi, bc_grid_z,
                    True, False,
                    0.0, 0.0, 1.0, 1.0,
                    0.0, p_a, 1.0, 1.0,
                )
            else:
                for color in (0, 1):
                    rb_kernel(
                        grid, block,
                        (
                            P_old, theta_old,
                            H_curr, C_prev, A, B, C, D, E,
                            np.float64(d_phi), np.float64(d_Z),
                            np.float64(dt), np.float64(alpha),
                            np.float64(omega_p), np.float64(omega_theta),
                            np.int32(N_Z), np.int32(N_phi),
                            np.int32(1), np.int32(0),
                            np.int32(color),
                        ),
                    )
                    _launch_bc(
                        bc_phi_kernel, bc_z_kernel,
                        P_old, theta_old, N_Z, N_phi,
                        bc_block, bc_grid_phi, bc_grid_z,
                        True, False,
                        0.0, 0.0, 1.0, 1.0,
                        0.0, p_a, 1.0, 1.0,
                    )
            k_done = k + 1

            # (7) Convergence measurement (only on check iters).
            if check_iter:
                if scheme == "jacobi":
                    dP_dev = cp.sqrt(cp.sum((P_new - P_old) ** 2))
                    dth_dev = cp.sqrt(cp.sum((theta_new - theta_old) ** 2))
                else:
                    dP_dev = cp.sqrt(cp.sum((P_old - P_new) ** 2))
                    dth_dev = cp.sqrt(cp.sum((theta_old - theta_new) ** 2))
                dX_dev = cp.abs(X_k_dev - X_k_prev_dev)
                dY_dev = cp.abs(Y_k_dev - Y_k_prev_dev)
                # Single host sync per check (was 2 per iter before).
                residual = float(dP_dev + dth_dev + dX_dev + dY_dev)
                last_residual = residual
                if residual < tol_inner and k > 2:
                    converged = True

                # Cav-jump guard (lagged mode only): one extra sync
                # per check iter to decide whether mechanics drifted
                # enough to warrant an out-of-schedule refresh.
                if (
                    lagged_mode_active
                    and cav_jump_enabled
                    and did_mech_refresh
                    and np.isfinite(cav_at_last_refresh)
                ):
                    cav_now = float(
                        cp.mean(
                            (theta_old < 1.0 - 1e-6).astype(cp.float64)
                        )
                    )
                    if abs(cav_now - cav_at_last_refresh) > cav_tol:
                        cav_jump_pending = True

            # (8) Jacobi swap (no snapshot).
            if scheme == "jacobi":
                P_old, P_new = P_new, P_old
                theta_old, theta_new = theta_new, theta_old

            # Save current X_k for next iteration's |dX|/|dY| measure.
            # Rename-only: the old object stays alive as X_k_prev_dev
            # until the next rename overwrites it, then is GC'd.
            X_k_prev_dev = X_k_dev
            Y_k_prev_dev = Y_k_dev

            if converged:
                break

        # End-of-step velocity / position update — stays on device.
        U_dev = U_dev + dt_over_M * (WX_dev + WaX_n)
        V_dev = V_dev + dt_over_M * (WY_dev + WaY_n)
        X_dev = X_k_dev
        Y_dev = Y_k_dev

        # Advance H_prev <- H_curr (buffer swap).
        H_prev, H_curr = H_curr, H_prev

        # --- Scalar history: transfer to host ONCE per step, not per iter.
        t_hist[step - 1] = t_n
        X_hist[step - 1] = float(X_dev)
        Y_hist[step - 1] = float(Y_dev)
        U_hist[step - 1] = float(U_dev)
        V_hist[step - 1] = float(V_dev)
        WX_hist[step - 1] = float(WX_dev)
        WY_hist[step - 1] = float(WY_dev)
        WaX_hist[step - 1] = WaX_n
        WaY_hist[step - 1] = WaY_n
        p_max_hist[step - 1] = float(cp.max(P_old))
        h_min_hist[step - 1] = float(cp.min(H_prev[1:-1, 1:-1]))
        cav_hist[step - 1] = float(
            cp.mean((theta_old < 1.0 - 1e-6).astype(cp.float64))
        )
        n_inner_hist[step - 1] = k_done
        conv_hist[step - 1] = converged

        if verbose and (step <= 3 or step % max(NT // 20, 1) == 0):
            Xh = X_hist[step - 1]; Yh = Y_hist[step - 1]
            e_now = np.sqrt(Xh * Xh + Yh * Yh)
            print(
                f"  [step {n:>6d}] t={t_n:.4f} "
                f"X={Xh:+.3f} Y={Yh:+.3f} e={e_now:.3f} "
                f"WX={WX_hist[step-1]:+.2e} WY={WY_hist[step-1]:+.2e} "
                f"p_max={p_max_hist[step-1]:.3e} "
                f"h_min={h_min_hist[step-1]:.3e} "
                f"cav={cav_hist[step-1]:.3f} "
                f"inner={k_done:>4d} res={residual:.2e} "
                f"conv={'Y' if converged else 'N'}"
            )

        if checkpoints is not None and save_stride and (
            n % save_stride == 0 or step == NT
        ):
            checkpoints[n] = (cp.asnumpy(P_old), cp.asnumpy(theta_old))

        if field_callback is not None:
            field_callback(n, t_n, cp.asnumpy(P_old), cp.asnumpy(theta_old))

    # Build final state for restart (host copies of the last iterate).
    P_last = cp.asnumpy(P_old)
    theta_last = cp.asnumpy(theta_old)
    H_prev_last = cp.asnumpy(H_prev)
    final_state = AusasState(
        P=P_last,
        theta=theta_last,
        H_prev=H_prev_last,
        X=float(X_dev), Y=float(Y_dev),
        U=float(U_dev), V=float(V_dev),
        step_index=n_offset + NT,
        time=t_offset + NT * dt,
    )

    return AusasTransientResult(
        t=t_hist,
        p_max=p_max_hist,
        cav_frac=cav_hist,
        n_inner=n_inner_hist,
        converged=conv_hist,
        P_last=P_last,
        theta_last=theta_last,
        field_checkpoints=checkpoints,
        h_min=h_min_hist,
        X=X_hist, Y=Y_hist,
        U=U_hist, V=V_hist,
        WX=WX_hist, WY=WY_hist,
        WaX=WaX_hist, WaY=WaY_hist,
        final_state=final_state,
    )
