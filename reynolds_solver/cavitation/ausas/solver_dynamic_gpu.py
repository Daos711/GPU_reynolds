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
    `solve_ausas_prescribed_h_gpu`.

    Memory: scalars are O(NT) floats (NT = number of time steps).  The
    field `P_last` / `theta_last` is a single snapshot. `field_checkpoints`
    is populated only when `save_stride` is given; keys are 1-indexed step
    numbers (+ 0 for the initial state).
    """
    t: np.ndarray
    p_max: np.ndarray
    cav_frac: np.ndarray
    n_inner: np.ndarray
    converged: np.ndarray
    P_last: np.ndarray
    theta_last: np.ndarray
    field_checkpoints: Optional[dict]


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
    verbose: bool = False,
) -> AusasTransientResult:
    """
    Full time loop: for n = 1..NT advance (P, θ) under a prescribed gap
    h(t) supplied by `H_provider(n, t_n)`.

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
    # --- Initial gap and dimensions ----------------------------------------
    H_prev_cpu = np.asarray(H_provider(0, 0.0), dtype=np.float64)
    N_Z, N_phi = H_prev_cpu.shape
    if N_Z < 3 or N_phi < 3:
        raise ValueError(
            f"Grid too small: ({N_Z}, {N_phi}) — need at least 3 on each axis."
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

    # Initial (P, θ) ........................................................
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
    cav_hist = np.empty(NT, dtype=np.float64)
    n_inner_hist = np.empty(NT, dtype=np.int32)
    conv_hist = np.empty(NT, dtype=bool)

    checkpoints: Optional[dict] = None
    if save_stride is not None and save_stride > 0:
        checkpoints = {0: (cp.asnumpy(P_old), cp.asnumpy(theta_old))}

    # --- Time loop ---------------------------------------------------------
    for n in range(1, NT + 1):
        t_n = n * dt

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

        # Scalar history for this step.
        t_hist[n - 1] = t_n
        p_max_hist[n - 1] = float(cp.max(P_old))
        cav_hist[n - 1] = float(
            cp.mean((theta_old < 1.0 - 1e-6).astype(cp.float64))
        )
        n_inner_hist[n - 1] = k_done
        conv_hist[n - 1] = converged

        if verbose and (n <= 3 or n % max(NT // 20, 1) == 0):
            print(
                f"  [step {n:>6d}/{NT}] t={t_n:.5f} "
                f"p_max={p_max_hist[n-1]:.4e} "
                f"cav={cav_hist[n-1]:.3f} "
                f"inner={k_done:>4d} "
                f"res={residual:.2e} "
                f"conv={'Y' if converged else 'N'}"
            )

        if checkpoints is not None and save_stride and (n % save_stride == 0 or n == NT):
            checkpoints[n] = (cp.asnumpy(P_old), cp.asnumpy(theta_old))

        if field_callback is not None:
            field_callback(n, t_n, cp.asnumpy(P_old), cp.asnumpy(theta_old))

        # Advance H_prev ← H_curr (swap buffers so the next step overwrites
        # the stale one).
        H_prev, H_curr = H_curr, H_prev

    P_last = cp.asnumpy(P_old)
    theta_last = cp.asnumpy(theta_old)

    return AusasTransientResult(
        t=t_hist,
        p_max=p_max_hist,
        cav_frac=cav_hist,
        n_inner=n_inner_hist,
        converged=conv_hist,
        P_last=P_last,
        theta_last=theta_last,
        field_checkpoints=checkpoints,
    )
