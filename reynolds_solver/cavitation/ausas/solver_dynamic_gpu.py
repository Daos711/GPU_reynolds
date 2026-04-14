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
    h_min_hist = np.empty(NT, dtype=np.float64)
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
        h_min_hist[n - 1] = float(cp.min(H_curr[1:-1, 1:-1]))
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
        h_min=h_min_hist,
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
    check_every: int = 1,
    save_stride: Optional[int] = None,
    field_callback: Optional[Callable[[int, float, np.ndarray, np.ndarray], None]] = None,
    verbose: bool = False,
) -> AusasTransientResult:
    """
    Fully dynamical journal-bearing solver (Ausas, Jai, Buscaglia 2008,
    Table 2 / Section 5).

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

    # --- Initial state ---
    X = float(X0)
    Y = float(Y0)
    U = float(U0)
    V = float(V0)

    _build_gap_inplace(H_prev, X, Y, cos_phi_1d, sin_phi_1d, texture_gpu)
    # Periodic phi is already exact from the shifted sampling above.
    # Apply a defensive pack anyway (cheap).
    _pack_ghosts(H_prev, periodic_phi=True, periodic_z=False)

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
    block, grid, bc_block, bc_grid_phi, bc_grid_z = _launch_configs(N_Z, N_phi)

    # --- History arrays ---
    t_hist = np.empty(NT, dtype=np.float64)
    X_hist = np.empty(NT, dtype=np.float64)
    Y_hist = np.empty(NT, dtype=np.float64)
    U_hist = np.empty(NT, dtype=np.float64)
    V_hist = np.empty(NT, dtype=np.float64)
    WX_hist = np.empty(NT, dtype=np.float64)
    WY_hist = np.empty(NT, dtype=np.float64)
    p_max_hist = np.empty(NT, dtype=np.float64)
    h_min_hist = np.empty(NT, dtype=np.float64)
    cav_hist = np.empty(NT, dtype=np.float64)
    n_inner_hist = np.empty(NT, dtype=np.int32)
    conv_hist = np.empty(NT, dtype=bool)

    checkpoints = None
    if save_stride is not None and save_stride > 0:
        checkpoints = {0: (cp.asnumpy(P_old), cp.asnumpy(theta_old))}

    # Shorthand for the force-integral normalisation.
    # Sum is over INTERIOR cells so we don't double-count ghost wraps.
    def _compute_forces(Parr):
        inner = Parr[1:-1, 1:-1]
        cx = cos_phi_1d[1:-1][None, :]
        sx = sin_phi_1d[1:-1][None, :]
        WX = float(d_phi * d_Z * cp.sum(inner * cx))
        WY = float(d_phi * d_Z * cp.sum(inner * sx))
        return WX, WY

    # --- Time loop ---
    for n in range(1, NT + 1):
        t_n = n * dt
        WaX_n, WaY_n = load_fn(t_n)
        WaX_n = float(WaX_n)
        WaY_n = float(WaY_n)

        # Freeze c_prev from the previous converged state.
        cp.multiply(theta_old, H_prev, out=C_prev)

        # Iterative predictor: X_k evolves along with the pressure field.
        X_k = X
        Y_k = Y
        X_k_prev = X_k
        Y_k_prev = Y_k

        residual = float("inf")
        converged = False
        k_done = 0
        WX = 0.0
        WY = 0.0

        for k in range(max_inner):
            # (1) Forces from the CURRENT P iterate.
            WX, WY = _compute_forces(P_old)

            # (2) Predict shaft position with Newmark-type update.
            X_k = X + dt * U + (dt * dt) / (2.0 * mass_M) * (WX + WaX_n)
            Y_k = Y + dt * V + (dt * dt) / (2.0 * mass_M) * (WY + WaY_n)

            # (3) Rebuild gap for this predicted position.
            _build_gap_inplace(H_curr, X_k, Y_k, cos_phi_1d, sin_phi_1d, texture_gpu)
            # Ghosts are exact by construction, but pack defensively.
            _pack_ghosts(H_curr, periodic_phi=True, periodic_z=False)

            # (4) Rebuild stencil coefficients.
            A[:], B[:], C[:], D[:], E[:] = _build_coefficients_gpu(
                H_curr, d_phi, d_Z, R, L
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
                        np.int32(1), np.int32(0),   # periodic_phi=1, periodic_z=0
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

            # (7) Convergence measurement (L2 field norms + scalar shaft
            #     position deltas, per Table 2 of the paper).
            if k % check_every == 0 or k < 3:
                if scheme == "jacobi":
                    dP = float(cp.sqrt(cp.sum((P_new - P_old) ** 2)))
                    dth = float(cp.sqrt(cp.sum((theta_new - theta_old) ** 2)))
                else:
                    dP = float(cp.sqrt(cp.sum((P_old - P_new) ** 2)))
                    dth = float(cp.sqrt(cp.sum((theta_old - theta_new) ** 2)))
                dX = abs(X_k - X_k_prev)
                dY = abs(Y_k - Y_k_prev)
                residual = dP + dth + dX + dY
                if residual < tol_inner and k > 2:
                    converged = True

            # (8) Swap / snapshot.
            if scheme == "jacobi":
                P_old, P_new = P_new, P_old
                theta_old, theta_new = theta_new, theta_old
            else:
                P_new[:] = P_old
                theta_new[:] = theta_old

            X_k_prev = X_k
            Y_k_prev = Y_k

            if converged:
                break

        # End-of-step velocity update using the CONVERGED W_X, W_Y.
        U = U + dt / mass_M * (WX + WaX_n)
        V = V + dt / mass_M * (WY + WaY_n)
        X = X_k
        Y = Y_k

        # Advance H_prev <- H_curr (buffer swap) for the next step's c_prev.
        H_prev, H_curr = H_curr, H_prev

        # --- Scalar history ---
        t_hist[n - 1] = t_n
        X_hist[n - 1] = X
        Y_hist[n - 1] = Y
        U_hist[n - 1] = U
        V_hist[n - 1] = V
        WX_hist[n - 1] = WX
        WY_hist[n - 1] = WY
        p_max_hist[n - 1] = float(cp.max(P_old))
        h_min_hist[n - 1] = float(cp.min(H_prev[1:-1, 1:-1]))
        cav_hist[n - 1] = float(
            cp.mean((theta_old < 1.0 - 1e-6).astype(cp.float64))
        )
        n_inner_hist[n - 1] = k_done
        conv_hist[n - 1] = converged

        if verbose and (n <= 3 or n % max(NT // 20, 1) == 0):
            e_now = np.sqrt(X * X + Y * Y)
            print(
                f"  [step {n:>6d}/{NT}] t={t_n:.4f} "
                f"X={X:+.3f} Y={Y:+.3f} e={e_now:.3f} "
                f"WX={WX:+.2e} WY={WY:+.2e} "
                f"p_max={p_max_hist[n-1]:.3e} "
                f"h_min={h_min_hist[n-1]:.3e} "
                f"cav={cav_hist[n-1]:.3f} "
                f"inner={k_done:>4d} res={residual:.2e} "
                f"conv={'Y' if converged else 'N'}"
            )

        if checkpoints is not None and save_stride and (
            n % save_stride == 0 or n == NT
        ):
            checkpoints[n] = (cp.asnumpy(P_old), cp.asnumpy(theta_old))

        if field_callback is not None:
            field_callback(n, t_n, cp.asnumpy(P_old), cp.asnumpy(theta_old))

    return AusasTransientResult(
        t=t_hist,
        p_max=p_max_hist,
        cav_frac=cav_hist,
        n_inner=n_inner_hist,
        converged=conv_hist,
        P_last=cp.asnumpy(P_old),
        theta_last=cp.asnumpy(theta_old),
        field_checkpoints=checkpoints,
        h_min=h_min_hist,
        X=X_hist, Y=Y_hist,
        U=U_hist, V=V_hist,
        WX=WX_hist, WY=WY_hist,
    )
