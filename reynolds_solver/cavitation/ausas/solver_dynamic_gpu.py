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

# ---------------------------------------------------------------------------
# Debug / nonfinite-trace helpers (Task 28). All gated by debug_checks=True
# in the one-step wrapper so production path stays untouched.
# ---------------------------------------------------------------------------
def _classify_index(i: int, j: int, N_Z: int, N_phi: int):
    """
    Three-bool localization for a (i, j) cell on the padded one-step grid.

    `first_nan_is_ghost`           — index is inside the phi ghost column
                                     (j == 0 or j == N_phi - 1).
    `first_nan_is_axial_boundary`  — index is on the z-boundary row
                                     (i == 0 or i == N_Z - 1).
    `first_nan_is_phi_seam`        — index is in the ghost column OR in a
                                     physical seam-adjacent interior column
                                     (j == 1 or j == N_phi - 2).
    """
    is_ghost = (j == 0) or (j == N_phi - 1)
    is_axial_boundary = (i == 0) or (i == N_Z - 1)
    is_phi_seam = is_ghost or (j == 1) or (j == N_phi - 2)
    return bool(is_ghost), bool(is_axial_boundary), bool(is_phi_seam)


def _npy_first_violation(arr, lo=None, hi=None):
    """First (i, j) where `arr` is non-finite or out of [lo, hi], or None.

    `arr` must be a numpy ndarray. Used for input-state validation before
    we touch the GPU.
    """
    bad = ~np.isfinite(arr)
    if lo is not None:
        bad = bad | (arr < lo)
    if hi is not None:
        bad = bad | (arr > hi)
    if not bool(np.any(bad)):
        return None
    flat = int(np.argmax(bad.astype(np.uint8)))
    j = flat % arr.shape[1]
    i = flat // arr.shape[1]
    return (i, j)


def _gpu_first_violation(arr_gpu, lo=None, hi=None):
    """First (i, j) where `arr_gpu` is non-finite or out of [lo, hi], or None.

    Mirrors `_npy_first_violation` but runs on a CuPy device array.
    """
    bad = ~cp.isfinite(arr_gpu)
    if lo is not None:
        bad = bad | (arr_gpu < lo)
    if hi is not None:
        bad = bad | (arr_gpu > hi)
    if not bool(cp.any(bad)):
        return None
    flat = int(cp.argmax(bad.astype(cp.uint8)))
    j = flat % arr_gpu.shape[1]
    i = flat // arr_gpu.shape[1]
    return (i, j)


def _empty_failure_metadata():
    """Default (no-failure) values for the debug metadata fields."""
    return {
        "failure_kind": None,
        "first_nan_field": None,
        "first_nan_index": None,
        "first_nan_is_ghost": False,
        "first_nan_is_axial_boundary": False,
        "first_nan_is_phi_seam": False,
        "nan_iter": None,
    }


def _validate_one_step_inputs(
    H_curr,
    H_prev,
    P_prev,
    theta_prev,
    dt,
    d_phi,
    d_Z,
    omega_p,
    omega_theta,
    p_eps,
    theta_eps,
):
    """
    Run on the host side before any GPU work. Returns (failure_meta, None)
    on the first detected violation (no GPU side effects), or
    (None, None) if everything is valid.

    `failure_meta` matches the layout produced by
    `_empty_failure_metadata` / consumed by the dict-return path.
    """
    N_Z, N_phi = H_curr.shape

    # Scalar checks first — these don't have a (i, j) to report.
    for name, val, lo in [
        ("dt", dt, 0.0),
        ("d_phi", d_phi, 0.0),
        ("d_Z", d_Z, 0.0),
    ]:
        if not np.isfinite(val) or val <= lo:
            meta = _empty_failure_metadata()
            meta.update(
                failure_kind="invalid_input",
                first_nan_field=name,
                nan_iter=0,
            )
            return meta
    for name, val in [("omega_p", omega_p), ("omega_theta", omega_theta)]:
        if not np.isfinite(val):
            meta = _empty_failure_metadata()
            meta.update(
                failure_kind="invalid_input",
                first_nan_field="omega",
                nan_iter=0,
            )
            return meta

    # Field checks — H must be finite and strictly positive, P must be
    # >= -p_eps, theta must be in [-theta_eps, 1 + theta_eps].
    for name, arr in [
        ("H_prev", H_prev),
        ("H_curr", H_curr),
    ]:
        idx = _npy_first_violation(arr, lo=0.0)
        if idx is not None:
            i, j = idx
            is_ghost, is_axial, is_seam = _classify_index(i, j, N_Z, N_phi)
            meta = _empty_failure_metadata()
            meta.update(
                failure_kind="invalid_input",
                first_nan_field=name,
                first_nan_index=(int(i), int(j)),
                first_nan_is_ghost=is_ghost,
                first_nan_is_axial_boundary=is_axial,
                first_nan_is_phi_seam=is_seam,
                nan_iter=0,
            )
            return meta

    idx = _npy_first_violation(P_prev, lo=-p_eps)
    if idx is not None:
        i, j = idx
        is_ghost, is_axial, is_seam = _classify_index(i, j, N_Z, N_phi)
        meta = _empty_failure_metadata()
        meta.update(
            failure_kind="invalid_input",
            first_nan_field="P_prev",
            first_nan_index=(int(i), int(j)),
            first_nan_is_ghost=is_ghost,
            first_nan_is_axial_boundary=is_axial,
            first_nan_is_phi_seam=is_seam,
            nan_iter=0,
        )
        return meta

    idx = _npy_first_violation(theta_prev, lo=-theta_eps, hi=1.0 + theta_eps)
    if idx is not None:
        i, j = idx
        is_ghost, is_axial, is_seam = _classify_index(i, j, N_Z, N_phi)
        meta = _empty_failure_metadata()
        meta.update(
            failure_kind="invalid_input",
            first_nan_field="theta_prev",
            first_nan_index=(int(i), int(j)),
            first_nan_is_ghost=is_ghost,
            first_nan_is_axial_boundary=is_axial,
            first_nan_is_phi_seam=is_seam,
            nan_iter=0,
        )
        return meta

    return None


def _check_state_gpu(P_gpu, theta_gpu, p_eps, theta_eps):
    """
    Locate the first violation on the GPU latest-iterate buffer.

    Returns (failure_kind, field_name, (i, j)) on the first violation, or
    None if everything is finite and within bounds. Priority order:
        1. P non-finite                -> ("nonfinite_state", "P", idx)
        2. theta non-finite            -> ("nonfinite_state", "theta", idx)
        3. P out of [-p_eps, +inf]     -> ("out_of_range", "P", idx)
        4. theta out of [-eps, 1+eps]  -> ("out_of_range", "theta", idx)
    """
    idx = _gpu_first_violation(P_gpu)
    if idx is not None:
        return "nonfinite_state", "P", idx
    idx = _gpu_first_violation(theta_gpu)
    if idx is not None:
        return "nonfinite_state", "theta", idx
    idx = _gpu_first_violation(P_gpu, lo=-p_eps)
    if idx is not None:
        return "out_of_range", "P", idx
    idx = _gpu_first_violation(theta_gpu, lo=-theta_eps, hi=1.0 + theta_eps)
    if idx is not None:
        return "out_of_range", "theta", idx
    return None


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
    scheme: str = "rb",
    residual_norm: str = "linf",
    legacy_return: bool = False,
    debug_checks: bool = False,
    debug_check_every: int = 50,
    debug_check_start_iter: int = 0,
    debug_stop_on_nonfinite: bool = True,
    debug_return_bad_state: bool = False,
    debug_return_last_finite_state: bool = True,
    debug_theta_eps: float = 1e-8,
    debug_p_eps: float = 1e-12,
):
    """
    Advance (P, θ) by one real time step Δt using the unsteady Ausas
    kernel. See module docstring for discretization details.

    Axis-BC semantics: each axis (φ, Z) is either periodic or Dirichlet.
    For a Dirichlet axis, (p_bc_<axis>0, p_bc_<axis>L) and the matching
    θ-Dirichlet values are written into the ghost row/column by the BC
    kernel after each sweep.

    Parameters
    ----------
    scheme : {"rb", "jacobi"}
        Inner-loop relaxation. "rb" (default) is the in-place Red-Black
        SOR path used by the Stage-2 prescribed-h solver — converges in
        O(N) sweeps and is the recommended path for diesel transients.
        "jacobi" keeps the legacy frozen-iterate Jacobi sweep for
        reference / regression checks.
    residual_norm : {"linf", "rms"}
        Stopping criterion. Both norms are computed and reported; this
        flag selects which one is checked against `tol`. The legacy
        absolute L2 sum (sqrt(sum dP^2) + sqrt(sum dtheta^2)) — which
        scales with sqrt(N_interior) and is the source of false
        non-convergence reports on finer grids — is reported as
        `residual_l2_abs` for diagnostic comparison only and is NEVER
        used as a stopping criterion.
    legacy_return : bool
        If True, return the historical 4-tuple
        (P, theta, residual, n_inner) for callers that haven't been
        migrated to the structured-dict API yet.
    debug_checks : bool
        Enable host-side input validation and periodic GPU-side
        nonfinite / out-of-range checks during the inner loop. Default
        False keeps the production path untouched. See Task 28 of the
        diesel-debug ladder.
    debug_check_every : int
        How often (in inner-loop iterations) to run the GPU-side debug
        check. The check is also run on iter 0. Has no effect when
        ``debug_checks=False``.
    debug_check_start_iter : int
        Defer GPU debug checks until iteration index >= this value.
        Default 0. Used by `scripts/replay_ausas_one_step_dump.py` to do
        a coarse-then-fine refinement of `nan_iter` without paying the
        per-sweep check cost on iterations preceding a known coarse
        window.
    debug_stop_on_nonfinite : bool
        If True (default), abort the inner loop the moment a
        non-finite or out-of-range cell is detected. The returned dict
        carries the failure metadata; the inner loop does not continue.
    debug_return_bad_state : bool
        If True, snapshot the offending GPU state to CPU and return it
        as ``P_bad`` / ``theta_bad`` in the result dict.
    debug_return_last_finite_state : bool
        If True (default), keep a CPU snapshot of the latest fully-finite
        in-bounds state and return it as ``P`` / ``theta`` even when the
        run aborts on a debug failure. Cannot silently propagate NaN
        into the result.
    debug_theta_eps, debug_p_eps : float
        Numerical slack for the input/state validators.

    Returns
    -------
    dict (default) with keys:
        "P", "theta" : (N_Z, N_phi) ndarray (CPU, float64) — last-finite
            iterate when debug_return_last_finite_state is True and a
            failure was detected; otherwise the latest iterate.
        "residual", "residual_linf", "residual_rms",
        "residual_l2_abs" : float
        "n_inner" : int — sweeps actually performed.
        "converged" : bool — explicit convergence flag.
        "failure_kind" : Optional[str] — None on success, otherwise one
            of {"invalid_input", "nonfinite_state", "out_of_range"}.
        "first_nan_field" : Optional[str] — None on success, otherwise
            one of {"P", "theta", "residual", "coeff", "H_prev",
            "H_curr", "P_prev", "theta_prev", "input", "dt", "d_phi",
            "d_Z", "omega"}.
        "first_nan_index" : Optional[Tuple[int, int]]
        "first_nan_is_ghost" : bool
        "first_nan_is_axial_boundary" : bool
        "first_nan_is_phi_seam" : bool
        "nan_iter" : Optional[int]
        "P_bad", "theta_bad" : present only when debug_return_bad_state
            is True AND a failure was recorded.
    """
    if scheme not in ("rb", "jacobi"):
        raise ValueError(f"Unknown scheme {scheme!r}. Use 'rb' or 'jacobi'.")
    if residual_norm not in ("linf", "rms"):
        raise ValueError(
            f"Unknown residual_norm {residual_norm!r}. Use 'linf' or 'rms'."
        )

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

    failure_meta = _empty_failure_metadata()

    if debug_checks:
        meta = _validate_one_step_inputs(
            H_curr, H_prev, P_prev, theta_prev,
            dt, d_phi, d_Z, omega_p, omega_theta,
            debug_p_eps, debug_theta_eps,
        )
        if meta is not None and debug_stop_on_nonfinite:
            # Bail out before allocating GPU memory. Return clean dict
            # with NaN residuals — there were no iterations to measure.
            zero_field = np.zeros((N_Z, N_phi), dtype=np.float64)
            result = {
                "P": zero_field,
                "theta": zero_field.copy(),
                "residual": float("nan"),
                "residual_linf": float("nan"),
                "residual_rms": float("nan"),
                "residual_l2_abs": float("nan"),
                "n_inner": 0,
                "converged": False,
            }
            result.update(meta)
            if debug_return_bad_state:
                result["P_bad"] = P_prev.copy()
                result["theta_bad"] = theta_prev.copy()
            return result

    H_curr_gpu = cp.asarray(H_curr)
    H_prev_gpu = cp.asarray(H_prev)
    _pack_ghosts(H_curr_gpu, periodic_phi, periodic_z)
    _pack_ghosts(H_prev_gpu, periodic_phi, periodic_z)

    theta_prev_gpu = cp.asarray(theta_prev)
    C_prev = theta_prev_gpu * H_prev_gpu

    A, B, C, D, E = _build_coefficients_gpu(H_curr_gpu, d_phi, d_Z, R, L)

    # Coefficient finite check (debug only). The CUDA kernel divides by E
    # internally; a non-finite coefficient produces silent NaN propagation
    # that shows up downstream as residual=NaN.
    if debug_checks:
        for cname, carr in (("A", A), ("B", B), ("C", C), ("D", D), ("E", E)):
            idx = _gpu_first_violation(carr)
            if idx is not None:
                i, j = idx
                is_ghost, is_axial, is_seam = _classify_index(
                    i, j, N_Z, N_phi
                )
                failure_meta.update(
                    failure_kind="nonfinite_state",
                    first_nan_field="coeff",
                    first_nan_index=(int(i), int(j)),
                    first_nan_is_ghost=is_ghost,
                    first_nan_is_axial_boundary=is_axial,
                    first_nan_is_phi_seam=is_seam,
                    nan_iter=0,
                )
                if debug_stop_on_nonfinite:
                    zero_field = np.zeros((N_Z, N_phi), dtype=np.float64)
                    result = {
                        "P": zero_field,
                        "theta": zero_field.copy(),
                        "residual": float("nan"),
                        "residual_linf": float("nan"),
                        "residual_rms": float("nan"),
                        "residual_l2_abs": float("nan"),
                        "n_inner": 0,
                        "converged": False,
                    }
                    result.update(failure_meta)
                    if debug_return_bad_state:
                        result["P_bad"] = P_prev.copy()
                        result["theta_bad"] = theta_prev.copy()
                    return result

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

    if scheme == "jacobi":
        jac_kernel = get_unsteady_ausas_kernel()
        rb_kernel = None
    else:
        jac_kernel = None
        rb_kernel = get_unsteady_ausas_rb_kernel()
    bc_phi_kernel = get_unsteady_ausas_bc_phi_kernel()
    bc_z_kernel = get_unsteady_ausas_bc_z_kernel()
    block, grid, bc_block, bc_grid_phi, bc_grid_z = _launch_configs(N_Z, N_phi)

    pphi = 1 if periodic_phi else 0
    pz = 1 if periodic_z else 0

    # Interior cell count for the RMS denominator. For both periodic and
    # Dirichlet boundaries the iterate-difference is well-defined on the
    # interior slice [1:-1, 1:-1]; ghost rows/cols carry either a copy of
    # the seam (periodic) or a constant (Dirichlet) and don't need to be
    # included in the residual.
    n_interior = max((N_Z - 2) * (N_phi - 2), 1)

    residual_linf = float("inf")
    residual_rms = float("inf")
    residual_l2_abs = float("inf")
    converged = False
    n_inner = 0

    # Last-finite-state snapshot machinery (Task 28). Initialised from the
    # clamped/BC-corrected starting iterate; refreshed after every clean
    # debug check.
    last_finite_P_cpu = None
    last_finite_theta_cpu = None
    if debug_checks and debug_return_last_finite_state:
        last_finite_P_cpu = cp.asnumpy(P_old)
        last_finite_theta_cpu = cp.asnumpy(theta_old)
    P_bad_cpu = None
    theta_bad_cpu = None
    debug_abort = False

    for k in range(max_inner):
        if scheme == "jacobi":
            jac_kernel(
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
        else:
            for color in (0, 1):
                rb_kernel(
                    grid, block,
                    (
                        P_old, theta_old,
                        H_curr_gpu, C_prev,
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
        n_inner += 1

        residual_just_computed = False
        if k % check_every == 0 or k < 3:
            residual_just_computed = True
            # For Jacobi the latest iterate is in (P_new, theta_new); for
            # RB the latest iterate is in (P_old, theta_old) and the
            # previous-iterate snapshot is in (P_new, theta_new).
            if scheme == "jacobi":
                dP_arr = P_new[1:-1, 1:-1] - P_old[1:-1, 1:-1]
                dth_arr = theta_new[1:-1, 1:-1] - theta_old[1:-1, 1:-1]
            else:
                dP_arr = P_old[1:-1, 1:-1] - P_new[1:-1, 1:-1]
                dth_arr = theta_old[1:-1, 1:-1] - theta_new[1:-1, 1:-1]

            max_abs_dP = float(cp.max(cp.abs(dP_arr)))
            max_abs_dth = float(cp.max(cp.abs(dth_arr)))
            sum_dP2 = float(cp.sum(dP_arr * dP_arr))
            sum_dth2 = float(cp.sum(dth_arr * dth_arr))

            residual_linf = max(max_abs_dP, max_abs_dth)
            residual_rms = float(np.sqrt((sum_dP2 + sum_dth2) / n_interior))
            residual_l2_abs = float(np.sqrt(sum_dP2) + np.sqrt(sum_dth2))

            primary = residual_linf if residual_norm == "linf" else residual_rms
            if verbose:
                print(
                    f"  [Ausas-dyn-GPU/{scheme}] inner={k:>5d}: "
                    f"res_linf={residual_linf:.3e}  "
                    f"res_rms={residual_rms:.3e}  "
                    f"res_l2_abs={residual_l2_abs:.3e}  "
                    f"maxP={float(cp.max(P_new if scheme == 'jacobi' else P_old)):.3e}"
                )
            if primary < tol and k > 2:
                converged = True

        if scheme == "jacobi":
            P_old, P_new = P_new, P_old
            theta_old, theta_new = theta_new, theta_old
        else:
            # Refresh the snapshot only when the NEXT iteration will run a
            # residual check. Saves a full-array copy on every sweep.
            if (k + 1) % check_every == 0 or (k + 1) < 3:
                P_new[:] = P_old
                theta_new[:] = theta_old

        # ------------------------------------------------------------------
        # Debug-mode in-loop checks (Task 28). After the post-iteration
        # swap (Jacobi) or in-place update (RB), the latest iterate is
        # always in (P_old, theta_old) — so the check is buffer-uniform.
        # ------------------------------------------------------------------
        if debug_checks and not debug_abort:
            do_debug_check = (
                k >= debug_check_start_iter
                and ((k - debug_check_start_iter) % debug_check_every == 0)
            )

            # Residual NaN/Inf is a real failure — flag it the moment it
            # appears, regardless of debug_check_every cadence.
            if residual_just_computed and not (
                np.isfinite(residual_linf)
                and np.isfinite(residual_rms)
                and np.isfinite(residual_l2_abs)
            ):
                failure_meta.update(
                    failure_kind="nonfinite_state",
                    first_nan_field="residual",
                    first_nan_index=None,
                    first_nan_is_ghost=False,
                    first_nan_is_axial_boundary=False,
                    first_nan_is_phi_seam=False,
                    nan_iter=int(k),
                )
                if debug_return_bad_state:
                    P_bad_cpu = cp.asnumpy(P_old)
                    theta_bad_cpu = cp.asnumpy(theta_old)
                if debug_stop_on_nonfinite:
                    debug_abort = True

            if not debug_abort and do_debug_check:
                state_violation = _check_state_gpu(
                    P_old, theta_old, debug_p_eps, debug_theta_eps,
                )
                if state_violation is None:
                    if debug_return_last_finite_state:
                        last_finite_P_cpu = cp.asnumpy(P_old)
                        last_finite_theta_cpu = cp.asnumpy(theta_old)
                else:
                    fkind, ffield, (fi, fj) = state_violation
                    is_ghost, is_axial, is_seam = _classify_index(
                        fi, fj, N_Z, N_phi
                    )
                    failure_meta.update(
                        failure_kind=fkind,
                        first_nan_field=ffield,
                        first_nan_index=(int(fi), int(fj)),
                        first_nan_is_ghost=is_ghost,
                        first_nan_is_axial_boundary=is_axial,
                        first_nan_is_phi_seam=is_seam,
                        nan_iter=int(k),
                    )
                    if debug_return_bad_state:
                        P_bad_cpu = cp.asnumpy(P_old)
                        theta_bad_cpu = cp.asnumpy(theta_old)
                    if debug_stop_on_nonfinite:
                        debug_abort = True

        if converged:
            if verbose:
                print(
                    f"  [Ausas-dyn-GPU/{scheme}] CONVERGED at inner={k}, "
                    f"res_linf={residual_linf:.3e} res_rms={residual_rms:.3e}"
                )
            break

        if debug_abort:
            break

    # If we aborted on a debug failure with debug_return_last_finite_state
    # set, prefer the snapshot to avoid leaking NaN into the result. Other
    # paths return the latest GPU iterate.
    if (
        debug_checks
        and failure_meta["failure_kind"] is not None
        and debug_return_last_finite_state
        and last_finite_P_cpu is not None
    ):
        P_cpu = last_finite_P_cpu
        theta_cpu = last_finite_theta_cpu
    else:
        P_cpu = cp.asnumpy(P_old)
        theta_cpu = cp.asnumpy(theta_old)

    primary_residual = (
        residual_linf if residual_norm == "linf" else residual_rms
    )

    if legacy_return:
        return P_cpu, theta_cpu, float(primary_residual), n_inner

    result = {
        "P": P_cpu,
        "theta": theta_cpu,
        "residual": float(primary_residual),
        "residual_linf": float(residual_linf),
        "residual_rms": float(residual_rms),
        "residual_l2_abs": float(residual_l2_abs),
        "n_inner": int(n_inner),
        "converged": bool(converged),
    }
    result.update(failure_meta)
    if debug_return_bad_state and P_bad_cpu is not None:
        result["P_bad"] = P_bad_cpu
        result["theta_bad"] = theta_bad_cpu
    return result


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
    accel: Optional["AusasAccelerationOptions"] = None,
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

    # ==================================================================
    # Adaptive-dt branch (Phase 5 Part 2)
    # ==================================================================
    if accel is not None and getattr(accel, "adaptive_dt", False):
        import warnings

        # (P, theta) rollback buffers.
        P_rb_buf = cp.empty(shape, dtype=cp.float64)
        theta_rb_buf = cp.empty(shape, dtype=cp.float64)

        if state is not None and state.dt_last > 0.0:
            dt_current = float(state.dt_last)
        else:
            dt_current = float(dt)
        dt_current = min(max(dt_current, accel.dt_min), accel.dt_max)

        t_hist_list: list = []
        p_max_hist_list: list = []
        h_min_hist_list: list = []
        cav_hist_list: list = []
        n_inner_hist_list: list = []
        conv_hist_list: list = []
        dt_hist_list: list = []

        t_current = t_offset
        t_end = t_offset + NT * dt
        step_local = 0
        rejected_steps = 0
        consecutive_rejects = 0
        last_successful_dt = dt_current

        while t_current < t_end - 1e-14:
            dt_step = min(dt_current, t_end - t_current)
            if dt_step <= 0.0:
                break
            t_n = t_current + dt_step
            n_global = n_offset + step_local + 1

            H_curr_cpu = np.asarray(
                H_provider(n_global, t_n), dtype=np.float64
            )
            if H_curr_cpu.shape != shape:
                raise ValueError(
                    f"H_provider returned shape {H_curr_cpu.shape}, "
                    f"expected {shape}"
                )

            # Rollback snapshot.
            P_rb_buf[:] = P_old
            theta_rb_buf[:] = theta_old

            H_curr[:] = cp.asarray(H_curr_cpu)
            _pack_ghosts(H_curr, periodic_phi, periodic_z)

            cp.multiply(theta_old, H_prev, out=C_prev)

            A[:], B[:], C[:], D[:], E[:] = _build_coefficients_gpu(
                H_curr, d_phi, d_Z, R, L
            )

            # Seed P_new / theta_new for the RB residual snapshot.
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
                            H_curr, C_prev, A, B, C, D, E,
                            np.float64(d_phi), np.float64(d_Z),
                            np.float64(dt_step), np.float64(alpha),
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
                    for color in (0, 1):
                        rb_kernel(
                            grid, block,
                            (
                                P_old, theta_old,
                                H_curr, C_prev, A, B, C, D, E,
                                np.float64(d_phi), np.float64(d_Z),
                                np.float64(dt_step), np.float64(alpha),
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
                        dP = float(cp.max(cp.abs(P_old - P_new)))
                        dth = float(cp.max(cp.abs(theta_old - theta_new)))
                    residual = dP + dth
                    if residual < tol_inner and k > 2:
                        converged = True

                if scheme == "jacobi":
                    P_old, P_new = P_new, P_old
                    theta_old, theta_new = theta_new, theta_old
                else:
                    if (k + 1) % check_every == 0 or k + 1 < 3:
                        P_new[:] = P_old
                        theta_new[:] = theta_old

                if converged:
                    break

            # Accept / reject
            accepted = True
            if accel.reject_if_not_converged and not converged:
                accepted = False
            h_min_step = float(cp.min(H_curr[1:-1, 1:-1]))
            if accepted and h_min_step <= 0.0:
                accepted = False

            if not accepted:
                P_old[:] = P_rb_buf
                theta_old[:] = theta_rb_buf
                rejected_steps += 1
                consecutive_rejects += 1
                if dt_current <= accel.dt_min * (1.0 + 1e-9):
                    if consecutive_rejects >= 10:
                        raise RuntimeError(
                            "Adaptive dt (prescribed_h): 10 consecutive "
                            f"rejects at dt_min {accel.dt_min:.3e} at "
                            f"t={t_current:.4e}."
                        )
                    warnings.warn(
                        f"Adaptive dt stuck at dt_min near t={t_current:.4e}",
                        RuntimeWarning,
                    )
                else:
                    dt_current = max(
                        dt_current * accel.dt_shrink, accel.dt_min
                    )
                continue

            # Commit
            consecutive_rejects = 0
            H_prev, H_curr = H_curr, H_prev
            t_current = t_n
            step_local += 1
            last_successful_dt = dt_step

            t_hist_list.append(t_n)
            p_max_hist_list.append(float(cp.max(P_old)))
            h_min_hist_list.append(h_min_step)
            cav_hist_list.append(
                float(cp.mean((theta_old < 1.0 - 1e-6).astype(cp.float64)))
            )
            n_inner_hist_list.append(k_done)
            conv_hist_list.append(converged)
            dt_hist_list.append(dt_step)

            if verbose and (step_local <= 3 or step_local % 20 == 0):
                print(
                    f"  [step {n_global:>6d}] t={t_n:.5f} dt={dt_step:.2e} "
                    f"p_max={p_max_hist_list[-1]:.4e} "
                    f"cav={cav_hist_list[-1]:.3f} "
                    f"inner={k_done} conv={'Y' if converged else 'N'}"
                )

            if checkpoints is not None and save_stride and (
                n_global % save_stride == 0
            ):
                checkpoints[n_global] = (
                    cp.asnumpy(P_old), cp.asnumpy(theta_old),
                )
            if field_callback is not None:
                field_callback(
                    n_global, t_n,
                    cp.asnumpy(P_old), cp.asnumpy(theta_old),
                )

            if dt_step >= dt_current - 1e-15:
                if k_done < accel.target_inner_low:
                    dt_current = min(
                        dt_current * accel.dt_grow, accel.dt_max
                    )
                elif k_done > accel.target_inner_high:
                    dt_current = max(
                        dt_current * accel.dt_shrink, accel.dt_min
                    )

        P_last = cp.asnumpy(P_old)
        theta_last = cp.asnumpy(theta_old)
        H_prev_last = cp.asnumpy(H_prev)
        final_state = AusasState(
            P=P_last,
            theta=theta_last,
            H_prev=H_prev_last,
            step_index=n_offset + step_local,
            time=t_current,
            dt_last=last_successful_dt,
        )
        return AusasTransientResult(
            t=np.asarray(t_hist_list, dtype=np.float64),
            p_max=np.asarray(p_max_hist_list, dtype=np.float64),
            cav_frac=np.asarray(cav_hist_list, dtype=np.float64),
            n_inner=np.asarray(n_inner_hist_list, dtype=np.int32),
            converged=np.asarray(conv_hist_list, dtype=bool),
            P_last=P_last,
            theta_last=theta_last,
            field_checkpoints=checkpoints,
            h_min=np.asarray(h_min_hist_list, dtype=np.float64),
            final_state=final_state,
        )

    # ==================================================================
    # Fixed-dt branch (Phase 4) — unchanged
    # ==================================================================
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
        dt_last=float(dt),
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
    accel: Optional["AusasAccelerationOptions"] = None,
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

    Adaptive dt (Phase 5 Part 2)
    ----------------------------
    Passing `accel=AusasAccelerationOptions(adaptive_dt=True, ...)`
    enables variable-dt time marching with rollback. Instead of
    stepping NT times at the caller's `dt`, the solver uses
    `t_end = t_offset + NT * dt` as the target final time and varies
    dt inside [dt_min, dt_max] based on the inner-iteration count
    (target window: target_inner_low .. target_inner_high). A step is
    REJECTED when `reject_if_not_converged=True` and the inner loop
    did not converge, or when a physical invariant is violated
    (h_min <= 0 or e >= 1); on reject the (P, theta) fields are
    restored from a rollback snapshot, dt is shrunk by `dt_shrink`,
    and the step is retried. The returned histories have variable
    spacing; inspect `result.t` (1-D array of the accepted step times)
    and `result.final_state.dt_last` to see the schedule.
    With `accel=None` or `adaptive_dt=False` the solver runs the
    classical fixed-dt loop bit-for-bit identical to Phase 4.

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
    # Phase 4.1 / 5.1B kernels (single-launch gap, coeffs, forces,
    # predictor rebuilds).
    from reynolds_solver.cavitation.ausas.kernels_dynamic import (
        get_build_coefficients_kernel,
        get_build_gap_kernel,
        get_forces_reduce_kernel,
        get_newmark_predictor_kernel,
    )
    build_coeffs_kernel = get_build_coefficients_kernel()
    build_gap_kernel = get_build_gap_kernel()
    forces_kernel = get_forces_reduce_kernel()
    predictor_kernel = get_newmark_predictor_kernel()
    # Single-block reduction for the forces kernel. 256 threads is
    # plenty for all realistic grids (<= ~64k interior cells).
    forces_block = (256, 1, 1)

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

    # --- Preallocated 0-d device buffers for the fused predictor +
    #     forces kernels (Phase 5 Part 1 variant B). Using ping-pong
    #     buffers for X_k / Y_k keeps the old-iter value alive for the
    #     residual check while the new value is being written by the
    #     predictor. WX, WY are overwritten in place each iter. ---
    WX_dev = cp.array(0.0, dtype=cp.float64)
    WY_dev = cp.array(0.0, dtype=cp.float64)
    X_k_buf_a = cp.empty((), dtype=cp.float64)
    X_k_buf_b = cp.empty((), dtype=cp.float64)
    Y_k_buf_a = cp.empty((), dtype=cp.float64)
    Y_k_buf_b = cp.empty((), dtype=cp.float64)

    # --- Dynamic check cadence (Phase 5 Part 3) --------------------------
    # When `accel.dynamic_check_every=True` the solver varies the
    # residual-measurement cadence inside [check_every_min, check_every_max]
    # based on the per-check residual drop ratio. When False (or no
    # accel), the fixed `check_every` loop behaviour is preserved
    # bit-for-bit.
    _use_dynamic_check = (
        accel is not None and getattr(accel, "dynamic_check_every", False)
    )
    if _use_dynamic_check:
        _check_min = max(1, int(accel.check_every_min))
        _check_max = max(_check_min, int(accel.check_every_max))
        _check_every_init = min(max(check_every, _check_min), _check_max)
    else:
        _check_min = check_every
        _check_max = check_every
        _check_every_init = check_every

    # ==================================================================
    # Adaptive-dt branch (Phase 5 Part 2)
    # ==================================================================
    adaptive_mode = (
        accel is not None and getattr(accel, "adaptive_dt", False)
    )
    if adaptive_mode:
        import math
        import warnings

        # Rollback buffers for (P, theta). U, V, X, Y, H_prev are only
        # modified on COMMIT, so they don't need rollback.
        P_rb_buf = cp.empty(shape, dtype=cp.float64)
        theta_rb_buf = cp.empty(shape, dtype=cp.float64)

        # Initial dt: resume from state.dt_last if available, else
        # caller's dt. Clamp to [dt_min, dt_max].
        if state is not None and state.dt_last > 0.0:
            dt_current = float(state.dt_last)
        else:
            dt_current = float(dt)
        dt_current = min(max(dt_current, accel.dt_min), accel.dt_max)

        # Histories are Python lists — number of steps is unknown.
        t_hist_list: list = []
        X_hist_list: list = []
        Y_hist_list: list = []
        U_hist_list: list = []
        V_hist_list: list = []
        WX_hist_list: list = []
        WY_hist_list: list = []
        WaX_hist_list: list = []
        WaY_hist_list: list = []
        p_max_hist_list: list = []
        h_min_hist_list: list = []
        cav_hist_list: list = []
        n_inner_hist_list: list = []
        conv_hist_list: list = []
        dt_hist_list: list = []

        checkpoints = None
        if save_stride is not None and save_stride > 0:
            checkpoints = {0: (cp.asnumpy(P_old), cp.asnumpy(theta_old))}

        t_current = t_offset
        t_end = t_offset + NT * dt
        step_local = 0
        rejected_steps = 0
        consecutive_rejects = 0
        last_successful_dt = dt_current

        while t_current < t_end - 1e-14:
            # Clip dt to exactly land on t_end.
            dt_step = min(dt_current, t_end - t_current)
            if dt_step <= 0.0:
                break
            t_n = t_current + dt_step
            n_global = n_offset + step_local + 1

            WaX_n, WaY_n = load_fn(t_n)
            WaX_n = float(WaX_n)
            WaY_n = float(WaY_n)

            # Save (P, theta) rollback state.
            P_rb_buf[:] = P_old
            theta_rb_buf[:] = theta_old

            # Per-step dt-dependent scalars.
            dt_sq_half_step = dt_step * dt_step / (2.0 * mass_M)
            dt_over_M_step = dt_step / mass_M

            cp.multiply(theta_old, H_prev, out=C_prev)

            # Ping-pong X_k / Y_k init.
            X_k_prev_dev = X_k_buf_a
            X_k_dev = X_k_buf_b
            Y_k_prev_dev = Y_k_buf_a
            Y_k_dev = Y_k_buf_b
            X_k_prev_dev[...] = X_dev
            Y_k_prev_dev[...] = Y_dev

            residual = float("inf")
            converged = False
            k_done = 0

            # Dynamic check-cadence bookkeeping (Phase 5.3). When
            # `accel.dynamic_check_every=False` this degenerates to
            # the fixed `check_every` behaviour.
            _last_k_checked = -1
            _last_residual_check = float("inf")
            _check_cur = _check_every_init

            for k in range(max_inner):
                if _use_dynamic_check:
                    check_iter = (k < 3) or (k - _last_k_checked >= _check_cur)
                else:
                    check_iter = (k % check_every == 0) or (k < 3)
                if check_iter and scheme == "rb":
                    P_new[:] = P_old
                    theta_new[:] = theta_old

                # (1) Forces (fused).
                forces_kernel(
                    (1, 1, 1), forces_block,
                    (
                        P_old, cos_phi_1d, sin_phi_1d,
                        WX_dev, WY_dev,
                        np.float64(d_phi_d_Z),
                        np.int32(N_Z), np.int32(N_phi),
                    ),
                    shared_mem=2 * forces_block[0] * 8,
                )
                # (2) Newmark predictor (fused, uses dt_step).
                predictor_kernel(
                    (1, 1, 1), (1, 1, 1),
                    (
                        X_dev, Y_dev, U_dev, V_dev,
                        WX_dev, WY_dev,
                        X_k_dev, Y_k_dev,
                        np.float64(dt_step),
                        np.float64(dt_sq_half_step),
                        np.float64(WaX_n), np.float64(WaY_n),
                    ),
                )
                # (3) Rebuild gap.
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
                # (4) Rebuild coefficients.
                build_coeffs_kernel(
                    full_grid, block,
                    (
                        H_curr, A, B, C, D, E,
                        np.float64(alpha_sq),
                        np.int32(N_Z), np.int32(N_phi),
                    ),
                )
                # (5) + (6) Sweep + BC — use dt_step.
                if scheme == "jacobi":
                    jac_kernel(
                        grid, block,
                        (
                            P_old, P_new, theta_old, theta_new,
                            H_curr, C_prev, A, B, C, D, E,
                            np.float64(d_phi), np.float64(d_Z),
                            np.float64(dt_step), np.float64(alpha),
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
                                np.float64(dt_step), np.float64(alpha),
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

                if check_iter:
                    if scheme == "jacobi":
                        dP_dev = cp.sqrt(cp.sum((P_new - P_old) ** 2))
                        dth_dev = cp.sqrt(cp.sum((theta_new - theta_old) ** 2))
                    else:
                        dP_dev = cp.sqrt(cp.sum((P_old - P_new) ** 2))
                        dth_dev = cp.sqrt(cp.sum((theta_old - theta_new) ** 2))
                    dX_dev = cp.abs(X_k_dev - X_k_prev_dev)
                    dY_dev = cp.abs(Y_k_dev - Y_k_prev_dev)
                    residual = float(dP_dev + dth_dev + dX_dev + dY_dev)
                    # Dynamic cadence adjustment (no-op if flag off).
                    if _use_dynamic_check:
                        if (
                            _last_residual_check > 0.0
                            and np.isfinite(_last_residual_check)
                        ):
                            drop = residual / _last_residual_check
                            if drop < 0.5:
                                # big drop: check LESS often
                                _check_cur = min(_check_cur * 2, _check_max)
                            elif drop > 0.9:
                                # stall: check MORE often
                                _check_cur = max(_check_cur // 2, _check_min)
                        _last_residual_check = residual
                        _last_k_checked = k
                    if residual < tol_inner and k > 2:
                        converged = True

                if scheme == "jacobi":
                    P_old, P_new = P_new, P_old
                    theta_old, theta_new = theta_new, theta_old

                if converged:
                    break

                X_k_prev_dev, X_k_dev = X_k_dev, X_k_prev_dev
                Y_k_prev_dev, Y_k_dev = Y_k_dev, Y_k_prev_dev

            # --- Accept / reject decision ---
            accepted = True
            if accel.reject_if_not_converged and not converged:
                accepted = False

            # Physical invariants (one sync batch for the three scalars).
            h_min_step = float(cp.min(H_curr[1:-1, 1:-1]))
            X_step = float(X_k_dev)
            Y_step = float(Y_k_dev)
            e_step = math.sqrt(X_step * X_step + Y_step * Y_step)
            if accepted and (h_min_step <= 0.0 or e_step >= 1.0):
                accepted = False

            if not accepted:
                # Rollback P, theta. U, V, X, Y, H_prev stay at
                # pre-step state (they haven't been committed yet).
                P_old[:] = P_rb_buf
                theta_old[:] = theta_rb_buf
                rejected_steps += 1
                consecutive_rejects += 1

                if dt_current <= accel.dt_min * (1.0 + 1e-9):
                    # Already at floor — cannot shrink further.
                    if consecutive_rejects >= 10:
                        raise RuntimeError(
                            "Adaptive dt: 10 consecutive rejects at "
                            f"dt_min = {accel.dt_min:.3e} at t = "
                            f"{t_current:.4e}. Increase max_inner or "
                            "relax tol_inner."
                        )
                    warnings.warn(
                        f"Adaptive dt stuck at dt_min {accel.dt_min:.3e} "
                        f"near t={t_current:.4e} (retry {consecutive_rejects})",
                        RuntimeWarning,
                    )
                else:
                    dt_current = max(
                        dt_current * accel.dt_shrink, accel.dt_min
                    )
                continue

            # --- Commit ---
            consecutive_rejects = 0
            # End-of-step velocity update (uses dt_step).
            U_dev = U_dev + dt_over_M_step * (WX_dev + WaX_n)
            V_dev = V_dev + dt_over_M_step * (WY_dev + WaY_n)
            X_dev[...] = X_k_dev
            Y_dev[...] = Y_k_dev
            H_prev, H_curr = H_curr, H_prev

            t_current = t_n
            step_local += 1
            last_successful_dt = dt_step

            # History (lists — variable spacing).
            t_hist_list.append(t_n)
            X_hist_list.append(X_step)
            Y_hist_list.append(Y_step)
            U_hist_list.append(float(U_dev))
            V_hist_list.append(float(V_dev))
            WX_hist_list.append(float(WX_dev))
            WY_hist_list.append(float(WY_dev))
            WaX_hist_list.append(WaX_n)
            WaY_hist_list.append(WaY_n)
            p_max_hist_list.append(float(cp.max(P_old)))
            h_min_hist_list.append(h_min_step)
            cav_hist_list.append(
                float(cp.mean((theta_old < 1.0 - 1e-6).astype(cp.float64)))
            )
            n_inner_hist_list.append(k_done)
            conv_hist_list.append(converged)
            dt_hist_list.append(dt_step)

            if verbose and (step_local <= 3 or step_local % 20 == 0):
                print(
                    f"  [step {n_global:>6d}] t={t_n:.4f} dt={dt_step:.2e} "
                    f"X={X_step:+.3f} Y={Y_step:+.3f} "
                    f"e={e_step:.3f} inner={k_done} "
                    f"conv={'Y' if converged else 'N'}"
                )

            if checkpoints is not None and save_stride and (
                n_global % save_stride == 0
            ):
                checkpoints[n_global] = (
                    cp.asnumpy(P_old), cp.asnumpy(theta_old),
                )
            if field_callback is not None:
                field_callback(
                    n_global, t_n,
                    cp.asnumpy(P_old), cp.asnumpy(theta_old),
                )

            # Adjust dt for next step. Skip the adjustment if this step
            # was clipped to land on t_end (we're about to exit anyway).
            if dt_step >= dt_current - 1e-15:
                if k_done < accel.target_inner_low:
                    dt_current = min(
                        dt_current * accel.dt_grow, accel.dt_max
                    )
                elif k_done > accel.target_inner_high:
                    dt_current = max(
                        dt_current * accel.dt_shrink, accel.dt_min
                    )
                # else: keep dt

        # --- Adaptive: assemble result ---
        P_last = cp.asnumpy(P_old)
        theta_last = cp.asnumpy(theta_old)
        H_prev_last = cp.asnumpy(H_prev)
        final_state = AusasState(
            P=P_last,
            theta=theta_last,
            H_prev=H_prev_last,
            X=float(X_dev), Y=float(Y_dev),
            U=float(U_dev), V=float(V_dev),
            step_index=n_offset + step_local,
            time=t_current,
            dt_last=last_successful_dt,
        )
        return AusasTransientResult(
            t=np.asarray(t_hist_list, dtype=np.float64),
            p_max=np.asarray(p_max_hist_list, dtype=np.float64),
            cav_frac=np.asarray(cav_hist_list, dtype=np.float64),
            n_inner=np.asarray(n_inner_hist_list, dtype=np.int32),
            converged=np.asarray(conv_hist_list, dtype=bool),
            P_last=P_last,
            theta_last=theta_last,
            field_checkpoints=checkpoints,
            h_min=np.asarray(h_min_hist_list, dtype=np.float64),
            X=np.asarray(X_hist_list, dtype=np.float64),
            Y=np.asarray(Y_hist_list, dtype=np.float64),
            U=np.asarray(U_hist_list, dtype=np.float64),
            V=np.asarray(V_hist_list, dtype=np.float64),
            WX=np.asarray(WX_hist_list, dtype=np.float64),
            WY=np.asarray(WY_hist_list, dtype=np.float64),
            WaX=np.asarray(WaX_hist_list, dtype=np.float64),
            WaY=np.asarray(WaY_hist_list, dtype=np.float64),
            final_state=final_state,
        )

    # ==================================================================
    # Fixed-dt branch (Phase 4.1 / 5.1B) — unchanged from pre-Phase-5.2
    # ==================================================================
    for step in range(1, NT + 1):
        n = n_offset + step                   # global step index
        t_n = t_offset + step * dt            # global time
        WaX_n, WaY_n = load_fn(t_n)
        WaX_n = float(WaX_n)
        WaY_n = float(WaY_n)

        # Freeze c_prev = theta^{n-1} * h^{n-1}.
        cp.multiply(theta_old, H_prev, out=C_prev)

        # Ping-pong X_k / Y_k 0-d buffers: X_k_prev_dev keeps the value
        # from the previous inner iteration alive for the residual
        # check, X_k_dev is what the fused Newmark predictor writes
        # into this iteration.
        X_k_prev_dev = X_k_buf_a
        X_k_dev = X_k_buf_b
        Y_k_prev_dev = Y_k_buf_a
        Y_k_dev = Y_k_buf_b
        # Seed the prev buffer with this step's starting (X, Y) (one
        # scalar copy per axis, per step — negligible).
        X_k_prev_dev[...] = X_dev
        Y_k_prev_dev[...] = Y_dev

        residual = float("inf")
        converged = False
        k_done = 0

        # Dynamic check-cadence bookkeeping (Phase 5.3).
        _last_k_checked = -1
        _last_residual_check = float("inf")
        _check_cur = _check_every_init

        for k in range(max_inner):
            if _use_dynamic_check:
                check_iter = (k < 3) or (k - _last_k_checked >= _check_cur)
            else:
                check_iter = (k % check_every == 0) or (k < 3)

            # Snapshot BEFORE the sweep, for RB post-sweep residual.
            if check_iter and scheme == "rb":
                P_new[:] = P_old
                theta_new[:] = theta_old

            # (1) Hydrodynamic forces — single fused kernel writes WX,
            # WY into preallocated 0-d device buffers (1 launch
            # replaces ~4 CuPy ops).
            forces_kernel(
                (1, 1, 1), forces_block,
                (
                    P_old, cos_phi_1d, sin_phi_1d,
                    WX_dev, WY_dev,
                    np.float64(d_phi_d_Z),
                    np.int32(N_Z), np.int32(N_phi),
                ),
                shared_mem=2 * forces_block[0] * 8,
            )

            # (2) Newmark predictor — single fused kernel writes X_k,
            # Y_k into the ping-pong buffers (1 launch replaces ~8
            # CuPy scalar ops).
            predictor_kernel(
                (1, 1, 1), (1, 1, 1),
                (
                    X_dev, Y_dev, U_dev, V_dev,
                    WX_dev, WY_dev,
                    X_k_dev, Y_k_dev,
                    np.float64(dt), np.float64(dt_sq_half),
                    np.float64(WaX_n), np.float64(WaY_n),
                ),
            )

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
                if _use_dynamic_check:
                    if (
                        _last_residual_check > 0.0
                        and np.isfinite(_last_residual_check)
                    ):
                        drop = residual / _last_residual_check
                        if drop < 0.5:
                            _check_cur = min(_check_cur * 2, _check_max)
                        elif drop > 0.9:
                            _check_cur = max(_check_cur // 2, _check_min)
                    _last_residual_check = residual
                    _last_k_checked = k
                if residual < tol_inner and k > 2:
                    converged = True

            # (8) Jacobi swap (no snapshot).
            if scheme == "jacobi":
                P_old, P_new = P_new, P_old
                theta_old, theta_new = theta_new, theta_old

            if converged:
                break

            # Swap ping-pong buffers for the NEXT iteration. After the
            # swap X_k_prev_dev points to the buffer we just wrote
            # (holding this iter's X_k value), and X_k_dev points to
            # the buffer whose contents we're free to overwrite with
            # next iter's predictor output. Break happens BEFORE the
            # swap so that outside the loop X_k_dev still points to
            # the latest predictor output.
            X_k_prev_dev, X_k_dev = X_k_dev, X_k_prev_dev
            Y_k_prev_dev, Y_k_dev = Y_k_dev, Y_k_prev_dev

        # End-of-step velocity / position update — stays on device.
        U_dev = U_dev + dt_over_M * (WX_dev + WaX_n)
        V_dev = V_dev + dt_over_M * (WY_dev + WaY_n)
        # Copy latest X_k / Y_k VALUES into dedicated X_dev / Y_dev
        # buffers so the ping-pong buffers can be reused next step
        # without aliasing.
        X_dev[...] = X_k_dev
        Y_dev[...] = Y_k_dev

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
        dt_last=float(dt),
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
