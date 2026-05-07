"""
Task 28 tests: nonfinite-trace debug API for ``ausas_unsteady_one_step_gpu``.

These tests cover:

1. ``debug_checks=True`` + invalid ``H_curr`` with NaN
2. ``debug_checks=True`` + ``theta_prev > 1 + eps``
3. ``debug_checks=True`` + ``P_prev < -debug_p_eps``
4. Returned dict carries every required failure-metadata field
5. ``debug_checks=False`` on a simple smoke-case keeps the production path
   and returns the expected structured dict

The first four tests exercise the host-side input validator which runs in
NumPy land before any GPU work — they therefore work even on machines
without a usable CUDA device. They still depend on importing CuPy (the
solver module imports it at top-level), so they skip cleanly when CuPy
is unavailable. The fifth test requires a GPU.

Run manually:
    python -m reynolds_solver.tests.test_ausas_one_step_nonfinite_debug
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np


REQUIRED_FAILURE_KEYS = (
    "failure_kind",
    "first_nan_field",
    "first_nan_index",
    "first_nan_is_ghost",
    "first_nan_is_axial_boundary",
    "first_nan_is_phi_seam",
    "nan_iter",
    "converged",
    "n_inner",
    "residual_linf",
    "residual_rms",
    "residual_l2_abs",
)


def _try_import():
    """Return (one_step_callable, skip_reason)."""
    try:
        import cupy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return None, f"cupy not available: {exc}"
    try:
        from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
            ausas_unsteady_one_step_gpu,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"solver import failed: {exc}"
    return ausas_unsteady_one_step_gpu, None


def _gpu_available() -> Optional[str]:
    try:
        import cupy as cp
        cp.zeros(1)
    except Exception as exc:  # noqa: BLE001
        return f"GPU not available: {exc}"
    return None


def _make_smooth_state(N_Z: int, N_phi: int, eps: float = 0.6):
    phi_1D = np.linspace(0.0, 2.0 * np.pi, N_phi)
    H = np.broadcast_to(
        (1.0 + eps * np.cos(phi_1D))[None, :], (N_Z, N_phi)
    ).copy()
    # Periodic ghost pack
    H[:, 0] = H[:, N_phi - 2]
    H[:, N_phi - 1] = H[:, 1]
    P = np.zeros((N_Z, N_phi), dtype=np.float64)
    theta = np.ones((N_Z, N_phi), dtype=np.float64)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = 1.0 / (N_Z - 2)
    return H, P, theta, d_phi, d_Z


def _has_required_keys(result) -> Optional[str]:
    """Return a description of the first missing key, or None on success."""
    if not isinstance(result, dict):
        return f"result is not a dict (got {type(result).__name__})"
    for key in REQUIRED_FAILURE_KEYS:
        if key not in result:
            return f"missing key: {key!r}"
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_invalid_input_h_curr_nan():
    one_step, skip = _try_import()
    if one_step is None:
        print(f"  [SKIP] {skip}")
        return True

    N_Z, N_phi = 8, 16
    H, P, theta, d_phi, d_Z = _make_smooth_state(N_Z, N_phi)
    H_bad = H.copy()
    H_bad[3, 5] = np.nan

    result = one_step(
        H_bad, H, P, theta,
        dt=1e-3, d_phi=d_phi, d_Z=d_Z, R=1.0, L=1.0,
        max_inner=10,
        check_every=1,
        debug_checks=True,
        debug_stop_on_nonfinite=True,
        debug_return_last_finite_state=True,
        debug_return_bad_state=True,
    )

    missing = _has_required_keys(result)
    if missing:
        print(f"  FAIL: required-keys: {missing}")
        return False
    if result["failure_kind"] != "invalid_input":
        print(f"  FAIL: failure_kind={result['failure_kind']!r}")
        return False
    if result["first_nan_field"] != "H_curr":
        print(f"  FAIL: first_nan_field={result['first_nan_field']!r}")
        return False
    if result["converged"] is not False:
        print(f"  FAIL: converged={result['converged']!r}")
        return False
    if result["first_nan_index"] != (3, 5):
        print(f"  FAIL: first_nan_index={result['first_nan_index']!r}")
        return False
    if result["n_inner"] != 0:
        print(f"  FAIL: n_inner={result['n_inner']!r}")
        return False
    # Bad-state should be present and equal to the bad input.
    if "P_bad" not in result or "theta_bad" not in result:
        print("  FAIL: missing P_bad/theta_bad despite debug_return_bad_state=True")
        return False
    print("  OK  invalid_input H_curr=NaN")
    return True


def test_invalid_input_theta_out_of_range():
    one_step, skip = _try_import()
    if one_step is None:
        print(f"  [SKIP] {skip}")
        return True

    N_Z, N_phi = 8, 16
    H, P, theta, d_phi, d_Z = _make_smooth_state(N_Z, N_phi)
    theta_bad = theta.copy()
    theta_bad[2, 7] = 1.5  # >> 1 + eps

    result = one_step(
        H, H, P, theta_bad,
        dt=1e-3, d_phi=d_phi, d_Z=d_Z, R=1.0, L=1.0,
        max_inner=10,
        debug_checks=True,
        debug_theta_eps=1e-8,
        debug_stop_on_nonfinite=True,
    )

    if _has_required_keys(result):
        print(f"  FAIL: {_has_required_keys(result)}")
        return False
    if result["failure_kind"] != "invalid_input":
        print(f"  FAIL: failure_kind={result['failure_kind']!r}")
        return False
    if result["first_nan_field"] != "theta_prev":
        print(f"  FAIL: first_nan_field={result['first_nan_field']!r}")
        return False
    if result["first_nan_index"] != (2, 7):
        print(f"  FAIL: first_nan_index={result['first_nan_index']!r}")
        return False
    print("  OK  invalid_input theta_prev>1+eps")
    return True


def test_invalid_input_P_prev_negative():
    one_step, skip = _try_import()
    if one_step is None:
        print(f"  [SKIP] {skip}")
        return True

    N_Z, N_phi = 8, 16
    H, P, theta, d_phi, d_Z = _make_smooth_state(N_Z, N_phi)
    P_bad = P.copy()
    P_bad[1, 3] = -1.0e-3   # well below -p_eps

    result = one_step(
        H, H, P_bad, theta,
        dt=1e-3, d_phi=d_phi, d_Z=d_Z, R=1.0, L=1.0,
        max_inner=10,
        debug_checks=True,
        debug_p_eps=1e-12,
        debug_stop_on_nonfinite=True,
    )

    if _has_required_keys(result):
        print(f"  FAIL: {_has_required_keys(result)}")
        return False
    if result["failure_kind"] != "invalid_input":
        print(f"  FAIL: failure_kind={result['failure_kind']!r}")
        return False
    if result["first_nan_field"] != "P_prev":
        print(f"  FAIL: first_nan_field={result['first_nan_field']!r}")
        return False
    if result["first_nan_index"] != (1, 3):
        print(f"  FAIL: first_nan_index={result['first_nan_index']!r}")
        return False
    # Three localization booleans must be present and bool-typed.
    for k in (
        "first_nan_is_ghost",
        "first_nan_is_axial_boundary",
        "first_nan_is_phi_seam",
    ):
        if not isinstance(result[k], bool):
            print(f"  FAIL: {k} is {type(result[k]).__name__}, want bool")
            return False
    print("  OK  invalid_input P_prev<-eps")
    return True


def test_required_dict_fields_present():
    """All required failure-metadata keys must be present even on
    invalid-input-fail rows."""
    one_step, skip = _try_import()
    if one_step is None:
        print(f"  [SKIP] {skip}")
        return True

    N_Z, N_phi = 6, 12
    H, P, theta, d_phi, d_Z = _make_smooth_state(N_Z, N_phi)
    H_bad = H.copy()
    H_bad[2, 2] = np.inf

    result = one_step(
        H_bad, H, P, theta,
        dt=1e-3, d_phi=d_phi, d_Z=d_Z, R=1.0, L=1.0,
        max_inner=5,
        debug_checks=True,
        debug_stop_on_nonfinite=True,
    )
    missing = _has_required_keys(result)
    if missing:
        print(f"  FAIL: {missing}")
        return False
    print("  OK  required-keys present")
    return True


def test_smoke_debug_off():
    """Production path with debug_checks=False on a small smooth state."""
    one_step, skip = _try_import()
    if one_step is None:
        print(f"  [SKIP] {skip}")
        return True
    gpu_skip = _gpu_available()
    if gpu_skip is not None:
        print(f"  [SKIP] {gpu_skip}")
        return True

    N_Z, N_phi = 8, 24
    H, P, theta, d_phi, d_Z = _make_smooth_state(N_Z, N_phi, eps=0.4)
    result = one_step(
        H, H, P, theta,
        dt=1e-2, d_phi=d_phi, d_Z=d_Z, R=1.0, L=1.0,
        max_inner=200,
        tol=1e-6,
        check_every=10,
        debug_checks=False,
    )
    if not isinstance(result, dict):
        print(f"  FAIL: result not dict: {type(result).__name__}")
        return False
    for key in (
        "P", "theta", "residual", "residual_linf", "residual_rms",
        "residual_l2_abs", "n_inner", "converged",
    ):
        if key not in result:
            print(f"  FAIL: missing key {key!r}")
            return False
    if not np.all(np.isfinite(result["P"])):
        print("  FAIL: P contains nonfinite")
        return False
    if not np.all(np.isfinite(result["theta"])):
        print("  FAIL: theta contains nonfinite")
        return False
    print(
        f"  OK  smoke debug-off: converged={result['converged']} "
        f"n_inner={result['n_inner']} "
        f"linf={result['residual_linf']:.3e}"
    )
    return True


def test_smoke_debug_on_finite_state():
    """debug_checks=True on a finite smooth state must not falsely trigger
    a nonfinite failure and must still return a dict with the new keys."""
    one_step, skip = _try_import()
    if one_step is None:
        print(f"  [SKIP] {skip}")
        return True
    gpu_skip = _gpu_available()
    if gpu_skip is not None:
        print(f"  [SKIP] {gpu_skip}")
        return True

    N_Z, N_phi = 8, 24
    H, P, theta, d_phi, d_Z = _make_smooth_state(N_Z, N_phi, eps=0.3)
    result = one_step(
        H, H, P, theta,
        dt=1e-2, d_phi=d_phi, d_Z=d_Z, R=1.0, L=1.0,
        max_inner=500,
        tol=1e-6,
        check_every=10,
        debug_checks=True,
        debug_check_every=10,
    )
    if _has_required_keys(result):
        print(f"  FAIL: {_has_required_keys(result)}")
        return False
    if result["failure_kind"] is not None:
        print(
            f"  FAIL: spurious failure_kind={result['failure_kind']!r} "
            f"on finite smoke state"
        )
        return False
    if not np.isfinite(result["residual_linf"]):
        print("  FAIL: residual_linf not finite")
        return False
    print(
        f"  OK  smoke debug-on-finite: converged={result['converged']} "
        f"n_inner={result['n_inner']}"
    )
    return True


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
ALL_TESTS = (
    test_invalid_input_h_curr_nan,
    test_invalid_input_theta_out_of_range,
    test_invalid_input_P_prev_negative,
    test_required_dict_fields_present,
    test_smoke_debug_off,
    test_smoke_debug_on_finite_state,
)


def main():
    results = []
    for fn in ALL_TESTS:
        print(f"[{fn.__name__}]")
        try:
            ok = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  EXC  {type(exc).__name__}: {exc}")
            ok = False
        results.append(ok)
    n_ok = sum(1 for r in results if r)
    print(f"\n{n_ok}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
