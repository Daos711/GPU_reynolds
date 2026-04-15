"""
Restart consistency under adaptive dt (Phase 5 Part 2).

When the adaptive-dt path is enabled, `state.dt_last` carries the
last successful dt across the save / load boundary. A split run
(A steps -> save -> load -> B steps) must produce a trajectory
close to the continuous (A+B) steps run. Because the adaptive path
makes per-step floating-point decisions based on host-scalar
comparisons (inner-iter count, residual), the accept/reject DECISION
sequence after the restart may differ by one step relative to the
continuous run. We therefore allow a looser tolerance than the
fixed-dt restart test (~1 % instead of ~1e-14).

Skipped on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_ausas_restart_variable_dt
"""
import os
import sys
import tempfile

import numpy as np


def _available():
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return False
    return True


def _interp(t_src, y_src, t_ref):
    t_ref = np.clip(t_ref, t_src[0], t_src[-1])
    return np.interp(t_ref, t_src, y_src)


def test_variable_dt_restart():
    """Continuous run vs. split(A) + save/load + split(B)."""
    print("\n=== Test: adaptive-dt restart consistency ===")
    if not _available():
        return True

    from reynolds_solver import AusasAccelerationOptions, save_state, load_state
    from reynolds_solver.cavitation.ausas.benchmark_dynamic_journal import (
        run_journal_benchmark,
    )

    accel = AusasAccelerationOptions(
        adaptive_dt=True,
        dt_min=5e-4, dt_max=5e-3,
        dt_grow=1.25, dt_shrink=0.5,
        target_inner_low=80, target_inner_high=250,
        reject_if_not_converged=True,
    )
    common = dict(
        N1=60, N2=8, dt=2e-3,
        omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-6, max_inner=5000,
        scheme="rb",
    )

    # Continuous run: 200 target steps worth of time.
    res_full = run_journal_benchmark(NT=200, accel=accel, **common)

    # Split: 100 target steps + save + load + 100 target steps.
    res_a = run_journal_benchmark(NT=100, accel=accel, **common)
    assert res_a.final_state is not None

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp = f.name
    try:
        save_state(res_a.final_state, tmp)
        loaded = load_state(tmp)
    finally:
        os.unlink(tmp)

    # dt_last should round-trip.
    assert loaded.dt_last > 0.0, \
        "Loaded state should carry dt_last from the adaptive run"

    res_b = run_journal_benchmark(
        NT=100, accel=accel, state=loaded, **common,
    )

    # Stitch split trajectories (non-uniform time axis).
    t_split = np.concatenate([res_a.t, res_b.t])
    X_split = np.concatenate([res_a.X, res_b.X])
    Y_split = np.concatenate([res_a.Y, res_b.Y])
    WX_split = np.concatenate([res_a.WX, res_b.WX])

    # Monotonic check — times must strictly increase.
    if not np.all(np.diff(t_split) > 0.0):
        print("  [FAIL] split time axis is not strictly monotonic")
        return False

    # Interpolate continuous onto the stitched time grid (same samples).
    X_ref = _interp(res_full.t, res_full.X, t_split)
    Y_ref = _interp(res_full.t, res_full.Y, t_split)
    WX_ref = _interp(res_full.t, res_full.WX, t_split)

    scale_X = max(float(np.max(np.abs(X_ref))), 1e-30)
    scale_W = max(float(np.max(np.abs(WX_ref))), 1e-30)
    err_X = float(np.max(np.abs(X_ref - X_split))) / scale_X
    err_Y = float(np.max(np.abs(Y_ref - Y_split))) / scale_X
    err_WX = float(np.max(np.abs(WX_ref - WX_split))) / scale_W

    print(
        f"  continuous steps : {len(res_full.t)}"
    )
    print(
        f"  split steps a+b  : {len(res_a.t)} + {len(res_b.t)} "
        f"= {len(res_a.t) + len(res_b.t)}"
    )
    print(f"  loaded.dt_last    = {loaded.dt_last:.4e}")
    print(
        f"  trajectory rel-err: X={err_X:.2e}, Y={err_Y:.2e}, "
        f"WX={err_WX:.2e}"
    )

    ok = err_X < 1e-2 and err_Y < 1e-2 and err_WX < 1e-2
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] adaptive-dt restart within 1%")
    return ok


def main():
    ok = test_variable_dt_restart()
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
