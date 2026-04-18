"""
Persistent state for restart / checkpointing of the dynamic Ausas solvers.

Both `solve_ausas_prescribed_h_gpu` (Stage 2) and
`solve_ausas_journal_dynamic_gpu` (Stage 3) accept an optional
`state=AusasState(...)` kwarg to resume from a previously saved run.

An `AusasState` carries everything the time loop needs to pick up
exactly where it left off: the pressure and theta fields, the last
gap H (so c_prev = theta * H is correctly formed for the first
continuation step), the shaft position and velocity for the journal
case, and the current (step_index, time) so histories remain
continuous across restarts.

Save/load use `np.savez_compressed` — self-describing, numpy-version
stable, no external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AusasState:
    """
    Snapshot of the solver state at the end of a time step.

    Fields
    ------
    P, theta : (N_Z, N_phi) float64
        Pressure and cavitation-fraction fields at the last completed
        step.
    H_prev : (N_Z, N_phi) float64
        Gap field used to form `c_prev = theta * H_prev` on the NEXT
        step. For the prescribed-h solver this is the last
        H_provider(n, t) evaluation; for the journal solver it is the
        H rebuilt from (X, Y, texture) at the last step.
    X, Y, U, V : float
        Shaft position and velocity at the last step. Ignored by the
        prescribed-h solver; populated by the journal solver. Default
        zero.
    step_index : int
        One-indexed number of the last completed step in the full
        (possibly multi-restart) simulation. Used to offset histories
        after a restart.
    time : float
        Real time at the last completed step (= step_index * dt for a
        fixed dt run).
    dt_last : float
        Last dt value used by the solver. Meaningful only when the
        adaptive-dt mode is enabled (Phase 5 Part 2); otherwise stays
        at 0 and the solver uses the caller-supplied `dt` on restart.
    """

    P: np.ndarray
    theta: np.ndarray
    H_prev: np.ndarray
    X: float = 0.0
    Y: float = 0.0
    U: float = 0.0
    V: float = 0.0
    step_index: int = 0
    time: float = 0.0
    dt_last: float = 0.0


def save_state(state: AusasState, path: str) -> None:
    """
    Serialise an `AusasState` to disk via `np.savez_compressed`.

    The resulting file is self-describing and loads on any numpy
    version; no pickle is used.
    """
    np.savez_compressed(
        path,
        P=np.asarray(state.P, dtype=np.float64),
        theta=np.asarray(state.theta, dtype=np.float64),
        H_prev=np.asarray(state.H_prev, dtype=np.float64),
        X=np.float64(state.X),
        Y=np.float64(state.Y),
        U=np.float64(state.U),
        V=np.float64(state.V),
        step_index=np.int64(state.step_index),
        time=np.float64(state.time),
        dt_last=np.float64(state.dt_last),
    )


def load_state(path: str) -> AusasState:
    """Load a previously saved `AusasState` from `path` (.npz)."""
    with np.load(path) as f:
        # Backward compatibility: older .npz files (pre-Phase-5.2) do
        # not have `dt_last`. Default to 0 in that case.
        dt_last = float(f["dt_last"]) if "dt_last" in f.files else 0.0
        return AusasState(
            P=np.asarray(f["P"], dtype=np.float64),
            theta=np.asarray(f["theta"], dtype=np.float64),
            H_prev=np.asarray(f["H_prev"], dtype=np.float64),
            X=float(f["X"]),
            Y=float(f["Y"]),
            U=float(f["U"]),
            V=float(f["V"]),
            step_index=int(f["step_index"]),
            time=float(f["time"]),
            dt_last=dt_last,
        )
