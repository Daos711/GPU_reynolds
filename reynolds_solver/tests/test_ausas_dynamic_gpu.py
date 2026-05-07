"""
Unit test for the unsteady Ausas GPU kernel (Stage 1).

The test drives the CUDA kernel `unsteady_ausas_step` and a NumPy/Numba
CPU reference (existing `_ausas_relax_sweep_pt`) through the SAME fixed
number of Jacobi sweeps over ONE real time step Δt and compares the
resulting (P, θ) pointwise.

Passing criterion: max|ΔP| / max(|P_ref|) < 1e-8 and max|Δθ| < 1e-8
on the interior.

Run:
    python -m reynolds_solver.tests.test_ausas_dynamic_gpu
"""
import sys

import numpy as np


R = 0.035
L = 0.056


def _make_gap(N_Z, N_phi, epsilon):
    """Smooth journal-bearing gap with prescribed eccentricity."""
    phi_1D = np.linspace(0.0, 2.0 * np.pi, N_phi)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi)
    # Defensive periodic pack — CPU coefficient builder expects it.
    H[:, 0] = H[:, N_phi - 2]
    H[:, N_phi - 1] = H[:, 1]
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z


def _make_initial_state(N_Z, N_phi):
    """
    Build a non-trivial (P, θ) state with a visible cavitation region so
    that both Branch 1 (pressure) and Branch 2 (θ) are exercised.
    """
    phi_1D = np.linspace(0.0, 2.0 * np.pi, N_phi)
    Z = np.linspace(-1.0, 1.0, N_Z)
    Phi, Zm = np.meshgrid(phi_1D, Z)

    # Analytic-flavour half-Sommerfeld-like pressure: positive on the
    # converging wedge, zero on the diverging wedge.
    P = np.maximum(np.sin(Phi) * (1.0 - Zm ** 2), 0.0)
    P[0, :] = 0.0
    P[-1, :] = 0.0
    P[:, 0] = P[:, N_phi - 2]
    P[:, N_phi - 1] = P[:, 1]

    # θ: 1 where P > 0 (full film); partial film in the diverging region.
    theta = np.ones_like(P)
    cav_region = (P <= 0.0) & (Zm > -0.8) & (Zm < 0.8)
    theta[cav_region] = 0.5 + 0.3 * np.cos(Phi[cav_region])
    theta = np.clip(theta, 0.0, 1.0)
    theta[0, :] = 1.0
    theta[-1, :] = 1.0
    theta[:, 0] = theta[:, N_phi - 2]
    theta[:, N_phi - 1] = theta[:, 1]
    return P, theta


def _cpu_one_step(
    H_curr, H_prev, P_prev, theta_prev,
    d_phi, d_Z, R, L, dt,
    omega_p, omega_theta, n_sweeps, flooded_ends=True,
):
    """
    CPU reference: run exactly `n_sweeps` iterations of the existing
    `_ausas_relax_sweep_pt` (Jacobi Ausas with the temporal term), using
    β = 2 · d_phi^2 / dt and c_prev = θ^{n-1} · h^{n-1}.

    Mirrors the GPU kernel one-to-one for α = 1.
    """
    from reynolds_solver.cavitation.ausas.solver_cpu import (
        _ausas_relax_sweep_pt,
        _build_coefficients,
    )

    N_Z, N_phi = H_curr.shape
    A, B, C, D, E = _build_coefficients(H_curr, d_phi, d_Z, R, L)

    beta = 2.0 * d_phi * d_phi / dt
    c_prev = theta_prev * H_prev  # θ^{n-1} · h^{n-1}

    P_old = P_prev.astype(np.float64).copy()
    theta_old = theta_prev.astype(np.float64).copy()

    # Apply BCs to the seed.
    P_old[0, :] = 0.0
    P_old[-1, :] = 0.0
    P_old[:, 0] = P_old[:, N_phi - 2]
    P_old[:, N_phi - 1] = P_old[:, 1]
    if flooded_ends:
        theta_old[0, :] = 1.0
        theta_old[-1, :] = 1.0
    theta_old[:, 0] = theta_old[:, N_phi - 2]
    theta_old[:, N_phi - 1] = theta_old[:, 1]

    P_new = P_old.copy()
    theta_new = theta_old.copy()

    flooded_flag = 1 if flooded_ends else 0
    for _ in range(n_sweeps):
        _ausas_relax_sweep_pt(
            P_new, P_old, theta_new, theta_old,
            H_curr, c_prev,
            A, B, C, D, E,
            d_phi, beta, omega_p, omega_theta,
            N_Z, N_phi, flooded_flag,
        )
        P_old, P_new = P_new, P_old
        theta_old, theta_new = theta_new, theta_old

    return P_old, theta_old


def test_one_step_gpu_vs_cpu():
    """
    One unsteady Ausas step (fixed # of Jacobi sweeps, α = 1) must match
    the CPU reference to ~1e-10 in float64.
    """
    print("\n=== Test: unsteady Ausas GPU vs CPU — one time step ===")
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return True

    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        ausas_unsteady_one_step_gpu,
    )

    N_Z, N_phi = 40, 80

    eps_prev, eps_curr = 0.55, 0.60   # small squeeze between time steps
    dt = 0.05
    omega_p, omega_theta = 1.0, 1.0
    n_sweeps = 120

    H_prev, d_phi, d_Z = _make_gap(N_Z, N_phi, eps_prev)
    H_curr, _, _ = _make_gap(N_Z, N_phi, eps_curr)
    P_prev, theta_prev = _make_initial_state(N_Z, N_phi)

    # --- CPU reference ---
    P_cpu, theta_cpu = _cpu_one_step(
        H_curr, H_prev, P_prev, theta_prev,
        d_phi, d_Z, R, L, dt,
        omega_p, omega_theta, n_sweeps,
        flooded_ends=True,
    )

    # --- GPU kernel under test ---
    # Force the GPU loop to run the exact same number of sweeps as the CPU
    # reference (tol = -1 so the early-exit never triggers).
    out = ausas_unsteady_one_step_gpu(
        H_curr, H_prev, P_prev, theta_prev,
        dt, d_phi, d_Z, R, L,
        alpha=1.0,
        omega_p=omega_p, omega_theta=omega_theta,
        tol=-1.0,              # disable early-exit
        max_inner=n_sweeps,    # run exactly n_sweeps
        p_bc_z0=0.0, p_bc_zL=0.0,
        theta_bc_z0=1.0, theta_bc_zL=1.0,
        periodic_phi=True,
        check_every=max(n_sweeps // 4, 1),
        scheme="jacobi",       # bit-for-bit match to the CPU Jacobi ref
        verbose=False,
    )
    P_gpu = out["P"]
    theta_gpu = out["theta"]
    residual = out["residual"]
    n_inner = out["n_inner"]

    assert n_inner == n_sweeps, f"expected {n_sweeps} sweeps, got {n_inner}"

    # --- Compare (interior only, avoid ghost columns) ---
    P_cpu_i = P_cpu[1:-1, 1:-1]
    P_gpu_i = P_gpu[1:-1, 1:-1]
    theta_cpu_i = theta_cpu[1:-1, 1:-1]
    theta_gpu_i = theta_gpu[1:-1, 1:-1]

    ref_scale_P = max(np.max(np.abs(P_cpu_i)), 1e-30)
    err_P = float(np.max(np.abs(P_gpu_i - P_cpu_i)) / ref_scale_P)
    err_theta = float(np.max(np.abs(theta_gpu_i - theta_cpu_i)))

    cav_frac_cpu = float(np.mean(theta_cpu < 1.0 - 1e-6))
    cav_frac_gpu = float(np.mean(theta_gpu < 1.0 - 1e-6))

    print(
        f"  sweeps={n_inner}, residual={residual:.3e}, "
        f"max|P|_cpu={ref_scale_P:.3e}, "
        f"cav_frac cpu/gpu = {cav_frac_cpu:.3f}/{cav_frac_gpu:.3f}"
    )
    print(
        f"  max|ΔP|/max|P_cpu| = {err_P:.2e}, "
        f"max|Δθ| = {err_theta:.2e}"
    )

    tol_P = 1e-8
    tol_theta = 1e-8
    ok_P = err_P < tol_P
    ok_theta = err_theta < tol_theta
    ok_regime = abs(cav_frac_cpu - cav_frac_gpu) < 1e-6

    status = "PASS" if (ok_P and ok_theta and ok_regime) else "FAIL"
    print(
        f"  [{status}] err_P={err_P:.2e} (<{tol_P:.0e}), "
        f"err_theta={err_theta:.2e} (<{tol_theta:.0e}), "
        f"|Δcav_frac|={abs(cav_frac_cpu - cav_frac_gpu):.2e}"
    )
    return ok_P and ok_theta and ok_regime


def main():
    ok = test_one_step_gpu_vs_cpu()
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
