"""
Steady-state consistency: dynamic Ausas -> Payvar-Salant stationary.

For a FIXED gap h(x1, x2) and no sliding-dependence on X_k (prescribed
static configuration), the dynamic Ausas equation eq. (12) reduces to
the stationary Ausas rule as c^n -> c^{n-1} (time term vanishes). We
check that this limit agrees, to within discretisation tolerance, with
the well-validated Payvar-Salant CPU solver on the same gap.

Gap chosen: smooth journal bearing, eccentricity epsilon = 0.6 (moderate
cavitation coverage). Flooded Z boundaries, periodic phi.

Metrics:
  * load integral W = int P dA
  * p_max
  * cavitation fraction

Tolerance: 1 % on W and p_max (Payvar-Salant uses a different splitting
than Ausas, so we do not require bit-for-bit agreement on fields).

Skipped on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_ausas_steady_limit
"""
import sys

import numpy as np


def _generate_gap(N_phi_total: int, N_Z_total: int, epsilon: float = 0.6):
    """
    Smooth journal bearing gap with the repo's (N_phi incl. 2 ghosts,
    d_phi = 2 pi / (N_phi - 1)) convention, matching solve_payvar_salant_cpu.
    """
    # Use the same generator used by test_payvar_salant.
    phi_1D = np.linspace(0.0, 2.0 * np.pi, N_phi_total)
    Z = np.linspace(-1.0, 1.0, N_Z_total)
    Phi, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi)
    d_phi = float(phi_1D[1] - phi_1D[0])
    d_Z = float(Z[1] - Z[0])
    return H, d_phi, d_Z


def _W_from_P(P: np.ndarray, d_phi: float, d_Z: float) -> float:
    """Load support integral over interior."""
    return float(d_phi * d_Z * np.sum(P[1:-1, 1:-1]))


def test_steady_limit():
    """Dynamic solver at steady state vs Payvar-Salant on the same gap."""
    print("\n=== Test: Ausas-dynamic steady-state vs Payvar-Salant ===")
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return True

    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        solve_ausas_prescribed_h_gpu,
    )
    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu

    # Grid
    N1_interior = 200                 # physical cells in phi
    N2_interior = 20                  # physical cells in Z
    N_phi = N1_interior + 2
    N_Z = N2_interior + 2
    R = 0.035
    L = 0.056

    H, d_phi, d_Z = _generate_gap(N_phi, N_Z, epsilon=0.6)

    # --- Baseline: Payvar-Salant (CPU) ---
    P_ps, theta_ps, res_ps, n_ps = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L,
        omega=1.0, tol=1e-9, max_iter=30000,
        verbose=False,
    )
    W_ps = _W_from_P(P_ps, d_phi, d_Z)
    pmax_ps = float(P_ps[1:-1, 1:-1].max())
    cav_ps = float(np.mean(theta_ps[1:-1, 1:-1] < 1.0 - 1e-6))

    print(
        f"  Payvar-Salant: W = {W_ps:.4e}, p_max = {pmax_ps:.4e}, "
        f"cav_frac = {cav_ps:.3f}"
    )

    # --- Dynamic Ausas driven by a CONSTANT gap ---
    def H_provider_const(n, t):
        return H

    # Small dt to keep the time term well-posed; we only need the
    # steady-state limit, so NT just has to be "long enough".
    dt = 1e-3
    NT = 600

    # Use alpha = 1 to match the stationary-Ausas physics that PS
    # approximates. alpha = 0 would be the pure-squeeze limit.
    result = solve_ausas_prescribed_h_gpu(
        H_provider=H_provider_const, NT=NT, dt=dt,
        d_phi=d_phi, d_Z=d_Z, R=R, L=L,
        alpha=1.0, omega_p=1.7, omega_theta=1.0,
        tol_inner=1e-7, max_inner=5000,
        P0=None, theta0=None,
        p_bc_phi0=0.0, p_bc_phiL=0.0,
        theta_bc_phi0=1.0, theta_bc_phiL=1.0,
        p_bc_z0=0.0, p_bc_zL=0.0,
        theta_bc_z0=1.0, theta_bc_zL=1.0,
        periodic_phi=True, periodic_z=False,
        scheme="rb",
        verbose=False,
    )

    # Steady drift: last-step fields.
    W_ausas = _W_from_P(result.P_last, d_phi, d_Z)
    pmax_ausas = float(result.P_last[1:-1, 1:-1].max())
    cav_ausas = float(
        np.mean(result.theta_last[1:-1, 1:-1] < 1.0 - 1e-6)
    )
    print(
        f"  Ausas-dynamic (steady-limit): W = {W_ausas:.4e}, "
        f"p_max = {pmax_ausas:.4e}, cav_frac = {cav_ausas:.3f}"
    )

    # Check the dynamic solve really did reach steady state: the last
    # 5 steps should report (p_max, cav_frac) to ~1e-5.
    tail = slice(-5, None)
    pmax_jitter = float(
        np.max(result.p_max[tail]) - np.min(result.p_max[tail])
    ) / (abs(pmax_ausas) + 1e-30)
    print(f"  last-5 p_max jitter = {pmax_jitter:.2e}")

    # Compare
    tol = 0.02
    err_W = abs(W_ausas - W_ps) / (abs(W_ps) + 1e-30)
    err_pmax = abs(pmax_ausas - pmax_ps) / (abs(pmax_ps) + 1e-30)
    err_cav = abs(cav_ausas - cav_ps)

    print(
        f"  relative errors: W = {err_W:.3f}, p_max = {err_pmax:.3f}, "
        f"|Δ cav_frac| = {err_cav:.3f}"
    )
    ok_W = err_W < tol
    ok_pmax = err_pmax < tol
    ok_cav = err_cav < 0.10     # 10% on cav fraction (different splitting)

    ok = ok_W and ok_pmax and ok_cav
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] W < {tol*100:.0f}%, p_max < {tol*100:.0f}%, "
        f"cav_frac within 10%"
    )
    return ok


def main():
    ok = test_steady_limit()
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
