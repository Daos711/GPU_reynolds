"""
Smoke tests for the dynamic Ausas solver with surface texture.

These are NOT literature-validated runs; they only check that the
solver:
  * does not crash or diverge when `texture_relief` is non-zero,
  * preserves the physical invariants (P >= 0, theta in [0, 1],
    h_min > 0, eccentricity < 1 for journal),
  * returns finite, float64 histories.

Two scenarios:
  (1) uniform gap with one axisymmetric-ish dimple, subjected to a
      small oscillatory squeeze (prescribed h(t));
  (2) dynamic journal bearing (Ausas 2008 Section 5 parameters) with
      a weak texture relief added to the gap.

Skipped on CPU-only machines.

Run:
    python -m reynolds_solver.tests.test_ausas_dynamic_texture
"""
import sys

import numpy as np


def _available():
    try:
        import cupy  # noqa: F401
    except Exception as exc:
        print(f"  [SKIP] cupy not available: {exc}")
        return False
    return True


def _single_dimple(N_Z: int, N_phi: int,
                   depth: float = 0.2, sigma: float = 0.05) -> np.ndarray:
    """One Gaussian dimple centred on the interior."""
    x = np.linspace(-0.5, 0.5, N_phi)
    z = np.linspace(-0.5, 0.5, N_Z)
    Z, X = np.meshgrid(z, x, indexing="ij")
    dimple = -depth * np.exp(
        -(X ** 2 + Z ** 2) / (2.0 * sigma ** 2)
    )
    return dimple.astype(np.float64)


def test_prescribed_h_with_dimple():
    """Oscillating squeeze on a base uniform gap with a dimple."""
    print("\n=== Smoke 1: prescribed-h squeeze + single dimple ===")
    if not _available():
        return True

    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        solve_ausas_prescribed_h_gpu,
    )

    N_phi, N_Z = 82, 12
    d_phi = 1.0 / (N_phi - 2)
    d_Z = 0.1 / (N_Z - 2)
    R, L = 0.5, 1.0

    # Base gap: uniform 0.5, modulated in time by a small squeeze.
    dimple = _single_dimple(N_Z, N_phi, depth=0.15, sigma=0.08)

    def H_provider(n, t):
        h0 = 0.5 + 0.05 * np.cos(2.0 * np.pi * t)
        H = np.full((N_Z, N_phi), h0, dtype=np.float64) + dimple
        return H

    p0 = 0.01
    P0 = p0 * np.ones((N_Z, N_phi), dtype=np.float64)
    theta0 = np.ones((N_Z, N_phi), dtype=np.float64)

    result = solve_ausas_prescribed_h_gpu(
        H_provider=H_provider, NT=300, dt=2e-3,
        d_phi=d_phi, d_Z=d_Z, R=R, L=L,
        alpha=0.0, omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-5, max_inner=2000,
        P0=P0, theta0=theta0,
        p_bc_phi0=p0, p_bc_phiL=p0,
        theta_bc_phi0=1.0, theta_bc_phiL=1.0,
        p_bc_z0=p0, p_bc_zL=p0,
        theta_bc_z0=1.0, theta_bc_zL=1.0,
        periodic_phi=False, periodic_z=True,
    )

    # Invariants
    P = result.P_last
    theta = result.theta_last
    ok_P_nonneg = float(P.min()) >= -1e-12
    ok_theta_bounds = (
        float(theta.min()) >= -1e-12 and float(theta.max()) <= 1.0 + 1e-12
    )
    ok_h_min = float(result.h_min.min()) > 0.0
    ok_finite = bool(
        np.all(np.isfinite(result.p_max))
        and np.all(np.isfinite(result.cav_frac))
    )
    p_max = float(result.p_max.max())
    h_min = float(result.h_min.min())
    cav_max = float(result.cav_frac.max())
    print(
        f"  p_max range over run = {p_max:.3e}, "
        f"h_min = {h_min:.3e}, cav_frac_max = {cav_max:.3f}"
    )
    ok = ok_P_nonneg and ok_theta_bounds and ok_h_min and ok_finite
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] P >= 0 ({ok_P_nonneg}), θ in [0,1] ({ok_theta_bounds}), "
        f"h_min > 0 ({ok_h_min}), finite ({ok_finite})"
    )
    return ok


def test_journal_with_texture_relief():
    """Short dynamic journal run with a weak texture relief."""
    print("\n=== Smoke 2: dynamic journal + weak texture relief ===")
    if not _available():
        return True

    from reynolds_solver.cavitation.ausas.solver_dynamic_gpu import (
        solve_ausas_journal_dynamic_gpu,
    )
    from reynolds_solver.cavitation.ausas.benchmark_dynamic_journal import (
        journal_load,
    )

    N1, N2 = 60, 8
    N_phi = N1 + 2
    N_Z = N2 + 2
    d_phi = 1.0 / N1
    d_Z = 0.1 / N2
    R, L = 0.5, 1.0

    # Weak texture: 2 % relief, single axisymmetric dimple.
    texture = _single_dimple(N_Z, N_phi, depth=0.02, sigma=0.12)

    result = solve_ausas_journal_dynamic_gpu(
        NT=150, dt=1e-3,
        N_Z=N_Z, N_phi=N_phi,
        d_phi=d_phi, d_Z=d_Z, R=R, L=L,
        mass_M=1e-6,
        load_fn=journal_load,
        X0=0.5, Y0=0.5, U0=0.0, V0=0.0,
        p_a=0.0075, B_width=0.1,
        alpha=1.0, omega_p=1.0, omega_theta=1.0,
        tol_inner=1e-5, max_inner=2000,
        texture_relief=texture,
        scheme="rb",
        verbose=False,
    )

    e = result.X ** 2 + result.Y ** 2
    e_max = float(np.sqrt(e.max()))
    h_min = float(result.h_min.min())
    cav_max = float(result.cav_frac.max())
    print(
        f"  e_max = {e_max:.3f}, h_min = {h_min:.3e}, "
        f"cav_max = {cav_max:.3f}"
    )
    ok_bounds = e_max < 1.0 and h_min > 0.0
    ok_finite = bool(
        np.all(np.isfinite(result.X)) and np.all(np.isfinite(result.WX))
    )
    ok = ok_bounds and ok_finite
    status = "PASS" if ok else "FAIL"
    print(
        f"  [{status}] e < 1 and h_min > 0 ({ok_bounds}), "
        f"finite histories ({ok_finite})"
    )
    return ok


def main():
    ok = True
    ok = test_prescribed_h_with_dimple() and ok
    ok = test_journal_with_texture_relief() and ok
    print("\n" + ("ALL TESTS PASSED" if ok else "TEST FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
