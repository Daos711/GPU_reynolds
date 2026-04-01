"""
Diagnostic A/B test for JFO solver: sweep direction and F_theta sign.

Tests all 4 combinations on epsilon=0.1 and epsilon=0.6:
  A: forward sweep + normal F
  B: backward sweep + normal F
  C: forward sweep + flipped F
  D: backward sweep + flipped F

Reports: mass_err, cav_frac, rel_diff (JFO vs HS).

Run:
    python -m reynolds_solver.test_jfo_diagnostic
"""

import numpy as np
from reynolds_solver import solve_reynolds
from reynolds_solver.solver_jfo import solve_reynolds_gpu_jfo
from reynolds_solver.physics.closures import LaminarClosure
from reynolds_solver.utils import precompute_coefficients_gpu


def generate_test_case(N, epsilon):
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z, phi_1D, Z


def compute_mass_err(H, P, theta, d_phi, d_Z):
    H_face = 0.5 * (H[:, :-1] + H[:, 1:])
    theta_face = 0.5 * (theta[:, :-1] + theta[:, 1:])
    dP_face = (P[:, 1:] - P[:, :-1]) / d_phi
    Q_inner = H_face * theta_face - 0.5 * H_face ** 3 * dP_face

    H_wrap = 0.5 * (H[:, -1] + H[:, 0])
    theta_wrap = 0.5 * (theta[:, -1] + theta[:, 0])
    dP_wrap = (P[:, 0] - P[:, -1]) / d_phi
    Q_wrap = (H_wrap * theta_wrap - 0.5 * H_wrap ** 3 * dP_wrap)[:, None]

    Q_all = np.concatenate([Q_inner, Q_wrap], axis=1)
    Q_integrated = np.sum(Q_all, axis=0) * d_Z

    mass_err = (np.max(Q_integrated) - np.min(Q_integrated)) / (np.mean(np.abs(Q_integrated)) + 1e-12)
    return mass_err


def compute_load(P, d_phi, d_Z, phi_1D, N):
    Phi_mesh, _ = np.meshgrid(phi_1D, np.linspace(-1, 1, N))
    Wy = -np.sum(P * np.cos(Phi_mesh)) * d_phi * d_Z
    return Wy


def run_variant(H, d_phi, d_Z, R, L, sweep_direction, flip_F_sign, label,
                max_outer=500, verbose_outer=False):
    N = H.shape[0]
    P, theta, residual, n_outer, n_inner = solve_reynolds_gpu_jfo(
        H, d_phi, d_Z, R, L,
        closure=LaminarClosure(),
        max_outer=max_outer,
        sweep_direction=sweep_direction,
        flip_F_sign=flip_F_sign,
        verbose=verbose_outer,
    )
    return P, theta, residual, n_outer, n_inner


def main():
    R = 0.035
    L = 0.056
    N = 250

    variants = [
        (0, False, "A: fwd sweep, normal F"),
        (1, False, "B: bwd sweep, normal F"),
        (0, True,  "C: fwd sweep, flip F  "),
        (1, True,  "D: bwd sweep, flip F  "),
    ]

    for epsilon in [0.1, 0.6]:
        print(f"\n{'='*70}")
        print(f"  epsilon = {epsilon}")
        print(f"{'='*70}")

        H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

        # HS reference
        P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L)
        W_hs = compute_load(P_hs, d_phi, d_Z, phi_1D, N)
        print(f"  HS reference: W_hs = {W_hs:.6e}")

        print(f"\n  {'Variant':<28s} {'mass_err':>10s} {'cav_frac':>10s} "
              f"{'W_jfo':>12s} {'rel_diff':>10s} {'n_outer':>8s} {'residual':>10s}")
        print(f"  {'-'*90}")

        for sweep_dir, flip_f, label in variants:
            try:
                P, theta, residual, n_outer, n_inner = run_variant(
                    H, d_phi, d_Z, R, L,
                    sweep_direction=sweep_dir,
                    flip_F_sign=flip_f,
                    label=label,
                    max_outer=500,
                    verbose_outer=False,
                )

                mass_err = compute_mass_err(H, P, theta, d_phi, d_Z)
                mask = (P > 0).astype(int)
                cav_frac = 1.0 - np.mean(mask)
                W_jfo = compute_load(P, d_phi, d_Z, phi_1D, N)
                rel_diff = abs(W_jfo - W_hs) / (abs(W_hs) + 1e-30)

                print(f"  {label:<28s} {mass_err:10.4e} {cav_frac:10.3f} "
                      f"{W_jfo:12.6e} {rel_diff:10.4f} {n_outer:8d} {residual:10.2e}")
            except Exception as e:
                print(f"  {label:<28s} ERROR: {e}")

    # Detailed log for the best variant at epsilon=0.6
    print(f"\n{'='*70}")
    print(f"  Detailed log: best variant at epsilon=0.6")
    print(f"{'='*70}")

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, 0.6)

    for sweep_dir, flip_f, label in variants:
        print(f"\n--- {label} ---")
        try:
            P, theta, residual, n_outer, n_inner = run_variant(
                H, d_phi, d_Z, R, L,
                sweep_direction=sweep_dir,
                flip_F_sign=flip_f,
                label=label,
                max_outer=100,
                verbose_outer=True,
            )
            mass_err = compute_mass_err(H, P, theta, d_phi, d_Z)
            print(f"  mass_err={mass_err:.4e}, n_outer={n_outer}, residual={residual:.2e}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
