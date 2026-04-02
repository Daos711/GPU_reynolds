"""
Diagnostic test for JFO solver with original F as RHS.

Runs JFO at epsilon=0.1 and 0.6 with verbose output.

Run:
    python -m reynolds_solver.test_jfo_diagnostic
"""

import numpy as np
from reynolds_solver import solve_reynolds
from reynolds_solver.solver_jfo import solve_reynolds_gpu_jfo
from reynolds_solver.physics.closures import LaminarClosure


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


def main():
    R = 0.035
    L = 0.056
    N = 250

    for epsilon in [0.1, 0.6]:
        print(f"\n{'='*70}")
        print(f"  JFO with original F as RHS, epsilon = {epsilon}")
        print(f"{'='*70}")

        H, d_phi, d_Z, phi_1D, Z = generate_test_case(N, epsilon)

        P_hs, _, _ = solve_reynolds(H, d_phi, d_Z, R, L)
        W_hs = compute_load(P_hs, d_phi, d_Z, phi_1D, N)
        print(f"  HS reference: W_hs = {W_hs:.6e}")

        P, theta, residual, n_outer, n_inner = solve_reynolds_gpu_jfo(
            H, d_phi, d_Z, R, L,
            closure=LaminarClosure(),
            max_outer=500,
            verbose=True,
        )

        mass_err = compute_mass_err(H, P, theta, d_phi, d_Z)
        mask = (P > 0).astype(int)
        cav_frac = 1.0 - np.mean(mask)
        W_jfo = compute_load(P, d_phi, d_Z, phi_1D, N)
        rel_diff = abs(W_jfo - W_hs) / (abs(W_hs) + 1e-30)

        print(f"\n  Results: mass_err={mass_err:.4e}, cav_frac={cav_frac:.3f}, "
              f"W_jfo={W_jfo:.6e}, rel_diff={rel_diff:.4f}, "
              f"n_outer={n_outer}, residual={residual:.2e}")


if __name__ == "__main__":
    main()
