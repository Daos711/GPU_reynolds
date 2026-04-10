"""
Test stability of JFO solver: W_smooth must not depend on max_outer.
Each run starts from scratch (P=0, theta=1).
"""
import numpy as np

R, L = 0.035, 0.056
N_phi, N_Z = 2000, 200
epsilon = 0.6

phi_1D = np.linspace(0, 2 * np.pi, N_phi)
Z_1D = np.linspace(-1, 1, N_Z)
Phi, Zm = np.meshgrid(phi_1D, Z_1D)
d_phi = phi_1D[1] - phi_1D[0]
d_Z = Z_1D[1] - Z_1D[0]
H = 1.0 + epsilon * np.cos(Phi)

from reynolds_solver import solve_reynolds

# Reference: Half-Sommerfeld
P_hs, _, _, _ = solve_reynolds(H, d_phi, d_Z, R, L,
    cavitation="half_sommerfeld", return_converged=True)
W_hs = float(np.sqrt(
    np.trapezoid(np.trapezoid(P_hs * np.cos(Phi), phi_1D, axis=1), Z_1D)**2 +
    np.trapezoid(np.trapezoid(P_hs * np.sin(Phi), phi_1D, axis=1), Z_1D)**2
)) * R * L / 2

print(f"W_hs_nd = {W_hs:.6e}")
print()

# JFO with different max_outer — EACH TIME FROM SCRATCH
results = []
for max_out in [100, 300, 500, 1000]:
    P_jfo, theta, res, n_out, n_in = solve_reynolds(
        H, d_phi, d_Z, R, L, cavitation="jfo",
        omega=None,
        tol=1e-5,
        jfo_tol_theta=1e-5,
        jfo_tol_inner=1e-6,
        jfo_max_inner=500,
        jfo_max_outer=max_out,
        verbose=False)

    W_jfo = float(np.sqrt(
        np.trapezoid(np.trapezoid(P_jfo * np.cos(Phi), phi_1D, axis=1), Z_1D)**2 +
        np.trapezoid(np.trapezoid(P_jfo * np.sin(Phi), phi_1D, axis=1), Z_1D)**2
    )) * R * L / 2

    cav_frac = float(np.mean(theta < 1.0 - 1e-6))
    converged = (n_out < max_out)
    results.append((max_out, W_jfo, cav_frac, n_out, converged))
    print(f"max_outer={max_out:>5d}: W_nd={W_jfo:.6e}, "
          f"cav={cav_frac:.3f}, n_out={n_out}, "
          f"{'CONVERGED' if converged else 'HIT LIMIT'}")

print()

# === Criterion 1: W stability ===
W_values = [r[1] for r in results]
spread = (max(W_values) - min(W_values)) / (np.mean(W_values) + 1e-30)
pass_spread = (spread < 0.02)
print(f"W spread: {spread*100:.1f}% -> {'PASS' if pass_spread else 'FAIL'} (threshold: <2%)")

# === Criterion 2: convergence (not hitting limit) ===
last_three = [r for r in results if r[0] >= 500]
stable_convergence = all(r[4] for r in last_three)
print(f"Convergence (max_outer>=500): {'PASS' if stable_convergence else 'FAIL'} "
      f"({sum(r[4] for r in last_three)}/{len(last_three)} converged)")

# === Informational: physical sanity ===
ratio = np.mean(W_values) / (W_hs + 1e-30)
in_range = (0.7 < ratio < 1.05)
print(f"W_jfo/W_hs = {ratio:.3f} -> {'OK' if in_range else 'WARN'}")

print()
overall = pass_spread and stable_convergence
print(f"{'='*50}")
print(f"RESULT: {'PASS' if overall else 'FAIL'}")
print(f"{'='*50}")
