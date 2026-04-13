"""
Диагностика для препода: профили P(φ) для T2 vs T3
в одной конкретной лунке у кавитационной границы.

Запуск:
    python diagnostic_t2_t3_pressure_profile.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reynolds_solver.experiments.elrod_manser_t2_t3 import (
    add_wedge_dimples, generate_bearing, make_dimple_centers
)
from reynolds_solver.cavitation.elrod import solve_elrod_compressible

R, L, C = 0.02, 0.04, 50e-6
epsilon = 0.6
N_phi, N_Z = 200, 60

H_s, Phi_m, Z_m, phi_1d, z_1d, d_phi, d_Z = generate_bearing(
    N_phi, N_Z, epsilon, R, L, C
)

r_x = 3e-3 / R
r_z = 3e-3 / (L / 2)
r_y = 15e-6 / C

beta = 100e6
omega_shaft = 2 * np.pi * 3000 / 60
U = omega_shaft * R
mu = 0.05
beta_bar = beta * C**2 / (mu * U * R)

phi_c, Z_c = make_dimple_centers(14, 4, (0, 2 * np.pi), (-1, 1))
H_t2 = add_wedge_dimples(H_s, Phi_m, Z_m, phi_c, Z_c, r_x, r_y, r_z, "T2")
H_t3 = add_wedge_dimples(H_s, Phi_m, Z_m, phi_c, Z_c, r_x, r_y, r_z, "T3")

print("Solving smooth...", flush=True)
P_s, th_s, _, _ = solve_elrod_compressible(
    H_s, d_phi, d_Z, R, L, beta_bar=beta_bar,
    phi_bc="groove", tol=1e-6, max_iter=20000,
)
print(f"  Pmax = {P_s.max():.4f}")

print("Solving T2...", flush=True)
P_t2, th_t2, _, _ = solve_elrod_compressible(
    H_t2, d_phi, d_Z, R, L, beta_bar=beta_bar,
    phi_bc="groove", tol=1e-6, max_iter=20000,
)
print(f"  Pmax = {P_t2.max():.4f}")

print("Solving T3...", flush=True)
P_t3, th_t3, _, _ = solve_elrod_compressible(
    H_t3, d_phi, d_Z, R, L, beta_bar=beta_bar,
    phi_bc="groove", tol=1e-6, max_iter=20000,
)
print(f"  Pmax = {P_t3.max():.4f}")

# Z-slice where dimples exist
iz = np.argmin(np.abs(z_1d - 0.25))
print(f"\nZ-slice: z = {z_1d[iz]:.4f}")

# Smooth rupture angle
P_sl = P_s[iz, :]
j_pmax = int(np.argmax(P_sl))
j_rupt = j_pmax
for j in range(j_pmax, N_phi):
    if P_sl[j] < 1e-10:
        j_rupt = j
        break
print(f"Smooth: Pmax at {np.degrees(phi_1d[j_pmax]):.0f} deg, "
      f"rupture at {np.degrees(phi_1d[j_rupt]):.0f} deg")

# Find dimple near rupture and in full-film
mask_z = np.abs(Z_c - z_1d[iz]) < r_z
for label, j_target in [("NEAR RUPTURE", j_rupt), ("FULL-FILM (Pmax)", j_pmax)]:
    dists = np.where(mask_z, np.abs(phi_c - phi_1d[j_target]), 999)
    k = int(np.argmin(dists))
    pcx = phi_c[k]

    print(f"\n{'='*70}")
    print(f"  DIMPLE {label}: center = {np.degrees(pcx):.1f} deg")
    print(f"{'='*70}")

    win = 2.5 * r_x
    mj = np.where(np.abs(np.arctan2(
        np.sin(phi_1d - pcx), np.cos(phi_1d - pcx)
    )) < win)[0]

    print(f"{'phi':>8s} {'':>1s} {'H_s':>7s} {'H_T2':>7s} {'H_T3':>7s} "
          f"{'P_s':>8s} {'P_T2':>8s} {'P_T3':>8s} {'dP=T2-T3':>9s}")
    for j in mj:
        inside = "*" if abs(np.arctan2(
            np.sin(phi_1d[j] - pcx), np.cos(phi_1d[j] - pcx)
        )) < r_x else " "
        dp = P_t2[iz, j] - P_t3[iz, j]
        print(f"{np.degrees(phi_1d[j]):7.1f} {inside} "
              f"{H_s[iz,j]:7.4f} {H_t2[iz,j]:7.4f} {H_t3[iz,j]:7.4f} "
              f"{P_s[iz,j]:8.4f} {P_t2[iz,j]:8.4f} {P_t3[iz,j]:8.4f} "
              f"{dp:+9.5f}")

    mj_in = np.where(np.abs(np.arctan2(
        np.sin(phi_1d - pcx), np.cos(phi_1d - pcx)
    )) < r_x)[0]
    print(f"\n  INSIDE DIMPLE:")
    print(f"    max|H_T2 - H_T3| = {np.max(np.abs(H_t2[iz,mj_in] - H_t3[iz,mj_in])):.4f}")
    print(f"    max|P_T2 - P_T3| = {np.max(np.abs(P_t2[iz,mj_in] - P_t3[iz,mj_in])):.6f}")
    print(f"    Theta T2: {th_t2[iz,mj_in].min():.4f} to {th_t2[iz,mj_in].max():.4f}")
    print(f"    Theta T3: {th_t3[iz,mj_in].min():.4f} to {th_t3[iz,mj_in].max():.4f}")

print(f"\n{'='*70}")
print("ИТОГ: если max|P_T2 - P_T3| ≈ 0 внутри лунки — солвер")
print("не различает конвергентный и дивергентный клин.")
print("Если max|H_T2 - H_T3| > 0, но max|P_T2 - P_T3| ≈ 0 — проблема")
print("в том, КАК солвер обрабатывает разные профили h(φ).")
print(f"{'='*70}")
