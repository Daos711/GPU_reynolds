"""
Manser T2 (convergent) vs T3 (divergent) wedge-texture validation
for the compressible Elrod solver.

This is the key test of the Elrod MVP: does the finite-bulk-modulus
formulation reproduce the micro-wedge asymmetry observed by Manser
between convergent and divergent dimples?

Setup (Manser article-like, scenario 2):
  D = L = 40 mm (R = 20 mm, L/D = 1)
  C = 50 μm, ε = 0.6
  N = 3000 rpm, μ = 0.05 Pa·s, β = 100 MPa
  β̄ = β·C² / (μ·U·R) ≈ 39.8
  Texture: full 0-360°, P_tex = 40% → 14 × 4 dimples
  Dimple size: r_x = r_z = 3 mm, r_y = 15 μm

Reference (Manser 2019a Table 4):
  Smooth  W/W_smooth = 1.00   (baseline)
  T2 full W/W_smooth ≈ 1.43   (convergent: gain > 1)
  T3 full W/W_smooth ≈ 0.41   (divergent:  gain < 1)
  T2/T3 ratio       ≈ 3.44

The wedge geometry here is a simplified linear ramp inside an
elliptical footprint (not identical to Manser's exact profile).
The MVP criterion is therefore qualitative: T2 gain > 1 and
T2/T3 ratio substantially larger than 1.

Run:
    python -m reynolds_solver.experiments.elrod_manser_t2_t3

Configurable via environment variables for a quick smoke run:
    ELROD_MANSER_NPHI=200 ELROD_MANSER_NZ=60 python -m ...
"""
import os
import numpy as np


def make_dimple_centers(N_phi_d, N_Z_d, phi_range, Z_range):
    """
    Uniform grid of dimple centers inside phi_range × Z_range.
    Centers are at cell midpoints of a (N_phi_d × N_Z_d) subdivision.
    """
    phi_edges = np.linspace(phi_range[0], phi_range[1], N_phi_d + 1)
    phi_c = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    Z_edges = np.linspace(Z_range[0], Z_range[1], N_Z_d + 1)
    Z_c = 0.5 * (Z_edges[:-1] + Z_edges[1:])
    Phi_c, ZZ_c = np.meshgrid(phi_c, Z_c)
    return Phi_c.flatten(), ZZ_c.flatten()


def add_wedge_dimples(H, Phi, Z, phi_c_flat, Z_c_flat,
                      r_x, r_y_dimless, r_z, wedge_type):
    """
    Add simplified wedge-shaped depressions to H.

    Footprint: |Δφ| ≤ r_x AND |ΔZ| ≤ r_z (elliptical fall-off in Z).
    Depth profile along φ:
        T2 (convergent): deep at the −φ (leading) edge, zero at the
            +φ (trailing) edge. Linear ramp.
        T3 (divergent):  zero at the −φ edge, deep at the +φ edge.

    The +φ direction is the shaft rotation direction, so −φ is the
    inlet (leading) edge and +φ is the outlet (trailing) edge of
    each dimple footprint.

    Returns a new H; the input is not modified.
    """
    H_new = H.copy()
    for k in range(len(phi_c_flat)):
        phi_c = phi_c_flat[k]
        z_c = Z_c_flat[k]

        delta_phi = np.arctan2(np.sin(Phi - phi_c), np.cos(Phi - phi_c))
        inside_phi = np.abs(delta_phi) <= r_x

        d_z_ratio = (Z - z_c) / r_z
        z_factor = np.clip(1.0 - d_z_ratio * d_z_ratio, 0.0, 1.0)

        if wedge_type == "T2":
            # 1 at delta_phi=-r_x, 0 at delta_phi=+r_x
            ramp = np.clip(0.5 * (1.0 - delta_phi / r_x), 0.0, 1.0)
        elif wedge_type == "T3":
            # 0 at delta_phi=-r_x, 1 at delta_phi=+r_x
            ramp = np.clip(0.5 * (1.0 + delta_phi / r_x), 0.0, 1.0)
        else:
            raise ValueError(f"wedge_type must be T2 or T3, got {wedge_type}")

        depth = r_y_dimless * ramp * z_factor * inside_phi.astype(float)
        H_new += depth
    return H_new


def generate_bearing(N_phi, N_Z, epsilon, R, L, C):
    phi_1D = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)
    Phi_m, Z_m = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_m)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, Phi_m, Z_m, phi_1D, Z, d_phi, d_Z


def compute_load(P, Phi_m, phi_1D, Z):
    """Dimensionless load along y (radial direction)."""
    Wx = float(np.trapezoid(
        np.trapezoid(P * np.cos(Phi_m), phi_1D, axis=1), Z
    ))
    Wy = float(np.trapezoid(
        np.trapezoid(P * np.sin(Phi_m), phi_1D, axis=1), Z
    ))
    return float(np.hypot(Wx, Wy)), Wx, Wy


def run_case(name, H, phi_m, z_m, phi_1d, z_1d, d_phi, d_Z,
             R, L, beta_bar):
    from reynolds_solver.cavitation.elrod import solve_elrod_compressible

    P, theta, res, n_iter = solve_elrod_compressible(
        H, d_phi, d_Z, R, L,
        beta_bar=beta_bar,
        omega=1.0,
        tol=1e-6,
        max_iter=500_000,
        phi_bc="groove",
    )
    W, Wx, Wy = compute_load(P, phi_m, phi_1d, z_1d)
    print(
        f"  {name:<12s}  "
        f"W={W:.4e}  Pmax={P.max():.4e}  "
        f"Θ_max={theta.max():.4f}  Θ_min={theta.min():.4f}  "
        f"n_iter={n_iter:>6d}  res={res:.1e}"
    )
    return W, P, theta


def main():
    # Grid (configurable via env for quicker runs)
    N_phi = int(os.environ.get("ELROD_MANSER_NPHI", 441))
    N_Z = int(os.environ.get("ELROD_MANSER_NZ", 121))

    # Geometry and lubricant (Manser scenario 2)
    R = 0.02       # bearing radius, m
    L = 0.04       # bearing length, m
    C = 50e-6      # radial clearance, m
    epsilon = 0.6
    mu = 0.05
    N_rpm = 3000
    beta = 100e6   # bulk modulus, Pa

    # Derived
    omega_shaft = 2 * np.pi * N_rpm / 60
    U = omega_shaft * R
    beta_bar = beta * C * C / (mu * U * R)

    # Texture (dimensionless)
    r_x_dim = 3e-3 / R           # half-extent along phi (in radians)
    r_z_dim = 3e-3 / (L / 2)     # half-extent along Z (normalised by L/2)
    r_y_dim = 15e-6 / C          # dimensionless depth
    N_phi_d = 14                 # dimples along phi (Ptex=40% → 14 per 2π)
    N_Z_d = 4                    # dimples along Z

    print("=" * 72)
    print("  Manser T2 vs T3 wedge-texture comparison (compressible Elrod)")
    print("=" * 72)
    print(f"  grid        {N_phi} × {N_Z}")
    print(f"  R={R*1000:.1f} mm, L={L*1000:.1f} mm, C={C*1e6:.1f} μm, "
          f"ε={epsilon}")
    print(f"  μ={mu} Pa·s, N={N_rpm} rpm, β={beta/1e6:.0f} MPa")
    print(f"  U = ω·R = {U:.3f} m/s,  β̄ = β·C²/(μ·U·R) = {beta_bar:.2f}")
    print(f"  texture: {N_phi_d}×{N_Z_d} dimples, "
          f"r_x={r_x_dim:.4f} rad, r_z={r_z_dim:.4f}, "
          f"r_y/C={r_y_dim:.3f}")
    print()

    # Build base gap and dimple centers
    H_s, Phi_m, Z_m, phi_1d, z_1d, d_phi, d_Z = generate_bearing(
        N_phi, N_Z, epsilon, R, L, C,
    )
    phi_c_flat, Z_c_flat = make_dimple_centers(
        N_phi_d, N_Z_d,
        phi_range=(0.0, 2 * np.pi),
        Z_range=(-1.0, 1.0),
    )

    H_t2 = add_wedge_dimples(
        H_s, Phi_m, Z_m, phi_c_flat, Z_c_flat,
        r_x_dim, r_y_dim, r_z_dim, wedge_type="T2",
    )
    H_t3 = add_wedge_dimples(
        H_s, Phi_m, Z_m, phi_c_flat, Z_c_flat,
        r_x_dim, r_y_dim, r_z_dim, wedge_type="T3",
    )

    print(f"  {'case':<12s}  {'W':>10s}  {'Pmax':>10s}  "
          f"{'Θ_max':>8s}  {'Θ_min':>8s}  {'n_iter':>6s}  {'res':>7s}")

    W_s, _, _ = run_case(
        "Smooth", H_s, Phi_m, Z_m, phi_1d, z_1d, d_phi, d_Z,
        R, L, beta_bar,
    )
    W_t2, _, _ = run_case(
        "T2 full", H_t2, Phi_m, Z_m, phi_1d, z_1d, d_phi, d_Z,
        R, L, beta_bar,
    )
    W_t3, _, _ = run_case(
        "T3 full", H_t3, Phi_m, Z_m, phi_1d, z_1d, d_phi, d_Z,
        R, L, beta_bar,
    )

    gain_t2 = W_t2 / (W_s + 1e-30)
    gain_t3 = W_t3 / (W_s + 1e-30)
    ratio = gain_t2 / (gain_t3 + 1e-30)

    print()
    print("=" * 72)
    print(f"  gain_W  T2 = {gain_t2:.3f}   (Manser reference ≈ 1.43)")
    print(f"  gain_W  T3 = {gain_t3:.3f}   (Manser reference ≈ 0.41)")
    print(f"  T2 / T3    = {ratio:.3f}   (Manser reference ≈ 3.44)")
    print("=" * 72)

    if ratio > 1.5 and gain_t2 > 1.0:
        print("  MVP SUCCESS: T2 gain > 1 and T2/T3 > 1.5 — "
              "convergent wedge amplifies load, divergent one reduces it.")
    elif ratio > 1.0:
        print("  WEAK SIGNAL: T2/T3 > 1 but < 1.5 — the asymmetry is "
              "present but smaller than Manser reports.")
    else:
        print("  NO MICRO-WEDGE: T2/T3 ≤ 1. The compressible formulation "
              "is not reproducing the Manser asymmetry in this setup.")


if __name__ == "__main__":
    main()
