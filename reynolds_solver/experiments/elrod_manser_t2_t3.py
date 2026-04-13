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

Engines compared (default: all three):
    ELROD_MANSER_FORMULATIONS="ptheta:hard,theta_vk:hard,theta_vk:fk_soft"

Optional pre-checks (TZ §2):
    ELROD_MANSER_DEPTH={ry|2ry}     # depth scaling, default 2ry (Table 2)
    ELROD_MANSER_MIRROR=1           # sign-sanity / mirror test
    ELROD_MANSER_MAXITER=N          # iter cap per case
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
    Add Manser wedge-shaped depressions to H (Manser 2019 Table 2).

    Footprint: rectangular |Δφ| ≤ r_x AND |ΔZ| ≤ r_z.
    φ-profile: linear ramp, max depth Δh = 2·r_y at the deep edge:
        T2 (convergent): Δh(-r_x) = 2·r_y, Δh(+r_x) = 0.
        T3 (divergent):  Δh(-r_x) = 0,     Δh(+r_x) = 2·r_y.
    Z-profile: flat within |ΔZ| ≤ r_z (no parabolic fall-off).

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
        inside_phi = (np.abs(delta_phi) <= r_x).astype(float)

        # Rectangular (not parabolic) Z footprint — Manser Table 2.
        inside_z = (np.abs(Z - z_c) <= r_z).astype(float)

        # Linear φ-ramp between Δφ = -r_x and Δφ = +r_x.
        # Max ramp = 2.0 so that max depth Δh = 2·r_y at the deep edge,
        # matching Manser Table 2 (Δh(-r_x) = 2·r_y for T2; mirrored
        # for T3).
        if wedge_type == "T2":
            # ramp = 2 at delta_phi=-r_x, 0 at delta_phi=+r_x
            ramp = np.clip(1.0 - delta_phi / r_x, 0.0, 2.0)
        elif wedge_type == "T3":
            # ramp = 0 at delta_phi=-r_x, 2 at delta_phi=+r_x
            ramp = np.clip(1.0 + delta_phi / r_x, 0.0, 2.0)
        else:
            raise ValueError(f"wedge_type must be T2 or T3, got {wedge_type}")

        depth = r_y_dimless * ramp * inside_z * inside_phi
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
             R, L, beta_bar,
             formulation="ptheta", switch_backend="hard",
             max_iter=500_000, tol=1e-6):
    from reynolds_solver.cavitation.elrod import solve_elrod_compressible

    P, theta, res, n_iter = solve_elrod_compressible(
        H, d_phi, d_Z, R, L,
        beta_bar=beta_bar,
        omega=1.0,
        tol=tol,
        max_iter=max_iter,
        phi_bc="groove",
        formulation=formulation,
        switch_backend=switch_backend,
    )
    W, Wx, Wy = compute_load(P, phi_m, phi_1d, z_1d)
    cav = float(np.mean(theta[1:-1, 1:-1] < 1.0 - 1e-6))
    # phi index of P_max (midplane)
    iz_mid = theta.shape[0] // 2
    j_pmax = int(np.argmax(P[iz_mid, :]))
    phi_pmax = float(phi_1d[j_pmax])
    # rupture angle: first index past P_max where P drops to 0
    rupt = np.nan
    for j in range(j_pmax, len(phi_1d)):
        if P[iz_mid, j] <= 1e-10:
            rupt = float(phi_1d[j])
            break
    print(
        f"  {name:<22s}  "
        f"W={W:.4e}  Pmax={P.max():.4e}  "
        f"Θ_max={theta.max():.4f}  Θ_min={theta.min():.4f}  "
        f"cav={cav:.3f}  φ(Pmax)={np.degrees(phi_pmax):6.1f}°  "
        f"φ_rupt={np.degrees(rupt):6.1f}°  "
        f"n={n_iter:>6d}  res={res:.1e}"
    )
    return W, P, theta


def report_t2_t3(label, W_s, W_t2, W_t3):
    """Print Manser-style ratio block for one engine/backend."""
    gain_t2 = W_t2 / (W_s + 1e-30)
    gain_t3 = W_t3 / (W_s + 1e-30)
    ratio = gain_t2 / (gain_t3 + 1e-30)
    print()
    print(f"  --- {label} ---")
    print(f"  gain_W  T2 = {gain_t2:.3f}   (Manser ≈ 1.43)")
    print(f"  gain_W  T3 = {gain_t3:.3f}   (Manser ≈ 0.41)")
    print(f"  T2 / T3    = {ratio:.3f}   (Manser ≈ 3.44)")
    if ratio > 1.5 and gain_t2 > 1.0:
        verdict = "MVP SUCCESS (T2/T3 > 1.5, T2 gain > 1)"
    elif ratio > 1.5:
        verdict = (
            "ASYMMETRY PRESENT (T2/T3 > 1.5) but T2 gain < 1 "
            "(absolute scaling off)"
        )
    elif ratio > 1.0:
        verdict = "WEAK SIGNAL (1 < T2/T3 ≤ 1.5)"
    else:
        verdict = "NO MICRO-WEDGE (T2/T3 ≤ 1)"
    print(f"  → {verdict}")
    return gain_t2, gain_t3, ratio


def main():
    # Grid (configurable via env for quicker runs)
    N_phi = int(os.environ.get("ELROD_MANSER_NPHI", 441))
    N_Z = int(os.environ.get("ELROD_MANSER_NZ", 121))
    max_iter = int(os.environ.get("ELROD_MANSER_MAXITER", 500_000))

    # Engine selectors. Default: run all three combinations so the
    # report covers the headline result (theta_vk fk_soft reproduces
    # Manser asymmetry while ptheta does not).
    formulations = os.environ.get(
        "ELROD_MANSER_FORMULATIONS",
        "ptheta:hard,theta_vk:hard,theta_vk:fk_soft",
    ).split(",")
    formulations = [s.strip() for s in formulations if s.strip()]

    # Pre-check 2.2: depth scaling. "ry" means dimple depth = r_y/C,
    # "2ry" means depth = 2·r_y/C (matches Manser Table 2 ramp). The
    # add_wedge_dimples ramp already goes up to 2 — depth_mode="2ry"
    # uses r_y_dim as-is; depth_mode="ry" halves it.
    depth_mode = os.environ.get("ELROD_MANSER_DEPTH", "2ry")
    if depth_mode not in ("ry", "2ry"):
        raise ValueError(
            f"ELROD_MANSER_DEPTH must be 'ry' or '2ry', got {depth_mode!r}"
        )

    # Pre-check 2.1: mirror-test mode (sanity / sign-convention check)
    do_mirror = os.environ.get("ELROD_MANSER_MIRROR", "0") == "1"

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
    r_y_dim_raw = 15e-6 / C      # base dimensionless depth (15 μm / C)
    if depth_mode == "ry":
        # ramp already maxes at 2 inside add_wedge_dimples → halve r_y
        r_y_dim = 0.5 * r_y_dim_raw
    else:
        r_y_dim = r_y_dim_raw    # depth = 2·r_y/C at the deep edge
    N_phi_d = 14                 # dimples along phi (Ptex=40% → 14 per 2π)
    N_Z_d = 4                    # dimples along Z

    print("=" * 78)
    print("  Manser T2 vs T3 wedge-texture comparison (compressible Elrod)")
    print("=" * 78)
    # V0 setup report
    print("  --- V0: setup ---")
    print(f"  grid          {N_phi} × {N_Z},  max_iter={max_iter}")
    print(f"  R={R*1000:.1f} mm, L={L*1000:.1f} mm, C={C*1e6:.1f} μm, "
          f"ε={epsilon}")
    print(f"  μ={mu} Pa·s, N={N_rpm} rpm, β={beta/1e6:.0f} MPa")
    print(f"  U = ω·R = {U:.3f} m/s,  β̄ = β·C²/(μ·U·R) = {beta_bar:.2f}")
    print(f"  texture       {N_phi_d}×{N_Z_d} dimples, "
          f"r_x={r_x_dim:.4f} rad, r_z={r_z_dim:.4f}")
    print(f"  depth_mode    {depth_mode!r}  →  effective r_y/C = "
          f"{r_y_dim:.3f}  (max depth Δh = {2*r_y_dim:.3f})")
    print(f"  formulations  {formulations}")
    print(f"  phi_bc        'groove'   mirror_test={do_mirror}")
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

    # Pre-check 2.1: mirror sanity. Build H_t2_mirror by reflecting the
    # T2 gap field along the φ axis (rotation direction). Under the
    # convention "+φ = rotation direction, T2 = convergent at -r_x", a
    # φ-reflection should turn T2 into T3 geometrically. The solver
    # equation is NOT symmetric under φ → -φ alone (the Couette term
    # ∂(Θh)/∂θ flips sign), but the load magnitude obtained by running
    # the solver on the φ-reflected T2 field with φ_bc='groove' should
    # be close to the load on the equivalent T3 field. A large
    # discrepancy (>10 %) signals a sign-convention bug in the wedge or
    # in the solver's Couette term.
    if do_mirror:
        # Reflect H_t2 in φ: H_mirror[i, j] = H_t2[i, N_phi-1-j]
        H_t2_mirror = H_t2[:, ::-1].copy()

    results = {}    # (formulation, backend) -> (W_s, W_t2, W_t3)

    for spec in formulations:
        if ":" in spec:
            formulation, backend = spec.split(":", 1)
        else:
            formulation, backend = spec, "hard"

        print()
        print("=" * 78)
        print(f"  ENGINE: formulation={formulation!r}, "
              f"switch_backend={backend!r}")
        print("=" * 78)
        print(f"  {'case':<22s}  {'W':>10s}  {'Pmax':>10s}  "
              f"{'Θ_max':>7s}  {'Θ_min':>7s}  {'cav':>6s}  "
              f"{'φ(Pmax)':>8s}  {'φ_rupt':>8s}  {'n':>6s}  {'res':>7s}")

        W_s, _, _ = run_case(
            f"Smooth [{formulation}/{backend}]", H_s, Phi_m, Z_m,
            phi_1d, z_1d, d_phi, d_Z, R, L, beta_bar,
            formulation=formulation, switch_backend=backend,
            max_iter=max_iter,
        )
        W_t2, _, _ = run_case(
            f"T2 full [{formulation}/{backend}]", H_t2, Phi_m, Z_m,
            phi_1d, z_1d, d_phi, d_Z, R, L, beta_bar,
            formulation=formulation, switch_backend=backend,
            max_iter=max_iter,
        )
        W_t3, _, _ = run_case(
            f"T3 full [{formulation}/{backend}]", H_t3, Phi_m, Z_m,
            phi_1d, z_1d, d_phi, d_Z, R, L, beta_bar,
            formulation=formulation, switch_backend=backend,
            max_iter=max_iter,
        )
        if do_mirror:
            W_t2m, _, _ = run_case(
                f"T2_mirror [{formulation}/{backend}]", H_t2_mirror,
                Phi_m, Z_m, phi_1d, z_1d, d_phi, d_Z, R, L, beta_bar,
                formulation=formulation, switch_backend=backend,
                max_iter=max_iter,
            )
            mirror_err = abs(W_t2m - W_t3) / max(W_t3, 1e-30)
            print(f"  → mirror sanity: |W(T2_mirror) - W(T3)| / W(T3) "
                  f"= {mirror_err:.3%}")

        report_t2_t3(f"{formulation} / {backend}", W_s, W_t2, W_t3)
        results[(formulation, backend)] = (W_s, W_t2, W_t3)

    # Final cross-engine summary
    print()
    print("=" * 78)
    print("  FINAL SUMMARY: Manser T2/T3 across engines/backends")
    print("=" * 78)
    print(f"  {'engine/backend':<22s}  {'gain_T2':>8s}  {'gain_T3':>8s}  "
          f"{'T2/T3':>8s}")
    for (form, bck), (W_s, W_t2, W_t3) in results.items():
        g2 = W_t2 / (W_s + 1e-30)
        g3 = W_t3 / (W_s + 1e-30)
        rt = g2 / (g3 + 1e-30)
        print(f"  {form+'/'+bck:<22s}  {g2:8.3f}  {g3:8.3f}  {rt:8.3f}")
    print(f"  {'Manser reference':<22s}  {1.43:8.3f}  {0.41:8.3f}  "
          f"{3.44:8.3f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
