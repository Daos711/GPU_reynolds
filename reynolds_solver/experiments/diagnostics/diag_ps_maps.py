"""
Pressure and θ maps for the Payvar-Salant solver.

Generates contour plots of P(φ, Z) and θ(φ, Z) at several
eccentricity ratios, plus a midplane P(φ) line-cut compared with
the Half-Sommerfeld solution. Used for visual sanity checks
(one connected cavitation zone, no checkerboard, smooth rupture
boundary, bell-shaped P profile).

Outputs: ps_maps_eps{e}.png for each ε.

Run:
    python -m reynolds_solver.experiments.diagnostics.diag_ps_maps
"""
import numpy as np


def generate_test_case(N_phi, N_Z, epsilon=0.6):
    phi_1D = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)
    Phi, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return H, d_phi, d_Z, phi_1D, Z


def hs_reference(H, d_phi, d_Z, R, L):
    """Local HS solver using the PS internals (no cupy dependency)."""
    from reynolds_solver.cavitation.payvar_salant.solver_cpu import (
        _build_coefficients,
        _hs_sor_sweep,
    )

    N_Z, N_phi = H.shape
    H_pack = H.copy()
    H_pack[:, 0] = H_pack[:, -2]
    H_pack[:, -1] = H_pack[:, 1]
    A, B, C, D, E = _build_coefficients(H_pack, d_phi, d_Z, R, L)

    F_hs = np.zeros_like(H_pack)
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jm = j - 1 if j - 1 >= 1 else N_phi - 2
            F_hs[i, j] = d_phi * (H_pack[i, j] - H_pack[i, jm])

    P = np.zeros_like(H_pack)
    for k in range(20000):
        res = _hs_sor_sweep(P, A, B, C, D, E, F_hs, 1.7, N_Z, N_phi)
        if res < 1e-9 and k > 10:
            break
    return P


def plot_one_eps(epsilon, R, L, N_phi, N_Z):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from reynolds_solver.cavitation.payvar_salant import solve_payvar_salant_cpu

    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N_phi, N_Z, epsilon)
    Phi, Z_grid = np.meshgrid(phi_1D, Z)

    # PS solve
    P_ps, theta_ps, res, n = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L,
        omega=1.0, tol=1e-7, max_iter=20000,
    )
    cav_frac = float(np.mean(theta_ps < 1.0 - 1e-6))

    # HS reference
    P_hs = hs_reference(H, d_phi, d_Z, R, L)

    # Recover g for rupture contour
    g = np.where(theta_ps >= 1.0 - 1e-12, P_ps, theta_ps - 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Payvar-Salant:  ε = {epsilon},  "
        f"maxP = {P_ps.max():.3e},  cav_frac = {cav_frac:.3f}",
        fontsize=13,
    )

    # (0,0) P contour
    ax = axes[0, 0]
    im = ax.pcolormesh(Phi, Z_grid, P_ps, cmap="jet", shading="auto")
    ax.contour(Phi, Z_grid, g, levels=[0.0], colors="white", linewidths=1.2)
    ax.set_title("P(φ, Z)")
    ax.set_xlabel("φ")
    ax.set_ylabel("Z")
    fig.colorbar(im, ax=ax, label="P")

    # (0,1) θ contour
    ax = axes[0, 1]
    im = ax.pcolormesh(
        Phi, Z_grid, theta_ps,
        cmap="coolwarm", vmin=0, vmax=1, shading="auto",
    )
    ax.contour(Phi, Z_grid, g, levels=[0.0], colors="black", linewidths=1.2)
    ax.set_title("θ(φ, Z)")
    ax.set_xlabel("φ")
    ax.set_ylabel("Z")
    fig.colorbar(im, ax=ax, label="θ")

    # (1,0) Midplane P(φ) — PS vs HS
    ax = axes[1, 0]
    mid_z = N_Z // 2
    ax.plot(phi_1D, P_ps[mid_z, :], "b-", lw=1.5, label="Payvar-Salant")
    ax.plot(phi_1D, P_hs[mid_z, :], "r--", lw=1.2, label="Half-Sommerfeld")
    ax.set_xlabel("φ")
    ax.set_ylabel("P")
    ax.set_title(f"Midplane P(φ) at Z=0  (ε={epsilon})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (1,1) Midplane θ(φ)
    ax = axes[1, 1]
    ax.plot(phi_1D, theta_ps[mid_z, :], "b-", lw=1.5)
    ax.set_xlabel("φ")
    ax.set_ylabel("θ")
    ax.set_title(f"Midplane θ(φ) at Z=0  (ε={epsilon})")
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    import os
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "results", "payvar_salant"
    )
    os.makedirs(out_dir, exist_ok=True)
    fname = f"ps_maps_eps{epsilon:.1f}.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved: {out_path}  (n_iter={n}, cav_frac={cav_frac:.3f})")


def main():
    print("=" * 60)
    print("  Payvar-Salant P / θ maps")
    print("=" * 60)

    R, L = 0.035, 0.056
    N_phi, N_Z = 200, 80

    for eps in (0.3, 0.6, 0.8):
        plot_one_eps(eps, R, L, N_phi, N_Z)

    print()
    print("  Visually check:")
    print("    - One connected cavitation zone (smooth bearing)")
    print("    - Rupture boundary ≈ π (divergent zone)")
    print("    - Reformation boundary continuous, no jagged teeth")
    print("    - Bell-shaped P(φ) in midplane")
    print("    - No checkerboard / stripes in θ")


if __name__ == "__main__":
    main()
