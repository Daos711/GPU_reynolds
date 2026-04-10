"""
PDE residual breakdown for the Payvar-Salant solver.

Verifies the hypothesis that the O(0.07) PDE residual seen after
convergence is concentrated on the 1-2 cell strip at the cavitation /
full-film boundary (the rupture line), and is negligible in the interior
of both the full-film and the cavitation zones.

Outputs:
  * Per-zone residual statistics (interior full-film, interior cav,
    boundary strip).
  * A heatmap PNG of |res(i,j)| with the cav_mask contour overlaid.

Run:
    python -m reynolds_solver.experiments.diagnostics.diag_pde_residual
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


def compute_nodal_pde_residual(g, H, A, B, C, D, E, d_phi, N_Z, N_phi):
    """Return (N_Z, N_phi) array of signed residual at every interior node."""
    res = np.zeros((N_Z, N_phi), dtype=np.float64)
    for i in range(1, N_Z - 1):
        for j in range(1, N_phi - 1):
            jp = j + 1 if j + 1 < N_phi - 1 else 1
            jm = j - 1 if j - 1 >= 1 else N_phi - 2

            g_ij = g[i, j]
            P_ij = max(g_ij, 0.0)
            theta_ij = 1.0 if g_ij >= 0.0 else max(1.0 + g_ij, 0.0)

            P_jp = max(g[i, jp], 0.0)
            P_jm = max(g[i, jm], 0.0)
            P_ip = max(g[i + 1, j], 0.0)
            P_im = max(g[i - 1, j], 0.0)

            g_jm = g[i, jm]
            theta_jm = 1.0 if g_jm >= 0.0 else max(1.0 + g_jm, 0.0)

            lhs = (
                A[i, j] * P_jp + B[i, j] * P_jm
                + C[i, j] * P_ip + D[i, j] * P_im
                - E[i, j] * P_ij
            )
            rhs = d_phi * (H[i, j] * theta_ij - H[i, jm] * theta_jm)

            res[i, j] = lhs - rhs
    return res


def main():
    from reynolds_solver.cavitation.payvar_salant.solver_cpu import (
        solve_payvar_salant_cpu,
        _build_coefficients,
    )

    R, L = 0.035, 0.056
    N_phi, N_Z = 100, 40
    epsilon = 0.6
    H, d_phi, d_Z, phi_1D, Z = generate_test_case(N_phi, N_Z, epsilon)

    print("=" * 60)
    print("  PDE residual breakdown (Payvar-Salant, ε=0.6)")
    print("=" * 60)
    print()

    P, theta, update_res, n_iter = solve_payvar_salant_cpu(
        H, d_phi, d_Z, R, L,
        omega=1.0, tol=1e-7, max_iter=20000, verbose=True,
    )

    # Recover g from (P, θ)
    g = np.where(theta >= 1.0 - 1e-12, P, theta - 1.0)

    # Ghost-pack H and build coefficients (same as solver does internally)
    H_pack = H.copy()
    H_pack[:, 0] = H_pack[:, -2]
    H_pack[:, -1] = H_pack[:, 1]
    A, B, C, D, E = _build_coefficients(H_pack, d_phi, d_Z, R, L)

    # Nodal residual
    res = compute_nodal_pde_residual(g, H_pack, A, B, C, D, E, d_phi, N_Z, N_phi)
    abs_res = np.abs(res)

    # Interior classification based on g value
    g_eps = 0.01
    interior = np.zeros((N_Z, N_phi), dtype=np.int32)  # 0=boundary
    interior[1:-1, 1:-1] = 1  # start with all interior

    ff_interior = (g > g_eps) & (interior == 1)
    cav_interior = (g < -g_eps) & (interior == 1)
    boundary_strip = (np.abs(g) <= g_eps) & (interior == 1)

    n_ff = int(ff_interior.sum())
    n_cav = int(cav_interior.sum())
    n_bnd = int(boundary_strip.sum())

    print()
    print(f"  Classification (g_eps={g_eps}):")
    print(f"    full-film interior (g > {g_eps}):  {n_ff} nodes")
    print(f"    cavitation interior (g < -{g_eps}): {n_cav} nodes")
    print(f"    boundary strip (|g| ≤ {g_eps}):    {n_bnd} nodes")
    print()

    def zone_stats(mask, label):
        vals = abs_res[mask]
        if vals.size == 0:
            print(f"  {label}: no nodes")
            return
        print(
            f"  {label}: max={vals.max():.3e}, mean={vals.mean():.3e}, "
            f"median={np.median(vals):.3e}, nodes={vals.size}"
        )

    zone_stats(ff_interior, "full-film interior")
    zone_stats(cav_interior, "cavitation interior")
    zone_stats(boundary_strip, "boundary strip")
    zone_stats(interior == 1, "all interior")

    # Global max
    max_abs = float(abs_res.max())
    max_idx = np.unravel_index(np.argmax(abs_res), abs_res.shape)
    print()
    print(
        f"  Global max |res| = {max_abs:.3e} at node {max_idx}, "
        f"g={g[max_idx]:.4f}"
    )

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # Left: |res| heatmap
        ax = axes[0]
        Phi_grid, Z_grid = np.meshgrid(phi_1D, Z)
        im = ax.pcolormesh(
            Phi_grid, Z_grid, abs_res,
            cmap="hot", shading="auto",
        )
        # Cav mask contour (g = 0 line)
        ax.contour(
            Phi_grid, Z_grid, g,
            levels=[0.0], colors="cyan", linewidths=1.5,
        )
        ax.set_xlabel("φ")
        ax.set_ylabel("Z")
        ax.set_title("|PDE residual| with cav boundary (cyan)")
        fig.colorbar(im, ax=ax, label="|res|")

        # Right: histogram of log10(|res|) by zone
        ax2 = axes[1]
        for mask, label, color in [
            (ff_interior, "full-film", "blue"),
            (cav_interior, "cavitation", "green"),
            (boundary_strip, "boundary", "red"),
        ]:
            vals = abs_res[mask]
            if vals.size == 0:
                continue
            vals_log = np.log10(vals + 1e-20)
            ax2.hist(
                vals_log, bins=40, alpha=0.5, label=label, color=color,
            )
        ax2.set_xlabel("log₁₀(|PDE residual|)")
        ax2.set_ylabel("count")
        ax2.set_title("Residual distribution by zone")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        import os
        out_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "results", "payvar_salant"
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "pde_residual_breakdown.png")
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"\n  [plot] saved to: {out_path}")
    except Exception as e:
        print(f"\n  [plot] skipped: {e}")


if __name__ == "__main__":
    main()
