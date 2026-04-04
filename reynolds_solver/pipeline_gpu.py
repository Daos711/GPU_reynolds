"""
Full GPU pipeline for journal bearing analysis.

Computes: F, mu, Q vs eccentricity, 8 K/C coefficients,
stability parameters, and shaft orbit.

Run: python -m reynolds_solver.pipeline_gpu

Replaces old CPU Numba pipeline with GPU solver (~100x speedup).
"""

import os
import time
import numpy as np

# ===================================================================
# Block 0: Parameters
# ===================================================================
R = 0.035        # Bearing radius (m)
c = 0.00005      # Radial clearance (m)
L = 0.056        # Bearing length (m)
n = 2980         # Shaft speed (rpm)
omega_shaft = 2 * np.pi * n / 60
U = omega_shaft * (R - c)
eta = 0.01105    # Viscosity (Pa·s)
h_p = 0.00001    # Depression depth (m)
H_p = h_p / c    # Dimensionless depth

# Ellipsoidal depression semi-axes
a_dim = 0.00241
b_dim = 0.002214
A_tex = 2 * a_dim / L   # dimensionless semi-axis along Z
B_tex = b_dim / R        # dimensionless semi-axis along phi

# Texture pattern
N_phi_tex = 8
N_Z_tex = 11
phi_start_deg = 90
phi_end_deg = 270

# Scaling
pressure_scale = (6 * eta * U * R) / (c**2)
load_scale = pressure_scale * (R * L) / 2
friction_scale = (eta * U * R * L) / c

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "pipeline"
)


def setup_grid(N):
    """Create phi, Z grid and meshes."""
    phi_1D = np.linspace(0, 2 * np.pi, N)
    Z = np.linspace(-1, 1, N)
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]
    return phi_1D, Z, Phi_mesh, Z_mesh, d_phi, d_Z


def setup_texture_centers():
    """Compute depression center coordinates."""
    phi_start = np.deg2rad(phi_start_deg)
    phi_end = np.deg2rad(phi_end_deg)
    phi_centers = np.linspace(phi_start, phi_end, N_phi_tex)
    Z_centers = np.linspace(-0.8, 0.8, N_Z_tex)
    phi_c, Z_c = np.meshgrid(phi_centers, Z_centers)
    return phi_c.ravel(), Z_c.ravel()


def make_H(epsilon, Phi_mesh, Z_mesh, textured=False):
    """Build gap field H (smooth or textured)."""
    from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
    H0 = 1.0 + epsilon * np.cos(Phi_mesh)
    if not textured:
        return H0
    phi_c, Z_c = setup_texture_centers()
    return create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi_mesh, Z_mesh, phi_c, Z_c, A_tex, B_tex)


def compute_forces(P, Phi_mesh, phi_1D, Z):
    """Compute Fx, Fy from pressure field."""
    Fx = np.trapezoid(np.trapezoid(P * np.cos(Phi_mesh), phi_1D, axis=1), Z)
    Fy = np.trapezoid(np.trapezoid(P * np.sin(Phi_mesh), phi_1D, axis=1), Z)
    return Fx, Fy


def compute_FmuQ(P, H, Phi_mesh, phi_1D, Z, d_phi):
    """Compute load F, friction coeff mu, flow rate Q."""
    Fx, Fy = compute_forces(P, Phi_mesh, phi_1D, Z)
    F = np.sqrt(Fx**2 + Fy**2) * load_scale

    # Friction
    dP_dphi = np.zeros_like(P)
    dP_dphi[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * d_phi)
    dP_dphi[:, 0] = (P[:, 1] - P[:, -2]) / (2 * d_phi)
    dP_dphi[:, -1] = dP_dphi[:, 0]
    tau = (1.0 / H) + 3.0 * H * dP_dphi
    F_friction = np.trapezoid(np.trapezoid(tau, phi_1D, axis=1), Z) * friction_scale
    mu_coeff = F_friction / (F + 1e-30)

    # Flow rate
    q_local = H - 0.5 * H**3 * dP_dphi
    Q = np.trapezoid(q_local[len(Z)//2, :], phi_1D)

    return F, mu_coeff, Q


def solve_static(H, d_phi, d_Z, P_init=None, tol=1e-5):
    """Solve static Reynolds on GPU."""
    from reynolds_solver import solve_reynolds
    P, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure="laminar", cavitation="half_sommerfeld",
        omega=1.5, tol=tol, max_iter=50000,
        P_init=P_init)
    return P


def solve_dynamic(H, d_phi, d_Z, xprime, yprime, P_init=None, tol=1e-5):
    """Solve dynamic Reynolds on GPU."""
    from reynolds_solver import solve_reynolds
    P, _, _ = solve_reynolds(
        H, d_phi, d_Z, R, L,
        closure="laminar", cavitation="half_sommerfeld",
        xprime=xprime, yprime=yprime, beta=2.0,
        omega=1.5, tol=tol, max_iter=50000,
        P_init=P_init)
    return P


# ===================================================================
# Block 1: F, mu, Q vs eccentricity
# ===================================================================
def block1_FmuQ(N, epsilon_values, Phi_mesh, Z_mesh, phi_1D, Z, d_phi, d_Z):
    print("\n=== Block 1: F, mu, Q vs eccentricity ===")
    n_eps = len(epsilon_values)
    F_s, mu_s, Q_s = np.zeros(n_eps), np.zeros(n_eps), np.zeros(n_eps)
    F_t, mu_t, Q_t = np.zeros(n_eps), np.zeros(n_eps), np.zeros(n_eps)

    P_prev_s, P_prev_t = None, None
    for i, eps in enumerate(epsilon_values):
        H_smooth = make_H(eps, Phi_mesh, Z_mesh, textured=False)
        H_tex = make_H(eps, Phi_mesh, Z_mesh, textured=True)

        P_s = solve_static(H_smooth, d_phi, d_Z, P_init=P_prev_s)
        P_t = solve_static(H_tex, d_phi, d_Z, P_init=P_prev_t)
        P_prev_s, P_prev_t = P_s, P_t

        F_s[i], mu_s[i], Q_s[i] = compute_FmuQ(P_s, H_smooth, Phi_mesh, phi_1D, Z, d_phi)
        F_t[i], mu_t[i], Q_t[i] = compute_FmuQ(P_t, H_tex, Phi_mesh, phi_1D, Z, d_phi)
        print(f"  eps={eps:.3f}: F_s={F_s[i]:.2f}, F_t={F_t[i]:.2f}")

    return F_s, mu_s, Q_s, F_t, mu_t, Q_t


# ===================================================================
# Block 2: 8 K/C coefficients
# ===================================================================
def block2_KC(N, eps_range, Phi_mesh, Z_mesh, phi_1D, Z, d_phi, d_Z,
              dx=1e-5, dxp=1e-5, textured=False):
    label = "textured" if textured else "smooth"
    print(f"\n=== Block 2: K/C coefficients ({label}) ===")

    sin_phi = np.sin(Phi_mesh)
    cos_phi = np.cos(Phi_mesh)
    n_eps = len(eps_range)

    Kxx = np.zeros(n_eps); Kxy = np.zeros(n_eps)
    Kyx = np.zeros(n_eps); Kyy = np.zeros(n_eps)
    Cxx = np.zeros(n_eps); Cxy = np.zeros(n_eps)
    Cyx = np.zeros(n_eps); Cyy = np.zeros(n_eps)

    # Dimensionalization scales
    F_dim_scale = pressure_scale * R * L / 2   # = load_scale (nd force -> N)
    K_to_dim = F_dim_scale / c                  # N/m (stiffness)
    C_to_dim = F_dim_scale / (c * omega_shaft)  # N·s/m (damping)

    # Use tight tolerance for perturbation solves (dx ~ 1e-5, need tol << dx)
    tol_kc = 1e-8

    for i, eps0 in enumerate(eps_range):
        H_base = make_H(eps0, Phi_mesh, Z_mesh, textured=textured)
        P_base = solve_static(H_base, d_phi, d_Z, tol=tol_kc)

        # --- Stiffness (static perturbations) ---
        # +x: eps -> eps+dx
        H_px = make_H(eps0 + dx, Phi_mesh, Z_mesh, textured=textured)
        P_px = solve_static(H_px, d_phi, d_Z, P_init=P_base, tol=tol_kc)
        Fx_px, Fy_px = compute_forces(P_px, Phi_mesh, phi_1D, Z)

        # -x: eps -> eps-dx
        H_mx = make_H(eps0 - dx, Phi_mesh, Z_mesh, textured=textured)
        P_mx = solve_static(H_mx, d_phi, d_Z, P_init=P_base, tol=tol_kc)
        Fx_mx, Fy_mx = compute_forces(P_mx, Phi_mesh, phi_1D, Z)

        # +y: H_base + dx*sin(phi)
        H_py = H_base + dx * sin_phi
        P_py = solve_static(H_py, d_phi, d_Z, P_init=P_base, tol=tol_kc)
        Fx_py, Fy_py = compute_forces(P_py, Phi_mesh, phi_1D, Z)

        # -y: H_base - dx*sin(phi)
        H_my = H_base - dx * sin_phi
        P_my = solve_static(H_my, d_phi, d_Z, P_init=P_base, tol=tol_kc)
        Fx_my, Fy_my = compute_forces(P_my, Phi_mesh, phi_1D, Z)

        # --- Damping (dynamic perturbations) ---
        # +x' velocity → physical cos(φ) → solver yprime
        P_pxp = solve_dynamic(H_base, d_phi, d_Z, xprime=0, yprime=dxp,
                              P_init=P_base, tol=tol_kc)
        Fx_pxp, Fy_pxp = compute_forces(P_pxp, Phi_mesh, phi_1D, Z)

        # -x' velocity → physical cos(φ) → solver yprime
        P_mxp = solve_dynamic(H_base, d_phi, d_Z, xprime=0, yprime=-dxp,
                              P_init=P_base, tol=tol_kc)
        Fx_mxp, Fy_mxp = compute_forces(P_mxp, Phi_mesh, phi_1D, Z)

        # +y' velocity → physical sin(φ) → solver xprime
        P_pyp = solve_dynamic(H_base, d_phi, d_Z, xprime=dxp, yprime=0,
                              P_init=P_base, tol=tol_kc)
        Fx_pyp, Fy_pyp = compute_forces(P_pyp, Phi_mesh, phi_1D, Z)

        # -y' velocity → physical sin(φ) → solver xprime
        P_myp = solve_dynamic(H_base, d_phi, d_Z, xprime=-dxp, yprime=0,
                              P_init=P_base, tol=tol_kc)
        Fx_myp, Fy_myp = compute_forces(P_myp, Phi_mesh, phi_1D, Z)

        # K = -dF/dq (N/m), C = -dF/dqdot (N·s/m)
        Kxx[i] = -(Fx_px - Fx_mx) / (2 * dx) * K_to_dim
        Kyx[i] = -(Fy_px - Fy_mx) / (2 * dx) * K_to_dim
        Kxy[i] = -(Fx_py - Fx_my) / (2 * dx) * K_to_dim
        Kyy[i] = -(Fy_py - Fy_my) / (2 * dx) * K_to_dim
        Cxx[i] = -(Fx_pxp - Fx_mxp) / (2 * dxp) * C_to_dim
        Cyx[i] = -(Fy_pxp - Fy_mxp) / (2 * dxp) * C_to_dim
        Cxy[i] = -(Fx_pyp - Fx_myp) / (2 * dxp) * C_to_dim
        Cyy[i] = -(Fy_pyp - Fy_myp) / (2 * dxp) * C_to_dim

        print(f"  eps={eps0:.2f}: Kxx={Kxx[i]:.2e} Kyy={Kyy[i]:.2e} "
              f"Cxx={Cxx[i]:.2e} Cyy={Cyy[i]:.2e}")

    return Kxx, Kxy, Kyx, Kyy, Cxx, Cxy, Cyx, Cyy


# ===================================================================
# Block 3: Stability parameters
# ===================================================================
def block3_stability(Kxx, Kxy, Kyx, Kyy, Cxx, Cxy, Cyx, Cyy):
    """Stability params from old model formulas (4.19)-(4.21)."""
    n_eps = len(Kxx)
    Keq = np.zeros(n_eps)
    gamma_sq = np.zeros(n_eps)
    omega_st = np.zeros(n_eps)

    for i in range(n_eps):
        # (4.19) Equivalent stiffness
        denom = Cxx[i] + Cyy[i]
        if abs(denom) < 1e-12:
            Keq[i] = 0.0
        else:
            Keq[i] = (Kxx[i]*Cyy[i] + Kyy[i]*Cxx[i]
                     - Kxy[i]*Cyx[i] - Kyx[i]*Cxy[i]) / denom

        # (4.20) Logarithmic decrement
        num = (Keq[i] - Kxx[i]) * (Keq[i] - Kyy[i]) - Kxy[i]*Kyx[i]
        den = Cxx[i]*Cyy[i] - Cxy[i]*Cyx[i]
        if abs(den) < 1e-12:
            gamma_sq[i] = 0.0
        else:
            gamma_sq[i] = num / den

        # (4.21) Critical speed
        if abs(gamma_sq[i]) < 1e-12:
            omega_st[i] = 0.0
        else:
            omega_st[i] = Keq[i] / gamma_sq[i]

    return Keq, gamma_sq, omega_st


# ===================================================================
# Block 4: Orbit integration
# ===================================================================
def block4_orbit(Kxx_val, Kxy_val, Kyx_val, Kyy_val,
                 Cxx_val, Cxy_val, Cyx_val, Cyy_val):
    from scipy.integrate import solve_ivp

    # Dimensionless parameters
    psi = c / R
    K_scale_nd = eta * omega_shaft * L / psi**3
    C_scale_nd = eta * L / psi**3

    # Dimensionless coefficients
    Kxx_nd = Kxx_val / K_scale_nd
    Kxy_nd = Kxy_val / K_scale_nd
    Kyx_nd = Kyx_val / K_scale_nd
    Kyy_nd = Kyy_val / K_scale_nd
    Cxx_nd = Cxx_val / C_scale_nd
    Cxy_nd = Cxy_val / C_scale_nd
    Cyx_nd = Cyx_val / C_scale_nd
    Cyy_nd = Cyy_val / C_scale_nd

    m_rotor = 1.0      # rotor mass, kg
    F0 = 1.0e4          # external force amplitude, N
    m_nd = m_rotor * omega_shaft**2 / K_scale_nd
    F_nd = F0 / (K_scale_nd * c)

    def ode(t_star, state):
        x, y, vx, vy = state
        Fx_nd = F_nd * np.cos(t_star)
        Fy_nd = F_nd * np.sin(t_star)
        ax = (Fx_nd - Cxx_nd*vx - Cxy_nd*vy - Kxx_nd*x - Kxy_nd*y) / m_nd
        ay = (Fy_nd - Cyx_nd*vx - Cyy_nd*vy - Kyx_nd*x - Kyy_nd*y) / m_nd
        return [vx, vy, ax, ay]

    t_star_max = 1000
    n_pts = 200000
    t_eval = np.linspace(0, t_star_max, n_pts)
    sol = solve_ivp(ode, (0, t_star_max), [0, 0, 0, 0],
                    t_eval=t_eval, method='BDF', rtol=1e-8, atol=1e-10)

    # Convert to meters
    return sol.t, sol.y[0] * c, sol.y[1] * c


# ===================================================================
# Plotting
# ===================================================================
def plot_two_curves(x, y1, y2, xlabel, ylabel, title, fname, label1="Smooth", label2="Textured"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y1, "b-o", markersize=4, label=label1)
    ax.plot(x, y2, "r-s", markersize=4, label=label2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


def plot_orbit(t, x, y, fname, title="Shaft orbit"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x, y, "b-", linewidth=0.5)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    axes[0].set_title(title + " (full)"); axes[0].set_aspect("equal")
    # Last portion
    n_last = min(len(t), len(t)//3)
    axes[1].plot(x[-n_last:], y[-n_last:], "b-", linewidth=0.5)
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    axes[1].set_title(title + " (last third)"); axes[1].set_aspect("equal")
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ===================================================================
# Block 5: Performance benchmark
# ===================================================================
def block5_benchmark(N):
    print(f"\n=== Block 5: Performance benchmark (N={N}) ===")
    from reynolds_solver import solve_reynolds

    phi_1D, Z, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N)
    H = make_H(0.6, Phi_mesh, Z_mesh, textured=False)

    # Warmup
    solve_reynolds(H, d_phi, d_Z, R, L,
                   closure="laminar", cavitation="half_sommerfeld",
                   omega=1.5, tol=1e-5, max_iter=50000)

    # Timed run
    t0 = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        solve_reynolds(H, d_phi, d_Z, R, L,
                       closure="laminar", cavitation="half_sommerfeld",
                       omega=1.5, tol=1e-5, max_iter=50000)
    t_gpu = (time.perf_counter() - t0) / n_runs
    print(f"  GPU: {t_gpu:.3f} s per solve ({N}x{N})")
    return t_gpu


# ===================================================================
# Main
# ===================================================================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    N = 300
    epsilon_FmuQ = np.linspace(0.05, 0.8, 16)
    eps_KC = np.linspace(0.2, 0.8, 10)

    phi_1D, Z, Phi_mesh, Z_mesh, d_phi, d_Z = setup_grid(N)

    print("=" * 60)
    print(f"  GPU bearing pipeline (N={N})")
    print("=" * 60)

    # --- Checkpoint: eps=0.6 comparison ---
    print("\n--- Checkpoint: eps=0.6 ---")
    H_check = make_H(0.6, Phi_mesh, Z_mesh, textured=False)
    P_check = solve_static(H_check, d_phi, d_Z)
    F_check, mu_check, Q_check = compute_FmuQ(
        P_check, H_check, Phi_mesh, phi_1D, Z, d_phi)
    print(f"  GPU: F={F_check:.4f}, mu={mu_check:.6f}, Q={Q_check:.6f}")

    t_total = time.perf_counter()

    # --- Block 1: F, mu, Q ---
    F_s, mu_s, Q_s, F_t, mu_t, Q_t = block1_FmuQ(
        N, epsilon_FmuQ, Phi_mesh, Z_mesh, phi_1D, Z, d_phi, d_Z)

    # --- Block 2: K/C coefficients ---
    print("\n--- K/C: Smooth ---")
    Kxx_s, Kxy_s, Kyx_s, Kyy_s, Cxx_s, Cxy_s, Cyx_s, Cyy_s = \
        block2_KC(N, eps_KC, Phi_mesh, Z_mesh, phi_1D, Z, d_phi, d_Z,
                  textured=False)

    print("\n--- K/C: Textured ---")
    Kxx_t, Kxy_t, Kyx_t, Kyy_t, Cxx_t, Cxy_t, Cyx_t, Cyy_t = \
        block2_KC(N, eps_KC, Phi_mesh, Z_mesh, phi_1D, Z, d_phi, d_Z,
                  textured=True)

    # --- Block 3: Stability ---
    print("\n=== Block 3: Stability parameters ===")
    Keq_s, gamma_s, omega_st_s = block3_stability(
        Kxx_s, Kxy_s, Kyx_s, Kyy_s, Cxx_s, Cxy_s, Cyx_s, Cyy_s)
    Keq_t, gamma_t, omega_st_t = block3_stability(
        Kxx_t, Kxy_t, Kyx_t, Kyy_t, Cxx_t, Cxy_t, Cyx_t, Cyy_t)
    for i, eps in enumerate(eps_KC):
        print(f"  eps={eps:.2f}: Keq_s={Keq_s[i]:.1f} gamma_s={gamma_s[i]:.2f} "
              f"Keq_t={Keq_t[i]:.1f} gamma_t={gamma_t[i]:.2f}")

    # --- Block 4: Orbit ---
    print("\n=== Block 4: Orbit (eps=0.6) ===")
    # Find index closest to 0.6
    idx06 = np.argmin(np.abs(eps_KC - 0.6))
    for label, Kxx, Kxy, Kyx, Kyy, Cxx, Cxy, Cyx, Cyy, tag in [
        ("Smooth", Kxx_s, Kxy_s, Kyx_s, Kyy_s, Cxx_s, Cxy_s, Cyx_s, Cyy_s, "smooth"),
        ("Textured", Kxx_t, Kxy_t, Kyx_t, Kyy_t, Cxx_t, Cxy_t, Cyx_t, Cyy_t, "textured"),
    ]:
        try:
            t_orb, x_orb, y_orb = block4_orbit(
                Kxx[idx06], Kxy[idx06], Kyx[idx06], Kyy[idx06],
                Cxx[idx06], Cxy[idx06], Cyx[idx06], Cyy[idx06])
            plot_orbit(t_orb, x_orb, y_orb, f"orbit_{tag}.png", f"Orbit ({label})")
            print(f"  {label}: max|x|={np.max(np.abs(x_orb)):.4e}, "
                  f"max|y|={np.max(np.abs(y_orb)):.4e}")
        except Exception as e:
            print(f"  {label}: orbit integration failed: {e}")

    # --- Block 5: Benchmark ---
    t_gpu = block5_benchmark(N)

    t_total = time.perf_counter() - t_total
    print(f"\n  Total pipeline time: {t_total:.1f} s")

    # --- Plots ---
    print("\n=== Saving plots ===")
    plot_two_curves(epsilon_FmuQ, F_s, F_t, "epsilon", "F (N)", "Load capacity", "F_vs_eps.png")
    plot_two_curves(epsilon_FmuQ, mu_s, mu_t, "epsilon", "mu", "Friction coefficient", "mu_vs_eps.png")
    plot_two_curves(epsilon_FmuQ, Q_s, Q_t, "epsilon", "Q", "Flow rate", "Q_vs_eps.png")

    for name, ys, yt in [
        ("Kxx", Kxx_s, Kxx_t), ("Kxy", Kxy_s, Kxy_t),
        ("Kyx", Kyx_s, Kyx_t), ("Kyy", Kyy_s, Kyy_t),
        ("Cxx", Cxx_s, Cxx_t), ("Cxy", Cxy_s, Cxy_t),
        ("Cyx", Cyx_s, Cyx_t), ("Cyy", Cyy_s, Cyy_t),
    ]:
        plot_two_curves(eps_KC, ys, yt, "epsilon", name, f"{name} vs epsilon", f"{name}_vs_eps.png")

    plot_two_curves(eps_KC, Keq_s, Keq_t, "epsilon", "K_eq", "Equivalent stiffness", "Keq_vs_eps.png")
    plot_two_curves(eps_KC, gamma_s, gamma_t, "epsilon", "gamma^2", "Stability margin", "gamma_vs_eps.png")
    plot_two_curves(eps_KC, omega_st_s, omega_st_t, "epsilon", "omega_st", "Critical speed", "omega_st_vs_eps.png")

    # --- Save data ---
    data_path = os.path.join(RESULTS_DIR, "data.npz")
    np.savez(data_path,
        eps_FmuQ=epsilon_FmuQ,
        F_smooth=F_s, F_textured=F_t,
        mu_smooth=mu_s, mu_textured=mu_t,
        Q_smooth=Q_s, Q_textured=Q_t,
        eps_KC=eps_KC,
        Kxx_smooth=Kxx_s, Kxy_smooth=Kxy_s, Kyx_smooth=Kyx_s, Kyy_smooth=Kyy_s,
        Cxx_smooth=Cxx_s, Cxy_smooth=Cxy_s, Cyx_smooth=Cyx_s, Cyy_smooth=Cyy_s,
        Kxx_textured=Kxx_t, Kxy_textured=Kxy_t, Kyx_textured=Kyx_t, Kyy_textured=Kyy_t,
        Cxx_textured=Cxx_t, Cxy_textured=Cxy_t, Cyx_textured=Cyx_t, Cyy_textured=Cyy_t,
        Keq_smooth=Keq_s, gamma_smooth=gamma_s, omega_st_smooth=omega_st_s,
        Keq_textured=Keq_t, gamma_textured=gamma_t, omega_st_textured=omega_st_t,
        t_gpu=t_gpu,
    )
    print(f"  Data saved: {data_path}")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
