"""
Utility functions for the Reynolds equation GPU solver.

- GPU-side coefficient precomputation
- Dynamic RHS modification
- Gap field with ellipsoidal depressions
- Auto SOR relaxation parameter
"""

import numpy as np
import cupy as cp


def compute_auto_omega(N_phi: int, N_Z: int, R: float, L: float,
                       cap: float = 1.97) -> float:
    """
    Recommended SOR omega based on anisotropic Young (1954) estimate.

    Not a strict optimum for the full nonlinear Reynolds equation
    (variable H³, cavitation, piezoviscosity), but a good automatic
    choice that prevents false convergence on fine grids.

    Parameters
    ----------
    N_phi, N_Z : int — grid dimensions
    R, L : float — bearing radius and length (m)
    cap : float — upper limit on omega (default 1.97).
        For nonlinear paths (piezoviscous, JFO) use 1.93–1.95.

    Returns
    -------
    float — omega in [1.0, cap]
    """
    d_phi = 2 * np.pi / N_phi
    d_Z = 2.0 / (N_Z - 1)
    D_over_L = 2.0 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    cos_z = np.cos(np.pi / (N_Z - 1))
    cos_phi = np.cos(np.pi / N_phi)
    rho_J = (cos_z + alpha_sq * cos_phi) / (1.0 + alpha_sq)

    omega_raw = 2.0 / (1.0 + np.sqrt(max(1.0 - rho_J ** 2, 1e-30)))
    return float(max(1.0, min(omega_raw, cap)))


def precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L, closure=None):
    """
    Compute discretization coefficients A, B, C, D, E, F on GPU.

    All operations via CuPy -- no CPU transfer.

    Parameters
    ----------
    H_gpu : cupy.ndarray, shape (N_Z, N_phi), float64
    d_phi : float
    d_Z : float
    R : float
    L : float
    closure : Closure or None
        Conductance model. None defaults to LaminarClosure.

    Returns
    -------
    A, B, C, D, E, F : cupy.ndarray, each shape (N_Z, N_phi), float64
    """
    if closure is None:
        from reynolds_solver.physics.closures import LaminarClosure
        closure = LaminarClosure()

    N_Z, N_phi = H_gpu.shape

    H_i_plus_half, H_j_plus_half, A_half, C_half_raw = \
        closure.modify_conductances(H_gpu, d_phi, d_Z, R, L)

    H_i_minus_half = cp.empty_like(H_i_plus_half)
    H_i_minus_half[:, 1:] = H_i_plus_half[:, :-1]
    H_i_minus_half[:, 0] = H_i_plus_half[:, -1]

    # B_half: conductance at i-1/2 faces
    # For laminar: H_i_minus_half**3. For turbulent: need closure on minus-half faces.
    # Since closure returns A_half at i+1/2, B_half is A_half shifted by one column.
    B_half = cp.empty_like(A_half)
    B_half[:, 1:] = A_half[:, :-1]
    B_half[:, 0] = A_half[:, -1]

    D_over_L = 2.0 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    A_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    B_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    C_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    D_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)

    A_full[:, :-1] = A_half
    A_full[:, -1] = A_half[:, 0]

    B_full[:, 1:] = B_half
    B_full[:, 0] = B_half[:, -1]

    # Z-direction: Dirichlet BC (P=0 at boundaries), no periodic wrap.
    # For internal nodes i = 1..N_Z-2:
    #   C[i] uses interface (i, i+1) -> C_half_raw[i]
    #   D[i] uses interface (i-1, i) -> C_half_raw[i-1]
    C_full[1:-1, :] = alpha_sq * C_half_raw[1:, :]
    D_full[1:-1, :] = alpha_sq * C_half_raw[:-1, :]

    E_full = A_full + B_full + C_full + D_full

    F_half = d_phi * (H_i_plus_half - H_i_minus_half)
    F_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    F_full[:, :-1] = F_half
    F_full[:, -1] = F_half[:, 0]

    return A_full, B_full, C_full, D_full, E_full, F_full


def build_F_theta_gpu(H_gpu, theta_gpu, d_phi):
    """
    Build JFO RHS: F_theta = d(H*theta)/dphi using face-based fluxes.

    Uses the same ghost/physical indexing and face scheme as F_orig
    in precompute_coefficients_gpu.

    When theta=1 everywhere, F_theta == F_orig to machine precision.

    Parameters
    ----------
    H_gpu : cupy.ndarray, shape (N_Z, N_phi), float64
    theta_gpu : cupy.ndarray, shape (N_Z, N_phi), float64
    d_phi : float

    Returns
    -------
    F_theta : cupy.ndarray, shape (N_Z, N_phi), float64
    """
    N_Z, N_phi = H_gpu.shape
    Hth = H_gpu * theta_gpu

    S_plus_half = 0.5 * (Hth[:, :-1] + Hth[:, 1:])

    S_minus_half = cp.empty_like(S_plus_half)
    S_minus_half[:, 1:] = S_plus_half[:, :-1]
    S_minus_half[:, 0] = S_plus_half[:, -1]

    F_half = d_phi * (S_plus_half - S_minus_half)

    F_theta = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    F_theta[:, :-1] = F_half
    F_theta[:, -1] = F_half[:, 0]

    return F_theta


def add_dynamic_rhs_gpu(F_full, d_phi, N_Z, N_phi, xprime, yprime, beta, phase_shift=0.0):
    """
    Add dynamic contribution to RHS F on GPU (in-place).

    F[i,j] += beta * (xprime * sin(phi_global) + yprime * cos(phi_global))
    where phi_global = j * d_phi + phase_shift.

    Parameters
    ----------
    F_full : cupy.ndarray, shape (N_Z, N_phi)
    d_phi : float
    N_Z, N_phi : int
    xprime, yprime : float
    beta : float
    phase_shift : float
        Phase offset added to phi (default 0.0).
        Use np.pi/4 ONLY to reproduce legacy behavior for validation.
    """
    j_indices = cp.arange(N_phi, dtype=cp.float64)
    phi_local = j_indices * d_phi
    phi_global = phi_local + phase_shift

    dyn_term = beta * (xprime * cp.sin(phi_global) + yprime * cp.cos(phi_global))
    F_full += dyn_term[cp.newaxis, :]


def create_H_with_ellipsoidal_depressions(H0, H_p, Phi_mesh, Z_mesh,
                                           phi_c_flat, Z_c_flat, A, B,
                                           profile="sqrt"):
    """
    Create gap field H with ellipsoidal surface depressions (CPU, numpy).

    Parameters
    ----------
    H0 : np.ndarray -- base gap
    H_p : float -- dimensionless depression depth
    Phi_mesh, Z_mesh : np.ndarray -- coordinate meshes
    phi_c_flat, Z_c_flat : np.ndarray -- depression center coordinates
    A, B : float -- dimensionless semi-axes
    profile : str -- depression shape: "sqrt" (default) or "smoothcap"

    Returns
    -------
    H : np.ndarray -- gap with depressions
    """
    H = H0.copy()
    for k in range(len(phi_c_flat)):
        phi_c = phi_c_flat[k]
        Z_c = Z_c_flat[k]
        delta_phi = np.arctan2(np.sin(Phi_mesh - phi_c), np.cos(Phi_mesh - phi_c))
        expr = (delta_phi / B) ** 2 + ((Z_mesh - Z_c) / A) ** 2
        inside = expr <= 1
        if profile == "sqrt":
            H[inside] += H_p * np.sqrt(1 - expr[inside])
        elif profile == "smoothcap":
            H[inside] += H_p * (1 - expr[inside])**2
        else:
            raise ValueError(f"Unknown profile: {profile}")
    return H
