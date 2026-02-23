"""
Utility functions for the Reynolds equation GPU solver.

- GPU-side coefficient precomputation
- Dynamic RHS modification
- Gap field with ellipsoidal depressions
"""

import numpy as np
import cupy as cp


def precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L):
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

    Returns
    -------
    A, B, C, D, E, F : cupy.ndarray, each shape (N_Z, N_phi), float64
    """
    N_Z, N_phi = H_gpu.shape

    H_i_plus_half = 0.5 * (H_gpu[:, :-1] + H_gpu[:, 1:])
    H_i_minus_half = cp.empty_like(H_i_plus_half)
    H_i_minus_half[:, 1:] = H_i_plus_half[:, :-1]
    H_i_minus_half[:, 0] = H_i_plus_half[:, -1]

    H_j_plus_half = 0.5 * (H_gpu[:-1, :] + H_gpu[1:, :])
    H_j_minus_half = cp.empty_like(H_j_plus_half)
    H_j_minus_half[1:, :] = H_j_plus_half[:-1, :]
    H_j_minus_half[0, :] = H_j_plus_half[-1, :]

    D_over_L = 2.0 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    A_half = H_i_plus_half ** 3
    B_half = H_i_minus_half ** 3
    C_half = alpha_sq * H_j_plus_half ** 3
    D_half = alpha_sq * H_j_minus_half ** 3

    A_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    B_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    C_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    D_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)

    A_full[:, :-1] = A_half
    A_full[:, -1] = A_half[:, 0]

    B_full[:, 1:] = B_half
    B_full[:, 0] = B_half[:, -1]

    C_full[:-1, :] = C_half
    C_full[-1, :] = C_half[0, :]

    D_full[1:, :] = D_half
    D_full[0, :] = D_half[-1, :]

    E_full = A_full + B_full + C_full + D_full

    F_half = d_phi * (H_i_plus_half - H_i_minus_half)
    F_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    F_full[:, :-1] = F_half
    F_full[:, -1] = F_half[:, 0]

    return A_full, B_full, C_full, D_full, E_full, F_full


def add_dynamic_rhs_gpu(F_full, d_phi, N_Z, N_phi, xprime, yprime, beta):
    """
    Add dynamic contribution to RHS F on GPU (in-place).

    F[i,j] += beta * (xprime * sin(phi_global) + yprime * cos(phi_global))
    where phi_global = j * d_phi + pi/4
    """
    j_indices = cp.arange(N_phi, dtype=cp.float64)
    phi_local = j_indices * d_phi
    phi_global = phi_local + cp.pi / 4.0

    dyn_term = beta * (xprime * cp.sin(phi_global) + yprime * cp.cos(phi_global))
    F_full += dyn_term[cp.newaxis, :]


def create_H_with_ellipsoidal_depressions(H0, H_p, Phi_mesh, Z_mesh,
                                           phi_c_flat, Z_c_flat, A, B):
    """
    Create gap field H with ellipsoidal surface depressions (CPU, numpy).

    Parameters
    ----------
    H0 : np.ndarray -- base gap
    H_p : float -- dimensionless depression depth
    Phi_mesh, Z_mesh : np.ndarray -- coordinate meshes
    phi_c_flat, Z_c_flat : np.ndarray -- depression center coordinates
    A, B : float -- dimensionless semi-axes

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
        H[inside] += H_p * np.sqrt(1 - expr[inside])
    return H
