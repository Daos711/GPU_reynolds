"""
Sparse CSR matrix assembly from 5-point stencil coefficients (vectorized).

Matrix is built for INTERIOR nodes of the grid:
    i in [1, N_Z-2], j in [1, N_phi-2]

Total unknowns: N_inner = (N_Z - 2) * (N_phi - 2)

Linear index: k = (i - 1) * (N_phi - 2) + (j - 1)

Boundary conditions:
    - Z (Dirichlet P=0): rows for i=1 and i=N_Z-2 have no links
      to i=0 and i=N_Z-1 (P=0 there, contribution to RHS = 0)
    - phi (periodic): j=1 links to j=N_phi-2, and vice versa
"""

import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cusparse
import scipy.sparse as sp


def _build_sparse_core(A_cpu, B_cpu, C_cpu, D_cpu, E_cpu, F_cpu, N_Z, N_phi):
    """
    Build sparse matrix from numpy stencil coefficient arrays.

    Returns
    -------
    M_cpu : scipy.sparse.csr_matrix, shape (N_inner, N_inner)
    f_cpu : np.ndarray, shape (N_inner,)
    """
    N_inner_Z = N_Z - 2
    N_inner_phi = N_phi - 2
    N_total = N_inner_Z * N_inner_phi

    # Vectorized index grids for interior nodes
    i_idx, j_idx = np.mgrid[1:N_Z - 1, 1:N_phi - 1]
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()

    # Linear index of each interior node
    k_flat = (i_flat - 1) * N_inner_phi + (j_flat - 1)

    # --- Diagonal: -E ---
    diag_vals = -E_cpu[i_flat, j_flat]

    # --- Right neighbor j+1: coefficient A ---
    j_right = np.where(j_flat + 1 > N_phi - 2, 1, j_flat + 1)
    right_cols = (i_flat - 1) * N_inner_phi + (j_right - 1)
    right_vals = A_cpu[i_flat, j_flat]

    # --- Left neighbor j-1: coefficient B ---
    j_left = np.where(j_flat - 1 < 1, N_phi - 2, j_flat - 1)
    left_cols = (i_flat - 1) * N_inner_phi + (j_left - 1)
    left_vals = B_cpu[i_flat, j_flat]

    # --- Below neighbor i+1: coefficient C ---
    # Only exists when i+1 <= N_Z-2 (i.e. i < N_Z-2)
    mask_below = i_flat + 1 <= N_Z - 2
    below_rows = k_flat[mask_below]
    below_cols = (i_flat[mask_below]) * N_inner_phi + (j_flat[mask_below] - 1)
    below_vals = C_cpu[i_flat[mask_below], j_flat[mask_below]]

    # --- Above neighbor i-1: coefficient D ---
    # Only exists when i-1 >= 1 (i.e. i > 1)
    mask_above = i_flat - 1 >= 1
    above_rows = k_flat[mask_above]
    above_cols = (i_flat[mask_above] - 2) * N_inner_phi + (j_flat[mask_above] - 1)
    above_vals = D_cpu[i_flat[mask_above], j_flat[mask_above]]

    # Concatenate all entries
    all_rows = np.concatenate([k_flat, k_flat, k_flat, below_rows, above_rows])
    all_cols = np.concatenate([k_flat, right_cols, left_cols, below_cols, above_cols])
    all_vals = np.concatenate([diag_vals, right_vals, left_vals, below_vals, above_vals])

    # Build CSR on CPU
    M_cpu = sp.coo_matrix(
        (all_vals, (all_rows, all_cols)), shape=(N_total, N_total)
    ).tocsr()

    # RHS vector
    f_cpu = F_cpu[i_flat, j_flat]

    return M_cpu, f_cpu


def build_sparse_matrix_gpu(A, B, C, D, E, F, N_Z, N_phi):
    """
    Build sparse matrix M and RHS vector f from stencil coefficients.

    Returns (cupyx CSR, cupy array) for GPU solvers.

    Parameters
    ----------
    A, B, C, D, E, F : cp.ndarray, shape (N_Z, N_phi)
    N_Z, N_phi : int

    Returns
    -------
    M : cupyx.scipy.sparse.csr_matrix, shape (N_inner, N_inner)
    f : cp.ndarray, shape (N_inner,)
    """
    A_cpu = cp.asnumpy(A)
    B_cpu = cp.asnumpy(B)
    C_cpu = cp.asnumpy(C)
    D_cpu = cp.asnumpy(D)
    E_cpu = cp.asnumpy(E)
    F_cpu = cp.asnumpy(F)

    M_cpu, f_cpu = _build_sparse_core(A_cpu, B_cpu, C_cpu, D_cpu, E_cpu, F_cpu, N_Z, N_phi)

    M_gpu = cusparse.csr_matrix(M_cpu)
    f_gpu = cp.asarray(f_cpu)

    return M_gpu, f_gpu


def build_sparse_matrix_cpu(A, B, C, D, E, F, N_Z, N_phi):
    """
    Build sparse matrix M and RHS vector f from stencil coefficients.

    Returns (scipy CSR, numpy array) for CPU solvers (AMG, direct).

    Parameters
    ----------
    A, B, C, D, E, F : cp.ndarray, shape (N_Z, N_phi)
    N_Z, N_phi : int

    Returns
    -------
    M : scipy.sparse.csr_matrix, shape (N_inner, N_inner)
    f : np.ndarray, shape (N_inner,)
    """
    A_cpu = cp.asnumpy(A)
    B_cpu = cp.asnumpy(B)
    C_cpu = cp.asnumpy(C)
    D_cpu = cp.asnumpy(D)
    E_cpu = cp.asnumpy(E)
    F_cpu = cp.asnumpy(F)

    return _build_sparse_core(A_cpu, B_cpu, C_cpu, D_cpu, E_cpu, F_cpu, N_Z, N_phi)
