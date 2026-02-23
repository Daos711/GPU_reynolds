"""
CUDA C kernels for Red-Black SOR solution of the Reynolds equation.

Contains:
  - rb_sor_step:  one half-step of Red-Black SOR (no residual computation)
  - apply_bc:     boundary conditions (periodic in phi, Dirichlet in Z)

All kernels are compiled via CuPy RawKernel on first call and cached.
"""

import cupy as cp

# ---------------------------------------------------------------------------
# CUDA kernel: one Red-Black SOR half-step (NO atomicAdd — pure update)
# ---------------------------------------------------------------------------
_RB_SOR_KERNEL_CODE = r"""
extern "C" __global__ void rb_sor_step(
    double* __restrict__ P,
    const double* __restrict__ A_arr,
    const double* __restrict__ B_arr,
    const double* __restrict__ C_arr,
    const double* __restrict__ D_arr,
    const double* __restrict__ E_arr,
    const double* __restrict__ F_arr,
    const int N_Z,
    const int N_phi,
    const double omega_sor,
    const int color
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= N_Z - 1 || j >= N_phi - 1) return;

    // checkerboard coloring: skip points of the other color
    if ((i + j) % 2 != color) return;

    int idx = i * N_phi + j;

    // periodic neighbors along phi
    int j_plus  = (j + 1 < N_phi - 1) ? j + 1 : 1;
    int j_minus = (j - 1 >= 1)        ? j - 1 : N_phi - 2;

    double P_old = P[idx];

    double P_new = (A_arr[idx] * P[i * N_phi + j_plus]
                  + B_arr[idx] * P[i * N_phi + j_minus]
                  + C_arr[idx] * P[(i + 1) * N_phi + j]
                  + D_arr[idx] * P[(i - 1) * N_phi + j]
                  - F_arr[idx]) / E_arr[idx];

    // cavitation condition
    if (P_new < 0.0) P_new = 0.0;

    // SOR relaxation
    P[idx] = P_old + omega_sor * (P_new - P_old);
}
"""

# ---------------------------------------------------------------------------
# CUDA kernel: boundary conditions
# ---------------------------------------------------------------------------
_APPLY_BC_KERNEL_CODE = r"""
extern "C" __global__ void apply_bc(
    double* __restrict__ P,
    const int N_Z,
    const int N_phi
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // periodic BC along phi: for each row i
    if (tid < N_Z) {
        int i = tid;
        P[i * N_phi + 0]           = P[i * N_phi + (N_phi - 2)];
        P[i * N_phi + (N_phi - 1)] = P[i * N_phi + 1];
    }

    // Dirichlet BC along Z: for each column j
    if (tid < N_phi) {
        int j = tid;
        P[0 * N_phi + j]           = 0.0;
        P[(N_Z - 1) * N_phi + j]   = 0.0;
    }
}
"""

# ---------------------------------------------------------------------------
# Compiled kernels (lazy init)
# ---------------------------------------------------------------------------
_rb_sor_kernel = None
_apply_bc_kernel = None


def get_rb_sor_kernel():
    """Returns compiled CUDA Red-Black SOR kernel (cached)."""
    global _rb_sor_kernel
    if _rb_sor_kernel is None:
        _rb_sor_kernel = cp.RawKernel(_RB_SOR_KERNEL_CODE, "rb_sor_step")
    return _rb_sor_kernel


def get_apply_bc_kernel():
    """Returns compiled CUDA boundary conditions kernel (cached)."""
    global _apply_bc_kernel
    if _apply_bc_kernel is None:
        _apply_bc_kernel = cp.RawKernel(_APPLY_BC_KERNEL_CODE, "apply_bc")
    return _apply_bc_kernel
