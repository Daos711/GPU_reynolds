"""
CUDA kernels for Ausas-style mass-conserving JFO cavitation solver.

Reference: Ausas, Jai, Buscaglia (2009), Table 1.

Kernels:
  - ausas_rb_step: one Red-Black half-step with per-node complementarity
                   update of (P, theta) — replaces standard SOR step.
                   Uses cell-centered H for mass-content terms.
  - apply_bc_ausas: periodic phi sync for both P and theta;
                    Dirichlet P=0 at Z boundaries; theta forced to 1
                    for flooded bearing (default) or clamped to [0,1].
"""

import cupy as cp


# ---------------------------------------------------------------------------
# Ausas Red-Black step kernel
# ---------------------------------------------------------------------------
_AUSAS_RB_KERNEL_CODE = r"""
extern "C" __global__ void ausas_rb_step(
    double* __restrict__ P,
    double* __restrict__ theta,
    const double* __restrict__ A_arr,
    const double* __restrict__ B_arr,
    const double* __restrict__ C_arr,
    const double* __restrict__ D_arr,
    const double* __restrict__ E_arr,
    const double* __restrict__ H,
    const int N_Z,
    const int N_phi,
    const double d_phi,
    const double omega_p,
    const double omega_theta,
    const int color
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= N_Z - 1 || j >= N_phi - 1) return;

    // Red-Black coloring
    if ((i + j) % 2 != color) return;

    int idx = i * N_phi + j;

    // Periodic neighbors along phi
    int jp = (j + 1 < N_phi - 1) ? (j + 1) : 1;
    int jm = (j - 1 >= 1) ? (j - 1) : (N_phi - 2);

    double P_old  = P[idx];
    double th_old = theta[idx];
    double th_up  = theta[i * N_phi + jm];

    // Cell-centered gap (NOT face) for mass-content terms
    double h_ij = H[idx];
    double h_jm = H[i * N_phi + jm];

    double A_l = A_arr[idx];
    double B_l = B_arr[idx];
    double C_l = C_arr[idx];
    double D_l = D_arr[idx];
    double E_l = E_arr[idx];

    double Pjp = P[i * N_phi + jp];
    double Pjm = P[i * N_phi + jm];
    double Pip = P[(i + 1) * N_phi + j];
    double Pim = P[(i - 1) * N_phi + j];

    double P_cur  = P_old;
    double th_cur = th_old;

    // Branch 1: pressure update if currently full-film
    if (P_old > 0.0 || th_old >= 1.0 - 1e-12) {
        // Full-film: theta_{i,j} = 1 locally, cell-centered upwind RHS
        double F_full = d_phi * (h_ij - h_jm * th_up);
        double P_trial = (
            A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim - F_full
        ) / (E_l + 1e-30);
        double P_new = omega_p * P_trial + (1.0 - omega_p) * P_old;

        if (P_new >= 0.0) {
            P_cur  = P_new;
            th_cur = 1.0;
        } else {
            P_cur = 0.0;
            // th_cur stays as th_old
        }
    }

    // Branch 2: theta update if cavitation or partial
    if (P_cur <= 0.0 || th_cur < 1.0 - 1e-12) {
        // Cell-centered mass balance, solve for theta_{i,j}:
        //   d_phi*h_{i,j}*theta_{i,j} = stencil(P) - E*P_cur + d_phi*h_{i,j-1}*th_up
        double stencil = A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim - E_l * P_cur;
        double Theta_trial = (stencil + d_phi * h_jm * th_up) / (d_phi * h_ij + 1e-30);
        double th_new = omega_theta * Theta_trial + (1.0 - omega_theta) * th_cur;

        if (th_new < 1.0) {
            if (th_new < 0.0) th_new = 0.0;
            th_cur = th_new;
            P_cur  = 0.0;
        } else {
            th_cur = 1.0;
            // P_cur stays
        }
    }

    P[idx]     = P_cur;
    theta[idx] = th_cur;
}
"""


# ---------------------------------------------------------------------------
# Boundary conditions kernel for Ausas solver
# ---------------------------------------------------------------------------
_APPLY_BC_AUSAS_KERNEL_CODE = r"""
extern "C" __global__ void apply_bc_ausas(
    double* __restrict__ P,
    double* __restrict__ theta,
    const int N_Z,
    const int N_phi,
    const int flooded_ends
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Periodic phi: ghost columns 0 and N_phi-1 for both P and theta
    if (tid < N_Z) {
        int i = tid;
        P[i * N_phi + 0]            = P[i * N_phi + (N_phi - 2)];
        P[i * N_phi + (N_phi - 1)]  = P[i * N_phi + 1];
        theta[i * N_phi + 0]            = theta[i * N_phi + (N_phi - 2)];
        theta[i * N_phi + (N_phi - 1)]  = theta[i * N_phi + 1];
    }

    // Dirichlet Z: P=0 on top/bottom.
    // theta: for flooded bearing (default) force theta=1; otherwise clamp to [0,1].
    if (tid < N_phi) {
        int j = tid;
        P[0 * N_phi + j]           = 0.0;
        P[(N_Z - 1) * N_phi + j]   = 0.0;

        if (flooded_ends) {
            theta[0 * N_phi + j]           = 1.0;
            theta[(N_Z - 1) * N_phi + j]   = 1.0;
        } else {
            double th_top = theta[0 * N_phi + j];
            if (th_top < 0.0) theta[0 * N_phi + j] = 0.0;
            else if (th_top > 1.0) theta[0 * N_phi + j] = 1.0;

            double th_bot = theta[(N_Z - 1) * N_phi + j];
            if (th_bot < 0.0) theta[(N_Z - 1) * N_phi + j] = 0.0;
            else if (th_bot > 1.0) theta[(N_Z - 1) * N_phi + j] = 1.0;
        }
    }
}
"""


# ---------------------------------------------------------------------------
# Compiled kernels (lazy init)
# ---------------------------------------------------------------------------
_ausas_rb_kernel = None
_apply_bc_ausas_kernel = None


def get_ausas_rb_kernel():
    """Returns compiled CUDA Ausas Red-Black kernel (cached)."""
    global _ausas_rb_kernel
    if _ausas_rb_kernel is None:
        _ausas_rb_kernel = cp.RawKernel(_AUSAS_RB_KERNEL_CODE, "ausas_rb_step")
    return _ausas_rb_kernel


def get_apply_bc_ausas_kernel():
    """Returns compiled CUDA Ausas BC kernel (cached)."""
    global _apply_bc_ausas_kernel
    if _apply_bc_ausas_kernel is None:
        _apply_bc_ausas_kernel = cp.RawKernel(_APPLY_BC_AUSAS_KERNEL_CODE, "apply_bc_ausas")
    return _apply_bc_ausas_kernel
