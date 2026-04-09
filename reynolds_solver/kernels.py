"""
CUDA C kernels for Red-Black SOR solution of the Reynolds equation.

Contains:
  - rb_sor_step:      one half-step of Red-Black SOR (Half-Sommerfeld)
  - rb_sor_jfo_step:  one half-step of Red-Black SOR (JFO active-set)
  - update_theta_sweep: upwind theta line-sweep along phi (one thread per Z-row)
  - apply_bc:          boundary conditions (periodic in phi, Dirichlet in Z)

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
# CUDA kernel: one Red-Black SOR half-step for JFO (active-set only)
# ---------------------------------------------------------------------------
_RB_SOR_JFO_KERNEL_CODE = r"""
extern "C" __global__ void rb_sor_jfo_step(
    double* __restrict__ P,
    const double* __restrict__ A_arr,
    const double* __restrict__ B_arr,
    const double* __restrict__ C_arr,
    const double* __restrict__ D_arr,
    const double* __restrict__ E_arr,
    const double* __restrict__ F_arr,
    const int* __restrict__ zone_mask,
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

    if (zone_mask[idx] == 1) {
        // Active zone: standard SOR update with F_theta as RHS
        int j_plus  = (j + 1 < N_phi - 1) ? j + 1 : 1;
        int j_minus = (j - 1 >= 1)        ? j - 1 : N_phi - 2;

        double P_old = P[idx];

        double P_new = (A_arr[idx] * P[i * N_phi + j_plus]
                      + B_arr[idx] * P[i * N_phi + j_minus]
                      + C_arr[idx] * P[(i + 1) * N_phi + j]
                      + D_arr[idx] * P[(i - 1) * N_phi + j]
                      - F_arr[idx]) / E_arr[idx];

        // Clamp negative pressure in active zone (same as HS kernel)
        if (P_new < 0.0) P_new = 0.0;

        // SOR relaxation
        P[idx] = P_old + omega_sor * (P_new - P_old);
    } else {
        // Cavitation zone: P = 0
        P[idx] = 0.0;
    }
}
"""

# ---------------------------------------------------------------------------
# CUDA kernel: rupture-anchored theta march along phi (one thread per Z-row)
# Uses face-based H values for mass-conservative transport.
#
# Algorithm per row:
#   1. Set full-film nodes (zone_mask=1) to theta=1; detect if any cavitation
#   2. Find rupture point: first transition full-film -> cavitation
#   3. March theta through cavitation zone: theta_j = H_face_m[j]/H_face_p[j] * theta_{j-1}
#   4. Stop at reformation (first zone_mask=1 encountered)
# ---------------------------------------------------------------------------
_UPDATE_THETA_SWEEP_KERNEL_CODE = r"""
extern "C" __global__ void update_theta_sweep(
    double* __restrict__ theta,
    const double* __restrict__ H_face_p,
    const double* __restrict__ H_face_m,
    const int* __restrict__ zone_mask,
    const int N_Z,
    const int N_phi
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_Z) return;

    // Boundary rows: theta = 1 everywhere
    if (i == 0 || i == N_Z - 1) {
        for (int j = 0; j < N_phi; j++)
            theta[i * N_phi + j] = 1.0;
        return;
    }

    int n_phys = N_phi - 2;  // physical columns: j = 1 .. N_phi-2

    // Pass 1: set full-film nodes to theta=1, detect any cavitation
    int any_cav = 0;
    for (int j = 1; j <= N_phi - 2; j++) {
        int idx = i * N_phi + j;
        if (zone_mask[idx] == 1) {
            theta[idx] = 1.0;
        } else {
            any_cav = 1;
        }
    }

    if (!any_cav) {
        // Entire row is full-film
        theta[i * N_phi + 0]           = theta[i * N_phi + (N_phi - 2)];
        theta[i * N_phi + (N_phi - 1)] = theta[i * N_phi + 1];
        return;
    }

    // Find rupture point: first transition full-film(1) -> cavitation(0)
    int rupture_j = -1;
    for (int jj = 0; jj < n_phys; jj++) {
        int j = 1 + jj;
        int j_next = (j + 1 <= N_phi - 2) ? (j + 1) : 1;
        int idx_j    = i * N_phi + j;
        int idx_next = i * N_phi + j_next;
        if (zone_mask[idx_j] == 1 && zone_mask[idx_next] == 0) {
            rupture_j = j;
            break;
        }
    }

    if (rupture_j < 0) {
        // Entire row cavitated (no full-film anchor)
        // March one full revolution from column 1 with theta_prev = 1.0
        double theta_prev = 1.0;
        int j = 1;
        for (int step = 0; step < n_phys; step++) {
            int idx = i * N_phi + j;
            if (zone_mask[idx] == 1) {
                theta_prev = 1.0;
            } else {
                double Hfp = H_face_p[idx];
                double Hfm = H_face_m[idx];
                double val = 0.0;
                if (Hfp > 1e-30) {
                    val = Hfm * theta_prev / Hfp;
                    if (val < 0.0) val = 0.0;
                    if (val > 1.0) val = 1.0;
                }
                theta[idx] = val;
                theta_prev = val;
            }
            j++;
            if (j > N_phi - 2) j = 1;
        }
    } else {
        // March from rupture point through cavitation zone
        double theta_prev = 1.0;  // entering cavitation from full-film
        int j = rupture_j + 1;
        if (j > N_phi - 2) j = 1;  // periodic wrap

        for (int step = 0; step < n_phys; step++) {
            int idx = i * N_phi + j;
            if (zone_mask[idx] == 1) {
                break;  // reached reformation point
            }

            double Hfp = H_face_p[idx];
            double Hfm = H_face_m[idx];
            double val = 0.0;
            if (Hfp > 1e-30) {
                val = Hfm * theta_prev / Hfp;
                if (val < 0.0) val = 0.0;
                if (val > 1.0) val = 1.0;
            }
            theta[idx] = val;
            theta_prev = val;

            j++;
            if (j > N_phi - 2) j = 1;
        }
    }

    // Sync ghost columns (periodic)
    theta[i * N_phi + 0]           = theta[i * N_phi + (N_phi - 2)];
    theta[i * N_phi + (N_phi - 1)] = theta[i * N_phi + 1];
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
_rb_sor_jfo_kernel = None
_update_theta_sweep_kernel = None
_apply_bc_kernel = None


def get_rb_sor_kernel():
    """Returns compiled CUDA Red-Black SOR kernel (cached)."""
    global _rb_sor_kernel
    if _rb_sor_kernel is None:
        _rb_sor_kernel = cp.RawKernel(_RB_SOR_KERNEL_CODE, "rb_sor_step")
    return _rb_sor_kernel


def get_rb_sor_jfo_kernel():
    """Returns compiled CUDA Red-Black SOR kernel for JFO (cached)."""
    global _rb_sor_jfo_kernel
    if _rb_sor_jfo_kernel is None:
        _rb_sor_jfo_kernel = cp.RawKernel(_RB_SOR_JFO_KERNEL_CODE, "rb_sor_jfo_step")
    return _rb_sor_jfo_kernel


def get_update_theta_sweep_kernel():
    """Returns compiled CUDA theta line-sweep kernel (cached)."""
    global _update_theta_sweep_kernel
    if _update_theta_sweep_kernel is None:
        _update_theta_sweep_kernel = cp.RawKernel(
            _UPDATE_THETA_SWEEP_KERNEL_CODE, "update_theta_sweep"
        )
    return _update_theta_sweep_kernel


def get_apply_bc_kernel():
    """Returns compiled CUDA boundary conditions kernel (cached)."""
    global _apply_bc_kernel
    if _apply_bc_kernel is None:
        _apply_bc_kernel = cp.RawKernel(_APPLY_BC_KERNEL_CODE, "apply_bc")
    return _apply_bc_kernel
