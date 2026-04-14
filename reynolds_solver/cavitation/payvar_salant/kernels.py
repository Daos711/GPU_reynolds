"""
CUDA kernels for Payvar-Salant Red-Black SOR.

Kernels:
  ps_rb_sor_step    — one half-step (red or black) of the unified-
      variable SOR with pinned or nonlinear dispatch. Supports a
      `groove` flag that switches the φ neighbour stencil from periodic
      wrap to a plain index step (j=0 and j=N_phi-1 become Dirichlet
      boundary nodes for the supply-groove case).
  apply_bc_ps       — periodic φ ghost columns (groove=0) or Dirichlet
      g=0 on j=0 and j=N_phi-1 (groove=1); flooded Dirichlet Z ends in
      both cases.
  hs_rb_sor_step_pv — HS warmup SOR step (clone of the common
      rb_sor_step with the same groove extension). Used by the PS
      solver for its own HS warmup when phi_bc="groove"; leaves the
      common kernels.py unchanged.
  hs_apply_bc_pv    — HS BC kernel with groove extension.

All four compiled via CuPy RawKernel on first call and cached.
"""
import cupy as cp

# ---------------------------------------------------------------------------
# CUDA kernel: one Red-Black SOR half-step for Payvar-Salant
# ---------------------------------------------------------------------------
_PS_RB_SOR_KERNEL_CODE = r"""
extern "C" __global__ void ps_rb_sor_step(
    double* __restrict__ g,
    const double* __restrict__ H,
    const double* __restrict__ A_arr,
    const double* __restrict__ B_arr,
    const double* __restrict__ C_arr,
    const double* __restrict__ D_arr,
    const double* __restrict__ E_arr,
    const int* __restrict__ cav_mask,
    const int N_Z,
    const int N_phi,
    const double d_phi,
    const double omega,
    const int color,
    const int pinned,
    const int groove
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= N_Z - 1 || j >= N_phi - 1) return;
    if ((i + j) % 2 != color) return;

    int idx = i * N_phi + j;

    /* phi neighbours: periodic (groove=0) or plain (groove=1) */
    int jp, jm;
    if (groove) {
        jp = j + 1;
        jm = j - 1;
    } else {
        jp = (j + 1 < N_phi - 1) ? j + 1 : 1;
        jm = (j - 1 >= 1)        ? j - 1 : N_phi - 2;
    }

    double g_old = g[idx];

    /* P = max(g, 0) for all neighbours */
    double g_jp = g[i * N_phi + jp];
    double g_jm_val = g[i * N_phi + jm];
    double g_ip = g[(i + 1) * N_phi + j];
    double g_im = g[(i - 1) * N_phi + j];

    double P_jp = g_jp > 0.0 ? g_jp : 0.0;
    double P_jm = g_jm_val > 0.0 ? g_jm_val : 0.0;
    double P_ip = g_ip > 0.0 ? g_ip : 0.0;
    double P_im = g_im > 0.0 ? g_im : 0.0;

    /* upstream theta from current g */
    double theta_jm = g_jm_val >= 0.0 ? 1.0 : fmax(1.0 + g_jm_val, 0.0);

    double h_ij = H[idx];
    double h_jm = H[i * N_phi + jm];

    double diff = A_arr[idx] * P_jp + B_arr[idx] * P_jm
                + C_arr[idx] * P_ip + D_arr[idx] * P_im;
    double couette = d_phi * (h_ij - h_jm * theta_jm);
    double numerator = diff - couette;

    double g_new;
    if (pinned) {
        if (cav_mask[idx]) {
            /* cavitation branch */
            g_new = numerator / (d_phi * h_ij + 1e-30);
            if (g_new > 0.0)  g_new = 0.0;
            if (g_new < -1.0) g_new = -1.0;
        } else {
            /* full-film branch */
            g_new = numerator / (E_arr[idx] + 1e-30);
            if (g_new < 0.0) g_new = 0.0;
        }
    } else {
        /* nonlinear Elrod dispatch */
        double g_ff = numerator / (E_arr[idx] + 1e-30);
        if (g_ff >= 0.0) {
            g_new = g_ff;
        } else {
            g_new = numerator / (d_phi * h_ij + 1e-30);
            if (g_new < -1.0) g_new = -1.0;
        }
    }

    double g_relax = g_old + omega * (g_new - g_old);

    /* project sign after relaxation in pinned mode */
    if (pinned) {
        if (cav_mask[idx]) {
            if (g_relax > 0.0)  g_relax = 0.0;
        } else {
            if (g_relax < 0.0)  g_relax = 0.0;
        }
    }
    if (g_relax < -1.0) g_relax = -1.0;

    g[idx] = g_relax;
}
"""

# ---------------------------------------------------------------------------
# CUDA kernel: boundary conditions for g
# ---------------------------------------------------------------------------
_APPLY_BC_PS_KERNEL_CODE = r"""
extern "C" __global__ void apply_bc_ps(
    double* __restrict__ g,
    const int N_Z,
    const int N_phi,
    const int groove
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    /* phi BC: periodic wrap (groove=0) or Dirichlet g=0 (groove=1) */
    if (tid < N_Z) {
        int i = tid;
        if (groove) {
            g[i * N_phi + 0]           = 0.0;
            g[i * N_phi + (N_phi - 1)] = 0.0;
        } else {
            g[i * N_phi + 0]           = g[i * N_phi + (N_phi - 2)];
            g[i * N_phi + (N_phi - 1)] = g[i * N_phi + 1];
        }
    }

    /* Dirichlet Z ends: g=0 (flooded, P=0, theta=1) */
    if (tid < N_phi) {
        int j = tid;
        g[0 * N_phi + j]           = 0.0;
        g[(N_Z - 1) * N_phi + j]   = 0.0;
    }
}
"""

# ---------------------------------------------------------------------------
# CUDA kernel: HS warmup SOR step with groove option (PS-local copy)
# ---------------------------------------------------------------------------
_HS_RB_SOR_PV_KERNEL_CODE = r"""
extern "C" __global__ void hs_rb_sor_step_pv(
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
    const int color,
    const int groove
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= N_Z - 1 || j >= N_phi - 1) return;
    if ((i + j) % 2 != color) return;

    int idx = i * N_phi + j;

    int jp, jm;
    if (groove) {
        jp = j + 1;
        jm = j - 1;
    } else {
        jp = (j + 1 < N_phi - 1) ? j + 1 : 1;
        jm = (j - 1 >= 1)        ? j - 1 : N_phi - 2;
    }

    double P_old = P[idx];
    double P_new = (A_arr[idx] * P[i * N_phi + jp]
                  + B_arr[idx] * P[i * N_phi + jm]
                  + C_arr[idx] * P[(i + 1) * N_phi + j]
                  + D_arr[idx] * P[(i - 1) * N_phi + j]
                  - F_arr[idx]) / E_arr[idx];

    if (P_new < 0.0) P_new = 0.0;
    P[idx] = P_old + omega_sor * (P_new - P_old);
}
"""

# ---------------------------------------------------------------------------
# CUDA kernel: HS warmup BC with groove option
# ---------------------------------------------------------------------------
_HS_APPLY_BC_PV_KERNEL_CODE = r"""
extern "C" __global__ void hs_apply_bc_pv(
    double* __restrict__ P,
    const int N_Z,
    const int N_phi,
    const int groove
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N_Z) {
        int i = tid;
        if (groove) {
            P[i * N_phi + 0]           = 0.0;
            P[i * N_phi + (N_phi - 1)] = 0.0;
        } else {
            P[i * N_phi + 0]           = P[i * N_phi + (N_phi - 2)];
            P[i * N_phi + (N_phi - 1)] = P[i * N_phi + 1];
        }
    }

    if (tid < N_phi) {
        int j = tid;
        P[0 * N_phi + j]           = 0.0;
        P[(N_Z - 1) * N_phi + j]   = 0.0;
    }
}
"""

# ---------------------------------------------------------------------------
# Compiled kernel singletons
# ---------------------------------------------------------------------------
_ps_rb_sor_kernel = None
_apply_bc_ps_kernel = None
_hs_rb_sor_pv_kernel = None
_hs_apply_bc_pv_kernel = None


def get_ps_rb_sor_kernel():
    """Returns compiled PS Red-Black SOR kernel (cached)."""
    global _ps_rb_sor_kernel
    if _ps_rb_sor_kernel is None:
        _ps_rb_sor_kernel = cp.RawKernel(
            _PS_RB_SOR_KERNEL_CODE, "ps_rb_sor_step"
        )
    return _ps_rb_sor_kernel


def get_apply_bc_ps_kernel():
    """Returns compiled PS boundary-conditions kernel (cached)."""
    global _apply_bc_ps_kernel
    if _apply_bc_ps_kernel is None:
        _apply_bc_ps_kernel = cp.RawKernel(
            _APPLY_BC_PS_KERNEL_CODE, "apply_bc_ps"
        )
    return _apply_bc_ps_kernel


def get_hs_rb_sor_pv_kernel():
    """Returns compiled PS-local HS Red-Black SOR kernel (cached)."""
    global _hs_rb_sor_pv_kernel
    if _hs_rb_sor_pv_kernel is None:
        _hs_rb_sor_pv_kernel = cp.RawKernel(
            _HS_RB_SOR_PV_KERNEL_CODE, "hs_rb_sor_step_pv"
        )
    return _hs_rb_sor_pv_kernel


def get_hs_apply_bc_pv_kernel():
    """Returns compiled PS-local HS BC kernel (cached)."""
    global _hs_apply_bc_pv_kernel
    if _hs_apply_bc_pv_kernel is None:
        _hs_apply_bc_pv_kernel = cp.RawKernel(
            _HS_APPLY_BC_PV_KERNEL_CODE, "hs_apply_bc_pv"
        )
    return _hs_apply_bc_pv_kernel
