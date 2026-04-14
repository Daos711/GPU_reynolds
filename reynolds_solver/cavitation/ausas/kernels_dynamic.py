"""
CUDA kernels for the UNSTEADY Ausas-style mass-conserving JFO solver.

Reference: Ausas, Jai, Buscaglia (2009),
  "A Mass-Conserving Algorithm for Dynamical Lubrication Problems With
  Cavitation", ASME J. Tribology, 131(3), 031702.

Discretization (implicit Euler in time, eq. (12) in the paper, generalized
with a slip-velocity scaling α so that α = 1 reproduces the classical
journal-bearing form):

    2·d_phi^2·(c^n - c^{n-1}) / dt + α · d_phi · (c^n_ij - c^n_{i,j-1})
        = A_ij · P^n_{i,j+1} + B_ij · P^n_{i,j-1}
        + C_ij · P^n_{i+1,j} + D_ij · P^n_{i-1,j} - E_ij · P^n_{i,j}

where c = θ · h is the cell-centred mass content and the coefficients
A, B, C, D, E are the average-of-cubes Poiseuille conductance (identical
to the stationary Ausas scheme — see cavitation/ausas/solver_cpu.py).

Per-node two-branch update (Ausas Table 1, eqs. (17)/(18)):

  Branch 1 — pressure update (if P_old > 0 OR θ_old == 1, meaning the
      node is currently in the full-film regime, θ_ij = 1 ⇒ c_ij = h_ij):
        F_time    = β · (h_ij - c_prev_ij),      β = 2·d_phi^2/dt
        F_couette = α · d_phi · (h_ij - h_jm · θ_up)
        P_trial   = (diff − F_time − F_couette) / E_ij
        P_relax   = ω_p · P_trial + (1 − ω_p) · P_old
        if P_relax ≥ 0:  (P_cur, θ_cur) = (P_relax, 1)
        else:            (P_cur, θ_cur) = (0,        θ_old)

  Branch 2 — θ update (if P_cur ≤ 0 OR θ_cur < 1):
        Θ_num   = stencil_signed + β · c_prev_ij + α · d_phi · h_jm · θ_up
        Θ_den   = (β + α · d_phi) · h_ij
        Θ_trial = Θ_num / Θ_den
        Θ_relax = ω_θ · Θ_trial + (1 − ω_θ) · θ_old
        if Θ_relax < 1:  (P_cur, θ_cur) = (0, clamp[0,1] Θ_relax)
        else:            θ_cur = 1

The kernel is strictly Jacobi (frozen-iterate): every stencil neighbour
is read from the *_old buffers and results are written only to the *_new
buffers. There is NO cav_mask, NO frozen active set, NO pinned cells —
each cell freely switches between full-film and cavitation every sweep
(two-sided complementarity).

Both axes (φ and Z) independently support periodic wrap or Dirichlet
ghost rows, selected via the `periodic_phi` / `periodic_z` integer flags
of the kernel. When periodic on an axis, the first/last index of that
axis is a ghost copy of the opposite physical seam (same convention as
the stationary Ausas solver). When Dirichlet, the BC kernel writes the
prescribed (P, θ) values into those ghost rows/columns.
"""

import cupy as cp


# ---------------------------------------------------------------------------
# Unsteady Ausas Jacobi step kernel
# ---------------------------------------------------------------------------
_UNSTEADY_AUSAS_KERNEL_CODE = r"""
extern "C" __global__ void unsteady_ausas_step(
    const double* __restrict__ P_old,
    double* __restrict__ P_new,
    const double* __restrict__ theta_old,
    double* __restrict__ theta_new,
    const double* __restrict__ H_curr,
    const double* __restrict__ C_prev,
    const double* __restrict__ A_arr,
    const double* __restrict__ B_arr,
    const double* __restrict__ C_coeff,
    const double* __restrict__ D_coeff,
    const double* __restrict__ E_coeff,
    const double dx1,
    const double dx2,
    const double dt,
    const double alpha,
    const double omega_p,
    const double omega_theta,
    const int N_Z,
    const int N_phi,
    const int periodic_phi,
    const int periodic_z
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= N_Z - 1 || j >= N_phi - 1) return;

    int idx = i * N_phi + j;

    // phi-neighbours with periodic wrap (matches solver_cpu.py)
    int jp, jm;
    if (periodic_phi) {
        jp = (j + 1 < N_phi - 1) ? (j + 1) : 1;
        jm = (j - 1 >= 1)         ? (j - 1) : (N_phi - 2);
    } else {
        jp = j + 1;
        jm = j - 1;
    }

    // Z-neighbours (with periodic wrap if requested).
    int ip, im;
    if (periodic_z) {
        ip = (i + 1 < N_Z - 1) ? (i + 1) : 1;
        im = (i - 1 >= 1)       ? (i - 1) : (N_Z - 2);
    } else {
        ip = i + 1;
        im = i - 1;
    }

    // All stencil values from OLD buffers (Jacobi / frozen iterate).
    double P_old_ij = P_old[idx];
    double th_old_ij = theta_old[idx];
    double Pjp = P_old[i * N_phi + jp];
    double Pjm = P_old[i * N_phi + jm];
    double Pip = P_old[ip * N_phi + j];
    double Pim = P_old[im * N_phi + j];
    double th_up = theta_old[i * N_phi + jm];

    double h_ij = H_curr[idx];
    double h_jm = H_curr[i * N_phi + jm];
    double cp_ij = C_prev[idx];

    double A_l = A_arr[idx];
    double B_l = B_arr[idx];
    double C_l = C_coeff[idx];
    double D_l = D_coeff[idx];
    double E_l = E_coeff[idx];

    // Temporal coefficient beta = 2 * dx1^2 / dt
    double beta = 2.0 * dx1 * dx1 / dt;
    double ad1  = alpha * dx1;

    double P_cur  = P_old_ij;
    double th_cur = th_old_ij;

    // ---- Branch 1 : pressure update if currently in full-film regime ----
    if (P_old_ij > 0.0 || th_old_ij >= 1.0 - 1e-12) {
        double F_time    = beta * (h_ij - cp_ij);
        double F_couette = ad1  * (h_ij - h_jm * th_up);
        double diff = A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim;
        double P_trial = (diff - F_time - F_couette) / (E_l + 1e-30);
        double P_relaxed = omega_p * P_trial + (1.0 - omega_p) * P_old_ij;

        if (P_relaxed >= 0.0) {
            P_cur  = P_relaxed;
            th_cur = 1.0;
        } else {
            P_cur = 0.0;
            // th_cur stays as th_old_ij -- fall through to Branch 2
        }
    }

    // ---- Branch 2 : theta update if cavitation or partial film ----
    if (P_cur <= 0.0 || th_cur < 1.0 - 1e-12) {
        double stencil_signed =
            A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim - E_l * P_cur;
        double Theta_num = stencil_signed + beta * cp_ij + ad1 * h_jm * th_up;
        double Theta_den = (beta + ad1) * h_ij + 1e-30;
        double Theta_trial = Theta_num / Theta_den;
        double th_relaxed =
            omega_theta * Theta_trial + (1.0 - omega_theta) * th_old_ij;

        if (th_relaxed < 1.0) {
            if (th_relaxed < 0.0) th_relaxed = 0.0;
            th_cur = th_relaxed;
            P_cur  = 0.0;
        } else {
            th_cur = 1.0;
            // P_cur stays
        }
    }

    P_new[idx]     = P_cur;
    theta_new[idx] = th_cur;
}
"""


# ---------------------------------------------------------------------------
# Red-Black Gauss-Seidel variant (used by the Stage-2 time loop)
# ---------------------------------------------------------------------------
#
# Same per-cell two-branch update as `unsteady_ausas_step` but performed
# IN PLACE on a single (P, theta) pair. One pass updates only cells whose
# (i + j) parity matches `color` (0 = red, 1 = black). Because each
# 5-point stencil neighbour has opposite parity, a red pass reads only
# black cells and vice-versa — so within a colour there is no
# write-after-read race.
#
# One "iteration" = red pass + BC + black pass + BC. Convergence rate
# is O(rho_J^2) which is much faster than pure Jacobi for Poisson-like
# problems; with omega_p > 1 this becomes SOR and converges in O(N)
# instead of O(N^2) iterations.
#
# The Stage-1 kernel `unsteady_ausas_step` above (Jacobi, separate
# *_old/*_new buffers) is preserved for the bit-for-bit CPU parity test.
_UNSTEADY_AUSAS_RB_KERNEL_CODE = r"""
extern "C" __global__ void unsteady_ausas_rb_step(
    double* __restrict__ P,
    double* __restrict__ theta,
    const double* __restrict__ H_curr,
    const double* __restrict__ C_prev,
    const double* __restrict__ A_arr,
    const double* __restrict__ B_arr,
    const double* __restrict__ C_coeff,
    const double* __restrict__ D_coeff,
    const double* __restrict__ E_coeff,
    const double dx1,
    const double dx2,
    const double dt,
    const double alpha,
    const double omega_p,
    const double omega_theta,
    const int N_Z,
    const int N_phi,
    const int periodic_phi,
    const int periodic_z,
    const int color
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (i >= N_Z - 1 || j >= N_phi - 1) return;
    if (((i + j) & 1) != color) return;

    int idx = i * N_phi + j;

    int jp, jm, ip, im;
    if (periodic_phi) {
        jp = (j + 1 < N_phi - 1) ? (j + 1) : 1;
        jm = (j - 1 >= 1)         ? (j - 1) : (N_phi - 2);
    } else {
        jp = j + 1;
        jm = j - 1;
    }
    if (periodic_z) {
        ip = (i + 1 < N_Z - 1) ? (i + 1) : 1;
        im = (i - 1 >= 1)       ? (i - 1) : (N_Z - 2);
    } else {
        ip = i + 1;
        im = i - 1;
    }

    double P_old_ij  = P[idx];
    double th_old_ij = theta[idx];
    double Pjp = P[i * N_phi + jp];
    double Pjm = P[i * N_phi + jm];
    double Pip = P[ip * N_phi + j];
    double Pim = P[im * N_phi + j];
    double th_up = theta[i * N_phi + jm];

    double h_ij = H_curr[idx];
    double h_jm = H_curr[i * N_phi + jm];
    double cp_ij = C_prev[idx];

    double A_l = A_arr[idx];
    double B_l = B_arr[idx];
    double C_l = C_coeff[idx];
    double D_l = D_coeff[idx];
    double E_l = E_coeff[idx];

    double beta = 2.0 * dx1 * dx1 / dt;
    double ad1  = alpha * dx1;

    double P_cur  = P_old_ij;
    double th_cur = th_old_ij;

    if (P_old_ij > 0.0 || th_old_ij >= 1.0 - 1e-12) {
        double F_time    = beta * (h_ij - cp_ij);
        double F_couette = ad1  * (h_ij - h_jm * th_up);
        double diff = A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim;
        double P_trial = (diff - F_time - F_couette) / (E_l + 1e-30);
        double P_relaxed = omega_p * P_trial + (1.0 - omega_p) * P_old_ij;

        if (P_relaxed >= 0.0) {
            P_cur  = P_relaxed;
            th_cur = 1.0;
        } else {
            P_cur = 0.0;
        }
    }

    if (P_cur <= 0.0 || th_cur < 1.0 - 1e-12) {
        double stencil_signed =
            A_l * Pjp + B_l * Pjm + C_l * Pip + D_l * Pim - E_l * P_cur;
        double Theta_num = stencil_signed + beta * cp_ij + ad1 * h_jm * th_up;
        double Theta_den = (beta + ad1) * h_ij + 1e-30;
        double Theta_trial = Theta_num / Theta_den;
        double th_relaxed =
            omega_theta * Theta_trial + (1.0 - omega_theta) * th_old_ij;

        if (th_relaxed < 1.0) {
            if (th_relaxed < 0.0) th_relaxed = 0.0;
            th_cur = th_relaxed;
            P_cur  = 0.0;
        } else {
            th_cur = 1.0;
        }
    }

    P[idx]     = P_cur;
    theta[idx] = th_cur;
}
"""


# ---------------------------------------------------------------------------
# Boundary conditions kernel for the unsteady Ausas solver
# ---------------------------------------------------------------------------
#
# Split into two separate kernels (phi-pass, z-pass) so that CUDA's
# implicit cross-kernel ordering prevents corner-cell races when BOTH
# axes have non-trivial BC (e.g., Dirichlet-phi + periodic-z in the
# squeeze benchmark). The caller must launch the phi kernel first,
# then the z kernel.
#
_UNSTEADY_AUSAS_BC_PHI_KERNEL_CODE = r"""
extern "C" __global__ void unsteady_ausas_bc_phi(
    double* __restrict__ P,
    double* __restrict__ theta,
    const int N_Z,
    const int N_phi,
    const int periodic_phi,
    const double p_phi0,
    const double p_phiL,
    const double theta_phi0,
    const double theta_phiL
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N_Z) return;

    if (periodic_phi) {
        P[i * N_phi + 0]            = P[i * N_phi + (N_phi - 2)];
        P[i * N_phi + (N_phi - 1)]  = P[i * N_phi + 1];
        theta[i * N_phi + 0]            = theta[i * N_phi + (N_phi - 2)];
        theta[i * N_phi + (N_phi - 1)]  = theta[i * N_phi + 1];
    } else {
        P[i * N_phi + 0]            = p_phi0;
        P[i * N_phi + (N_phi - 1)]  = p_phiL;
        theta[i * N_phi + 0]            = theta_phi0;
        theta[i * N_phi + (N_phi - 1)]  = theta_phiL;
    }
}
"""


_UNSTEADY_AUSAS_BC_Z_KERNEL_CODE = r"""
extern "C" __global__ void unsteady_ausas_bc_z(
    double* __restrict__ P,
    double* __restrict__ theta,
    const int N_Z,
    const int N_phi,
    const int periodic_z,
    const double p_z0,
    const double p_zL,
    const double theta_z0,
    const double theta_zL
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_phi) return;

    if (periodic_z) {
        P[0 * N_phi + j]           = P[(N_Z - 2) * N_phi + j];
        P[(N_Z - 1) * N_phi + j]   = P[1 * N_phi + j];
        theta[0 * N_phi + j]           = theta[(N_Z - 2) * N_phi + j];
        theta[(N_Z - 1) * N_phi + j]   = theta[1 * N_phi + j];
    } else {
        P[0 * N_phi + j]           = p_z0;
        P[(N_Z - 1) * N_phi + j]   = p_zL;
        theta[0 * N_phi + j]           = theta_z0;
        theta[(N_Z - 1) * N_phi + j]   = theta_zL;
    }
}
"""


# ---------------------------------------------------------------------------
# Compiled kernels (lazy init)
# ---------------------------------------------------------------------------
_unsteady_ausas_kernel = None
_unsteady_ausas_rb_kernel = None
_unsteady_ausas_bc_phi_kernel = None
_unsteady_ausas_bc_z_kernel = None


def get_unsteady_ausas_kernel():
    """Returns compiled CUDA unsteady Ausas Jacobi kernel (cached)."""
    global _unsteady_ausas_kernel
    if _unsteady_ausas_kernel is None:
        _unsteady_ausas_kernel = cp.RawKernel(
            _UNSTEADY_AUSAS_KERNEL_CODE, "unsteady_ausas_step"
        )
    return _unsteady_ausas_kernel


def get_unsteady_ausas_rb_kernel():
    """Returns compiled CUDA unsteady Ausas red-black GS/SOR kernel (cached)."""
    global _unsteady_ausas_rb_kernel
    if _unsteady_ausas_rb_kernel is None:
        _unsteady_ausas_rb_kernel = cp.RawKernel(
            _UNSTEADY_AUSAS_RB_KERNEL_CODE, "unsteady_ausas_rb_step"
        )
    return _unsteady_ausas_rb_kernel


def get_unsteady_ausas_bc_phi_kernel():
    """Returns compiled CUDA phi-only BC kernel (cached)."""
    global _unsteady_ausas_bc_phi_kernel
    if _unsteady_ausas_bc_phi_kernel is None:
        _unsteady_ausas_bc_phi_kernel = cp.RawKernel(
            _UNSTEADY_AUSAS_BC_PHI_KERNEL_CODE, "unsteady_ausas_bc_phi"
        )
    return _unsteady_ausas_bc_phi_kernel


def get_unsteady_ausas_bc_z_kernel():
    """Returns compiled CUDA z-only BC kernel (cached)."""
    global _unsteady_ausas_bc_z_kernel
    if _unsteady_ausas_bc_z_kernel is None:
        _unsteady_ausas_bc_z_kernel = cp.RawKernel(
            _UNSTEADY_AUSAS_BC_Z_KERNEL_CODE, "unsteady_ausas_bc_z"
        )
    return _unsteady_ausas_bc_z_kernel


# Back-compat helper: old call site used a single BC kernel. Now we
# launch the two split kernels in sequence (phi first, then z).
def apply_unsteady_ausas_bc(
    P, theta, N_Z, N_phi,
    periodic_phi, periodic_z,
    p_phi0, p_phiL, theta_phi0, theta_phiL,
    p_z0, p_zL, theta_z0, theta_zL,
    bc_block=(256, 1, 1),
    bc_grid_phi=None,
    bc_grid_z=None,
):
    import numpy as _np
    if bc_grid_phi is None:
        bc_grid_phi = ((N_Z + bc_block[0] - 1) // bc_block[0], 1, 1)
    if bc_grid_z is None:
        bc_grid_z = ((N_phi + bc_block[0] - 1) // bc_block[0], 1, 1)

    k_phi = get_unsteady_ausas_bc_phi_kernel()
    k_z = get_unsteady_ausas_bc_z_kernel()

    k_phi(
        bc_grid_phi, bc_block,
        (
            P, theta,
            _np.int32(N_Z), _np.int32(N_phi),
            _np.int32(1 if periodic_phi else 0),
            _np.float64(p_phi0), _np.float64(p_phiL),
            _np.float64(theta_phi0), _np.float64(theta_phiL),
        ),
    )
    k_z(
        bc_grid_z, bc_block,
        (
            P, theta,
            _np.int32(N_Z), _np.int32(N_phi),
            _np.int32(1 if periodic_z else 0),
            _np.float64(p_z0), _np.float64(p_zL),
            _np.float64(theta_z0), _np.float64(theta_zL),
        ),
    )
