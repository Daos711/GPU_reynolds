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
    const int periodic_phi
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

    // All stencil values from OLD buffers (Jacobi / frozen iterate).
    double P_old_ij = P_old[idx];
    double th_old_ij = theta_old[idx];
    double Pjp = P_old[i * N_phi + jp];
    double Pjm = P_old[i * N_phi + jm];
    double Pip = P_old[(i + 1) * N_phi + j];
    double Pim = P_old[(i - 1) * N_phi + j];
    double th_up = theta_old[i * N_phi + jm];

    double h_ij = H_curr[idx];
    double h_jm = H_curr[i * N_phi + jm];
    double cp_ij = C_prev[idx];

    double A_l = A_arr[idx];
    double B_l = B_arr[idx];
    double C_l = C_coeff[idx];
    double D_l = D_coeff[idx];
    double E_l = E_coeff[idx];

    // Temporal coefficient β = 2 · dx1^2 / dt
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
            // th_cur stays as th_old_ij — fall through to Branch 2
        }
    }

    // ---- Branch 2 : θ update if cavitation or partial film ----
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
# Boundary conditions kernel for the unsteady Ausas solver
# ---------------------------------------------------------------------------
_UNSTEADY_AUSAS_BC_KERNEL_CODE = r"""
extern "C" __global__ void unsteady_ausas_bc(
    double* __restrict__ P,
    double* __restrict__ theta,
    const int N_Z,
    const int N_phi,
    const int periodic_phi,
    const double p_z0,
    const double p_zL,
    const double theta_z0,
    const double theta_zL
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Periodic phi: ghost columns 0 and N_phi-1.
    if (periodic_phi && tid < N_Z) {
        int i = tid;
        P[i * N_phi + 0]            = P[i * N_phi + (N_phi - 2)];
        P[i * N_phi + (N_phi - 1)]  = P[i * N_phi + 1];
        theta[i * N_phi + 0]            = theta[i * N_phi + (N_phi - 2)];
        theta[i * N_phi + (N_phi - 1)]  = theta[i * N_phi + 1];
    }

    // Dirichlet Z-ends: prescribed P and θ.
    if (tid < N_phi) {
        int j = tid;
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
_unsteady_ausas_bc_kernel = None


def get_unsteady_ausas_kernel():
    """Returns compiled CUDA unsteady Ausas Jacobi kernel (cached)."""
    global _unsteady_ausas_kernel
    if _unsteady_ausas_kernel is None:
        _unsteady_ausas_kernel = cp.RawKernel(
            _UNSTEADY_AUSAS_KERNEL_CODE, "unsteady_ausas_step"
        )
    return _unsteady_ausas_kernel


def get_unsteady_ausas_bc_kernel():
    """Returns compiled CUDA unsteady Ausas BC kernel (cached)."""
    global _unsteady_ausas_bc_kernel
    if _unsteady_ausas_bc_kernel is None:
        _unsteady_ausas_bc_kernel = cp.RawKernel(
            _UNSTEADY_AUSAS_BC_KERNEL_CODE, "unsteady_ausas_bc"
        )
    return _unsteady_ausas_bc_kernel
