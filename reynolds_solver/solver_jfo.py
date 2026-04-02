"""
GPU solver for the Reynolds equation with JFO cavitation model.

Implements the Jakobsson-Floberg-Olsson mass-conserving cavitation model
via an active-set iteration with Red-Black SOR inner solver.

Method:
  - Outer loop: active-set iteration (zone mask updates, theta transport)
  - Inner loop: Red-Black SOR on active zone only (rb_sor_jfo_step kernel)
  - Theta transport: line-sweep along +phi (one thread per Z-row, in-place)
  - RHS rebuilt every outer iteration: F_theta = d(H*theta)/dphi
"""

import numpy as np
import cupy as cp

from reynolds_solver.kernels import (
    get_rb_sor_jfo_kernel,
    get_update_theta_sweep_kernel,
    get_apply_bc_kernel,
)
from reynolds_solver.utils import precompute_coefficients_gpu


class SolverJFO:
    """
    JFO cavitation solver with GPU buffer caching.

    Usage
    -----
    solver = SolverJFO((N_Z, N_phi))
    P, theta, residual, n_outer, n_inner_total = solver.solve(
        H_gpu, A, B, C, D, E, d_phi,
    )
    """

    def __init__(self, shape):
        N_Z, N_phi = shape
        self.N_Z = N_Z
        self.N_phi = N_phi

        # Working GPU buffers
        self._P = cp.zeros(shape, dtype=cp.float64)
        self._P_old = cp.empty(shape, dtype=cp.float64)
        self._theta = cp.ones(shape, dtype=cp.float64)
        self._theta_prev = cp.empty(shape, dtype=cp.float64)
        self._mask = cp.ones(shape, dtype=cp.int32)
        self._mask_old = cp.empty(shape, dtype=cp.int32)
        self._F_theta = cp.zeros(shape, dtype=cp.float64)

        # Stencil coefficient buffers
        self._A = cp.empty(shape, dtype=cp.float64)
        self._B = cp.empty(shape, dtype=cp.float64)
        self._C = cp.empty(shape, dtype=cp.float64)
        self._D = cp.empty(shape, dtype=cp.float64)
        self._E = cp.empty(shape, dtype=cp.float64)

        # CUDA launch config for interior points (SOR kernel)
        self._block = (32, 8, 1)
        self._grid = (
            (N_phi - 2 + self._block[0] - 1) // self._block[0],
            (N_Z - 2 + self._block[1] - 1) // self._block[1],
            1,
        )
        # CUDA launch config for theta line-sweep (one thread per Z-row)
        self._sweep_block = (256, 1, 1)
        self._sweep_grid = ((N_Z + 255) // 256, 1, 1)
        # For boundary conditions
        max_dim = max(N_Z, N_phi)
        self._bc_block = (256, 1, 1)
        self._bc_grid = ((max_dim + 255) // 256, 1, 1)

    def _build_F_theta(self, H_gpu, d_phi):
        """
        Rebuild RHS: F_theta = d(H*theta)/dphi using the same face scheme.
        """
        H_theta = H_gpu * self._theta

        HT_i_plus_half = 0.5 * (H_theta[:, :-1] + H_theta[:, 1:])
        HT_i_minus_half = cp.empty_like(HT_i_plus_half)
        HT_i_minus_half[:, 1:] = HT_i_plus_half[:, :-1]
        HT_i_minus_half[:, 0] = HT_i_plus_half[:, -1]

        F_half = d_phi * (HT_i_plus_half - HT_i_minus_half)

        self._F_theta[:] = 0.0
        self._F_theta[:, :-1] = F_half
        self._F_theta[:, -1] = F_half[:, 0]

    def _run_jfo_sor_iteration(self, sor_kernel, bc_kernel, omega):
        """Run one Red-Black SOR iteration on active zone only."""
        N_Z, N_phi = self.N_Z, self.N_phi
        # Red pass
        sor_kernel(
            self._grid, self._block,
            (
                self._P, self._A, self._B, self._C, self._D,
                self._E, self._F_theta, self._mask,
                np.int32(N_Z), np.int32(N_phi),
                np.float64(omega), np.int32(0),
            ),
        )
        # Black pass
        sor_kernel(
            self._grid, self._block,
            (
                self._P, self._A, self._B, self._C, self._D,
                self._E, self._F_theta, self._mask,
                np.int32(N_Z), np.int32(N_phi),
                np.float64(omega), np.int32(1),
            ),
        )
        # Boundary conditions
        bc_kernel(
            self._bc_grid, self._bc_block,
            (self._P, np.int32(N_Z), np.int32(N_phi)),
        )

    def _sync_periodic(self):
        """Sync ghost columns for theta and mask (periodic in phi)."""
        N_phi = self.N_phi
        self._theta[:, 0] = self._theta[:, N_phi - 2]
        self._theta[:, N_phi - 1] = self._theta[:, 1]
        self._mask[:, 0] = self._mask[:, N_phi - 2]
        self._mask[:, N_phi - 1] = self._mask[:, 1]

    def _run_theta_sweep(self, sweep_kernel, H_gpu, direction=0):
        """Run theta line-sweep: one thread per Z-row, sequential along phi."""
        N_Z, N_phi = self.N_Z, self.N_phi
        sweep_kernel(
            self._sweep_grid, self._sweep_block,
            (
                self._theta, H_gpu, self._mask,
                np.int32(N_Z), np.int32(N_phi),
                np.int32(direction),
            ),
        )

    def _update_zone_mask(self, H_gpu, p_off, p_on):
        """
        Update zone_mask based on hysteresis thresholds.

        Active -> cavitation: P <= p_off
        Cavitation -> active: P_trial > p_on (computed via local stencil)
        """
        N_Z, N_phi = self.N_Z, self.N_phi

        # Active nodes going to cavitation: P <= p_off
        go_cavitation = (self._mask == 1) & (self._P <= p_off)
        self._mask[go_cavitation] = 0
        self._P[go_cavitation] = 0.0

        # Cavitation nodes potentially returning to active: compute P_trial
        cav_mask = (self._mask == 0)
        interior = cp.zeros((N_Z, N_phi), dtype=cp.bool_)
        interior[1:-1, 1:-1] = True
        candidates = cav_mask & interior

        if cp.any(candidates):
            P_jp1 = cp.roll(self._P, -1, axis=1)
            P_jm1 = cp.roll(self._P, 1, axis=1)
            P_ip1 = cp.roll(self._P, -1, axis=0)
            P_im1 = cp.roll(self._P, 1, axis=0)

            P_trial = (
                self._A * P_jp1 + self._B * P_jm1 +
                self._C * P_ip1 + self._D * P_im1 -
                self._F_theta
            ) / (self._E + 1e-30)

            reactivate = candidates & (P_trial > p_on)
            self._mask[reactivate] = 1
            self._theta[reactivate] = 1.0

    def solve(
        self,
        H_gpu,
        A, B, C, D, E,
        d_phi,
        omega=1.5,
        tol_P=1e-5,
        tol_theta=1e-5,
        tol_inner=None,
        max_outer=500,
        max_inner=500,
        p_off=0.0,
        p_on=1e-6,
        P_init=None,
        theta_init=None,
        mask_init=None,
        verbose=False,
        sweep_direction=0,
        flip_F_sign=False,
        omega_theta=1.0,
    ):
        """
        Solve Reynolds equation with JFO cavitation.

        Returns
        -------
        P : numpy.ndarray, float64, (N_Z, N_phi)
        theta : numpy.ndarray, float64, (N_Z, N_phi)
        residual : float
            max(||delta_P||_inf, ||delta_theta||_inf) at last outer iteration.
        n_outer : int
        n_inner_total : int
        """
        if tol_inner is None:
            tol_inner = tol_P

        if p_on <= p_off:
            raise ValueError(
                f"p_on ({p_on}) must be strictly greater than p_off ({p_off})"
            )

        N_Z, N_phi = self.N_Z, self.N_phi

        # Load stencil coefficients
        self._A[:] = A
        self._B[:] = B
        self._C[:] = C
        self._D[:] = D
        self._E[:] = E

        # Initialize P
        if P_init is not None:
            P_arr = cp.asarray(P_init, dtype=cp.float64)
            if P_arr.shape != (N_Z, N_phi):
                raise ValueError(
                    f"P_init shape {P_arr.shape} != expected {(N_Z, N_phi)}"
                )
            cp.maximum(P_arr, 0.0, out=P_arr)
            self._P[:] = P_arr
        else:
            self._P[:] = 0.0

        # Initialize theta
        if theta_init is not None:
            t_arr = cp.asarray(theta_init, dtype=cp.float64)
            if t_arr.shape != (N_Z, N_phi):
                raise ValueError(
                    f"theta_init shape {t_arr.shape} != expected {(N_Z, N_phi)}"
                )
            if float(cp.min(t_arr)) < 0.0 or float(cp.max(t_arr)) > 1.0:
                raise ValueError("theta_init values must be in [0, 1]")
            self._theta[:] = t_arr
        else:
            self._theta[:] = 1.0

        # Initialize mask
        if mask_init is not None:
            m_arr = cp.asarray(mask_init, dtype=cp.int32)
            if m_arr.shape != (N_Z, N_phi):
                raise ValueError(
                    f"mask_init shape {m_arr.shape} != expected {(N_Z, N_phi)}"
                )
            unique_vals = cp.unique(m_arr)
            if not cp.all((unique_vals == 0) | (unique_vals == 1)):
                raise ValueError("mask_init values must be 0 or 1")
            self._mask[:] = m_arr
        elif P_init is not None:
            self._mask[:] = (self._P > 0.0).astype(cp.int32)
        else:
            self._mask[:] = 1

        # Get compiled kernels
        sor_kernel = get_rb_sor_jfo_kernel()
        sweep_kernel = get_update_theta_sweep_kernel()
        bc_kernel = get_apply_bc_kernel()

        n_inner_total = 0
        residual_P = 1.0
        residual_theta = 1.0

        # Sync ghost columns and build initial F_theta
        self._sync_periodic()
        self._build_F_theta(H_gpu, d_phi)
        if flip_F_sign:
            self._F_theta *= -1.0

        for outer in range(max_outer):
            # Save state for convergence check
            self._mask_old[:] = self._mask
            self._theta_prev[:] = self._theta
            self._P_old[:] = self._P

            # (a) Inner SOR: solve P at fixed mask/theta
            #     F_theta was built at end of previous iteration (or initial)
            inner_iters = 0
            dP_inner_last = 0.0
            dP_inner_best = float('inf')
            for inner in range(max_inner):
                P_before = self._P.copy()
                self._run_jfo_sor_iteration(sor_kernel, bc_kernel, omega)
                n_inner_total += 1
                inner_iters += 1

                delta_P_inner = float(cp.max(cp.abs(self._P - P_before)))
                dP_inner_last = delta_P_inner
                if delta_P_inner < dP_inner_best:
                    dP_inner_best = delta_P_inner
                if delta_P_inner < tol_inner:
                    break
            hit_max_inner = (inner_iters == max_inner)

            # (b) Update zone mask with hysteresis (uses current P and F_theta)
            self._update_zone_mask(H_gpu, p_off, p_on)

            # (c) Strict projection: enforce invariants immediately
            cav = (self._mask == 0)
            self._P[cav] = 0.0
            act = (self._mask == 1)
            self._theta[act] = 1.0
            cp.maximum(self._P, 0.0, out=self._P)
            self._sync_periodic()

            # (d) Theta sweep in cavitation zone (using NEW mask)
            #     Save pre-sweep theta for under-relaxation
            if omega_theta < 1.0:
                self._theta_prev[:] = self._theta
            self._run_theta_sweep(sweep_kernel, H_gpu, direction=sweep_direction)
            if omega_theta < 1.0:
                # Under-relax: theta = omega*theta_sweep + (1-omega)*theta_prev
                self._theta[:] = omega_theta * self._theta + (1.0 - omega_theta) * self._theta_prev
                # Re-enforce active zone invariant after blending
                act = (self._mask == 1)
                self._theta[act] = 1.0

            # (e) Periodic sync + rebuild F_theta on consistent state
            self._sync_periodic()
            self._build_F_theta(H_gpu, d_phi)
            if flip_F_sign:
                self._F_theta *= -1.0

            # Apply BCs
            bc_kernel(
                self._bc_grid, self._bc_block,
                (self._P, np.int32(N_Z), np.int32(N_phi)),
            )

            # (f) Check outer convergence
            diff_mask = self._mask != self._mask_old
            mask_changed_count = int(cp.sum(diff_mask))
            n_0to1 = int(cp.sum((self._mask_old == 0) & (self._mask == 1)))
            n_1to0 = int(cp.sum((self._mask_old == 1) & (self._mask == 0)))
            residual_P = float(cp.max(cp.abs(self._P - self._P_old)))
            residual_theta = float(cp.max(cp.abs(self._theta - self._theta_prev)))
            cav_frac = float(cp.mean((self._mask == 0).astype(cp.float64)))

            if verbose and (outer % 20 == 0 or outer < 5):
                N_phi = self.N_phi
                seam_theta = max(
                    float(cp.max(cp.abs(self._theta[:, 0] - self._theta[:, N_phi - 2]))),
                    float(cp.max(cp.abs(self._theta[:, N_phi - 1] - self._theta[:, 1]))),
                )
                seam_mask = int(cp.sum(self._mask[:, 0] != self._mask[:, N_phi - 2])) + \
                            int(cp.sum(self._mask[:, N_phi - 1] != self._mask[:, 1]))
                hit_flag = "!" if hit_max_inner else " "
                print(
                    f"    outer={outer:>4d}: dP={residual_P:.2e}, "
                    f"dtheta={residual_theta:.2e}, "
                    f"mask={mask_changed_count} "
                    f"(0\u21921={n_0to1}, 1\u21920={n_1to0}), "
                    f"cav={cav_frac:.3f}, "
                    f"inner={inner_iters}{hit_flag} "
                    f"dPi_last={dP_inner_last:.2e} best={dP_inner_best:.2e}, "
                    f"seam={seam_theta:.1e}/{seam_mask}"
                )

            if (mask_changed_count == 0) and residual_P < tol_P and residual_theta < tol_theta:
                if verbose:
                    print(f"    Converged at outer={outer}")
                break

        residual = max(residual_P, residual_theta)
        n_outer = outer + 1

        P_cpu = cp.asnumpy(self._P)
        theta_cpu = cp.asnumpy(self._theta)
        return P_cpu, theta_cpu, float(residual), n_outer, n_inner_total


# ---------------------------------------------------------------------------
# Global cache of JFO solver instances (by grid size)
# ---------------------------------------------------------------------------
_jfo_solver_cache: dict[tuple[int, int], SolverJFO] = {}


def _get_jfo_solver(N_Z: int, N_phi: int) -> SolverJFO:
    """Returns cached JFO solver instance for given grid size."""
    key = (N_Z, N_phi)
    if key not in _jfo_solver_cache:
        _jfo_solver_cache[key] = SolverJFO((N_Z, N_phi))
    return _jfo_solver_cache[key]


def solve_reynolds_gpu_jfo(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    closure=None,
    omega: float = 1.5,
    tol_P: float = 1e-5,
    tol_theta: float = 1e-5,
    tol_inner: float = None,
    max_outer: int = 500,
    max_inner: int = 100,
    p_off: float = 0.0,
    p_on: float = 1e-6,
    P_init=None,
    theta_init=None,
    mask_init=None,
    verbose: bool = False,
    sweep_direction: int = 0,
    flip_F_sign: bool = False,
    omega_theta: float = 1.0,
) -> tuple:
    """
    Solve Reynolds equation with JFO cavitation on GPU.

    Returns
    -------
    P : numpy.ndarray, float64, (N_Z, N_phi)
    theta : numpy.ndarray, float64, (N_Z, N_phi)
    residual : float
    n_outer : int
    n_inner_total : int
    """
    N_Z, N_phi = H.shape
    solver = _get_jfo_solver(N_Z, N_phi)

    H_gpu = cp.asarray(H, dtype=cp.float64)
    A, B, C, D, E, _F = precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L, closure=closure)

    return solver.solve(
        H_gpu, A, B, C, D, E, d_phi,
        omega=omega,
        tol_P=tol_P,
        tol_theta=tol_theta,
        tol_inner=tol_inner,
        max_outer=max_outer,
        max_inner=max_inner,
        p_off=p_off,
        p_on=p_on,
        P_init=P_init,
        theta_init=theta_init,
        mask_init=mask_init,
        verbose=verbose,
        sweep_direction=sweep_direction,
        flip_F_sign=flip_F_sign,
        omega_theta=omega_theta,
    )
