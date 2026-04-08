"""
GPU solver for the Reynolds equation with JFO cavitation model.

Implements the Jakobsson-Floberg-Olsson mass-conserving cavitation model
via operator splitting with Red-Black SOR inner solver.

Method:
  - Outer loop: operator splitting (P solve, zone update, theta transport)
  - Inner loop: Red-Black SOR on ENTIRE domain with P>=0 clamp (rb_sor_step)
  - Theta transport: rupture-anchored march (one thread per Z-row, in-place)
  - RHS = F_theta = d(H*theta)/dphi, rebuilt every outer iteration with blending
  - When use_F_theta=False (diagnostic): uses F_orig = d(H)/dphi instead
"""

import numpy as np
import cupy as cp

from reynolds_solver.kernels import (
    get_rb_sor_kernel,
    get_rb_sor_jfo_kernel,
    get_update_theta_sweep_kernel,
    get_apply_bc_kernel,
)
from reynolds_solver.utils import precompute_coefficients_gpu, build_F_theta_gpu


class SolverJFO:
    """
    JFO cavitation solver with GPU buffer caching.

    Usage
    -----
    solver = SolverJFO((N_Z, N_phi))
    P, theta, residual, n_outer, n_inner_total = solver.solve(
        H_gpu, A, B, C, D, E, F_orig, d_phi,
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

        # Stencil coefficient buffers
        self._A = cp.empty(shape, dtype=cp.float64)
        self._B = cp.empty(shape, dtype=cp.float64)
        self._C = cp.empty(shape, dtype=cp.float64)
        self._D = cp.empty(shape, dtype=cp.float64)
        self._E = cp.empty(shape, dtype=cp.float64)
        self._F = cp.empty(shape, dtype=cp.float64)

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

    def _run_sor_iteration(self, sor_kernel, bc_kernel, omega):
        """Run one Red-Black SOR iteration on entire domain with P>=0 clamp."""
        N_Z, N_phi = self.N_Z, self.N_phi
        # Red pass
        sor_kernel(
            self._grid, self._block,
            (
                self._P, self._A, self._B, self._C, self._D,
                self._E, self._F,
                np.int32(N_Z), np.int32(N_phi),
                np.float64(omega), np.int32(0),
            ),
        )
        # Black pass
        sor_kernel(
            self._grid, self._block,
            (
                self._P, self._A, self._B, self._C, self._D,
                self._E, self._F,
                np.int32(N_Z), np.int32(N_phi),
                np.float64(omega), np.int32(1),
            ),
        )
        # Boundary conditions
        bc_kernel(
            self._bc_grid, self._bc_block,
            (self._P, np.int32(N_Z), np.int32(N_phi)),
        )

    def _run_jfo_sor_iteration(self, sor_kernel, bc_kernel, omega):
        """Run one Red-Black SOR iteration on active zone only (diagnostic)."""
        N_Z, N_phi = self.N_Z, self.N_phi
        # Red pass
        sor_kernel(
            self._grid, self._block,
            (
                self._P, self._A, self._B, self._C, self._D,
                self._E, self._F, self._mask,
                np.int32(N_Z), np.int32(N_phi),
                np.float64(omega), np.int32(0),
            ),
        )
        # Black pass
        sor_kernel(
            self._grid, self._block,
            (
                self._P, self._A, self._B, self._C, self._D,
                self._E, self._F, self._mask,
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

    def _run_theta_sweep(self, sweep_kernel, H_face_p, H_face_m):
        """Run rupture-anchored theta march: one thread per Z-row, face-based H."""
        N_Z, N_phi = self.N_Z, self.N_phi
        sweep_kernel(
            self._sweep_grid, self._sweep_block,
            (
                self._theta, H_face_p, H_face_m, self._mask,
                np.int32(N_Z), np.int32(N_phi),
            ),
        )

    def _update_zone_mask(self, p_off, p_on):
        """
        Update zone_mask based on hysteresis thresholds.

        Since P is solved on the ENTIRE domain (with P>=0 clamp),
        the actual P values are meaningful everywhere. This matches
        the CPU-reference logic (solver_jfo_splitting_cpu._update_zone_state):
          P > p_on  -> full-film (mask=1)
          P < p_off -> cavitation (mask=0)
          p_off <= P <= p_on -> keep previous state (hysteresis band)

        Only updates interior nodes; boundary rows stay unchanged.
        """
        N_Z, N_phi = self.N_Z, self.N_phi

        interior = cp.zeros((N_Z, N_phi), dtype=cp.bool_)
        interior[1:-1, 1:-1] = True

        to_full = interior & (self._P > p_on)
        to_cav = interior & (self._P < p_off)

        self._mask[to_full] = 1
        self._mask[to_cav] = 0

        self._theta[to_full] = 1.0

    def solve(
        self,
        H_gpu,
        A, B, C, D, E, F_orig,
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
        use_F_theta=True,
        update_mask=True,
        run_theta_sweep=True,
    ):
        """
        Solve Reynolds equation with JFO cavitation.

        Parameters
        ----------
        use_F_theta : bool
            If True (default), rebuild F_theta = d(H*theta)/dphi each outer
            iteration and use as SOR RHS. If False, use F_orig = dH/dphi
            (diagnostic mode, decouples theta from pressure).
        update_mask : bool
            If True (default), update zone mask each outer iteration.
            Set False for frozen-state diagnostics.
        run_theta_sweep : bool
            If True (default), run theta sweep each outer iteration.
            Set False for frozen-state diagnostics.

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

        N_Z, N_phi = self.N_Z, self.N_phi

        # Load stencil coefficients
        self._A[:] = A
        self._B[:] = B
        self._C[:] = C
        self._D[:] = D
        self._E[:] = E
        self._F[:] = F_orig

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
        sor_kernel = get_rb_sor_kernel()       # solve P everywhere with P>=0 clamp
        jfo_sor_kernel = get_rb_sor_jfo_kernel()  # active-set (for frozen diagnostics)
        sweep_kernel = get_update_theta_sweep_kernel()
        bc_kernel = get_apply_bc_kernel()

        # Precompute face H values for theta sweep kernel
        H_face_p = cp.empty((N_Z, N_phi), dtype=cp.float64)
        H_face_p[:, :-1] = 0.5 * (H_gpu[:, :-1] + H_gpu[:, 1:])
        H_face_p[:, -1] = 0.5 * (H_gpu[:, -1] + H_gpu[:, 0])

        H_face_m = cp.empty((N_Z, N_phi), dtype=cp.float64)
        H_face_m[:, 1:] = H_face_p[:, :-1]
        H_face_m[:, 0] = H_face_p[:, -1]

        n_inner_total = 0
        residual_P = 1.0
        residual_theta = 1.0
        W_prev = 0.0

        # Sync ghost columns before iteration
        self._sync_periodic()

        # Build initial F_theta if using theta-coupled RHS
        if use_F_theta:
            self._F[:] = build_F_theta_gpu(H_gpu, self._theta, d_phi)

        for outer in range(max_outer):
            # Save state for convergence check
            self._mask_old[:] = self._mask
            self._theta_prev[:] = self._theta
            self._P_old[:] = self._P

            # (a) Inner SOR: solve P on entire domain with P>=0 clamp
            inner_iters = 0
            dP_inner_last = 0.0
            active_kernel = sor_kernel if update_mask else jfo_sor_kernel
            run_inner = self._run_sor_iteration if update_mask else self._run_jfo_sor_iteration
            for inner in range(max_inner):
                P_before = self._P.copy()
                run_inner(active_kernel, bc_kernel, omega)
                n_inner_total += 1
                inner_iters += 1

                delta_P_inner = float(cp.max(cp.abs(self._P - P_before)))
                dP_inner_last = delta_P_inner
                if delta_P_inner < tol_inner:
                    break
            hit_max_inner = (inner_iters == max_inner)

            # (b) Update zone mask with adaptive hysteresis thresholds
            if update_mask:
                maxP = float(cp.max(self._P))
                adaptive_p_on = 1e-5 * maxP if maxP > 0 else 1e-10
                adaptive_p_off = 1e-6 * maxP if maxP > 0 else 1e-11
                self._update_zone_mask(adaptive_p_off, adaptive_p_on)

            # (c) Projection: enforce theta=1 in active zone, P>=0
            act = (self._mask == 1)
            self._theta[act] = 1.0
            cp.maximum(self._P, 0.0, out=self._P)
            self._sync_periodic()

            # (d) Theta sweep in cavitation zone
            if run_theta_sweep:
                self._run_theta_sweep(sweep_kernel, H_face_p, H_face_m)
                self._sync_periodic()

            # (e) Rebuild F_theta from current theta with blending for stability
            if use_F_theta:
                F_theta_new = build_F_theta_gpu(H_gpu, self._theta, d_phi)
                if outer == 0:
                    self._F[:] = F_theta_new
                else:
                    self._F[:] = 0.5 * F_theta_new + 0.5 * self._F

            # Apply BCs
            bc_kernel(
                self._bc_grid, self._bc_block,
                (self._P, np.int32(N_Z), np.int32(N_phi)),
            )

            # (f) Check outer convergence (dW_rel like CPU reference)
            diff_mask = self._mask != self._mask_old
            mask_changed_count = int(cp.sum(diff_mask))
            n_0to1 = int(cp.sum((self._mask_old == 0) & (self._mask == 1)))
            n_1to0 = int(cp.sum((self._mask_old == 1) & (self._mask == 0)))
            residual_P = float(cp.max(cp.abs(self._P - self._P_old)))
            residual_theta = float(cp.max(cp.abs(self._theta - self._theta_prev)))
            cav_frac = float(cp.mean((self._theta < 1.0 - 1e-6).astype(cp.float64)))

            W = float(cp.sum(self._P))
            dW_rel = abs(W - W_prev) / (abs(W) + 1e-30) if outer > 0 else 1.0
            W_prev = W

            if verbose and (outer % 20 == 0 or outer < 5):
                hit_flag = "!" if hit_max_inner else " "
                print(
                    f"    outer={outer:>4d}: dP={residual_P:.2e}, "
                    f"dtheta={residual_theta:.2e}, "
                    f"mask={mask_changed_count} "
                    f"(0\u21921={n_0to1}, 1\u21920={n_1to0}), "
                    f"cav={cav_frac:.3f}, "
                    f"inner={inner_iters}{hit_flag} "
                    f"dPi={dP_inner_last:.2e}"
                )

            # Convergence by dW_rel (like CPU reference) — mask stability
            # is NOT required since boundary nodes may oscillate indefinitely
            # with full-domain SOR + P>=0 clamp
            converged = (dW_rel < tol_P and residual_P < tol_P and outer > 5)
            if converged:
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
    max_inner: int = 500,
    p_off: float = 0.0,
    p_on: float = 1e-6,
    P_init=None,
    theta_init=None,
    mask_init=None,
    verbose: bool = False,
    sweep_direction: int = 0,
    use_F_theta: bool = True,
    update_mask: bool = True,
    run_theta_sweep: bool = True,
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
    A, B, C, D, E, F_orig = precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L, closure=closure)

    return solver.solve(
        H_gpu, A, B, C, D, E, F_orig, d_phi,
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
        use_F_theta=use_F_theta,
        update_mask=update_mask,
        run_theta_sweep=run_theta_sweep,
    )
