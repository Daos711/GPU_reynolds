"""
GPU solver for the Reynolds equation with Ausas-style mass-conserving JFO.

Reference: Ausas, Jai, Buscaglia (2009),
"A Mass-Conserving Algorithm for Dynamical Lubrication Problems With
Cavitation", ASME J. Tribology, 131(3), 031702.

This solver uses a single relaxation loop where P and theta are updated
together at each node with complementarity check, replacing the operator-
splitting approach of solver_jfo.py. This avoids mask chattering and
provides more stable convergence.

Method:
  - Single Red-Black relaxation loop (no outer/inner separation)
  - Per-node Ausas update: try P, then theta, with complementarity
  - Convergence: ||dP||_2 + ||dtheta||_2 < tol
"""

import numpy as np
import cupy as cp

from reynolds_solver.kernels_ausas import (
    get_ausas_rb_kernel,
    get_apply_bc_ausas_kernel,
)
from reynolds_solver.kernels import (
    get_rb_sor_kernel,
    get_apply_bc_kernel,
)
from reynolds_solver.utils import precompute_coefficients_gpu


class SolverJFOAusas:
    """
    Ausas-style mass-conserving JFO solver with GPU buffer caching.

    Usage
    -----
    solver = SolverJFOAusas((N_Z, N_phi))
    P, theta, residual, n_iter = solver.solve(
        H_gpu, d_phi, d_Z, R, L,
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
        self._theta_old = cp.empty(shape, dtype=cp.float64)

        # Stencil coefficient buffers
        self._A = cp.empty(shape, dtype=cp.float64)
        self._B = cp.empty(shape, dtype=cp.float64)
        self._C = cp.empty(shape, dtype=cp.float64)
        self._D = cp.empty(shape, dtype=cp.float64)
        self._E = cp.empty(shape, dtype=cp.float64)

        # Face H buffers
        self._H_face_p = cp.empty(shape, dtype=cp.float64)
        self._H_face_m = cp.empty(shape, dtype=cp.float64)

        # CUDA launch config for interior points
        self._block = (32, 8, 1)
        self._grid = (
            (N_phi - 2 + self._block[0] - 1) // self._block[0],
            (N_Z - 2 + self._block[1] - 1) // self._block[1],
            1,
        )
        # For boundary conditions
        max_dim = max(N_Z, N_phi)
        self._bc_block = (256, 1, 1)
        self._bc_grid = ((max_dim + 255) // 256, 1, 1)

    def _run_color_pass(self, kernel, bc_kernel, d_phi, omega_p, omega_theta, color):
        """Run one color (red/black) pass + BC sync."""
        N_Z, N_phi = self.N_Z, self.N_phi
        kernel(
            self._grid, self._block,
            (
                self._P, self._theta,
                self._A, self._B, self._C, self._D, self._E,
                self._H_face_p, self._H_face_m,
                np.int32(N_Z), np.int32(N_phi),
                np.float64(d_phi),
                np.float64(omega_p), np.float64(omega_theta),
                np.int32(color),
            ),
        )
        bc_kernel(
            self._bc_grid, self._bc_block,
            (self._P, self._theta, np.int32(N_Z), np.int32(N_phi)),
        )

    def solve(
        self,
        H_gpu, d_phi, d_Z, R, L,
        omega_p=1.0,
        omega_theta=1.0,
        omega_hs=1.7,
        tol=1e-6,
        max_iter=50000,
        check_every=50,
        hs_warmup_iter=2000,
        hs_warmup_tol=1e-7,
        P_init=None,
        theta_init=None,
        verbose=False,
    ):
        """
        Solve Reynolds equation with Ausas-style JFO cavitation.

        Performs an HS-like warmup (theta=1 fixed) before Ausas relaxation
        to establish a good pressure profile.

        Parameters
        ----------
        H_gpu : cupy.ndarray, (N_Z, N_phi), float64
        d_phi, d_Z : float
        R, L : float
        omega_p, omega_theta : float — Ausas relaxation factors
        omega_hs : float — SOR omega for HS warmup
        tol : float — convergence on ||dP||_2 + ||dtheta||_2
        max_iter : int — max Ausas iterations
        check_every : int — convergence check frequency
        hs_warmup_iter : int — max HS warmup iterations (0 to skip)
        hs_warmup_tol : float — HS warmup convergence tolerance
        P_init, theta_init : array-like or None — warm start
        verbose : bool

        Returns
        -------
        P : numpy.ndarray, (N_Z, N_phi), float64
        theta : numpy.ndarray, (N_Z, N_phi), float64
        residual : float
        n_iter : int (HS warmup + Ausas combined)
        """
        N_Z, N_phi = self.N_Z, self.N_phi

        # Precompute stencil coefficients and F_orig (for HS warmup)
        A, B, C, D, E, F_orig = precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)
        self._A[:] = A
        self._B[:] = B
        self._C[:] = C
        self._D[:] = D
        self._E[:] = E

        # Precompute face H values for upwind RHS
        self._H_face_p[:, :-1] = 0.5 * (H_gpu[:, :-1] + H_gpu[:, 1:])
        self._H_face_p[:, -1] = 0.5 * (H_gpu[:, -1] + H_gpu[:, 0])
        self._H_face_m[:, 1:] = self._H_face_p[:, :-1]
        self._H_face_m[:, 0] = self._H_face_p[:, -1]

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

        # Apply initial BCs
        self._P[0, :] = 0.0
        self._P[-1, :] = 0.0

        ausas_kernel = get_ausas_rb_kernel()
        bc_kernel = get_apply_bc_ausas_kernel()
        hs_sor_kernel = get_rb_sor_kernel()
        hs_bc_kernel = get_apply_bc_kernel()

        n_iter = 0

        # HS warmup: solve P with theta=1 fixed (only if no warm-start P)
        if hs_warmup_iter > 0 and P_init is None:
            F_warmup = cp.asarray(F_orig, dtype=cp.float64)
            for k in range(hs_warmup_iter):
                P_before = self._P.copy()
                # Red pass
                hs_sor_kernel(
                    self._grid, self._block,
                    (self._P, self._A, self._B, self._C, self._D,
                     self._E, F_warmup,
                     np.int32(N_Z), np.int32(N_phi),
                     np.float64(omega_hs), np.int32(0)),
                )
                # Black pass
                hs_sor_kernel(
                    self._grid, self._block,
                    (self._P, self._A, self._B, self._C, self._D,
                     self._E, F_warmup,
                     np.int32(N_Z), np.int32(N_phi),
                     np.float64(omega_hs), np.int32(1)),
                )
                hs_bc_kernel(
                    self._bc_grid, self._bc_block,
                    (self._P, np.int32(N_Z), np.int32(N_phi)),
                )
                n_iter += 1

                if k % 200 == 0 or k < 3:
                    dP = float(cp.max(cp.abs(self._P - P_before)))
                    if verbose:
                        maxP = float(cp.max(self._P))
                        print(f"  [HS warmup] iter={k:>5d}: dP={dP:.4e}, maxP={maxP:.4e}")
                    if dP < hs_warmup_tol and k > 5:
                        if verbose:
                            print(f"  [HS warmup] CONVERGED at iter={k}, dP={dP:.4e}")
                        break

        # Ausas relaxation
        residual = 1.0
        for k in range(max_iter):
            self._P_old[:] = self._P
            self._theta_old[:] = self._theta

            # Red pass
            self._run_color_pass(ausas_kernel, bc_kernel, d_phi, omega_p, omega_theta, 0)
            # Black pass
            self._run_color_pass(ausas_kernel, bc_kernel, d_phi, omega_p, omega_theta, 1)

            n_iter += 1

            if k % check_every == 0 or k < 5:
                dP = float(cp.sqrt(cp.sum((self._P - self._P_old) ** 2)))
                dth = float(cp.sqrt(cp.sum((self._theta - self._theta_old) ** 2)))
                residual = dP + dth
                cav_frac = float(cp.mean((self._theta < 1.0 - 1e-6).astype(cp.float64)))

                if verbose:
                    maxP = float(cp.max(self._P))
                    print(
                        f"  [Ausas] iter={k:>5d}: residual={residual:.4e}, "
                        f"dP={dP:.2e}, dth={dth:.2e}, cav={cav_frac:.3f}, maxP={maxP:.4e}"
                    )

                if residual < tol and k > 5:
                    if verbose:
                        print(f"  [Ausas] CONVERGED at iter={k}, residual={residual:.4e}")
                    break

        P_cpu = cp.asnumpy(self._P)
        theta_cpu = cp.asnumpy(self._theta)
        return P_cpu, theta_cpu, float(residual), n_iter


# ---------------------------------------------------------------------------
# Global cache of Ausas solver instances (by grid size)
# ---------------------------------------------------------------------------
_ausas_solver_cache: dict[tuple[int, int], SolverJFOAusas] = {}


def _get_ausas_solver(N_Z: int, N_phi: int) -> SolverJFOAusas:
    """Returns cached Ausas solver instance for given grid size."""
    key = (N_Z, N_phi)
    if key not in _ausas_solver_cache:
        _ausas_solver_cache[key] = SolverJFOAusas((N_Z, N_phi))
    return _ausas_solver_cache[key]


def solve_reynolds_gpu_jfo_ausas(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    omega_p: float = 1.0,
    omega_theta: float = 1.0,
    omega_hs: float = 1.7,
    tol: float = 1e-6,
    max_iter: int = 50000,
    check_every: int = 50,
    hs_warmup_iter: int = 2000,
    hs_warmup_tol: float = 1e-7,
    P_init=None,
    theta_init=None,
    verbose: bool = False,
) -> tuple:
    """
    Solve Reynolds equation with Ausas-style JFO cavitation on GPU.

    Returns
    -------
    P : numpy.ndarray, (N_Z, N_phi), float64
    theta : numpy.ndarray, (N_Z, N_phi), float64
    residual : float
    n_iter : int (HS warmup + Ausas combined)
    """
    N_Z, N_phi = H.shape
    solver = _get_ausas_solver(N_Z, N_phi)

    H_gpu = cp.asarray(H, dtype=cp.float64)

    return solver.solve(
        H_gpu, d_phi, d_Z, R, L,
        omega_p=omega_p,
        omega_theta=omega_theta,
        omega_hs=omega_hs,
        tol=tol,
        max_iter=max_iter,
        check_every=check_every,
        hs_warmup_iter=hs_warmup_iter,
        hs_warmup_tol=hs_warmup_tol,
        P_init=P_init,
        theta_init=theta_init,
        verbose=verbose,
    )
