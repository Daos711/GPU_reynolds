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


def _compute_ausas_coefficients_gpu(H_gpu, d_phi, d_Z, R, L):
    """
    Compute Ausas discretization coefficients on GPU (average-of-cubes).

    A_{i,j} = 0.5 * (h^3_{i,j}   + h^3_{i,j+1}),       phi face (+)
    B_{i,j} = 0.5 * (h^3_{i,j-1} + h^3_{i,j}  ),       phi face (-)
    C_{i,j} = alpha_sq * 0.5 * (h^3_{i,j}   + h^3_{i+1,j}),  Z face (+)
    D_{i,j} = alpha_sq * 0.5 * (h^3_{i-1,j} + h^3_{i,j}  ),  Z face (-)
    E = A + B + C + D
    alpha_sq = (2R/L * d_phi/d_Z)^2

    Also builds F_hs, the HS-warmup RHS with cell-centered upwind dh/dphi:
        F_hs[i, j] = d_phi * (h_{i,j} - h_{i,j-1})
    (periodic in j).

    Returns
    -------
    A, B, C, D, E, F_hs : cupy.ndarray, shape (N_Z, N_phi), float64
    """
    N_Z, N_phi = H_gpu.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    H3 = H_gpu ** 3

    # phi-direction face conductance (average of cubes).
    # Ah[:, k] = face between cells k and k+1, for k in [0, N_phi-2].
    # Requires H_gpu to be ghost-packed so the wrap-around face at
    # j=1/j=N_phi-2 is computed from the correct physical neighbours.
    Ah = 0.5 * (H3[:, :-1] + H3[:, 1:])           # shape (N_Z, N_phi-1)

    A = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    B = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    # A[:, j] = Ah[:, j] = face between j and j+1 (plus face of cell j).
    A[:, :-1] = Ah
    A[:, -1] = Ah[:, 0]   # ghost col (unused by the sweep)
    # B[:, j] = Ah[:, j-1] = face between j-1 and j (minus face of cell j).
    B[:, 1:] = Ah
    B[:, 0] = Ah[:, -1]   # ghost col (unused by the sweep)

    # Z-direction face conductance (average of cubes)
    H_jph3 = 0.5 * (H3[:-1, :] + H3[1:, :])       # shape (N_Z-1, N_phi)

    C = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    D = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    C[1:-1, :] = alpha_sq * H_jph3[1:, :]
    D[1:-1, :] = alpha_sq * H_jph3[:-1, :]

    E = A + B + C + D

    # HS warmup RHS: cell-centered upwind F_hs[i, j] = d_phi * (h_{i,j} - h_{i,j-1})
    F_hs = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    # j = 1 wraps to jm = N_phi - 2
    F_hs[:, 1] = d_phi * (H_gpu[:, 1] - H_gpu[:, N_phi - 2])
    # j = 2 .. N_phi - 2: jm = j - 1
    F_hs[:, 2:N_phi - 1] = d_phi * (
        H_gpu[:, 2:N_phi - 1] - H_gpu[:, 1:N_phi - 2]
    )

    return A, B, C, D, E, F_hs


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

        # Cell-centered gap buffer (used by ausas_rb_step for mass content)
        self._H = cp.empty(shape, dtype=cp.float64)

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

    def _run_color_pass(self, kernel, bc_kernel, d_phi,
                        omega_p, omega_theta, color, flooded_ends):
        """Run one color (red/black) pass + BC sync."""
        N_Z, N_phi = self.N_Z, self.N_phi
        kernel(
            self._grid, self._block,
            (
                self._P, self._theta,
                self._A, self._B, self._C, self._D, self._E,
                self._H,
                np.int32(N_Z), np.int32(N_phi),
                np.float64(d_phi),
                np.float64(omega_p), np.float64(omega_theta),
                np.int32(color),
            ),
        )
        bc_kernel(
            self._bc_grid, self._bc_block,
            (self._P, self._theta,
             np.int32(N_Z), np.int32(N_phi),
             np.int32(flooded_ends)),
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
        flooded_ends=True,
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
        flooded_ends : bool — if True (default), force theta=1 on Z boundaries
            (flooded bearing); otherwise clamp theta to [0, 1].
        verbose : bool

        Returns
        -------
        P : numpy.ndarray, (N_Z, N_phi), float64
        theta : numpy.ndarray, (N_Z, N_phi), float64
        residual : float
        n_iter : int (HS warmup + Ausas combined)
        """
        N_Z, N_phi = self.N_Z, self.N_phi
        flooded_flag = 1 if flooded_ends else 0

        # Defensive H ghost packing: force periodic wrap on columns 0 and
        # N_phi-1 before computing coefficients. Matches the CPU reference.
        H_gpu = H_gpu.copy()
        H_gpu[:, 0] = H_gpu[:, N_phi - 2]
        H_gpu[:, N_phi - 1] = H_gpu[:, 1]

        # Compute Ausas stencil coefficients (average-of-cubes conductance)
        # and cell-centered HS warmup RHS.
        A, B, C, D, E, F_hs = _compute_ausas_coefficients_gpu(
            H_gpu, d_phi, d_Z, R, L
        )
        self._A[:] = A
        self._B[:] = B
        self._C[:] = C
        self._D[:] = D
        self._E[:] = E

        # Store cell-centered H for the Ausas kernel
        self._H[:] = H_gpu

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
        if flooded_ends:
            self._theta[0, :] = 1.0
            self._theta[-1, :] = 1.0

        ausas_kernel = get_ausas_rb_kernel()
        bc_kernel = get_apply_bc_ausas_kernel()
        hs_sor_kernel = get_rb_sor_kernel()
        hs_bc_kernel = get_apply_bc_kernel()

        n_iter = 0

        # HS warmup: solve P with theta=1 fixed (only if no warm-start P).
        # Uses F_hs built with cell-centered upwind dh/dphi.
        if hs_warmup_iter > 0 and P_init is None:
            F_warmup = F_hs
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
            self._run_color_pass(
                ausas_kernel, bc_kernel, d_phi,
                omega_p, omega_theta, 0, flooded_flag,
            )
            # Black pass
            self._run_color_pass(
                ausas_kernel, bc_kernel, d_phi,
                omega_p, omega_theta, 1, flooded_flag,
            )

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
    flooded_ends: bool = True,
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
        flooded_ends=flooded_ends,
        verbose=verbose,
    )
