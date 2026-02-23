"""
Red-Black SOR solver -- wrapper around existing CuPy RawKernel code.

Provides:
  - ReynoldsSolverGPU: class with cached GPU buffers
  - solve_reynolds_sor: stateless static solver
  - solve_reynolds_sor_dynamic: stateless dynamic solver
"""

import numpy as np
import cupy as cp

from reynolds_solver.kernels import get_rb_sor_kernel, get_apply_bc_kernel
from reynolds_solver.utils import precompute_coefficients_gpu, add_dynamic_rhs_gpu


class ReynoldsSolverGPU:
    """
    Caches GPU buffers between calls with the same grid size.

    Avoids repeated memory allocation when called multiple times
    (e.g. parametric sweeps over eccentricity).

    Usage
    -----
    solver = ReynoldsSolverGPU(500, 500)
    P, delta, n_iter = solver.solve(H, d_phi, d_Z, R, L)
    P2, delta2, n_iter2 = solver.solve(H2, d_phi, d_Z, R, L)  # buffers reused
    """

    def __init__(self, N_Z: int, N_phi: int):
        self.N_Z = N_Z
        self.N_phi = N_phi

        # Working GPU buffers
        self._P = cp.zeros((N_Z, N_phi), dtype=cp.float64)
        self._P_old = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._A = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._B = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._C = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._D = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._E = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._F = cp.empty((N_Z, N_phi), dtype=cp.float64)

        # CUDA launch config: 32x8 block for better memory coalescing
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

    def _run_sor_iteration(self, sor_kernel, bc_kernel, N_Z, N_phi, omega):
        """Run one full Red-Black SOR iteration (red + black + BC)."""
        # Red pass (color = 0)
        sor_kernel(
            self._grid, self._block,
            (
                self._P, self._A, self._B, self._C, self._D,
                self._E, self._F,
                np.int32(N_Z), np.int32(N_phi),
                np.float64(omega), np.int32(0),
            ),
        )
        # Black pass (color = 1)
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

    def _compute_residual(self):
        """Compute relative residual: sum|P - P_old| / (sum|P| + eps)."""
        diff = cp.sum(cp.abs(self._P - self._P_old))
        norm = cp.sum(cp.abs(self._P))
        return float(diff) / (float(norm) + 1e-8)

    def solve(
        self,
        H: np.ndarray,
        d_phi: float,
        d_Z: float,
        R: float,
        L: float,
        omega: float = 1.5,
        tol: float = 1e-5,
        max_iter: int = 50000,
        check_every: int = 500,
    ) -> tuple:
        """
        Solve the static Reynolds equation on GPU.

        Returns
        -------
        P : np.ndarray, shape (N_Z, N_phi), float64
        delta : float
        n_iter : int
        """
        N_Z, N_phi = H.shape
        assert N_Z == self.N_Z and N_phi == self.N_phi, \
            f"Grid size mismatch: solver ({self.N_Z}x{self.N_phi}) vs input ({N_Z}x{N_phi})"

        H_gpu = cp.asarray(H, dtype=cp.float64)
        A, B, C, D, E, F = precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)

        self._A[:] = A
        self._B[:] = B
        self._C[:] = C
        self._D[:] = D
        self._E[:] = E
        self._F[:] = F
        self._P[:] = 0.0

        sor_kernel = get_rb_sor_kernel()
        bc_kernel = get_apply_bc_kernel()

        delta = 1.0
        iteration = 0

        while iteration < max_iter:
            need_check = ((iteration + 1) % check_every == 0) or (iteration == 0)

            if need_check:
                self._P_old[:] = self._P

            self._run_sor_iteration(sor_kernel, bc_kernel, N_Z, N_phi, omega)
            iteration += 1

            if need_check:
                delta = self._compute_residual()
                if delta < tol:
                    break

        P_cpu = cp.asnumpy(self._P)
        return P_cpu, float(delta), iteration

    def solve_with_rhs(
        self,
        H_gpu: cp.ndarray,
        F_full: cp.ndarray,
        A: cp.ndarray,
        B: cp.ndarray,
        C: cp.ndarray,
        D: cp.ndarray,
        E: cp.ndarray,
        omega: float = 1.5,
        tol: float = 1e-5,
        max_iter: int = 50000,
        check_every: int = 500,
    ) -> tuple:
        """
        Internal method: solve with pre-computed coefficients on GPU.
        Used by the dynamic solver variant.

        Returns (P_gpu, delta, n_iter) -- P stays on GPU.
        """
        N_Z, N_phi = self.N_Z, self.N_phi

        self._A[:] = A
        self._B[:] = B
        self._C[:] = C
        self._D[:] = D
        self._E[:] = E
        self._F[:] = F_full
        self._P[:] = 0.0

        sor_kernel = get_rb_sor_kernel()
        bc_kernel = get_apply_bc_kernel()

        delta = 1.0
        iteration = 0

        while iteration < max_iter:
            need_check = ((iteration + 1) % check_every == 0) or (iteration == 0)

            if need_check:
                self._P_old[:] = self._P

            self._run_sor_iteration(sor_kernel, bc_kernel, N_Z, N_phi, omega)
            iteration += 1

            if need_check:
                delta = self._compute_residual()
                if delta < tol:
                    break

        return self._P.copy(), float(delta), iteration


# ---------------------------------------------------------------------------
# Global cache of solver instances (by grid size)
# ---------------------------------------------------------------------------
_solver_cache: dict[tuple[int, int], ReynoldsSolverGPU] = {}


def _get_solver(N_Z: int, N_phi: int) -> ReynoldsSolverGPU:
    """Returns cached solver instance for given grid size."""
    key = (N_Z, N_phi)
    if key not in _solver_cache:
        _solver_cache[key] = ReynoldsSolverGPU(N_Z, N_phi)
    return _solver_cache[key]


def solve_reynolds_sor(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 500,
) -> tuple:
    """
    Solve the static Reynolds equation on GPU via Red-Black SOR.

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64
    delta : float
    n_iter : int
    """
    N_Z, N_phi = H.shape
    solver = _get_solver(N_Z, N_phi)
    return solver.solve(H, d_phi, d_Z, R, L, omega, tol, max_iter, check_every)


def solve_reynolds_sor_dynamic(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    xprime: float = 0.0,
    yprime: float = 0.0,
    beta: float = 2.0,
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 500,
) -> tuple:
    """
    Solve the dynamic Reynolds equation on GPU via Red-Black SOR.

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64
    delta : float
    n_iter : int
    """
    N_Z, N_phi = H.shape
    solver = _get_solver(N_Z, N_phi)

    H_gpu = cp.asarray(H, dtype=cp.float64)
    A, B, C, D, E, F_full = precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)

    add_dynamic_rhs_gpu(F_full, d_phi, N_Z, N_phi, xprime, yprime, beta)

    P_gpu, delta, n_iter = solver.solve_with_rhs(
        H_gpu, F_full, A, B, C, D, E,
        omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
    )

    P_cpu = cp.asnumpy(P_gpu)
    return P_cpu, delta, n_iter
