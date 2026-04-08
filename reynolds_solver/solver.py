"""
GPU solver for the Reynolds equation (static version).

Method: Red-Black SOR (Successive Over-Relaxation) via CuPy RawKernel.
Residual computed on host side via CuPy array ops every check_every iterations.
Caches GPU buffers between calls with the same grid size.
"""

import numpy as np
import cupy as cp

from reynolds_solver.kernels import get_rb_sor_kernel, get_apply_bc_kernel
from reynolds_solver.utils import precompute_coefficients_gpu


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
        max_iter: int = 200000,
        check_every: int = 500,
        closure=None,
        P_init=None,
    ) -> tuple:
        """
        Solve the static Reynolds equation on GPU.

        Parameters
        ----------
        H : np.ndarray, shape (N_Z, N_phi), float64
        d_phi, d_Z : float
        R, L : float
        omega : float
            SOR relaxation parameter (1.0-1.9).
        tol : float
            Convergence criterion (relative residual).
        max_iter : int
        check_every : int
            How often to check convergence (iterations).
        closure : Closure or None
            Conductance model. None defaults to LaminarClosure.
        P_init : np.ndarray or None
            Initial pressure field for warm start.

        Returns
        -------
        P : np.ndarray, shape (N_Z, N_phi), float64
        delta : float
        n_iter : int
        """
        N_Z, N_phi = H.shape
        if N_Z != self.N_Z or N_phi != self.N_phi:
            raise ValueError(
                f"Grid size mismatch: solver ({self.N_Z}x{self.N_phi}) "
                f"vs input ({N_Z}x{N_phi})"
            )

        # 1. Transfer H to GPU and precompute coefficients
        H_gpu = cp.asarray(H, dtype=cp.float64)
        A, B, C, D, E, F = precompute_coefficients_gpu(
            H_gpu, d_phi, d_Z, R, L, closure=closure
        )

        # Copy into cached buffers
        self._A[:] = A
        self._B[:] = B
        self._C[:] = C
        self._D[:] = D
        self._E[:] = E
        self._F[:] = F

        if P_init is not None:
            P_arr = cp.asarray(P_init, dtype=cp.float64)
            if P_arr.shape != (N_Z, N_phi):
                raise ValueError(
                    f"P_init shape {P_arr.shape} != H shape {(N_Z, N_phi)}"
                )
            self._P[:] = P_arr
        else:
            self._P[:] = 0.0

        # 2. Get compiled kernels
        sor_kernel = get_rb_sor_kernel()
        bc_kernel = get_apply_bc_kernel()

        # 3. Iterative Red-Black SOR loop (block-based convergence check)
        delta = 1.0
        iteration = 0
        min_iter = max(100, check_every)
        converged = False

        while iteration < max_iter:
            # Save P before block
            self._P_old[:] = self._P

            # Run a block of check_every iterations
            n_block = min(check_every, max_iter - iteration)
            for _ in range(n_block):
                self._run_sor_iteration(sor_kernel, bc_kernel, N_Z, N_phi, omega)
                iteration += 1

            # Check convergence over entire block
            if iteration >= min_iter:
                delta = self._compute_residual()
                if delta < tol:
                    converged = True
                    break

        if not converged:
            import warnings
            warnings.warn(
                f"SOR did not converge: {iteration} iters, delta={delta:.2e}, "
                f"tol={tol:.1e}, omega={omega:.4f}",
                RuntimeWarning,
                stacklevel=3,
            )

        # 4. Transfer result to CPU
        P_cpu = cp.asnumpy(self._P)
        return P_cpu, float(delta), iteration, converged

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
        P_init=None,
    ) -> tuple:
        """
        Internal method: solve with pre-computed coefficients on GPU.
        Used by the dynamic solver variant.

        Returns (P_gpu, delta, n_iter) — P stays on GPU.
        """
        N_Z, N_phi = self.N_Z, self.N_phi

        self._A[:] = A
        self._B[:] = B
        self._C[:] = C
        self._D[:] = D
        self._E[:] = E
        self._F[:] = F_full

        if P_init is not None:
            P_arr = cp.asarray(P_init, dtype=cp.float64)
            if P_arr.shape != (N_Z, N_phi):
                raise ValueError(
                    f"P_init shape {P_arr.shape} != expected shape {(N_Z, N_phi)}"
                )
            self._P[:] = P_arr
        else:
            self._P[:] = 0.0

        sor_kernel = get_rb_sor_kernel()
        bc_kernel = get_apply_bc_kernel()

        delta = 1.0
        iteration = 0
        min_iter = max(100, check_every)
        converged = False

        while iteration < max_iter:
            self._P_old[:] = self._P

            n_block = min(check_every, max_iter - iteration)
            for _ in range(n_block):
                self._run_sor_iteration(sor_kernel, bc_kernel, N_Z, N_phi, omega)
                iteration += 1

            if iteration >= min_iter:
                delta = self._compute_residual()
                if delta < tol:
                    converged = True
                    break

        return self._P.copy(), float(delta), iteration, converged


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


def solve_reynolds_gpu(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 200000,
    check_every: int = 500,
    closure=None,
    P_init=None,
) -> tuple:
    """
    Drop-in replacement for solve_reynolds_gauss_seidel_numba().

    Solves the static Reynolds equation on GPU via Red-Black SOR.

    Parameters
    ----------
    H : np.ndarray, shape (N_Z, N_phi), float64
    d_phi, d_Z : float
    R, L : float
    omega : float
    tol : float
    max_iter : int
    check_every : int
    closure : Closure or None
        Conductance model. None defaults to LaminarClosure.
    P_init : np.ndarray or None
        Initial pressure field for warm start.

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64
    delta : float
    n_iter : int
    """
    N_Z, N_phi = H.shape
    solver = _get_solver(N_Z, N_phi)
    return solver.solve(
        H, d_phi, d_Z, R, L, omega, tol, max_iter, check_every,
        closure=closure, P_init=P_init,
    )
