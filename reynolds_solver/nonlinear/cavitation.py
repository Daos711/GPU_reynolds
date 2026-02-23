"""
Outer loop for cavitation condition P >= 0.

Algorithm:
1. Solve M @ p = f
2. Find all nodes where p < 0
3. Fix P = 0 at those nodes (modify matrix via penalty)
4. Solve again
5. Repeat until cavitation zone stabilizes

Typically converges in 3-8 outer iterations.
"""

import cupy as cp
import cupyx.scipy.sparse as cusparse


def solve_with_cavitation(linear_solver, M_original, f_original,
                          max_outer=20, tol_cav=1e-4):
    """
    Solve M @ p = f with constraint p >= 0.

    Parameters
    ----------
    linear_solver : object with method solve(M, f) -> (p, info)
    M_original : cupyx.scipy.sparse.csr_matrix
    f_original : cp.ndarray
    max_outer : int
    tol_cav : float
        Cavitation zone is stable when number of changed nodes
        is < tol_cav * N_total.

    Returns
    -------
    p : cp.ndarray
        Solution with p >= 0.
    n_outer : int
        Number of outer iterations.
    """
    N = f_original.shape[0]

    # Initial solve
    p, info = linear_solver.solve(M_original, f_original)

    # Cavitation mask (True = P fixed to 0)
    cav_mask = p < 0
    p[cav_mask] = 0.0

    if not cp.any(cav_mask):
        return p, 1

    for outer in range(max_outer):
        cav_mask_old = cav_mask.copy()

        # Penalty approach: for cavitation nodes, add large penalty
        # to diagonal so the solution is forced to ~0
        f_mod = f_original.copy()

        if cp.any(cav_mask):
            penalty = 1e20
            diag_penalty = cp.zeros(N, dtype=cp.float64)
            diag_penalty[cav_mask] = penalty
            M_penalty = M_original + cusparse.diags(diag_penalty)
            f_mod[cav_mask] = 0.0
        else:
            M_penalty = M_original

        # Solve modified system
        p, info = linear_solver.solve(M_penalty, f_mod)

        # Update cavitation zone
        cav_mask = p < 0
        p[cav_mask] = 0.0

        # Check stability of cavitation zone
        changed = int(cp.sum(cav_mask != cav_mask_old))
        if changed < tol_cav * N:
            return p, outer + 2  # +2: 1 for initial solve + (outer+1) for this iteration

    return p, max_outer + 1
