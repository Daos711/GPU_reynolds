"""
Outer loop for cavitation condition P >= 0.

Algorithm:
1. Solve M @ p = f (unconstrained linear system)
2. Find all nodes where p < 0
3. Replace cavitation rows with identity (P_k = 0) via row scaling
4. Solve modified system
5. Repeat until cavitation zone stabilizes

Typically converges in 3-8 outer iterations.
"""

import cupy as cp
import cupyx.scipy.sparse as cusparse


def solve_with_cavitation(linear_solver, M_original, f_original,
                          max_outer=20, tol_cav=1e-4):
    """
    Solve M @ p = f with constraint p >= 0.

    For cavitation nodes (p < 0), the row is replaced with an identity row
    (M[k,:] = 0, M[k,k] = 1, f[k] = 0) which forces P[k] = 0.
    This preserves matrix conditioning unlike penalty methods.

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

    # Initial solve (unconstrained)
    p, info = linear_solver.solve(M_original, f_original)

    # Cavitation mask (True = P fixed to 0)
    cav_mask = p < 0
    p[cav_mask] = 0.0

    if not cp.any(cav_mask):
        return p, 1

    for outer in range(max_outer):
        cav_mask_old = cav_mask.copy()

        # Row elimination: for cavitation nodes, replace row with identity
        # active[k] = 1 if not cavitation, 0 if cavitation
        # M_mod = diag(active) @ M + diag(cav)
        # This zeros out cavitation rows and sets their diagonal to 1
        active = (~cav_mask).astype(cp.float64)
        cav_float = cav_mask.astype(cp.float64)

        M_mod = cusparse.diags(active) @ M_original + cusparse.diags(cav_float)

        f_mod = f_original.copy()
        f_mod[cav_mask] = 0.0

        # Solve modified system
        p, info = linear_solver.solve(M_mod, f_mod)

        # Update cavitation zone
        cav_mask = p < 0
        p[cav_mask] = 0.0

        # Check stability of cavitation zone
        changed = int(cp.sum(cav_mask != cav_mask_old))
        if changed < tol_cav * N:
            return p, outer + 2  # +2: 1 for initial + (outer+1) for this

    return p, max_outer + 1
