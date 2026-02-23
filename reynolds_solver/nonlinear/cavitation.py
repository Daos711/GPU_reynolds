"""
Outer loop for cavitation condition P >= 0 (monotonic active set method).

Algorithm:
1. Solve M @ p = f (unconstrained)
2. Identify cavitation zone: nodes where p < 0
3. Fix P = 0 at cavitation nodes (replace rows with identity)
4. Solve modified system
5. If new negative nodes found → add to cavitation, repeat
6. If no new negative nodes → converged

The cavitation set only grows (monotonic). This guarantees convergence
in a small number of iterations (typically 3-8).

Typically converges in 3-8 outer iterations.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def solve_with_cavitation_cpu(M_original, f_original,
                              max_outer=20, tol_cav=1e-4):
    """
    Solve M @ p = f with constraint p >= 0 using monotonic active set.

    For cavitation nodes (p < 0), the row is replaced with an identity row
    (M[k,:] = 0, M[k,k] = 1, f[k] = 0) which forces P[k] = 0.

    The cavitation set only grows: once a node is cavitated, it stays.
    Converges when no new negative nodes appear after solving.

    Parameters
    ----------
    M_original : scipy.sparse.csr_matrix
    f_original : np.ndarray
    max_outer : int
    tol_cav : float
        Convergence tolerance: stop when number of new cavitation nodes
        is less than tol_cav * N.

    Returns
    -------
    p : np.ndarray
    n_outer : int
    """
    N = f_original.shape[0]

    # Initial unconstrained solve
    p = spsolve(M_original, f_original)

    cav_mask = p < 0
    p[cav_mask] = 0.0

    if not np.any(cav_mask):
        return p, 1

    for outer in range(max_outer):
        # Build modified system: identity rows for cavitation nodes
        active = (~cav_mask).astype(np.float64)
        cav_float = cav_mask.astype(np.float64)

        M_mod = sp.diags(active) @ M_original + sp.diags(cav_float)

        f_mod = f_original.copy()
        f_mod[cav_mask] = 0.0

        p = spsolve(M_mod, f_mod)

        # Check for new negative nodes (monotonic: only add, never remove)
        new_neg = (~cav_mask) & (p < 0)
        n_new = int(np.sum(new_neg))

        if n_new <= tol_cav * N:
            # Converged: clamp remaining negatives and return
            p[p < 0] = 0.0
            return p, outer + 2

        # Add new negative nodes to cavitation set
        cav_mask = cav_mask | new_neg
        p[cav_mask] = 0.0

    return p, max_outer + 1
