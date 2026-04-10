"""
Payvar-Salant / Elrod-Adams GPU solver for steady mass-conserving JFO.

Uses Red-Black SOR on CUDA with the same frozen-active-set strategy
as the CPU reference (solver_cpu.py):

  1. HS warmup on GPU (reuses the existing rb_sor_step kernel).
  2. Classify cav_mask from HS result.
  3. PS inner loop: Red pass → BC → Black pass → BC → residual check.
  4. Outer loop: update cav_mask if any cells flipped.

Coefficients are built on GPU via CuPy (average-of-cubes), NOT using
the old precompute_coefficients_gpu (which uses cube-of-average).
"""
import numpy as np
import cupy as cp

from reynolds_solver.cavitation.payvar_salant.kernels import (
    get_ps_rb_sor_kernel,
    get_apply_bc_ps_kernel,
)
from reynolds_solver.kernels import get_rb_sor_kernel, get_apply_bc_kernel


# -----------------------------------------------------------------------
# Coefficient build on GPU (average-of-cubes)
# -----------------------------------------------------------------------
def _build_coefficients_gpu(H_gpu, d_phi, d_Z, R, L):
    """
    Average-of-cubes conductance on GPU. Mirrors the CPU
    _build_coefficients from solver_cpu.py exactly.
    """
    N_Z, N_phi = H_gpu.shape
    alpha_sq = (2.0 * R / L * d_phi / d_Z) ** 2

    H3 = H_gpu ** 3
    # phi-direction face conductance
    Ah = 0.5 * (H3[:, :-1] + H3[:, 1:])  # (N_Z, N_phi - 1)

    A = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    A[:, :-1] = Ah
    A[:, -1] = Ah[:, 0]

    B = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    B[:, 1:] = Ah
    B[:, 0] = Ah[:, -1]

    # Z-direction face conductance
    H3_jph = 0.5 * (H3[:-1, :] + H3[1:, :])  # (N_Z - 1, N_phi)
    C = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    D = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    C[1:-1, :] = alpha_sq * H3_jph[1:, :]
    D[1:-1, :] = alpha_sq * H3_jph[:-1, :]

    E = A + B + C + D
    return A, B, C, D, E


# -----------------------------------------------------------------------
# GPU solver
# -----------------------------------------------------------------------
def solve_payvar_salant_gpu(
    H, d_phi, d_Z, R, L,
    omega=1.0,
    tol=1e-6,
    max_iter=50000,
    check_every=100,
    hs_warmup_iter=2000,
    hs_warmup_tol=1e-7,
    hs_warmup_omega=1.7,
    pin_active_set=True,
    max_outer_active_set=10,
    cav_threshold=1e-10,
    verbose=False,
):
    """
    GPU Payvar-Salant solver — same algorithm as the CPU reference.

    Parameters match `solve_payvar_salant_cpu`. Returns (P, theta,
    residual, n_iter) as numpy arrays on host.
    """
    N_Z, N_phi = H.shape

    # Ghost-pack H on host, then upload
    H = np.ascontiguousarray(H, dtype=np.float64).copy()
    H[:, 0] = H[:, N_phi - 2]
    H[:, N_phi - 1] = H[:, 1]
    H_gpu = cp.asarray(H)

    # Coefficients on GPU (average-of-cubes)
    A, B, C, D, E = _build_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)

    # Launch config (same 32×8 block as the HS solver)
    block = (32, 8, 1)
    grid = (
        (N_phi - 2 + block[0] - 1) // block[0],
        (N_Z - 2 + block[1] - 1) // block[1],
        1,
    )
    max_dim = max(N_Z, N_phi)
    bc_block = (256, 1, 1)
    bc_grid = ((max_dim + 255) // 256, 1, 1)

    n_iter_total = 0

    # ------------------------------------------------------------------
    # HS warmup on GPU (reuse existing rb_sor_step kernel)
    # ------------------------------------------------------------------
    hs_kernel = get_rb_sor_kernel()
    hs_bc_kernel = get_apply_bc_kernel()

    # Build F_hs = d_phi * (H[i,j] - H[i,jm]) on GPU
    F_hs = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    F_hs[:, 1:] = d_phi * (H_gpu[:, 1:] - H_gpu[:, :-1])
    F_hs[:, 0] = d_phi * (H_gpu[:, 0] - H_gpu[:, -1])
    # Correct the wrap at column 1: jm=N_phi-2
    F_hs[:, 1] = d_phi * (H_gpu[:, 1] - H_gpu[:, N_phi - 2])

    P_hs = cp.zeros((N_Z, N_phi), dtype=cp.float64)

    if verbose:
        print("  [PS-GPU] HS warmup starting...")

    for k in range(hs_warmup_iter):
        for color in (0, 1):
            hs_kernel(
                grid, block,
                (
                    P_hs, A, B, C, D, E, F_hs,
                    np.int32(N_Z), np.int32(N_phi),
                    np.float64(hs_warmup_omega), np.int32(color),
                ),
            )
        hs_bc_kernel(bc_grid, bc_block, (P_hs, np.int32(N_Z), np.int32(N_phi)))
        n_iter_total += 1

        if k % 50 == 0 or k == hs_warmup_iter - 1:
            hs_max = float(cp.max(P_hs))
            if k > 5 and hs_max > 0:
                # Quick convergence estimate from max change
                pass

        if k > 5:
            # Use a simple max|ΔP| check every 50 iters
            if k % 50 == 0:
                # For efficiency, just check max(P) stability
                pass

    if verbose:
        print(
            f"  [PS-GPU] HS warmup done: iter={n_iter_total}, "
            f"maxP={float(cp.max(P_hs)):.4e}"
        )

    # ------------------------------------------------------------------
    # g = P_hs (full-film seed)
    # ------------------------------------------------------------------
    g = P_hs  # reuse buffer

    # ------------------------------------------------------------------
    # Cav mask from HS
    # ------------------------------------------------------------------
    cav_mask = (g < cav_threshold).astype(cp.int32)
    cav_mask[0, :] = 0
    cav_mask[-1, :] = 0
    cav_mask[:, 0] = cav_mask[:, N_phi - 2]
    cav_mask[:, N_phi - 1] = cav_mask[:, 1]

    # Seed g = 0 in cavitation set
    if pin_active_set:
        g[cav_mask.astype(bool)] = 0.0

    # ------------------------------------------------------------------
    # PS iteration kernels
    # ------------------------------------------------------------------
    ps_kernel = get_ps_rb_sor_kernel()
    ps_bc_kernel = get_apply_bc_ps_kernel()

    pinned_flag = np.int32(1 if pin_active_set else 0)
    n_outer = max_outer_active_set if pin_active_set else 1

    if verbose:
        cav0 = int(cav_mask[1:-1, 1:-1].sum())
        tot0 = (N_Z - 2) * (N_phi - 2)
        print(
            f"  [PS-GPU] N_Z={N_Z}, N_phi={N_phi}, ω={omega}, "
            f"pinned={'yes' if pin_active_set else 'no'}, "
            f"initial cav={cav0}/{tot0}"
        )

    g_old = cp.empty_like(g)
    residual = 1.0

    for outer in range(n_outer):
        # Inner SOR
        inner_k = 0
        for k in range(max_iter):
            # Save for residual
            if k % check_every == 0:
                g_old[:] = g

            # Red pass
            ps_kernel(
                grid, block,
                (
                    g, H_gpu, A, B, C, D, E, cav_mask,
                    np.int32(N_Z), np.int32(N_phi),
                    np.float64(d_phi), np.float64(omega),
                    np.int32(0), pinned_flag,
                ),
            )
            # Black pass
            ps_kernel(
                grid, block,
                (
                    g, H_gpu, A, B, C, D, E, cav_mask,
                    np.int32(N_Z), np.int32(N_phi),
                    np.float64(d_phi), np.float64(omega),
                    np.int32(1), pinned_flag,
                ),
            )
            # BC
            ps_bc_kernel(
                bc_grid, bc_block,
                (g, np.int32(N_Z), np.int32(N_phi)),
            )

            inner_k = k + 1
            n_iter_total += 1

            # Residual check
            if k % check_every == 0 and k > 0:
                residual = float(cp.max(cp.abs(g - g_old)))
                if residual < tol:
                    break
                if verbose:
                    maxP = float(cp.max(cp.where(g >= 0, g, 0)))
                    cav_frac = float(
                        cp.mean(g[1:-1, 1:-1] < 0)
                    )
                    print(
                        f"    outer={outer} inner={k}: "
                        f"update={residual:.3e}, maxP={maxP:.4e}, "
                        f"cav={cav_frac:.3f}"
                    )

        # Update active set
        if not pin_active_set:
            break

        new_cav = (g < 0).astype(cp.int32)
        new_cav[0, :] = 0
        new_cav[-1, :] = 0
        new_cav[:, 0] = new_cav[:, N_phi - 2]
        new_cav[:, N_phi - 1] = new_cav[:, 1]

        flips = int(cp.sum(new_cav[1:-1, 1:-1] != cav_mask[1:-1, 1:-1]))
        if verbose:
            print(
                f"  [PS-GPU] outer={outer}: inner_k={inner_k}, "
                f"flips={flips}"
            )
        if flips == 0:
            break
        cav_mask = new_cav

    # ------------------------------------------------------------------
    # Recover P, θ and transfer to host
    # ------------------------------------------------------------------
    P_gpu = cp.where(g >= 0, g, 0.0)
    theta_gpu = cp.where(g >= 0, 1.0, 1.0 + g)
    cp.clip(theta_gpu, 0.0, 1.0, out=theta_gpu)

    P = cp.asnumpy(P_gpu)
    theta = cp.asnumpy(theta_gpu)

    return P, theta, residual, n_iter_total
