"""
Closure models for lubricant film conductance.

Closures modify the effective conductance of the gap:
  - LaminarClosure: H³ (standard laminar model)
  - ConstantinescuClosure: 12·G·H³ (turbulent model, Constantinescu)
"""

from abc import ABC, abstractmethod
import cupy as cp


class Closure(ABC):
    """Base class for lubricant film conductance models."""

    @abstractmethod
    def modify_conductances(self, H_gpu, d_phi, d_Z, R, L, **kwargs):
        """
        Compute face-averaged gap heights and conductances.

        Returns
        -------
        H_i_plus_half : cp.ndarray, shape (N_Z, N_phi-1)
            Gap at i+1/2 faces (phi direction).
        H_j_plus_half : cp.ndarray, shape (N_Z-1, N_phi)
            Gap at j+1/2 faces (Z direction).
        A_half : cp.ndarray, shape (N_Z, N_phi-1)
            Conductance in phi direction at i+1/2 faces.
        C_half_raw : cp.ndarray, shape (N_Z-1, N_phi)
            Conductance in Z direction at j+1/2 faces (before alpha_sq scaling).
        """
        pass


class LaminarClosure(Closure):
    """H³ — standard laminar conductance model.

    Parameters
    ----------
    subcell_quad : bool
        If True, use subcell quadrature for conductance. Default False.
    n_sub : int
        Number of sub-points per direction (n_sub² per cell). Default 4.
    H_smooth_gpu : cp.ndarray or None
        Smooth gap field (without texture) on GPU. Required for analytical
        texture quadrature.
    texture_params : dict or None
        Analytical texture definition. Keys: phi_c, Z_c, A, B, H_p, profile.
    phi_1D : np.ndarray or None
        Node coordinates in phi direction.
    Z_1D : np.ndarray or None
        Node coordinates in Z direction.
    """

    def __init__(self, subcell_quad: bool = False, n_sub: int = 4,
                 H_smooth_gpu=None, texture_params=None,
                 phi_1D=None, Z_1D=None):
        self.subcell_quad = subcell_quad
        self.n_sub = n_sub
        self.H_smooth_gpu = H_smooth_gpu
        self.texture_params = texture_params
        self.phi_1D = phi_1D
        self.Z_1D = Z_1D

        if texture_params is not None:
            _validate_texture_params(texture_params)

    def modify_conductances(self, H_gpu, d_phi, d_Z, R, L, **kwargs):
        # H at faces (simple average) — used for wedge RHS, always simple
        H_i_ph = 0.5 * (H_gpu[:, :-1] + H_gpu[:, 1:])
        H_j_ph = 0.5 * (H_gpu[:-1, :] + H_gpu[1:, :])

        if not self.subcell_quad:
            A_half = H_i_ph ** 3
            C_half_raw = H_j_ph ** 3
        elif self.texture_params is not None and self.H_smooth_gpu is not None:
            A_half, C_half_raw = self._subcell_analytical(
                H_gpu, d_phi, d_Z)
        else:
            A_half, C_half_raw = self._subcell_bilinear(H_gpu)

        return H_i_ph, H_j_ph, A_half, C_half_raw

    def _subcell_bilinear(self, H_gpu):
        """Subcell quadrature via bilinear interpolation of nodal H."""
        N_Z, N_phi = H_gpu.shape
        n = self.n_sub
        t = cp.linspace(0.5 / n, 1.0 - 0.5 / n, n, dtype=cp.float64)

        H00 = H_gpu[:-1, :-1]
        H01 = H_gpu[:-1, 1:]
        H10 = H_gpu[1:, :-1]
        H11 = H_gpu[1:, 1:]

        K_cell = cp.zeros_like(H00)
        for p in range(n):
            for q in range(n):
                tp = float(t[p])
                tq = float(t[q])
                H_sub = (H00 * (1 - tp) * (1 - tq)
                       + H01 * tp * (1 - tq)
                       + H10 * (1 - tp) * tq
                       + H11 * tp * tq)
                K_cell += H_sub ** 3
        K_cell /= (n * n)

        return self._cells_to_faces(K_cell, N_Z, N_phi)

    def _subcell_analytical(self, H_gpu, d_phi, d_Z):
        """Subcell quadrature with analytical texture — GPU-vectorized.

        Loop only over pits (72 iterations). All sub-points computed via
        CuPy broadcasting on 4D arrays (N_Z-1, N_phi, n_sub, n_sub).
        """
        N_Z, N_phi = H_gpu.shape
        n = self.n_sub
        tp = self.texture_params
        H_sm = self.H_smooth_gpu

        A_tex = float(tp["A"])
        B_tex = float(tp["B"])
        H_p = float(tp["H_p"])
        profile = tp["profile"]
        # Pit centers as CPU lists (avoid repeated GPU→CPU transfers)
        phi_c_cpu = [float(x) for x in tp["phi_c"]]
        Z_c_cpu = [float(x) for x in tp["Z_c"]]
        n_pits = len(phi_c_cpu)

        # Sub-point weights within [0, 1]
        t = cp.linspace(0.5 / n, 1.0 - 0.5 / n, n, dtype=cp.float64)

        # --- H_smooth at sub-points via bilinear interpolation ---
        # Wrap last→first column for periodicity: N_phi cells around the ring
        H_sm_wrap = cp.concatenate([H_sm, H_sm[:, :1]], axis=1)  # (N_Z, N_phi+1)
        H00 = H_sm_wrap[:-1, :N_phi]       # (N_Z-1, N_phi)
        H01 = H_sm_wrap[:-1, 1:N_phi + 1]
        H10 = H_sm_wrap[1:, :N_phi]
        H11 = H_sm_wrap[1:, 1:N_phi + 1]

        # Broadcast weights: (1, 1, n, 1) for phi, (1, 1, 1, n) for Z
        tp_w = t[None, None, :, None]
        tq_w = t[None, None, None, :]

        # H_sub shape: (N_Z-1, N_phi, n, n)
        H_sub = (H00[:, :, None, None] * (1 - tp_w) * (1 - tq_w)
               + H01[:, :, None, None] * tp_w * (1 - tq_w)
               + H10[:, :, None, None] * (1 - tp_w) * tq_w
               + H11[:, :, None, None] * tp_w * tq_w)

        # --- Sub-point physical coordinates ---
        phi_nodes = cp.asarray(self.phi_1D, dtype=cp.float64)
        Z_nodes = cp.asarray(self.Z_1D, dtype=cp.float64)

        # phi_sub: (N_phi, n) → broadcast to (1, N_phi, n, 1)
        phi_sub = phi_nodes[:, None] + t[None, :] * d_phi  # (N_phi, n)
        phi_sub_4d = phi_sub[None, :, :, None]

        # Z_sub: (N_Z-1, n) → broadcast to (N_Z-1, 1, 1, n)
        Z_sub = Z_nodes[:-1, None] + t[None, :] * d_Z  # (N_Z-1, n)
        Z_sub_4d = Z_sub[:, None, None, :]

        # --- Texture: loop over pits, vectorized per pit ---
        for k in range(n_pits):
            dphi = cp.arctan2(
                cp.sin(phi_sub_4d - phi_c_cpu[k]),
                cp.cos(phi_sub_4d - phi_c_cpu[k]))
            expr = (dphi / B_tex) ** 2 + ((Z_sub_4d - Z_c_cpu[k]) / A_tex) ** 2

            contrib = cp.maximum(1.0 - expr, 0.0)
            if profile == "smoothcap":
                H_sub += H_p * contrib ** 2
            elif profile == "sqrt":
                H_sub += H_p * cp.sqrt(contrib)

        # --- K_cell = mean(H³) over sub-points ---
        K_cell = cp.mean(H_sub ** 3, axis=(2, 3))  # (N_Z-1, N_phi)

        # --- Convert to faces (N_phi cells → N_phi-1 face array) ---
        # K_cell has N_phi columns (periodic). Need A_half with N_phi-1 columns.
        # A_half[i, j] = face between columns j and j+1:
        #   average of K_cell for cell-j (covers [j, j+1]) from above and below rows
        # For the standard A_half shape (N_Z, N_phi-1):
        #   use K_cell[:, :N_phi-1] (cells 0..N_phi-2)
        K_cell_trim = K_cell[:, :N_phi - 1]
        A_half = cp.zeros((N_Z, N_phi - 1), dtype=cp.float64)
        A_half[1:-1, :] = 0.5 * (K_cell_trim[:-1, :] + K_cell_trim[1:, :])
        A_half[0, :] = K_cell_trim[0, :]
        A_half[-1, :] = K_cell_trim[-1, :]

        # C_half_raw (N_Z-1, N_phi): face between rows, all columns
        # Use all N_phi columns of K_cell, average phi-neighbors
        C_half_raw = cp.zeros((N_Z - 1, N_phi), dtype=cp.float64)
        C_half_raw[:, 1:-1] = 0.5 * (K_cell[:, :-1][:, :N_phi - 2]
                                     + K_cell[:, 1:][:, :N_phi - 2])
        # But we have N_phi columns; handle boundary with periodic wrap
        C_half_raw[:, 1:N_phi - 1] = 0.5 * (K_cell[:, :N_phi - 2] + K_cell[:, 1:N_phi - 1])
        C_half_raw[:, 0] = K_cell[:, N_phi - 1]   # periodic wrap
        C_half_raw[:, N_phi - 1] = K_cell[:, 0]    # periodic wrap

        return A_half, C_half_raw

    @staticmethod
    def _cells_to_faces(K_cell, N_Z, N_phi):
        """Convert cell-averaged K to face conductances A_half, C_half_raw."""
        # Phi-direction faces
        A_half = cp.zeros((N_Z, N_phi - 1), dtype=cp.float64)
        A_half[1:-1, :] = 0.5 * (K_cell[:-1, :] + K_cell[1:, :])
        A_half[0, :] = K_cell[0, :]
        A_half[-1, :] = K_cell[-1, :]

        # Z-direction faces
        C_half_raw = cp.zeros((N_Z - 1, N_phi), dtype=cp.float64)
        C_half_raw[:, 1:-1] = 0.5 * (K_cell[:, :-1] + K_cell[:, 1:])
        C_half_raw[:, 0] = K_cell[:, -1]
        C_half_raw[:, -1] = K_cell[:, 0]

        return A_half, C_half_raw


def _validate_texture_params(tp):
    """Validate texture_params dict."""
    import numpy as np
    required = {"phi_c", "Z_c", "A", "B", "H_p", "profile"}
    missing = required - set(tp.keys())
    if missing:
        raise ValueError(f"texture_params missing keys: {missing}")

    phi_c = np.asarray(tp["phi_c"])
    Z_c = np.asarray(tp["Z_c"])
    if phi_c.ndim != 1 or Z_c.ndim != 1:
        raise ValueError("phi_c and Z_c must be 1D arrays")
    if len(phi_c) != len(Z_c):
        raise ValueError(f"phi_c ({len(phi_c)}) and Z_c ({len(Z_c)}) must have same length")

    for name in ("A", "B", "H_p"):
        val = tp[name]
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValueError(f"texture_params['{name}'] must be scalar > 0, got {val}")

    if tp["profile"] not in ("sqrt", "smoothcap"):
        raise ValueError(f"Unknown profile: {tp['profile']}")



class ConstantinescuClosure(Closure):
    """12·Gx·H³, 12·Gz·H³ — turbulent model (Constantinescu).

    Parameters
    ----------
    rho : float
        Lubricant density, kg/m³.
    U_velocity : float
        Shaft surface linear velocity, m/s.
    mu : float
        Dynamic viscosity, Pa·s.
    c_clearance : float
        Radial clearance, m (h = H * c_clearance).
    """

    def __init__(self, rho: float, U_velocity: float,
                 mu: float, c_clearance: float):
        self.rho = rho
        self.U = U_velocity
        self.mu = mu
        self.c = c_clearance

    def modify_conductances(self, H_gpu, d_phi, d_Z, R, L, **kwargs):
        # 1. Dimensional gap
        h_dim = H_gpu * self.c

        # 2. Local Reynolds number
        Re = self.rho * self.U * h_dim / self.mu

        # 3. Constantinescu coefficients (all in float64)
        Re = Re.astype(cp.float64)
        Gx = cp.float64(1.0) / (cp.float64(12.0) + cp.float64(0.0139) * Re ** cp.float64(0.96))
        Gz = cp.float64(1.0) / (cp.float64(12.0) + cp.float64(0.01) * Re ** cp.float64(0.75))

        # 4. Face-averaged values (arithmetic mean)
        H_i_ph = 0.5 * (H_gpu[:, :-1] + H_gpu[:, 1:])
        H_j_ph = 0.5 * (H_gpu[:-1, :] + H_gpu[1:, :])

        Gx_i_ph = 0.5 * (Gx[:, :-1] + Gx[:, 1:])
        Gz_j_ph = 0.5 * (Gz[:-1, :] + Gz[1:, :])

        # 5. Conductances: 12·G·H³ (at Re→0: G→1/12, so 12·G·H³ → H³)
        A_half = cp.float64(12.0) * Gx_i_ph * H_i_ph ** 3
        C_half_raw = cp.float64(12.0) * Gz_j_ph * H_j_ph ** 3

        return H_i_ph, H_j_ph, A_half, C_half_raw
