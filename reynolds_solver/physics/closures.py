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
        """Subcell quadrature with analytical texture evaluation."""
        import numpy as np

        N_Z, N_phi = H_gpu.shape
        n = self.n_sub
        tp = self.texture_params
        phi_1D = self.phi_1D
        Z_1D = self.Z_1D
        H_sm = self.H_smooth_gpu  # CuPy array

        phi_c = tp["phi_c"]
        Z_c = tp["Z_c"]
        A_tex = tp["A"]
        B_tex = tp["B"]
        H_p = tp["H_p"]
        profile = tp["profile"]
        n_pits = len(phi_c)

        # Precompute bounding box indices for each pit (CPU, once)
        pit_cells = []
        for k in range(n_pits):
            j_lo = max(0, int((phi_c[k] - B_tex) / d_phi) - 1)
            j_hi = min(N_phi - 1, int((phi_c[k] + B_tex) / d_phi) + 2)
            i_lo = max(0, int((Z_c[k] - (-1.0) - A_tex) / d_Z) - 1)
            # Z_1D starts at Z_1D[0], find index
            i_lo = max(0, int((Z_c[k] - A_tex - Z_1D[0]) / d_Z) - 1)
            i_hi = min(N_Z - 1, int((Z_c[k] + A_tex - Z_1D[0]) / d_Z) + 2)
            pit_cells.append((j_lo, j_hi, i_lo, i_hi))

        # Sub-point offsets within [0, 1]
        t_arr = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)

        # Smooth field corners (CuPy)
        # Cell (i,j) → corners at nodes (i,j), (i,j+1), (i+1,j), (i+1,j+1)
        # With periodic wrap for last phi column
        H_sm_np = cp.asnumpy(H_sm)

        # Compute K_cell on CPU (texture is CPU-side analytical)
        K_cell_np = np.zeros((N_Z - 1, N_phi - 1), dtype=np.float64)

        for i in range(N_Z - 1):
            for j in range(N_phi - 1):
                j_next = (j + 1) % N_phi
                # Corners of smooth field
                h00 = H_sm_np[i, j]
                h01 = H_sm_np[i, j_next]
                h10 = H_sm_np[i + 1, j]
                h11 = H_sm_np[i + 1, j_next]

                K_sum = 0.0
                for p in range(n):
                    for q in range(n):
                        tp_val = t_arr[p]
                        tq_val = t_arr[q]

                        # Bilinear smooth H
                        H_smooth_q = (h00 * (1 - tp_val) * (1 - tq_val)
                                    + h01 * tp_val * (1 - tq_val)
                                    + h10 * (1 - tp_val) * tq_val
                                    + h11 * tp_val * tq_val)

                        # Physical coordinates of sub-point
                        phi_q = phi_1D[j] + (p + 0.5) / n * d_phi
                        Z_q = Z_1D[i] + (q + 0.5) / n * d_Z

                        # Analytical texture contribution
                        H_text_q = 0.0
                        for k in range(n_pits):
                            jl, jh, il, ih = pit_cells[k]
                            if j < jl or j > jh or i < il or i > ih:
                                continue
                            dphi = np.arctan2(
                                np.sin(phi_q - phi_c[k]),
                                np.cos(phi_q - phi_c[k]))
                            expr = (dphi / B_tex)**2 + ((Z_q - Z_c[k]) / A_tex)**2
                            if expr <= 1.0:
                                if profile == "sqrt":
                                    H_text_q += H_p * np.sqrt(1.0 - expr)
                                elif profile == "smoothcap":
                                    H_text_q += H_p * (1.0 - expr)**2

                        H_q = H_smooth_q + H_text_q
                        K_sum += H_q ** 3

                K_cell_np[i, j] = K_sum / (n * n)

        K_cell = cp.asarray(K_cell_np, dtype=cp.float64)
        return self._cells_to_faces(K_cell, N_Z, N_phi)

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
