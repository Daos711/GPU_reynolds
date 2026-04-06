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
        If True, use subcell quadrature for conductance (avg(H³) instead
        of (avg H)³), with harmonic mean at faces. Default False.
    n_sub : int
        Number of sub-points per direction for quadrature (n_sub² per cell).
        Default 4.
    """

    def __init__(self, subcell_quad: bool = False, n_sub: int = 4):
        self.subcell_quad = subcell_quad
        self.n_sub = n_sub

    def modify_conductances(self, H_gpu, d_phi, d_Z, R, L, **kwargs):
        # H at faces (simple average) — used for wedge RHS, always simple
        H_i_ph = 0.5 * (H_gpu[:, :-1] + H_gpu[:, 1:])
        H_j_ph = 0.5 * (H_gpu[:-1, :] + H_gpu[1:, :])

        if not self.subcell_quad:
            A_half = H_i_ph ** 3
            C_half_raw = H_j_ph ** 3
        else:
            A_half, C_half_raw = self._subcell_conductance(H_gpu)

        return H_i_ph, H_j_ph, A_half, C_half_raw

    def _subcell_conductance(self, H_gpu):
        """Compute conductances via subcell quadrature + harmonic face mean."""
        N_Z, N_phi = H_gpu.shape
        n = self.n_sub

        # Sub-point positions within [0, 1] (midpoints of sub-intervals)
        t = cp.linspace(0.5 / n, 1.0 - 0.5 / n, n, dtype=cp.float64)

        # --- Cell-averaged K = avg(H³) via bilinear interpolation ---
        # Cell (i, j) has corners: H[i,j], H[i,j+1], H[i+1,j], H[i+1,j+1]
        # Interior cells: i=0..N_Z-2, j=0..N_phi-2
        H00 = H_gpu[:-1, :-1]  # (N_Z-1, N_phi-1)
        H01 = H_gpu[:-1, 1:]
        H10 = H_gpu[1:, :-1]
        H11 = H_gpu[1:, 1:]

        # Vectorized quadrature: broadcast sub-point weights
        # t_phi[p], t_z[q] → H_sub = bilinear(H00, H01, H10, H11, t_phi, t_z)
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
        # K_cell shape: (N_Z-1, N_phi-1)

        # --- Face conductance via harmonic mean of adjacent cells ---

        # Phi-direction faces (A_half): face between columns j and j+1
        # Shape: (N_Z, N_phi-1)
        # Interior faces (rows 1..N_Z-2): harmonic mean of K_cell[i-1,j] and K_cell[i,j]
        # But K_cell is indexed by cell (i,j) = corners (i,j)→(i+1,j+1)
        # Face between node columns j and j+1 at row i:
        #   left cell = (i-1, j) if i>0, right cell = (i, j) if i<N_Z-1
        # For simplicity: A_half[i, j] = K_cell average in phi direction
        # K_cell[i, j] covers phi interval [j, j+1] and Z interval [i, i+1]
        # Face j+1/2 at row i touches cells (i-1, j) above and (i, j) below
        # Use simple average of vertically adjacent cells for row-interior:
        A_half = cp.zeros((N_Z, N_phi - 1), dtype=cp.float64)
        # Rows 1..N_Z-2: average of cell above and below
        A_half[1:-1, :] = 0.5 * (K_cell[:-1, :] + K_cell[1:, :])
        # Boundary rows: use single adjacent cell
        A_half[0, :] = K_cell[0, :]
        A_half[-1, :] = K_cell[-1, :]

        # Z-direction faces (C_half_raw): face between rows i and i+1
        # Shape: (N_Z-1, N_phi)
        # Face between rows i and i+1 at column j:
        #   touches cells (i, j-1) and (i, j) in phi direction
        C_half_raw = cp.zeros((N_Z - 1, N_phi), dtype=cp.float64)
        # Columns 1..N_phi-2: average of phi-adjacent cells
        C_half_raw[:, 1:-1] = 0.5 * (K_cell[:, :-1] + K_cell[:, 1:])
        # Boundary columns: use single adjacent cell (periodic handled by caller)
        C_half_raw[:, 0] = K_cell[:, -1]  # periodic wrap
        C_half_raw[:, -1] = K_cell[:, 0]  # periodic wrap

        return A_half, C_half_raw


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
