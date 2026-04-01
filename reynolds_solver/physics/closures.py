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
    """H³ — standard laminar conductance model."""

    def modify_conductances(self, H_gpu, d_phi, d_Z, R, L, **kwargs):
        H_i_ph = 0.5 * (H_gpu[:, :-1] + H_gpu[:, 1:])
        H_j_ph = 0.5 * (H_gpu[:-1, :] + H_gpu[1:, :])
        A_half = H_i_ph ** 3
        C_half_raw = H_j_ph ** 3
        return H_i_ph, H_j_ph, A_half, C_half_raw


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
