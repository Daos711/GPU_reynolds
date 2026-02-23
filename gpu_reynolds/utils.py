"""
Вспомогательные функции для GPU-солвера уравнения Рейнольдса.

- Предвычисление коэффициентов дискретизации на GPU
- Создание зазора H с эллипсоидальными углублениями
"""

import numpy as np
import cupy as cp


def precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L):
    """
    Вычисляет коэффициенты дискретизации A, B, C, D, E, F на GPU.

    Все операции выполняются через CuPy — без пересылки на CPU.

    Parameters
    ----------
    H_gpu : cupy.ndarray, shape (N_Z, N_phi), float64
        Безразмерный зазор на GPU.
    d_phi : float
        Шаг сетки по φ.
    d_Z : float
        Шаг сетки по Z.
    R : float
        Радиус подшипника (м).
    L : float
        Длина подшипника (м).

    Returns
    -------
    A, B, C, D, E, F : cupy.ndarray, каждый shape (N_Z, N_phi), float64
    """
    N_Z, N_phi = H_gpu.shape

    # Зазор на полуцелых узлах по φ
    H_i_plus_half = 0.5 * (H_gpu[:, :-1] + H_gpu[:, 1:])       # (N_Z, N_phi-1)
    # Сдвиг: H_i_minus_half[j] = H_i_plus_half[j-1] (с периодичностью)
    H_i_minus_half = cp.empty_like(H_i_plus_half)
    H_i_minus_half[:, 1:] = H_i_plus_half[:, :-1]
    H_i_minus_half[:, 0] = H_i_plus_half[:, -1]

    # Зазор на полуцелых узлах по Z
    H_j_plus_half = 0.5 * (H_gpu[:-1, :] + H_gpu[1:, :])       # (N_Z-1, N_phi)
    H_j_minus_half = cp.empty_like(H_j_plus_half)
    H_j_minus_half[1:, :] = H_j_plus_half[:-1, :]
    H_j_minus_half[0, :] = H_j_plus_half[-1, :]

    # alpha^2 = (D/L * d_phi/d_Z)^2
    D_over_L = 2.0 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    # Коэффициенты на полуцелых узлах
    A_half = H_i_plus_half ** 3
    B_half = H_i_minus_half ** 3
    C_half = alpha_sq * H_j_plus_half ** 3
    D_half = alpha_sq * H_j_minus_half ** 3

    # Расширяем до полного размера (N_Z, N_phi) с периодическими/симметричными ГУ
    A_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    B_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    C_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    D_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)

    A_full[:, :-1] = A_half
    A_full[:, -1] = A_half[:, 0]

    B_full[:, 1:] = B_half
    B_full[:, 0] = B_half[:, -1]

    C_full[:-1, :] = C_half
    C_full[-1, :] = C_half[0, :]

    D_full[1:, :] = D_half
    D_full[0, :] = D_half[-1, :]

    E_full = A_full + B_full + C_full + D_full

    # Правая часть (статическая)
    F_half = d_phi * (H_i_plus_half - H_i_minus_half)
    F_full = cp.zeros((N_Z, N_phi), dtype=cp.float64)
    F_full[:, :-1] = F_half
    F_full[:, -1] = F_half[:, 0]

    return A_full, B_full, C_full, D_full, E_full, F_full


def add_dynamic_rhs_gpu(F_full, d_phi, N_Z, N_phi, xprime, yprime, beta):
    """
    Добавляет динамический вклад к правой части F на GPU (in-place).

    F[i,j] += beta * (xprime * sin(phi_global) + yprime * cos(phi_global))
    где phi_global = j * d_phi + pi/4

    Parameters
    ----------
    F_full : cupy.ndarray, shape (N_Z, N_phi)
    d_phi : float
    N_Z, N_phi : int
    xprime, yprime : float
    beta : float
    """
    j_indices = cp.arange(N_phi, dtype=cp.float64)
    phi_local = j_indices * d_phi
    phi_global = phi_local + cp.pi / 4.0

    dyn_term = beta * (xprime * cp.sin(phi_global) + yprime * cp.cos(phi_global))
    # Broadcast: dyn_term shape (N_phi,) -> добавляем ко всем строкам
    F_full += dyn_term[cp.newaxis, :]


def create_H_with_ellipsoidal_depressions(H0, H_p, Phi_mesh, Z_mesh,
                                           phi_c_flat, Z_c_flat, A, B):
    """
    Создаёт поле зазора H с эллипсоидальными углублениями (на CPU, numpy).

    Parameters
    ----------
    H0 : np.ndarray — базовый зазор
    H_p : float — безразмерная глубина углубления
    Phi_mesh, Z_mesh : np.ndarray — координатные сетки
    phi_c_flat, Z_c_flat : np.ndarray — координаты центров углублений
    A, B : float — безразмерные полуоси

    Returns
    -------
    H : np.ndarray — зазор с углублениями
    """
    H = H0.copy()
    for k in range(len(phi_c_flat)):
        phi_c = phi_c_flat[k]
        Z_c = Z_c_flat[k]
        delta_phi = np.arctan2(np.sin(Phi_mesh - phi_c), np.cos(Phi_mesh - phi_c))
        expr = (delta_phi / B) ** 2 + ((Z_mesh - Z_c) / A) ** 2
        inside = expr <= 1
        H[inside] += H_p * np.sqrt(1 - expr[inside])
    return H
