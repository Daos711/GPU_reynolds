"""
GPU-солвер уравнения Рейнольдса (динамическая версия).

Расширяет статический солвер динамическим вкладом в правую часть:
    F[i,j] += beta * (xprime * sin(phi_global) + yprime * cos(phi_global))
где phi_global = j * d_phi + pi/4.
"""

import numpy as np
import cupy as cp

from gpu_reynolds.solver import _get_solver
from gpu_reynolds.utils import precompute_coefficients_gpu, add_dynamic_rhs_gpu


def solve_reynolds_gpu_dynamic(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    xprime: float = 0.0,
    yprime: float = 0.0,
    beta: float = 2.0,
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 500,
) -> tuple:
    """
    Drop-in замена для solve_reynolds_gauss_seidel_numba_dynamic().

    Решает динамическое уравнение Рейнольдса на GPU методом Red-Black SOR.
    Правая часть включает динамический вклад от скоростей xprime, yprime.

    Parameters
    ----------
    H : np.ndarray, shape (N_Z, N_phi), float64
        Безразмерный зазор (numpy, CPU).
    d_phi, d_Z : float
        Шаги сетки.
    R, L : float
        Радиус и длина подшипника (м).
    xprime, yprime : float
        Безразмерные скорости (малые возмущения).
    beta : float
        Коэффициент для динамического члена.
    omega : float
        Параметр SOR-релаксации.
    tol : float
        Критерий сходимости.
    max_iter : int
        Максимальное число итераций.
    check_every : int
        Частота проверки сходимости.

    Returns
    -------
    P : np.ndarray, shape (N_Z, N_phi), float64
        Поле безразмерного давления (CPU).
    delta : float
        Финальная относительная невязка.
    n_iter : int
        Количество итераций.
    """
    N_Z, N_phi = H.shape
    solver = _get_solver(N_Z, N_phi)

    # 1. Перенос H на GPU и предвычисление коэффициентов
    H_gpu = cp.asarray(H, dtype=cp.float64)
    A, B, C, D, E, F_full = precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)

    # 2. Добавляем динамический вклад к правой части
    add_dynamic_rhs_gpu(F_full, d_phi, N_Z, N_phi, xprime, yprime, beta)

    # 3. Решаем с подготовленными коэффициентами
    P_gpu, delta, n_iter = solver.solve_with_rhs(
        H_gpu, F_full, A, B, C, D, E,
        omega=omega, tol=tol, max_iter=max_iter, check_every=check_every,
    )

    # 4. Перенос результата на CPU
    P_cpu = cp.asnumpy(P_gpu)
    return P_cpu, delta, n_iter
