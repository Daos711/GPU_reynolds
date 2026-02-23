"""
GPU-солвер уравнения Рейнольдса (статическая версия).

Метод: Red-Black SOR (Successive Over-Relaxation) на CUDA через CuPy RawKernel.
Кэширует GPU-буферы между вызовами с одинаковым размером сетки.
"""

import numpy as np
import cupy as cp

from gpu_reynolds.kernels import get_rb_sor_kernel, get_apply_bc_kernel
from gpu_reynolds.utils import precompute_coefficients_gpu


class ReynoldsSolverGPU:
    """
    Кэширует GPU-буферы между вызовами с одинаковым размером сетки.

    Позволяет избежать повторного выделения памяти при многократных вызовах
    (например, при параметрическом сканировании по эксцентриситету).

    Usage
    -----
    solver = ReynoldsSolverGPU(500, 500)
    P, delta, n_iter = solver.solve(H, d_phi, d_Z, R, L)
    # ... изменить H ...
    P2, delta2, n_iter2 = solver.solve(H2, d_phi, d_Z, R, L)  # буферы переиспользуются
    """

    def __init__(self, N_Z: int, N_phi: int):
        """
        Parameters
        ----------
        N_Z : int
            Количество узлов по Z.
        N_phi : int
            Количество узлов по φ.
        """
        self.N_Z = N_Z
        self.N_phi = N_phi

        # Рабочие GPU-буферы
        self._P = cp.zeros((N_Z, N_phi), dtype=cp.float64)
        self._A = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._B = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._C = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._D = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._E = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._F = cp.empty((N_Z, N_phi), dtype=cp.float64)
        self._delta_arr = cp.zeros(2, dtype=cp.float64)  # [sum|diff|, sum|P|]

        # Конфигурация CUDA-запуска
        self._block = (16, 16, 1)
        self._grid = (
            (N_phi - 2 + self._block[0] - 1) // self._block[0],
            (N_Z - 2 + self._block[1] - 1) // self._block[1],
            1,
        )
        # Для граничных условий
        max_dim = max(N_Z, N_phi)
        self._bc_block = (256, 1, 1)
        self._bc_grid = ((max_dim + 255) // 256, 1, 1)

    def solve(
        self,
        H: np.ndarray,
        d_phi: float,
        d_Z: float,
        R: float,
        L: float,
        omega: float = 1.5,
        tol: float = 1e-5,
        max_iter: int = 50000,
        check_every: int = 100,
    ) -> tuple:
        """
        Решает стационарное уравнение Рейнольдса на GPU.

        Parameters
        ----------
        H : np.ndarray, shape (N_Z, N_phi), float64
            Безразмерный зазор (numpy, CPU).
        d_phi, d_Z : float
            Шаги сетки.
        R, L : float
            Радиус и длина подшипника.
        omega : float
            Параметр SOR-релаксации (1.0–1.9).
        tol : float
            Критерий сходимости (относительная невязка).
        max_iter : int
            Максимальное число итераций.
        check_every : int
            Частота проверки сходимости (в итерациях).

        Returns
        -------
        P : np.ndarray, shape (N_Z, N_phi), float64
            Поле давления (numpy, CPU).
        delta : float
            Финальная относительная невязка.
        n_iter : int
            Количество выполненных итераций.
        """
        N_Z, N_phi = H.shape
        assert N_Z == self.N_Z and N_phi == self.N_phi, \
            f"Grid size mismatch: solver ({self.N_Z}x{self.N_phi}) vs input ({N_Z}x{N_phi})"

        # 1. Перенос H на GPU и предвычисление коэффициентов
        H_gpu = cp.asarray(H, dtype=cp.float64)
        A, B, C, D, E, F = precompute_coefficients_gpu(H_gpu, d_phi, d_Z, R, L)

        # Копируем в кэшированные буферы
        self._A[:] = A
        self._B[:] = B
        self._C[:] = C
        self._D[:] = D
        self._E[:] = E
        self._F[:] = F
        self._P[:] = 0.0  # Начальное приближение P = 0

        # 2. Получаем скомпилированные ядра
        sor_kernel = get_rb_sor_kernel()
        bc_kernel = get_apply_bc_kernel()

        # 3. Итерационный цикл Red-Black SOR
        delta = 1.0
        iteration = 0
        need_check = False

        while iteration < max_iter:
            # Определяем, нужно ли считать невязку на этом шаге
            need_check = ((iteration + 1) % check_every == 0) or (iteration == 0)

            if need_check:
                self._delta_arr[:] = 0.0

            # Red pass (color = 0)
            sor_kernel(
                self._grid, self._block,
                (
                    self._P, self._A, self._B, self._C, self._D,
                    self._E, self._F, self._delta_arr,
                    np.int32(N_Z), np.int32(N_phi),
                    np.float64(omega), np.int32(0),
                ),
            )

            # Black pass (color = 1)
            sor_kernel(
                self._grid, self._block,
                (
                    self._P, self._A, self._B, self._C, self._D,
                    self._E, self._F, self._delta_arr,
                    np.int32(N_Z), np.int32(N_phi),
                    np.float64(omega), np.int32(1),
                ),
            )

            # Граничные условия
            bc_kernel(
                self._bc_grid, self._bc_block,
                (self._P, np.int32(N_Z), np.int32(N_phi)),
            )

            iteration += 1

            # Проверка сходимости
            if need_check:
                delta_vals = self._delta_arr.get()  # GPU -> CPU (синхронизация)
                delta = delta_vals[0] / (delta_vals[1] + 1e-8)
                if delta < tol:
                    break

        # 4. Перенос результата на CPU
        P_cpu = cp.asnumpy(self._P)
        return P_cpu, float(delta), iteration

    def solve_with_rhs(
        self,
        H_gpu: cp.ndarray,
        F_full: cp.ndarray,
        A: cp.ndarray,
        B: cp.ndarray,
        C: cp.ndarray,
        D: cp.ndarray,
        E: cp.ndarray,
        omega: float = 1.5,
        tol: float = 1e-5,
        max_iter: int = 50000,
        check_every: int = 100,
    ) -> tuple:
        """
        Внутренний метод: решает с уже подготовленными коэффициентами на GPU.
        Используется динамической версией солвера.

        Returns (P_gpu, delta, n_iter) — P остаётся на GPU.
        """
        N_Z, N_phi = self.N_Z, self.N_phi

        self._A[:] = A
        self._B[:] = B
        self._C[:] = C
        self._D[:] = D
        self._E[:] = E
        self._F[:] = F_full
        self._P[:] = 0.0

        sor_kernel = get_rb_sor_kernel()
        bc_kernel = get_apply_bc_kernel()

        delta = 1.0
        iteration = 0

        while iteration < max_iter:
            need_check = ((iteration + 1) % check_every == 0) or (iteration == 0)

            if need_check:
                self._delta_arr[:] = 0.0

            # Red
            sor_kernel(
                self._grid, self._block,
                (
                    self._P, self._A, self._B, self._C, self._D,
                    self._E, self._F, self._delta_arr,
                    np.int32(N_Z), np.int32(N_phi),
                    np.float64(omega), np.int32(0),
                ),
            )
            # Black
            sor_kernel(
                self._grid, self._block,
                (
                    self._P, self._A, self._B, self._C, self._D,
                    self._E, self._F, self._delta_arr,
                    np.int32(N_Z), np.int32(N_phi),
                    np.float64(omega), np.int32(1),
                ),
            )
            # BC
            bc_kernel(
                self._bc_grid, self._bc_block,
                (self._P, np.int32(N_Z), np.int32(N_phi)),
            )

            iteration += 1

            if need_check:
                delta_vals = self._delta_arr.get()
                delta = delta_vals[0] / (delta_vals[1] + 1e-8)
                if delta < tol:
                    break

        return self._P.copy(), float(delta), iteration


# ---------------------------------------------------------------------------
# Глобальный кэш экземпляров солвера (по размеру сетки)
# ---------------------------------------------------------------------------
_solver_cache: dict[tuple[int, int], ReynoldsSolverGPU] = {}


def _get_solver(N_Z: int, N_phi: int) -> ReynoldsSolverGPU:
    """Возвращает кэшированный экземпляр солвера для заданного размера сетки."""
    key = (N_Z, N_phi)
    if key not in _solver_cache:
        _solver_cache[key] = ReynoldsSolverGPU(N_Z, N_phi)
    return _solver_cache[key]


def solve_reynolds_gpu(
    H: np.ndarray,
    d_phi: float,
    d_Z: float,
    R: float,
    L: float,
    omega: float = 1.5,
    tol: float = 1e-5,
    max_iter: int = 50000,
    check_every: int = 100,
) -> tuple:
    """
    Drop-in замена для solve_reynolds_gauss_seidel_numba().

    Решает стационарное уравнение Рейнольдса на GPU методом Red-Black SOR.

    Parameters
    ----------
    H : np.ndarray, shape (N_Z, N_phi), float64
        Безразмерный зазор (numpy, CPU).
    d_phi, d_Z : float
        Шаги сетки.
    R, L : float
        Радиус и длина подшипника (м).
    omega : float
        Параметр SOR-релаксации (рекомендуется 1.0–1.9).
    tol : float
        Критерий сходимости (относительная невязка).
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
    return solver.solve(H, d_phi, d_Z, R, L, omega, tol, max_iter, check_every)
