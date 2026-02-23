"""
Валидация GPU-солвера: сравнение результатов GPU (CuPy Red-Black SOR) vs CPU (Numba).

Проверяет:
  1. Статический солвер — поле давления и интегральные нагрузки
  2. Динамический солвер — то же с xprime, yprime != 0

Критерии приёмки:
  - max|P_cpu - P_gpu| / max(P_cpu) < 1e-3  (0.1% относительная ошибка)
  - |F_cpu - F_gpu| / F_cpu < 1e-3

Запуск:
    python -m gpu_reynolds.test_validation
"""

import sys
import numpy as np
from numba import njit


# ───────────────────────────────────────────────────────────────────────────
# CPU-референс: статический солвер
# ───────────────────────────────────────────────────────────────────────────
@njit
def solve_reynolds_cpu(H, d_phi, d_Z, R, L, omega=1.5, tol=1e-5, max_iter=20000):
    N_Z, N_phi = H.shape
    P = np.zeros((N_Z, N_phi))

    H_i_plus_half = 0.5 * (H[:, :-1] + H[:, 1:])
    H_i_minus_half = np.hstack((H_i_plus_half[:, -1:], H_i_plus_half[:, :-1]))
    H_j_plus_half = 0.5 * (H[:-1, :] + H[1:, :])
    H_j_minus_half = np.vstack((H_j_plus_half[-1:, :], H_j_plus_half[:-1, :]))

    D_over_L = 2 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    A = H_i_plus_half ** 3
    B = H_i_minus_half ** 3
    C = alpha_sq * H_j_plus_half ** 3
    D_coef = alpha_sq * H_j_minus_half ** 3

    A_full = np.zeros((N_Z, N_phi))
    B_full = np.zeros((N_Z, N_phi))
    C_full = np.zeros((N_Z, N_phi))
    D_full = np.zeros((N_Z, N_phi))

    A_full[:, :-1] = A
    A_full[:, -1] = A[:, 0]
    B_full[:, 1:] = B
    B_full[:, 0] = B[:, -1]
    C_full[:-1, :] = C
    C_full[-1, :] = C[0, :]
    D_full[1:, :] = D_coef
    D_full[0, :] = D_coef[-1, :]

    E = A_full + B_full + C_full + D_full

    F = d_phi * (H_i_plus_half - H_i_minus_half)
    F_full = np.zeros((N_Z, N_phi))
    F_full[:, :-1] = F
    F_full[:, -1] = F[:, 0]

    delta = 1.0
    iteration = 0
    while delta > tol and iteration < max_iter:
        delta = 0.0
        norm_P = 0.0
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                P_old_ij = P[i, j]
                P_new = (
                    A_full[i, j] * P[i, (j + 1) % N_phi]
                    + B_full[i, j] * P[i, (j - 1) % N_phi]
                    + C_full[i, j] * P[i + 1, j]
                    + D_full[i, j] * P[i - 1, j]
                    - F_full[i, j]
                ) / E[i, j]
                P_new = max(P_new, 0.0)
                P[i, j] = P_old_ij + omega * (P_new - P_old_ij)
                delta += abs(P[i, j] - P_old_ij)
                norm_P += abs(P[i, j])
        P[:, 0] = P[:, -2]
        P[:, -1] = P[:, 1]
        P[0, :] = 0.0
        P[-1, :] = 0.0
        delta /= norm_P + 1e-8
        iteration += 1
    return P, delta, iteration


# ───────────────────────────────────────────────────────────────────────────
# CPU-референс: динамический солвер
# ───────────────────────────────────────────────────────────────────────────
@njit
def solve_reynolds_cpu_dynamic(H, d_phi, d_Z, R, L,
                                xprime=0.0, yprime=0.0, beta=2.0,
                                omega=1.5, tol=1e-5, max_iter=20000):
    N_Z, N_phi = H.shape
    P = np.zeros((N_Z, N_phi))

    H_i_plus_half = 0.5 * (H[:, :-1] + H[:, 1:])
    H_i_minus_half = np.hstack((H_i_plus_half[:, -1:], H_i_plus_half[:, :-1]))
    H_j_plus_half = 0.5 * (H[:-1, :] + H[1:, :])
    H_j_minus_half = np.vstack((H_j_plus_half[-1:, :], H_j_plus_half[:-1, :]))

    D_over_L = 2 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    A = H_i_plus_half ** 3
    B = H_i_minus_half ** 3
    C = alpha_sq * H_j_plus_half ** 3
    D_coef = alpha_sq * H_j_minus_half ** 3

    A_full = np.zeros((N_Z, N_phi))
    B_full = np.zeros((N_Z, N_phi))
    C_full = np.zeros((N_Z, N_phi))
    D_full = np.zeros((N_Z, N_phi))

    A_full[:, :-1] = A
    A_full[:, -1] = A[:, 0]
    B_full[:, 1:] = B
    B_full[:, 0] = B[:, -1]
    C_full[:-1, :] = C
    C_full[-1, :] = C[0, :]
    D_full[1:, :] = D_coef
    D_full[0, :] = D_coef[-1, :]

    E = A_full + B_full + C_full + D_full

    # Статическая часть правой части
    static_part = d_phi * (H_i_plus_half - H_i_minus_half)
    F_full = np.zeros((N_Z, N_phi))
    F_full[:, :-1] = static_part
    F_full[:, -1] = static_part[:, 0]

    # Динамический вклад
    for j in range(N_phi):
        phi_local = j * d_phi
        phi_global = phi_local + np.pi / 4.0
        sin_phi_global = np.sin(phi_global)
        cos_phi_global = np.cos(phi_global)
        dyn_term = beta * (xprime * sin_phi_global + yprime * cos_phi_global)
        for i in range(N_Z):
            F_full[i, j] += dyn_term

    delta = 1.0
    iteration = 0
    while delta > tol and iteration < max_iter:
        delta = 0.0
        norm_P = 0.0
        for i in range(1, N_Z - 1):
            for j in range(1, N_phi - 1):
                P_old_ij = P[i, j]
                P_new = (
                    A_full[i, j] * P[i, (j + 1) % N_phi]
                    + B_full[i, j] * P[i, (j - 1) % N_phi]
                    + C_full[i, j] * P[i + 1, j]
                    + D_full[i, j] * P[i - 1, j]
                    - F_full[i, j]
                ) / E[i, j]
                if P_new < 0.0:
                    P_new = 0.0
                P[i, j] = P_old_ij + omega * (P_new - P_old_ij)
                delta += abs(P[i, j] - P_old_ij)
                norm_P += abs(P[i, j])
        P[:, 0] = P[:, -2]
        P[:, -1] = P[:, 1]
        P[0, :] = 0.0
        P[-1, :] = 0.0
        delta /= norm_P + 1e-12
        iteration += 1
    return P, delta, iteration


# ───────────────────────────────────────────────────────────────────────────
# Вспомогательные функции
# ───────────────────────────────────────────────────────────────────────────
def compute_loads(P, phi_1D, Z):
    """Вычисляет интегральные нагрузки F_r, F_t, F_total."""
    Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z)
    cos_phi = np.cos(Phi_mesh)
    sin_phi = np.sin(Phi_mesh)
    F_r = np.trapz(np.trapz(P * cos_phi, phi_1D, axis=1), Z)
    F_t = np.trapz(np.trapz(P * sin_phi, phi_1D, axis=1), Z)
    F_total = np.sqrt(F_r**2 + F_t**2)
    return F_r, F_t, F_total


def run_test(test_name, passed, details=""):
    """Печатает результат теста."""
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_name}")
    if details:
        print(f"         {details}")
    return passed


# ───────────────────────────────────────────────────────────────────────────
# Основные тесты
# ───────────────────────────────────────────────────────────────────────────
def test_static_solver():
    """Тест 1: Статический солвер — сравнение GPU vs CPU."""
    print("\n=== Тест 1: Статический солвер ===")

    from gpu_reynolds.solver import solve_reynolds_gpu

    # Параметры
    R = 0.035
    L = 0.056
    epsilon = 0.6
    N_Z, N_phi = 500, 500
    omega_sor = 1.5
    tol = 1e-5
    max_iter = 50000

    phi_1D = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]

    # CPU
    print("  Решаем на CPU (Numba)...")
    P_cpu, delta_cpu, iter_cpu = solve_reynolds_cpu(
        H, d_phi, d_Z, R, L, omega_sor, tol, max_iter
    )
    print(f"  CPU: {iter_cpu} итераций, delta = {delta_cpu:.2e}")

    # GPU
    print("  Решаем на GPU (CuPy)...")
    P_gpu, delta_gpu, iter_gpu = solve_reynolds_gpu(
        H, d_phi, d_Z, R, L, omega_sor, tol, max_iter, check_every=100
    )
    print(f"  GPU: {iter_gpu} итераций, delta = {delta_gpu:.2e}")

    all_passed = True

    # Тест 1a: Относительная ошибка поля давления
    P_max = np.max(P_cpu)
    if P_max > 0:
        max_err = np.max(np.abs(P_cpu - P_gpu)) / P_max
        mean_err = np.mean(np.abs(P_cpu - P_gpu)) / P_max
    else:
        max_err = np.max(np.abs(P_cpu - P_gpu))
        mean_err = np.mean(np.abs(P_cpu - P_gpu))

    passed = max_err < 1e-3
    all_passed &= run_test(
        "Поле давления: max|P_cpu - P_gpu| / max(P_cpu)",
        passed,
        f"max_err = {max_err:.2e}, mean_err = {mean_err:.2e}"
    )

    # Тест 1b: Интегральные нагрузки
    _, _, F_cpu = compute_loads(P_cpu, phi_1D, Z)
    _, _, F_gpu = compute_loads(P_gpu, phi_1D, Z)

    if F_cpu > 0:
        load_err = abs(F_cpu - F_gpu) / F_cpu
    else:
        load_err = abs(F_cpu - F_gpu)

    passed = load_err < 1e-3
    all_passed &= run_test(
        "Интегральная нагрузка: |F_cpu - F_gpu| / F_cpu",
        passed,
        f"F_cpu = {F_cpu:.6f}, F_gpu = {F_gpu:.6f}, err = {load_err:.2e}"
    )

    return all_passed


def test_dynamic_solver():
    """Тест 2: Динамический солвер — сравнение GPU vs CPU."""
    print("\n=== Тест 2: Динамический солвер (xprime=0.001, yprime=0.001) ===")

    from gpu_reynolds.solver_dynamic import solve_reynolds_gpu_dynamic

    # Параметры
    R = 0.035
    L = 0.056
    epsilon = 0.6
    N_Z, N_phi = 500, 500
    omega_sor = 1.5
    tol = 1e-5
    max_iter = 50000
    xprime = 0.001
    yprime = 0.001
    beta = 2.0

    phi_1D = np.linspace(0, 2 * np.pi, N_phi)
    Z = np.linspace(-1, 1, N_Z)
    Phi_mesh, _ = np.meshgrid(phi_1D, Z)
    H = 1.0 + epsilon * np.cos(Phi_mesh)
    d_phi = phi_1D[1] - phi_1D[0]
    d_Z = Z[1] - Z[0]

    # CPU
    print("  Решаем на CPU (Numba)...")
    P_cpu, delta_cpu, iter_cpu = solve_reynolds_cpu_dynamic(
        H, d_phi, d_Z, R, L,
        xprime=xprime, yprime=yprime, beta=beta,
        omega=omega_sor, tol=tol, max_iter=max_iter
    )
    print(f"  CPU: {iter_cpu} итераций, delta = {delta_cpu:.2e}")

    # GPU
    print("  Решаем на GPU (CuPy)...")
    P_gpu, delta_gpu, iter_gpu = solve_reynolds_gpu_dynamic(
        H, d_phi, d_Z, R, L,
        xprime=xprime, yprime=yprime, beta=beta,
        omega=omega_sor, tol=tol, max_iter=max_iter, check_every=100
    )
    print(f"  GPU: {iter_gpu} итераций, delta = {delta_gpu:.2e}")

    all_passed = True

    # Тест 2a: Поле давления
    P_max = np.max(P_cpu)
    if P_max > 0:
        max_err = np.max(np.abs(P_cpu - P_gpu)) / P_max
        mean_err = np.mean(np.abs(P_cpu - P_gpu)) / P_max
    else:
        max_err = np.max(np.abs(P_cpu - P_gpu))
        mean_err = np.mean(np.abs(P_cpu - P_gpu))

    passed = max_err < 1e-3
    all_passed &= run_test(
        "Поле давления (dynamic): max|P_cpu - P_gpu| / max(P_cpu)",
        passed,
        f"max_err = {max_err:.2e}, mean_err = {mean_err:.2e}"
    )

    # Тест 2b: Интегральные нагрузки
    _, _, F_cpu = compute_loads(P_cpu, phi_1D, Z)
    _, _, F_gpu = compute_loads(P_gpu, phi_1D, Z)

    if F_cpu > 0:
        load_err = abs(F_cpu - F_gpu) / F_cpu
    else:
        load_err = abs(F_cpu - F_gpu)

    passed = load_err < 1e-3
    all_passed &= run_test(
        "Интегральная нагрузка (dynamic): |F_cpu - F_gpu| / F_cpu",
        passed,
        f"F_cpu = {F_cpu:.6f}, F_gpu = {F_gpu:.6f}, err = {load_err:.2e}"
    )

    return all_passed


def main():
    print("=" * 60)
    print("  Валидация GPU-солвера уравнения Рейнольдса")
    print("=" * 60)

    # Прогрев Numba
    print("\nПрогрев Numba JIT...")
    H_w = 1.0 + 0.6 * np.cos(np.linspace(0, 2*np.pi, 50)[np.newaxis, :] * np.ones((50, 1)))
    solve_reynolds_cpu(H_w, 0.1, 0.1, 0.035, 0.056, 1.5, 0.1, 10)
    solve_reynolds_cpu_dynamic(H_w, 0.1, 0.1, 0.035, 0.056,
                                xprime=0.001, yprime=0.001, beta=2.0,
                                omega=1.5, tol=0.1, max_iter=10)
    print("Numba JIT прогрет.")

    results = []
    results.append(test_static_solver())
    results.append(test_dynamic_solver())

    print("\n" + "=" * 60)
    all_ok = all(results)
    if all_ok:
        print("  ВСЕ ТЕСТЫ ПРОЙДЕНЫ (PASS)")
    else:
        print("  ЕСТЬ ПРОВАЛЫ (FAIL)")
    print("=" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
