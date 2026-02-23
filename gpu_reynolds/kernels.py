"""
CUDA C ядра для Red-Black SOR решения уравнения Рейнольдса.

Содержит:
  - rb_sor_step:  один полушаг Red-Black SOR (обновление точек одного цвета)
  - apply_bc:     применение граничных условий (периодичность по φ, Дирихле по Z)

Все ядра компилируются через CuPy RawKernel при первом вызове и кэшируются.
"""

import cupy as cp

# ---------------------------------------------------------------------------
# CUDA-ядро: один полушаг Red-Black SOR
# ---------------------------------------------------------------------------
_RB_SOR_KERNEL_CODE = r"""
extern "C" __global__ void rb_sor_step(
    double* __restrict__ P,
    const double* __restrict__ A_arr,
    const double* __restrict__ B_arr,
    const double* __restrict__ C_arr,
    const double* __restrict__ D_arr,
    const double* __restrict__ E_arr,
    const double* __restrict__ F_arr,
    double* __restrict__ delta_arr,
    const int N_Z,
    const int N_phi,
    const double omega_sor,
    const int color
)
{
    // j — индекс по φ (столбцы), i — индекс по Z (строки)
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;   // от 1 до N_phi-2
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;   // от 1 до N_Z-2

    if (i >= N_Z - 1 || j >= N_phi - 1) return;

    // Шахматная раскраска: пропускаем точки другого цвета
    if ((i + j) % 2 != color) return;

    int idx = i * N_phi + j;

    // Соседи по φ с учётом периодичности
    int j_plus  = (j + 1 < N_phi - 1) ? j + 1 : 1;
    int j_minus = (j - 1 >= 1)        ? j - 1 : N_phi - 2;

    double P_old = P[idx];

    double P_new = (A_arr[idx] * P[i * N_phi + j_plus]
                  + B_arr[idx] * P[i * N_phi + j_minus]
                  + C_arr[idx] * P[(i + 1) * N_phi + j]
                  + D_arr[idx] * P[(i - 1) * N_phi + j]
                  - F_arr[idx]) / E_arr[idx];

    // Условие кавитации
    if (P_new < 0.0) P_new = 0.0;

    // SOR-релаксация
    P[idx] = P_old + omega_sor * (P_new - P_old);

    // Невязка (атомарная аккумуляция)
    double diff = fabs(P[idx] - P_old);
    atomicAdd(&delta_arr[0], diff);
    atomicAdd(&delta_arr[1], fabs(P[idx]));
}
"""

# ---------------------------------------------------------------------------
# CUDA-ядро: применение граничных условий
# ---------------------------------------------------------------------------
_APPLY_BC_KERNEL_CODE = r"""
extern "C" __global__ void apply_bc(
    double* __restrict__ P,
    const int N_Z,
    const int N_phi
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Периодичность по φ: для каждой строки i
    if (tid < N_Z) {
        int i = tid;
        P[i * N_phi + 0]           = P[i * N_phi + (N_phi - 2)];
        P[i * N_phi + (N_phi - 1)] = P[i * N_phi + 1];
    }

    // Дирихле по Z: для каждого столбца j
    if (tid < N_phi) {
        int j = tid;
        P[0 * N_phi + j]           = 0.0;
        P[(N_Z - 1) * N_phi + j]   = 0.0;
    }
}
"""

# ---------------------------------------------------------------------------
# Скомпилированные ядра (ленивая инициализация)
# ---------------------------------------------------------------------------
_rb_sor_kernel = None
_apply_bc_kernel = None


def get_rb_sor_kernel():
    """Возвращает скомпилированное CUDA-ядро Red-Black SOR (кэшируется)."""
    global _rb_sor_kernel
    if _rb_sor_kernel is None:
        _rb_sor_kernel = cp.RawKernel(_RB_SOR_KERNEL_CODE, "rb_sor_step")
    return _rb_sor_kernel


def get_apply_bc_kernel():
    """Возвращает скомпилированное CUDA-ядро граничных условий (кэшируется)."""
    global _apply_bc_kernel
    if _apply_bc_kernel is None:
        _apply_bc_kernel = cp.RawKernel(_APPLY_BC_KERNEL_CODE, "apply_bc")
    return _apply_bc_kernel
