# Hydrodynamic force integration in JFO/Ausas — audit

Repo audited: `gpu-reynolds` only. The diesel runner ("Stage K2") lives in
`article-dump-truck`, which is **outside** the MCP scope of this agent —
questions whose answer lives there are flagged below.

---

## 1. Используется ли при интегрировании Fx/Fy маска `p >= p_cav`, или интегрируется всё поле давления?

**Маска не используется. Интегрируется всё поле P[1:-1, 1:-1].**

Канонический интегратор сил в gpu-reynolds — фьюзед CUDA-kernel
`forces_reduce_cos_sin`:

- `reynolds_solver/cavitation/ausas/kernels_dynamic.py:586-635`

Тело редукции (строки 608-618):

```cpp
double wx = 0.0;
double wy = 0.0;
for (int idx = tid; idx < total; idx += bs) {
    int ii = idx / N_j;
    int jj = idx - ii * N_j;
    int i = 1 + ii;
    int j = 1 + jj;
    double p_val = P[i * N_phi + j];   // <-- raw P, без маски
    wx += p_val * cos_phi[j];
    wy += p_val * sin_phi[j];
}
```

То же для стационарного пути (Stage 0, NumPy):

- `reynolds_solver/pipeline_gpu.py:83-87`

```python
def compute_forces(P, Phi_mesh, phi_1D, Z):
    Fx = np.trapezoid(np.trapezoid(P * np.cos(Phi_mesh), phi_1D, axis=1), Z)
    Fy = np.trapezoid(np.trapezoid(P * np.sin(Phi_mesh), phi_1D, axis=1), Z)
    return Fx, Fy
```

Никакого `(P >= p_cav)`-фильтра, никакой маски `theta >= 1`,
никакой `np.maximum(P, p_cav)` — суммируется всё.

---

## 2. Чему равен `p_cav` в production-команде Stage K2 (численно)?

**В коде gpu-reynolds нет именованной переменной `p_cav`.**
В Ausas-кернеле граница кавитации зашита численно как `0.0`:

- `reynolds_solver/cavitation/ausas/kernels_dynamic.py:145-149` (Jacobi)
- `reynolds_solver/cavitation/ausas/kernels_dynamic.py:277-282` (RB/SOR)

```cpp
if (P_relaxed >= 0.0) {
    P_cur  = P_relaxed;
    th_cur = 1.0;
} else {
    P_cur = 0.0;            // cavitation -> p clamped to 0 in solver
}
```

И при θ-обновлении (строки 167, 297):

```cpp
if (th_relaxed < 1.0) {
    if (th_relaxed < 0.0) th_relaxed = 0.0;
    th_cur = th_relaxed;
    P_cur  = 0.0;
}
```

То есть в **gauge gpu-reynolds** `p_cav ≡ 0` по построению. Это не
конфигурируемый параметр — это ноль литерал в .cu-коде.

> Конкретное численное значение `p_cav`, которое использует
> *production-команда Stage K2*, должен подтвердить дизель-агент по
> своему конфигу в `article-dump-truck`. Из gpu-reynolds на это
> ответить нельзя — solver принимает поле P в безразмерных единицах,
> диммесизация и сдвиг шкалы делается на стороне дизеля.

---

## 3. Где в коде задаётся `p_cav`, есть ли разница relative vs absolute?

**Именованной переменной `p_cav` в gpu-reynolds нет.**

Поиск по репо подтверждает: `grep -rn "p_cav\|p_cavitation\|cavitation_pressure"` находит только `experiments/diagnostics/diagnostic_jfo_splitting.py:308` (`sweep_cav_frac` — это про cavitation **fraction**, а не pressure).

В Stage 3 (`solve_ausas_journal_dynamic_gpu`) есть параметр **`p_a`** —
но это **не** `p_cav`, это аксиальный Dirichlet BC (supply pressure
на торце подшипника, z = B):

- `reynolds_solver/cavitation/ausas/solver_dynamic_gpu.py:1503`  `p_a: float = 0.0`
- `solver_dynamic_gpu.py:1601`  `"Z:    p = 0 at z = 0, p = p_a at z = B (feeding/supply)"`
- `solver_dynamic_gpu.py:1712,1987,2010,2308,2331`  передаётся как `p_zL=p_a`

**Gauge:** в gpu-reynolds используется **relative pressure**, отсчитываемое от
кавитационного уровня. В этой шкале `p_cav = 0`. Solver *никогда* не
оперирует абсолютным давлением. Дизель-агент должен сам делать
`p_abs = p_solver + p_cav_phys` если ему это нужно.

---

## 4. Производится ли усечение `p < p_cav` до `p_cav` при интегрировании сил?

**Явного усечения при интегрировании нет — берётся raw P из solver.**
Однако это эквивалентно усечению, потому что **solver сам уже клампит
P до ≥ 0 в каждом sweep** (см. строки 149/167/281/297 выше). После
последнего внутреннего sweep на выходе всегда `P ≥ 0` поэлементно — в
кавитированных ячейках `P = 0` строго.

То есть отдельной операции `np.maximum(P, p_cav)` или вычитания фона
перед `forces_reduce_cos_sin` — нет. Но и нужды в ней нет: гарантия
поэлементного `P ≥ 0` обеспечивается солвером.

Если дизель-сторона добавляет постпроцессинг, который раскладывает
solver-output в сетку с другим происхождением (ghosts / интерполяция /
сглаживание), там может появиться численный шум `< 0`. Это надо
проверить на стороне `article-dump-truck`.

---

## 5. Разница в маске между силами и матрицей сходимости JFO?

**Нет различия. Force integral и residual считаются над одним и тем же
интерьерным slice без масок.**

Residual:
- `solver_dynamic_gpu.py:702-712`  (одношаговый путь):
  `dP_arr = (P_new - P_old)[1:-1, 1:-1]; dth_arr = (theta_new - theta_old)[1:-1, 1:-1]`
- `solver_dynamic_gpu.py:851-861`  (Stage 2 / RB-путь): то же самое.

Forces (см. п. 1) тоже над `P[1:-1, 1:-1]`.

`theta` (cavitation fraction в ячейке) **используется внутри
solver-kernel** для выбора ветки уравнения (полная плёнка vs кавитация),
но **не** входит в force-интегра́л и не входит в residual-норму
напрямую (residual считает поточечную разность θ-итераций — это
*не* маска).

Отсюда: единственное место, где `theta` участвует как маска — диагностика
`cav_frac = mean(theta < 1 - 1e-6)` (`solver_dynamic_gpu.py:884-885`),
которая нигде не подаётся в физику.

---

## Итог

| Вопрос | Ответ | Файл:строка |
| --- | --- | --- |
| Маска при интегрировании Fx/Fy | нет | `kernels_dynamic.py:586-635`, `pipeline_gpu.py:83-87` |
| p_cav численно | `0.0` зашито в .cu (gpu-reynolds gauge) | `kernels_dynamic.py:145, 149, 167, 281, 297` |
| p_cav как переменная | отсутствует; есть `p_a` (другой смысл — supply BC) | `solver_dynamic_gpu.py:1503, 1601` |
| relative vs absolute | relative-to-cav; солвер всегда оперирует `p ≥ 0` | (по построению kernel) |
| Усечение `p < p_cav` при F | нет, но и не нужно — клампится в solver | `kernels_dynamic.py:149, 167, 281, 297` |
| Маска в residual vs F | нет различия, обе поверх `[1:-1,1:-1]` без маски | `solver_dynamic_gpu.py:702-712, 851-861, kernels_dynamic.py:586-635` |

**Что осталось проверить в `article-dump-truck` (вне scope этого
агента):**

1. **Численное значение** `p_cav` в дизельном Stage K2 конфиге — есть
   ли там сдвиг абсолют → relative перед вызовом `ausas_unsteady_one_step_gpu`.
2. **Диззельный собственный force-integral** — если diesel runner
   считает Fx/Fy сам (а не вызывает `solve_ausas_journal_dynamic_gpu`,
   который уже включает forces_reduce kernel), в его коде стоит
   проверить:
   * нет ли отдельного `p_cav` в конфиге как фон / нуля шкалы;
   * не делает ли он `(P - p_cav).clip(0, ∞)` или `(P >= p_cav)` перед
     суммой;
   * над каким slice он суммирует (с ghosts или без);
   * учитывается ли θ как маска (т.е. не считается ли F только над
     full-film cells).
