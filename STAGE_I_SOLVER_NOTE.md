# Stage I — Solver-Side Deliverables

## What was done

### S1. Masked Payvar-Salant + piezoviscosity wired

`solve_payvar_salant_piezoviscous` now accepts and forwards
`dirichlet_mask` + `g_bc` to every internal `solve_payvar_salant_gpu`
call (initial solve, each PV outer iteration, and the zero-alpha
fallback path). The `NotImplementedError` gate in `solve_reynolds`
has been replaced with a direct pass-through.

**Working call:**

```python
from reynolds_solver import solve_reynolds

P, theta, residual, n_iter = solve_reynolds(
    H, d_phi, d_Z, R, L,
    cavitation="payvar_salant",
    dirichlet_mask=mask,    # (N_Z, N_phi) bool
    g_bc=p_feed_nd,         # scalar float (nondimensional)
    alpha_pv=alpha_eff,     # Pa⁻¹
    p_scale=p_scale,        # Pa
)
```

**Mandatory path:** GPU (cupy required for PV+PS; CPU fallback only
when `alpha_pv ≈ 0`).

**Backward compat:** `mask=None, g_bc=None, alpha_pv=None` → bit-for-bit
identical to pre-Stage-I code.

### S2. Thermal / oil helper API

```python
from reynolds_solver.thermal import (
    OilModel,               # frozen dataclass: A_w, B_w, rho, cp, alpha_pv_base, gamma
    fit_walther_two_point,  # (T1, nu1, T2, nu2) → OilModel
    mu_at_T_C,              # (T_C, model) → Pa·s (scalar/ndarray safe)
    alpha_at_T_C,           # (T_C, model, mode="constant") → Pa⁻¹
    global_static_target_C, # (T_in, P_loss, mdot, cp, gamma) → T_eff °C
    global_relax_step_C,    # (T_prev, T_target, dt, tau) → T_new °C
)
```

All functions are **stateless** — the pipeline owns thermal state.
The solver repo owns only property evaluation and single-ODE helpers.

### Runnable example

```python
import numpy as np
from dataclasses import replace
from reynolds_solver import solve_reynolds
from reynolds_solver.thermal import (
    fit_walther_two_point, mu_at_T_C, alpha_at_T_C,
    global_static_target_C,
)

# 1) Oil model from two viscosity points
oil = fit_walther_two_point(40.0, 68.0, 100.0, 11.0)
oil = replace(oil, rho_kg_m3=860.0, cp_J_kgK=2000.0,
              alpha_pv_base=2.2e-8, gamma_mix=0.5)

# 2) Effective viscosity at some temperature
T_eff = 90.0   # °C
eta = mu_at_T_C(T_eff, oil)
alpha_pv = alpha_at_T_C(T_eff, oil)

# 3) Build gap + feed mask (pipeline responsibility)
N_phi, N_Z = 200, 60
phi = np.linspace(0, 2*np.pi, N_phi)
Z = np.linspace(-1, 1, N_Z)
Phi, _ = np.meshgrid(phi, Z)
H = 1.0 + 0.6 * np.cos(Phi)
d_phi = float(phi[1] - phi[0])
d_Z = float(Z[1] - Z[0])
R, L = 0.035, 0.056
mask = np.zeros((N_Z, N_phi), dtype=bool)
mask[:, 0] = True   # example: pin phi=0 column
g_bc = 0.01          # nondimensional supply pressure

# 4) Solve
P, theta, res, n = solve_reynolds(
    H, d_phi, d_Z, R, L,
    cavitation="payvar_salant",
    dirichlet_mask=mask, g_bc=g_bc,
    alpha_pv=alpha_pv, p_scale=1e7,
)
print(f"maxP = {P.max():.3e}, cav_frac = {(theta < 0.999).mean():.3f}")
```

## Constraints / limitations

- **No 2D energy equation.** Temperature is a single scalar managed
  by the pipeline via `global_static_target_C` / `global_relax_step_C`.
- **No engine-specific geometry.** Mask construction is pipeline's job.
- **GPU required** for the `alpha_pv + mask` path (uses cupy).
- **alpha_at_T_C** currently only supports `mode="constant"`. A
  Roelands T-dependent model can be added later.
- **Thermal state** lives in the pipeline, not in the solver.

## Tests

```bash
python -m reynolds_solver.tests.test_stage_i_solver
python -m reynolds_solver.tests.test_payvar_salant_dirichlet
python -m reynolds_solver.tests.test_payvar_salant
```
