# reynolds-solver

GPU-accelerated Reynolds equation solver for hydrodynamic bearings.

## Installation

```bash
pip install -e .
```

## Usage

```python
from reynolds_solver import solve_reynolds

# Static equation
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L)

# Dynamic equation
P, delta, n_iter = solve_reynolds(H, d_phi, d_Z, R, L,
                                   xprime=0.001, yprime=0.001)
```

## Performance (RTX 4090)

| Grid       | CPU Numba (s) | GPU SOR (s) | Speedup |
|------------|---------------|-------------|---------|
| 250x250    | 3.0           | 0.35        | 8x      |
| 500x500    | 32.0          | 0.50        | 65x     |
| 1000x1000  | ~300 (est)    | 2.2         | ~130x   |

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA
- CuPy (cupy-cuda12x)
