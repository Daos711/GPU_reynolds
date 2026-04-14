"""
reynolds_solver.cavitation.elrod — compressible Elrod / Vijayaraghavan-Keith
cavitation solver (Manser 2019 formulation).

Solves the unified Reynolds equation for the fractional film content Θ
with a finite bulk modulus β̄:

    (1/12)·∂/∂θ[β̄·g·h³·∂Θ/∂θ]
    + (R/L)²·(1/12)·∂/∂Z[β̄·g·h³·∂Θ/∂Z]
        = (1/2)·∂(Θ·h)/∂θ

Pressure is recovered as P = β̄·g·ln(Θ), where g(Θ) is the switch
function (g=1 in full film Θ≥1, g=0 in cavitation Θ<1). The finite
bulk modulus creates the nonlinear P↔Θ coupling that amplifies the
micro-wedge effect in textured bearings (Manser 2019 a/b).

Unlike the Payvar-Salant unified-variable formulation (incompressible,
g ≡ P), this solver tracks density / Θ as the primary unknown and
recovers P a posteriori, which is the Elrod-Adams family.
"""
from reynolds_solver.cavitation.elrod.solver_cpu import (
    solve_elrod_compressible,
)

__all__ = ["solve_elrod_compressible"]
