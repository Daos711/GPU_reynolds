"""
reynolds_solver.cavitation — mass-conserving cavitation solvers.

Submodules:
    ausas          — dynamic Ausas (2009) JFO solver (validated by the
                     squeeze benchmark). Use for transient problems.
    payvar_salant  — Payvar-Salant steady mass-conserving JFO solver
                     (incompressible unified variable g).
    elrod          — compressible Elrod / Vijayaraghavan-Keith
                     (Manser 2019) solver with finite bulk modulus β̄.
    legacy         — old non-working JFO solvers kept for reference /
                     diagnostics only.
"""
