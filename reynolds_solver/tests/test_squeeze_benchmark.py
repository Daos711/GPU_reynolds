"""
Regression test for the Ausas 2009 squeeze benchmark.

Runs the full benchmark (N1=450, dt=6.6e-4, one oscillation period)
and checks the acceptance criteria from the TZ:
  [1] rupture nucleates near t ≈ 0.25
  [2] Σ_num tracks Σ_exact on the rupture phase (after the rupture
      transient has settled) within a few staircase cells
  [3] reformation begins after the peak cavitation step
  [4] P ≥ 0 and 0 ≤ θ ≤ 1 invariants hold
  [5] symmetry around x = 0.5 below the round-off floor

This is the regression test for the dynamic-Ausas formulas. If the
squeeze benchmark starts failing, the Ausas equation bookkeeping has
regressed.

Run:
    python -m reynolds_solver.tests.test_squeeze_benchmark
"""
import sys

from reynolds_solver.cavitation.ausas.benchmark_squeeze import (
    run_benchmark,
    evaluate_success,
)


def main():
    print("=" * 72)
    print("  Regression: Ausas 2009 squeeze benchmark")
    print("=" * 72)
    print()

    result = run_benchmark(verbose=True)
    passed = evaluate_success(result)

    print()
    if passed:
        print("  REGRESSION TEST: PASS")
        sys.exit(0)
    else:
        print("  REGRESSION TEST: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
