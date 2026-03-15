# type: ignore
"""
Quick benchmark to compare ScipyParallelSolver with parallel vs serial processing.
Tests solving only 2 IVPs (the template initial conditions from pendulum case).
"""

import time

import torch

from case_studies.pendulum.pendulum_ode import PendulumODE, PendulumParams
from pybasin.solvers import ScipyParallelSolver


def benchmark_sklearn_solver():
    """
    Benchmark ScipyParallelSolver for 2 template IVPs.
    Compare parallel (n_jobs=-1) vs serial (n_jobs=1) processing.
    """
    print("\n" + "=" * 80)
    print("SKLEARN SOLVER BENCHMARK - 2 IVPs")
    print("=" * 80)

    # Setup pendulum ODE system
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    ode_system = PendulumODE(params)

    # Template initial conditions (2 IVPs)
    initial_conditions = torch.tensor(
        [
            [0.5, 0.0],  # FP: fixed point
            [2.7, 0.0],  # LC: limit cycle
        ],
        dtype=torch.float32,
    )

    print(f"\nInitial conditions:\n{initial_conditions}")
    print(f"Number of IVPs: {len(initial_conditions)}")

    # Solver configuration
    time_span = (0, 1000)
    n_steps = 25001  # 25 Hz equivalent: 25*1000 + 1 = 25001 points

    print(f"\nIntegration time span: {time_span}")
    print(f"Number of evaluation points: {n_steps}")

    # Test 1: Parallel processing
    print("\n" + "-" * 80)
    print("TEST 1: PARALLEL PROCESSING (n_jobs=-1)")
    print("-" * 80)

    solver_parallel = ScipyParallelSolver(
        t_span=time_span,
        t_steps=n_steps,
        n_jobs=-1,  # Use all available CPUs
        method="RK45",
        rtol=1e-8,
        atol=1e-6,
        cache_dir=None,  # Disable cache for accurate benchmark
    )

    start_time = time.perf_counter()
    with torch.no_grad():
        t_parallel, y_parallel = solver_parallel.integrate(ode_system, initial_conditions)
    parallel_time = time.perf_counter() - start_time

    print(f"\nParallel execution time: {parallel_time:.6f} seconds")
    print(f"Solution shape: {y_parallel.shape}")

    # Test 2: Serial processing
    print("\n" + "-" * 80)
    print("TEST 2: SERIAL PROCESSING (n_jobs=1)")
    print("-" * 80)

    solver_serial = ScipyParallelSolver(
        t_span=time_span,
        t_steps=n_steps,
        n_jobs=1,  # Force serial execution
        method="RK45",
        rtol=1e-8,
        atol=1e-6,
        cache_dir=None,  # Disable cache for accurate benchmark
    )

    start_time = time.perf_counter()
    with torch.no_grad():
        t_serial, y_serial = solver_serial.integrate(ode_system, initial_conditions)
    serial_time = time.perf_counter() - start_time

    print(f"\nSerial execution time: {serial_time:.6f} seconds")
    print(f"Solution shape: {y_serial.shape}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nParallel time: {parallel_time:.6f} seconds")
    print(f"Serial time:   {serial_time:.6f} seconds")
    print(f"Speedup:       {serial_time / parallel_time:.2f}x")
    print(f"Difference:    {abs(serial_time - parallel_time):.6f} seconds")

    # Verify results match
    max_diff = torch.abs(y_parallel - y_serial).max().item()
    print(f"\nMax solution difference: {max_diff:.2e} (should be ~0)")


def main():
    """Main function."""
    benchmark_sklearn_solver()


if __name__ == "__main__":
    main()
