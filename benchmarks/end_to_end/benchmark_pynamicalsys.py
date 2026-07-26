# pyright: basic
"""
Basin stability scaling benchmark using pynamicalsys.

Benchmarks the basin_of_attraction workflow for the damped driven pendulum
across different N values. Supports incremental runs: existing results are loaded
and only missing N values are benchmarked. Delete the output JSON to force a full re-run.

Pendulum dynamics:
    dθ/dt = θ̇
    dθ̇/dt = -α·θ̇ + T - K·sin(θ)

Run with:
    uv run python -m benchmarks.end_to_end.benchmark_pynamicalsys
"""

import json
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from numba import njit
from numpy.typing import NDArray
from pynamicalsys import ContinuousDynamicalSystem

N_VALUES: list[int] = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
NUM_ROUNDS: int = 5

ALPHA: float = 0.1
TORQUE: float = 0.5
K: float = 1.0

RESULTS_PATH: Path = (
    Path(__file__).parent / "results" / "pynamicalsys_basin_stability_estimator_scaling.json"
)

# Expected basin stability values (from main_pendulum_case1.json, N=10000)
EXPECTED_FP: float = 0.152
EXPECTED_LC: float = 0.848
BS_TOLERANCE: float = 0.02


@njit
def pendulum_equations(
    t: float,  # noqa: ARG001
    state: NDArray[np.float64],
    params: NDArray[np.float64],
) -> NDArray[np.float64]:
    alpha, torque, k = params[0], params[1], params[2]
    dudt = np.zeros_like(state)
    dudt[0] = state[1]
    dudt[1] = -alpha * state[1] + torque - k * np.sin(state[0])
    return dudt


def generate_initial_conditions(n: int, seed: int = 42) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    theta_offset: float = np.arcsin(TORQUE / K)
    theta_min: float = -np.pi + theta_offset
    theta_max: float = np.pi + theta_offset
    theta_vals: NDArray[np.float64] = rng.uniform(theta_min, theta_max, n)
    theta_dot_vals: NDArray[np.float64] = rng.uniform(-10.0, 10.0, n)
    return np.column_stack([theta_vals, theta_dot_vals])


def create_system() -> ContinuousDynamicalSystem:
    ds = ContinuousDynamicalSystem(
        equations_of_motion=pendulum_equations,
        system_dimension=2,
        number_of_parameters=3,
    )
    ds.set_parameters([ALPHA, TORQUE, K])
    ds.integrator("rk45", atol=np.float64(1e-6), rtol=np.float64(1e-8))
    return ds


def load_existing_results() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"num_rounds": NUM_ROUNDS, "benchmarks": []}


def get_completed_n_values(existing: dict) -> set[int]:
    return {bench["N"] for bench in existing["benchmarks"]}


def save_results(existing: dict, new_benchmarks: list[dict]) -> None:
    all_benchmarks: list[dict] = existing["benchmarks"] + new_benchmarks
    all_benchmarks.sort(key=lambda b: b["N"])
    existing["benchmarks"] = all_benchmarks
    existing["num_rounds"] = NUM_ROUNDS

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Results saved to: {RESULTS_PATH}")


def validate_labels(labels: NDArray[np.int32], n: int) -> bool:
    """Validate basin fractions against expected pendulum results.

    :param labels: Integer label array returned by basin_of_attraction.
    :param n: Number of initial conditions used.
    :return: True if valid or N < 10000, False if validation fails.
    """
    if n < 10_000:
        return True

    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique) != 2:
        print(f"  VALIDATION FAILED: Expected 2 attractors, got {len(unique)}")
        return False

    fractions = sorted(counts / n)
    fp_ok = abs(fractions[0] - EXPECTED_FP) <= BS_TOLERANCE
    lc_ok = abs(fractions[1] - EXPECTED_LC) <= BS_TOLERANCE

    if not fp_ok or not lc_ok:
        print(f"  VALIDATION FAILED for N={n}")
        print(f"    Expected: FP≈{EXPECTED_FP}, LC≈{EXPECTED_LC} (tolerance ±{BS_TOLERANCE})")
        print(f"    Got:      {[round(f, 4) for f in fractions]}")
        return False

    print(f"  Validation passed: {[round(f, 4) for f in fractions]} ✓")
    return True


def run_benchmark(ds: ContinuousDynamicalSystem, n: int) -> dict | None:
    num_intersections: int = 100
    transient_time: float = 900.0
    sampling_time: float = 2 * np.pi
    eps: float = 10.0
    min_samples: int = 100

    round_times: list[float] = []

    for round_idx in range(1, NUM_ROUNDS + 1):
        u: NDArray[np.float64] = generate_initial_conditions(n, seed=round_idx)

        start: float = time.perf_counter()
        labels: NDArray[np.int32] = ds.basin_of_attraction(
            u=u,
            num_intersections=num_intersections,
            transient_time=transient_time,
            map_type="SM",
            sampling_time=sampling_time,
            eps=eps,
            min_samples=min_samples,
        )
        elapsed: float = time.perf_counter() - start
        round_times.append(elapsed)
        print(f"  Round {round_idx}/{NUM_ROUNDS}: {elapsed:.3f}s")

        if round_idx == 1 and not validate_labels(labels, n):
            return None

    return {
        "N": n,
        "round_times": round_times,
        "mean_time": mean(round_times),
        "std_time": stdev(round_times) if len(round_times) > 1 else 0.0,
        "min_time": min(round_times),
        "max_time": max(round_times),
    }


def main() -> None:
    existing: dict = load_existing_results()
    completed: set[int] = get_completed_n_values(existing)
    pending: list[int] = sorted([n for n in N_VALUES if n not in completed])

    if not pending:
        print(f"All N values already benchmarked. Delete {RESULTS_PATH} to re-run.")
        return

    print(f"Already completed: {sorted(completed)}")
    print(f"Pending N values: {pending}")

    ds: ContinuousDynamicalSystem = create_system()

    # Warmup run (Numba JIT compilation) - excluded from timing
    print("\nWarmup run (N=50, excluded from timing)...")
    u_warmup: NDArray[np.float64] = generate_initial_conditions(50, seed=0)
    ds.basin_of_attraction(
        u=u_warmup,
        num_intersections=100,
        transient_time=900.0,
        map_type="SM",
        sampling_time=2 * np.pi,
        eps=10.0,
        min_samples=100,
    )
    print("Warmup complete.\n")

    for n in pending:
        print(f"Benchmarking N={n} ({NUM_ROUNDS} rounds)...")
        result: dict | None = run_benchmark(ds, n)
        if result is None:
            print("  Stopping benchmark due to validation failure.")
            print("  Check eps / min_samples parameters and re-run.")
            return
        print(f"  Mean: {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
        existing["benchmarks"].append(result)
        existing["benchmarks"].sort(key=lambda b: b["N"])
        save_results(existing, [])


if __name__ == "__main__":
    main()
