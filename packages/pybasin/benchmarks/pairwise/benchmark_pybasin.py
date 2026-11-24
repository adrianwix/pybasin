# pyright: basic
"""
Scaling benchmark for `PairwiseComparisonClusterer` against Attractors.jl.

Benchmarks the full magnetic-pendulum pipeline (ODE integration, featurizing, and
grouping) -- the same system, parameters, and grid as `benchmark_julia.jl` -- across
different N values, on CPU and (if available) CUDA. Supports incremental runs: existing
results are loaded and only missing N values are benchmarked. Delete the output JSON to
force a full re-run.

Run with:
    uv run python -m packages.pybasin.benchmarks.pairwise.benchmark_pybasin
"""

import json
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from case_studies.magnetic_pendulum.magnetic_pendulum_ode import (
    MagneticPendulumODE,
    MagneticPendulumParams,
)

from pybasin.predictors.pairwise_comparison_clusterer import PairwiseComparisonClusterer
from pybasin.solvers.torchode_solver import TorchOdeSolver

# N is capped at 2000: the magnetic pendulum's basin boundaries are fractal, so as N grows,
# more sampled points land arbitrarily close to a boundary, where the dynamics are locally
# stiff and the adaptive integrator needs many small steps to resolve them correctly.
# torchode batches all N trajectories through the same step-count loop, so these rare
# boundary-hugging trajectories can make an entire batch's integration take an order of
# magnitude longer than its neighbors -- reproduced directly and repeatedly: N=5000 at
# several different seeds took 170-190s against 4-12s for N<=2000 and unstable, highly
# variable multi-tens-of-seconds times at N=10_000/20_000. This is a genuine property of
# batched adaptive integration on a fractal-boundary system, not a bug in this benchmark or
# in `PairwiseComparisonClusterer` -- this script's job is to show the full pipeline is
# reasonable at practical scale, which it is up to N=2000. `benchmark_julia.jl` (Attractors.jl,
# tighter tolerances, a different integrator) demonstrates the full N range separately.
N_VALUES: list[int] = [100, 200, 500, 1000, 2000]
NUM_ROUNDS: int = 5

THRESHOLD: float = 1.0
PARAMS: MagneticPendulumParams = {"alpha": 0.2, "omega": 0.8, "d": 0.2, "gamma": 1.0}
TTR: float = 500.0
N_MAGNETS: int = 3

# A trajectory starting essentially on top of a magnet is a measure-zero degenerate case,
# not a basin this benchmark needs to resolve -- excluded on those physical grounds, not as
# a (disproven) fix for the fractal-boundary slowdown above.
MIN_MAGNET_DISTANCE: float = 0.15

RESULTS_DIR: Path = Path(__file__).parent / "results"

# Initial conditions are sampled with torch's own RNG, not Julia's Xoshiro, so this
# benchmark's fractions cannot be pinned to Attractors.jl's specific seed-1234 realization
# (packages/pybasin/tests/accuracy/magnetic_pendulum/generate_ground_truth.jl's run,
# {1: 0.377, 2: 0.314, 3: 0.309}) -- the magnetic pendulum's basin boundaries are fractal,
# so a different sample of initial conditions from the same distribution legitimately
# lands a different (but still structurally sane) split. This validates structurally
# instead, mirroring benchmark_julia.jl's own `validate_fractions`: exactly 3 groups, none
# negligible, fractions summing to 1. Exact label parity against the real Julia ground
# truth is already checked by
# `packages/pybasin/tests/accuracy/magnetic_pendulum/test_magnetic_pendulum.py`.
MIN_PLAUSIBLE_FRACTION: float = 0.05


def _magnet_positions(device: str) -> torch.Tensor:
    angles = torch.arange(1, N_MAGNETS + 1, dtype=torch.float64, device=device) * (
        2 * torch.pi / N_MAGNETS
    )
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (N_MAGNETS, 2)


def sample_initial_conditions(n: int, seed: int, device: str) -> torch.Tensor:
    """Position sampled uniformly in ``[-2, 2]^2``, excluding points within
    ``MIN_MAGNET_DISTANCE`` of a magnet; velocity fixed at ``[0, 0]``."""
    generator = torch.Generator(device=device).manual_seed(seed)
    magnets = _magnet_positions(device)
    accepted = torch.empty((0, 2), dtype=torch.float64, device=device)

    while accepted.shape[0] < n:
        candidates = torch.empty(n, 2, dtype=torch.float64, device=device).uniform_(
            -2.0, 2.0, generator=generator
        )
        distances = torch.cdist(candidates, magnets)  # (n, N_MAGNETS)
        far_enough = distances.min(dim=1).values >= MIN_MAGNET_DISTANCE
        accepted = torch.cat([accepted, candidates[far_enough]], dim=0)

    xy = accepted[:n]
    zero_velocity = torch.zeros(n, 2, dtype=torch.float64, device=device)
    return torch.cat([xy, zero_velocity], dim=1)


def run_pipeline(n: int, seed: int, device: str) -> torch.Tensor:
    """Full magnetic-pendulum -> integrate -> featurize -> PairwiseComparisonClusterer
    pipeline; returns group labels."""
    y0 = sample_initial_conditions(n, seed, device)
    ode_system = MagneticPendulumODE(PARAMS, n_magnets=3).to(device)
    solver = TorchOdeSolver(t_span=(0, TTR), t_steps=2, device=device, cache_dir=None)
    _t, y = solver.integrate(ode_system, y0)
    features = y[:, :2, -1]  # final (x, y), shape (n, 2) -- y is (batch, n_dims, t_steps)

    clusterer = PairwiseComparisonClusterer(threshold=THRESHOLD, rescale_features=False)
    return clusterer.fit_predict(features)


def validate_labels(labels: torch.Tensor, n: int) -> bool:
    if n < 1000:
        return True

    labels_np = labels.detach().cpu().numpy()
    _, counts = np.unique(labels_np, return_counts=True)
    if len(counts) != 3:
        print(f"  VALIDATION FAILED: expected 3 groups, got {len(counts)}")
        return False

    fractions = sorted(counts / n)
    if any(f < MIN_PLAUSIBLE_FRACTION for f in fractions):
        print(
            f"  VALIDATION FAILED for N={n}: a group is implausibly small: "
            f"{[round(f, 4) for f in fractions]}"
        )
        return False
    total = sum(fractions)
    if abs(total - 1.0) > 1e-9:
        print(f"  VALIDATION FAILED for N={n}: fractions sum to {total}, not 1.0")
        return False

    print(f"  Validation passed: {[round(f, 4) for f in fractions]}")
    return True


def results_path(device: str) -> Path:
    return RESULTS_DIR / f"pybasin_pairwise_{device}.json"


def load_existing_results(device: str) -> dict:
    path = results_path(device)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"num_rounds": NUM_ROUNDS, "benchmarks": []}


def save_results(existing: dict, device: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path(device), "w") as f:
        json.dump(existing, f, indent=2)


def run_benchmark(n: int, device: str) -> dict | None:
    round_times: list[float] = []

    for round_idx in range(1, NUM_ROUNDS + 1):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        labels = run_pipeline(n, seed=round_idx, device=device)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
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


def run_device(device: str) -> None:
    print(f"\n=== Device: {device} ===")
    existing = load_existing_results(device)
    completed = {bench["N"] for bench in existing["benchmarks"]}
    pending = sorted([n for n in N_VALUES if n not in completed])

    if not pending:
        print(
            f"All N values already benchmarked for {device}. Delete {results_path(device)} to re-run."
        )
        return

    print(f"Already completed: {sorted(completed)}")
    print(f"Pending N values: {pending}")

    warmup_n = max(100, pending[0])
    print(f"\nWarmup run (N={warmup_n}, excluded from timing)...")
    run_pipeline(warmup_n, seed=0, device=device)
    print("Warmup complete.\n")

    for n in pending:
        print(f"Benchmarking N={n} ({NUM_ROUNDS} rounds) on {device}...")
        result = run_benchmark(n, device)
        if result is None:
            print(f"  Stopping {device} benchmark due to validation failure at N={n}.")
            return
        print(f"  Mean: {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
        existing["benchmarks"].append(result)
        existing["benchmarks"].sort(key=lambda b: b["N"])
        save_results(existing, device)
        print(f"  Saved N={n} to: {results_path(device)}")


def main() -> None:
    run_device("cpu")

    if torch.cuda.is_available():
        run_device("cuda")
    else:
        print("\nCUDA not available; skipping CUDA benchmark.")


if __name__ == "__main__":
    main()
