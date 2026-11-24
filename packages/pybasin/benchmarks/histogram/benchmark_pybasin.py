# pyright: basic
"""
Histogram-grouping scaling benchmark using pybasin's `HistogramClusterer`.

Benchmarks the Hénon map + mean-state featurizer + `HistogramClusterer` workflow
(same system, same bin edges, same fraction-validation gate as
`benchmark_julia.jl`) across different N values, on CPU and (if available) CUDA.
Supports incremental runs: existing results are loaded and only missing N values
are benchmarked. Delete the output JSON to force a full re-run.

Run with:
    uv run python -m packages.pybasin.benchmarks.histogram.benchmark_pybasin
"""

import json
import time
from pathlib import Path
from statistics import mean, stdev

import torch
from case_studies.henon_map.henon_feature_extractor import henon_mean_state_features
from case_studies.henon_map.henon_map_ode import HENON_DEFAULT_P, HenonMapODE, HenonMapParams

from pybasin.predictors.histogram_clusterer import HistogramClusterer
from pybasin.solvers import DiscreteMapSolver

N_VALUES: list[int] = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000]
NUM_ROUNDS: int = 5

RESULTS_DIR: Path = Path(__file__).parent / "results"

BIN_EDGES: list[float] = [-10.0, 40.0, 90.0, 140.0, 190.0, 240.0]
N_STEPS: int = 200  # transient + recorded steps combined, matching a fixed-length trajectory

EXPECTED_FRAC_BOUNDED: float = 0.45
EXPECTED_FRAC_DIVERGENT: float = 0.55
FRAC_RTOL: float = 0.15


def generate_initial_conditions(n: int, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.empty(n, 2, dtype=torch.float64, device=device).uniform_(
        -2.0, 2.0, generator=generator
    )


def run_pipeline(n: int, seed: int, device: str) -> torch.Tensor:
    """Full Hénon -> featurize -> HistogramClusterer pipeline; returns bin labels."""
    y0 = generate_initial_conditions(n, seed, device)
    params: HenonMapParams = {"p1": HENON_DEFAULT_P[0], "p2": HENON_DEFAULT_P[1]}
    ode_system = HenonMapODE(params)
    solver = DiscreteMapSolver(t_steps=N_STEPS, device=device, cache_dir=None)
    _t, trajectory = solver.integrate(ode_system, y0)  # (n, 2, N_STEPS)
    features = henon_mean_state_features(trajectory)
    grouper = HistogramClusterer(bin_edges=[BIN_EDGES, BIN_EDGES])
    return grouper.fit_predict(features)


def validate_labels(labels: torch.Tensor, n: int) -> bool:
    if n < 10_000:
        return True

    unique_labels = torch.unique(labels[labels >= 0])
    if unique_labels.numel() != 2:
        print(f"  VALIDATION FAILED: Expected 2 occupied bins, got {unique_labels.numel()}")
        return False

    fractions = sorted(
        float((labels == label).sum().item()) / n for label in unique_labels.tolist()
    )
    lo_ok = abs(fractions[0] - EXPECTED_FRAC_BOUNDED) / EXPECTED_FRAC_BOUNDED <= FRAC_RTOL
    hi_ok = abs(fractions[1] - EXPECTED_FRAC_DIVERGENT) / EXPECTED_FRAC_DIVERGENT <= FRAC_RTOL
    if not lo_ok or not hi_ok:
        print(f"  VALIDATION FAILED for N={n}")
        print(
            f"    Expected: ~{EXPECTED_FRAC_BOUNDED} / ~{EXPECTED_FRAC_DIVERGENT} "
            f"(rtol {FRAC_RTOL})"
        )
        print(f"    Got:      {[round(f, 4) for f in fractions]}")
        return False

    print(f"  Validation passed: {[round(f, 4) for f in fractions]} ✓")
    return True


def results_path(device: str) -> Path:
    return RESULTS_DIR / f"pybasin_histogram_{device}.json"


def load_existing_results(device: str) -> dict:
    path = results_path(device)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"num_rounds": NUM_ROUNDS, "benchmarks": []}


def get_completed_n_values(existing: dict) -> set[int]:
    return {bench["N"] for bench in existing["benchmarks"]}


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
    completed = get_completed_n_values(existing)
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
