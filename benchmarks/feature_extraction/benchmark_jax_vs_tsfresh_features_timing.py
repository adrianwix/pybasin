# pyright: basic
"""
Benchmark to compare JAX vs tsfresh feature extraction timing.

This script times each feature in both JAX and tsfresh to compare performance.
It uses the same arguments for both implementations to ensure a fair comparison.

Feature Counts:
- MINIMAL_BATCH_FEATURES: 41 features (default, fast benchmarking)
- --all: JAX_COMPREHENSIVE_FC_PARAMETERS (JAX) vs EfficientFCParameters (tsfresh)
  Both exclude sample_entropy and approximate_entropy.

Usage:
    uv run python benchmarks/benchmark_jax_features_timing.py --individual-only
    uv run python benchmarks/benchmark_jax_features_timing.py --batch-only
    uv run python benchmarks/benchmark_jax_features_timing.py --batch-only --gpu
    uv run python benchmarks/benchmark_jax_features_timing.py --batch-only --batches=10000
    uv run python benchmarks/benchmark_jax_features_timing.py --batch-only --all
"""

import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress pandas FutureWarning from tsfresh
warnings.filterwarnings("ignore", category=FutureWarning, module="tsfresh")

# Parse --gpu flag BEFORE importing jax (env vars must be set first)
USE_GPU = "--gpu" in sys.argv
if not USE_GPU:
    os.environ["JAX_PLATFORMS"] = "cpu"

# Disable internal multithreading for numerical libraries
# This prevents thread contention when using multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import csv
import time
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Suppress numpy RankWarning from polyfit in tsfresh (must be after numpy import)
warnings.filterwarnings("ignore", category=np.exceptions.RankWarning)
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import feature_calculators as fc

# Import pendulum ODE system
from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE, PendulumParams
from pybasin.feature_extractors.jax.jax_feature_calculators import (
    ALL_FEATURE_FUNCTIONS,
    JAX_COMPREHENSIVE_FC_PARAMETERS,
)
from pybasin.sampler import GridSampler
from pybasin.solvers import JaxSolver

# Enable persistent compilation cache to speed up subsequent runs
cache_dir = Path(__file__).parent / "cache" / "jax_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(cache_dir))

# Parse --all flag
USE_ALL_FEATURES = "--all" in sys.argv

# Parse --multiprocessing flag
USE_MULTIPROCESSING = "--multiprocessing" in sys.argv


# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_sample(
    n_timesteps: int,
    n_batches: int,
    time_steady: float = 900.0,
    total_time: float = 1000.0,
) -> jax.Array:
    """Generate pendulum ODE trajectories for benchmarking.

    Only returns the steady-state portion of the trajectory (t >= time_steady).

    Args:
        n_timesteps: Number of integration steps (evenly spaced in [0, total_time])
        n_batches: Number of batches (trajectories)
        time_steady: Time threshold for filtering transients (time, not step)
        total_time: Total integration time

    Returns:
        JAX array of shape (steady_steps, n_batches, 2) where steady_steps
        is the number of points with t >= time_steady
    """
    print(f"  Generating {n_batches} pendulum trajectories...")

    # Define pendulum parameters
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    ode_system = PendulumJaxODE(params)

    # Create sampler with pendulum-appropriate limits (GridSampler is deterministic)
    sampler = GridSampler(
        min_limits=[-np.pi + np.arcsin(params["T"] / params["K"]), -10.0],
        max_limits=[np.pi + np.arcsin(params["T"] / params["K"]), 10.0],
        device="cpu",
    )

    # Sample initial conditions
    y0 = sampler.sample(n_batches)

    solver = JaxSolver(
        t_span=(0, total_time),
        t_steps=n_timesteps,
        device="cpu",
        rtol=1e-8,
        atol=1e-6,
    )

    # Integrate ODE system
    t, y = solver.integrate(ode_system, y0)

    # Filter to steady state portion (t >= time_steady)
    steady_mask = t >= time_steady
    y_steady = y[steady_mask, :, :]

    print(f"  Generated trajectory shape: {y_steady.shape} (t >= {time_steady})")
    return jnp.array(y_steady)


# =============================================================================
# MINIMAL FEATURE SET FOR FAST BATCH BENCHMARKS
# =============================================================================
# Subset of ~50 simpler/faster features (excludes slow features like CWT, AR, etc.)
# Format: {jax_name: kwargs} - None means no parameters
MINIMAL_BATCH_FEATURES: dict[str, dict | None] = {
    # MINIMAL_FEATURES (10)
    "sum_values": None,
    "median": None,
    "mean": None,
    "length": None,
    "standard_deviation": None,
    "variance": None,
    "root_mean_square": None,
    "maximum": None,
    "absolute_maximum": None,
    "minimum": None,
    # SIMPLE_STATISTICS_FEATURES (4)
    "abs_energy": None,
    "kurtosis": None,
    "skewness": None,
    "variation_coefficient": None,
    # CHANGE_FEATURES (4)
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "mean_change": None,
    "mean_second_derivative_central": None,
    # COUNTING_FEATURES (2)
    "count_above_mean": None,
    "count_below_mean": None,
    # BOOLEAN_FEATURES (4)
    "has_duplicate": None,
    "has_duplicate_max": None,
    "has_duplicate_min": None,
    "has_variance_larger_than_standard_deviation": None,
    # LOCATION_FEATURES (4)
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_maximum": None,
    "last_location_of_minimum": None,
    # STREAK_FEATURES (2)
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    # ENTROPY_FEATURES (2)
    "fourier_entropy": {"bins": 10},
    "cid_ce": {"normalize": True},
    # REOCCURRENCE_FEATURES (5)
    "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
    "percentage_of_reoccurring_values_to_all_values": None,
    "sum_of_reoccurring_data_points": None,
    "sum_of_reoccurring_values": None,
    "ratio_value_number_to_time_series_length": None,
    # ADVANCED_FEATURES (4)
    "benford_correlation": None,
    "mean_n_absolute_max": {"number_of_maxima": 1},
    "ratio_beyond_r_sigma": {"r": 1.0},
    "symmetry_looking": {"r": 0.1},
}


# =============================================================================
# FEATURE CONFIGURATIONS FOR INDIVIDUAL BENCHMARKS
# =============================================================================
# Each entry: (jax_func_name, jax_kwargs, tsfresh_func_name, tsfresh_kwargs)
# We use ONE representative parameter set per feature (not all permutations)
# This gives ~70 individual feature comparisons

INDIVIDUAL_FEATURE_CONFIGS: list[tuple[str, dict, str, dict]] = [
    # === NO-PARAMETER FEATURES (36 features) ===
    ("sum_values", {}, "sum_values", {}),
    ("median", {}, "median", {}),
    ("mean", {}, "mean", {}),
    ("length", {}, "length", {}),
    ("standard_deviation", {}, "standard_deviation", {}),
    ("variance", {}, "variance", {}),
    ("root_mean_square", {}, "root_mean_square", {}),
    ("maximum", {}, "maximum", {}),
    ("absolute_maximum", {}, "absolute_maximum", {}),
    ("minimum", {}, "minimum", {}),
    ("abs_energy", {}, "abs_energy", {}),
    ("kurtosis", {}, "kurtosis", {}),
    ("skewness", {}, "skewness", {}),
    ("variation_coefficient", {}, "variation_coefficient", {}),
    ("absolute_sum_of_changes", {}, "absolute_sum_of_changes", {}),
    ("mean_abs_change", {}, "mean_abs_change", {}),
    ("mean_change", {}, "mean_change", {}),
    ("mean_second_derivative_central", {}, "mean_second_derivative_central", {}),
    ("count_above_mean", {}, "count_above_mean", {}),
    ("count_below_mean", {}, "count_below_mean", {}),
    ("has_duplicate", {}, "has_duplicate", {}),
    ("has_duplicate_max", {}, "has_duplicate_max", {}),
    ("has_duplicate_min", {}, "has_duplicate_min", {}),
    (
        "has_variance_larger_than_standard_deviation",
        {},
        "variance_larger_than_standard_deviation",
        {},
    ),
    ("first_location_of_maximum", {}, "first_location_of_maximum", {}),
    ("first_location_of_minimum", {}, "first_location_of_minimum", {}),
    ("last_location_of_maximum", {}, "last_location_of_maximum", {}),
    ("last_location_of_minimum", {}, "last_location_of_minimum", {}),
    ("longest_strike_above_mean", {}, "longest_strike_above_mean", {}),
    ("longest_strike_below_mean", {}, "longest_strike_below_mean", {}),
    (
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        {},
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        {},
    ),
    (
        "percentage_of_reoccurring_values_to_all_values",
        {},
        "percentage_of_reoccurring_values_to_all_values",
        {},
    ),
    ("sum_of_reoccurring_data_points", {}, "sum_of_reoccurring_data_points", {}),
    ("sum_of_reoccurring_values", {}, "sum_of_reoccurring_values", {}),
    (
        "ratio_value_number_to_time_series_length",
        {},
        "ratio_value_number_to_time_series_length",
        {},
    ),
    ("benford_correlation", {}, "benford_correlation", {}),
    # === PARAMETERIZED FEATURES (one config each) ===
    # Time reversal / lag features
    (
        "time_reversal_asymmetry_statistic",
        {"lag": 1},
        "time_reversal_asymmetry_statistic",
        {"lag": 1},
    ),
    ("c3", {"lag": 1}, "c3", {"lag": 1}),
    # Complexity
    ("cid_ce", {"normalize": True}, "cid_ce", {"normalize": True}),
    # Symmetry / distribution
    ("symmetry_looking", {"r": 0.1}, "symmetry_looking", {"param": [{"r": 0.1}]}),
    ("large_standard_deviation", {"r": 0.25}, "large_standard_deviation", {"r": 0.25}),
    ("quantile", {"q": 0.5}, "quantile", {"q": 0.5}),
    # Autocorrelation
    ("autocorrelation", {"lag": 1}, "autocorrelation", {"lag": 1}),
    (
        "agg_autocorrelation",
        {"f_agg": "mean", "maxlag": 40},
        "agg_autocorrelation",
        {"param": [{"f_agg": "mean", "maxlag": 40}]},
    ),
    ("partial_autocorrelation", {"lag": 1}, "partial_autocorrelation", {"param": [{"lag": 1}]}),
    # Peaks
    ("number_cwt_peaks", {"max_width": 5}, "number_cwt_peaks", {"n": 5}),
    ("number_peaks", {"n": 3}, "number_peaks", {"n": 3}),
    # Entropy
    ("binned_entropy", {"max_bins": 10}, "binned_entropy", {"max_bins": 10}),
    ("fourier_entropy", {"bins": 10}, "fourier_entropy", {"bins": 10}),
    (
        "permutation_entropy",
        {"tau": 1, "dimension": 3},
        "permutation_entropy",
        {"tau": 1, "dimension": 3},
    ),
    ("lempel_ziv_complexity", {"bins": 2}, "lempel_ziv_complexity", {"bins": 2}),
    # Index/mass
    ("index_mass_quantile", {"q": 0.5}, "index_mass_quantile", {"param": [{"q": 0.5}]}),
    # FFT / frequency
    (
        "fft_coefficient",
        {"coeff": 0, "attr": "abs"},
        "fft_coefficient",
        {"param": [{"coeff": 0, "attr": "abs"}]},
    ),
    (
        "fft_aggregated",
        {"aggtype": "centroid"},
        "fft_aggregated",
        {"param": [{"aggtype": "centroid"}]},
    ),
    ("spkt_welch_density", {"coeff": 2}, "spkt_welch_density", {"param": [{"coeff": 2}]}),
    (
        "cwt_coefficients",
        {"widths": (2,), "coeff": 0, "w": 2},
        "cwt_coefficients",
        {"param": [{"widths": (2,), "coeff": 0, "w": 2}]},
    ),
    # AR / trend
    ("ar_coefficient", {"coeff": 0, "k": 10}, "ar_coefficient", {"param": [{"coeff": 0, "k": 10}]}),
    ("linear_trend", {"attr": "slope"}, "linear_trend", {"param": [{"attr": "slope"}]}),
    (
        "linear_trend_timewise",
        {"attr": "slope"},
        "linear_trend_timewise",
        {"param": [{"attr": "slope"}]},
    ),
    (
        "agg_linear_trend",
        {"attr": "slope", "chunk_size": 10, "f_agg": "mean"},
        "agg_linear_trend",
        {"param": [{"attr": "slope", "chunk_len": 10, "f_agg": "mean"}]},
    ),
    (
        "augmented_dickey_fuller",
        {"attr": "teststat"},
        "augmented_dickey_fuller",
        {"param": [{"attr": "teststat"}]},
    ),
    # Change quantiles
    (
        "change_quantiles",
        {"ql": 0.0, "qh": 0.5, "isabs": False, "f_agg": "mean"},
        "change_quantiles",
        {"ql": 0.0, "qh": 0.5, "isabs": False, "f_agg": "mean"},
    ),
    # Counting with threshold
    ("count_above", {"t": 0}, "count_above", {"t": 0}),
    ("count_below", {"t": 0}, "count_below", {"t": 0}),
    ("number_crossing_m", {"m": 0}, "number_crossing_m", {"m": 0}),
    # Energy / ratio
    (
        "energy_ratio_by_chunks",
        {"num_segments": 10, "segment_focus": 0},
        "energy_ratio_by_chunks",
        {"param": [{"num_segments": 10, "segment_focus": 0}]},
    ),
    ("ratio_beyond_r_sigma", {"r": 1.0}, "ratio_beyond_r_sigma", {"r": 1.0}),
    # Value counting
    ("value_count", {"value": 0}, "value_count", {"value": 0}),
    ("range_count", {"min_val": -1, "max_val": 1}, "range_count", {"min": -1, "max": 1}),
    # Advanced/physics
    (
        "friedrich_coefficients",
        {"coeff": 0, "m": 3, "r": 30},
        "friedrich_coefficients",
        {"param": [{"coeff": 0, "m": 3, "r": 30}]},
    ),
    ("max_langevin_fixed_point", {"m": 3, "r": 30}, "max_langevin_fixed_point", {"m": 3, "r": 30}),
    (
        "mean_n_absolute_max",
        {"number_of_maxima": 3},
        "mean_n_absolute_max",
        {"number_of_maxima": 3},
    ),
]


def time_jax_feature(func, x_jax, kwargs):
    """Time a JAX feature function.

    Returns:
        Tuple of (warmup_time, post_warmup_time, error_msg)
        warmup_time includes JIT compilation
        post_warmup_time is a single post-warmup run
    """
    try:
        # Warmup (includes JIT compilation)
        t0_warmup = time.perf_counter()
        result = func(x_jax, **kwargs)
        jax.block_until_ready(result)
        warmup_time = time.perf_counter() - t0_warmup

        # Single post-warmup run
        t0 = time.perf_counter()
        result = func(x_jax, **kwargs)
        jax.block_until_ready(result)
        run_time = time.perf_counter() - t0

        return warmup_time, run_time, None
    except Exception as e:
        return None, None, str(e)


def _tsfresh_worker(args):
    """Worker function for multiprocessing tsfresh feature extraction."""
    func_name, x_series, kwargs = args
    tsfresh_fn = getattr(fc, func_name)
    if "param" in kwargs:
        return list(tsfresh_fn(x_series, kwargs["param"]))
    elif kwargs:
        return tsfresh_fn(x_series, **kwargs)
    else:
        return tsfresh_fn(x_series)


def time_tsfresh_feature(func_name, x_np_batches, kwargs, n_batches, use_multiprocessing=False):
    """Time a tsfresh feature function.

    Args:
        func_name: Name of the tsfresh feature function
        x_np_batches: Array of shape (n_timesteps, n_batches) with all series
        kwargs: Arguments for the feature function
        n_batches: Number of series to process
        use_multiprocessing: If True, use multiprocessing pool
    """
    if not hasattr(fc, func_name):
        return None, f"Function {func_name} not found in tsfresh"

    tsfresh_fn = getattr(fc, func_name)

    try:
        if use_multiprocessing:
            work_items = [(func_name, x_np_batches[:, i], kwargs) for i in range(n_batches)]
            n_workers = os.cpu_count() or 1
            t0 = time.perf_counter()
            with mp.Pool(n_workers) as pool:
                _ = pool.map(_tsfresh_worker, work_items)
            run_time = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            for i in range(n_batches):
                x_series = x_np_batches[:, i]
                if "param" in kwargs:
                    _ = list(tsfresh_fn(x_series, kwargs["param"]))
                elif kwargs:
                    _ = tsfresh_fn(x_series, **kwargs)
                else:
                    _ = tsfresh_fn(x_series)
            run_time = time.perf_counter() - t0

        return run_time, None
    except Exception as e:
        return None, str(e)


def run_individual_benchmark(n_timesteps: int, n_batches: int):
    """Run individual feature benchmarks comparing JAX vs tsfresh."""
    print("\n" + "=" * 100)
    print("INDIVIDUAL FEATURE BENCHMARK")
    print("=" * 100)

    device = jax.devices()[0]
    n_states = 2  # Pendulum has 2 states
    total_series = n_batches * n_states

    print(f"\nDevice: {device}")
    print(
        f"Data: {n_timesteps} timesteps, {n_batches} batches x {n_states} states = {total_series} series"
    )
    mp_str = f"multiprocessing ({os.cpu_count()} workers)" if USE_MULTIPROCESSING else "sequential"
    print(f"JAX processes all {total_series} series at once, tsfresh uses {mp_str}")

    # Create test data from pendulum ODE
    x_jax = generate_sample(n_timesteps, n_batches)
    # Reshape to (n_timesteps, total_series) for tsfresh - flatten batch and state dims
    x_np_batches = np.array(x_jax.reshape(x_jax.shape[0], -1))
    actual_timesteps = x_np_batches.shape[0]
    actual_series = x_np_batches.shape[1]
    print(f"Actual data shape: {actual_timesteps} timesteps x {actual_series} series")

    # Setup CSV file for incremental results
    csv_path = Path(__file__).parent / "individual_benchmark_results.csv"
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "feature",
                "jax_kwargs",
                "jax_warmup_ms",
                "jax_run_ms",
                "tsfresh_ms",
                "speedup_with_compile",
                "speedup_without_compile",
            ]
        )
        csv_file.flush()
        print(f"Writing results to: {csv_path}\n")

        print("-" * 120)
        print(
            f"{'Feature':<40} {'JAX warmup':>12} {'JAX run':>10} {'tsfresh':>12} {'w/ compile':>12} {'w/o compile':>12}"
        )
        print("-" * 120)

        results = []
        jax_warmup_total = 0.0
        jax_run_total = 0.0
        tsfresh_total = 0.0
        n_success = 0
        n_jax_faster_with_compile = 0
        n_jax_faster_without_compile = 0

        for jax_name, jax_kwargs, tsfresh_name, tsfresh_kwargs in INDIVIDUAL_FEATURE_CONFIGS:
            # Get JAX function
            if jax_name not in ALL_FEATURE_FUNCTIONS:
                print(f"  {jax_name:<40} {'SKIP':>12} {'':>10} {'':>12} {'':>12} {'':>12}")
                continue

            jax_func = ALL_FEATURE_FUNCTIONS[jax_name]

            # Time JAX (returns warmup_time, run_time, error)
            jax_warmup, jax_run, jax_err = time_jax_feature(jax_func, x_jax, jax_kwargs)

            # Time tsfresh
            tsfresh_time, tsfresh_err = time_tsfresh_feature(
                tsfresh_name, x_np_batches, tsfresh_kwargs, actual_series, USE_MULTIPROCESSING
            )

            # Format results
            if jax_warmup is not None and jax_run is not None:
                jax_warmup_str = f"{jax_warmup * 1000:8.2f}ms"
                jax_run_str = f"{jax_run * 1000:6.2f}ms"
                jax_warmup_total += jax_warmup
                jax_run_total += jax_run
            else:
                jax_warmup_str = "ERROR"
                jax_run_str = "ERROR"

            if tsfresh_time is not None:
                tsfresh_str = f"{tsfresh_time * 1000:8.2f}ms"
                tsfresh_total += tsfresh_time
            else:
                tsfresh_str = "ERROR"

            if jax_warmup is not None and tsfresh_time is not None and jax_run is not None:
                # Speedup = tsfresh_time / jax_time
                # >1 means JAX is faster, <1 means JAX is slower
                speedup_with = tsfresh_time / jax_warmup
                speedup_without = tsfresh_time / jax_run

                speedup_with_str = f"{speedup_with:8.2f}x"
                speedup_without_str = f"{speedup_without:8.2f}x"

                if speedup_with >= 1:
                    n_jax_faster_with_compile += 1
                if speedup_without >= 1:
                    n_jax_faster_without_compile += 1

                n_success += 1
            else:
                speedup_with_str = "N/A"
                speedup_without_str = "N/A"

            # Display name with params
            display_name = jax_name
            if jax_kwargs:
                param_str = ", ".join(f"{k}={v}" for k, v in list(jax_kwargs.items())[:2])
                display_name = f"{jax_name}({param_str})"
            display_name = display_name[:39]

            print(
                f"  {display_name:<40} {jax_warmup_str:>12} {jax_run_str:>10} {tsfresh_str:>12} {speedup_with_str:>12} {speedup_without_str:>12}"
            )

            results.append(
                {
                    "feature": jax_name,
                    "jax_kwargs": jax_kwargs,
                    "jax_warmup": jax_warmup,
                    "jax_run": jax_run,
                    "tsfresh_time": tsfresh_time,
                    "speedup_with_compile": tsfresh_time / jax_warmup
                    if jax_warmup and tsfresh_time
                    else None,
                    "speedup_without_compile": tsfresh_time / jax_run
                    if jax_run and tsfresh_time
                    else None,
                }
            )

            # Write to CSV immediately
            csv_writer.writerow(
                [
                    jax_name,
                    str(jax_kwargs) if jax_kwargs else "",
                    f"{jax_warmup * 1000:.4f}" if jax_warmup else "",
                    f"{jax_run * 1000:.4f}" if jax_run else "",
                    f"{tsfresh_time * 1000:.4f}" if tsfresh_time else "",
                    f"{tsfresh_time / jax_warmup:.4f}" if jax_warmup and tsfresh_time else "",
                    f"{tsfresh_time / jax_run:.4f}" if jax_run and tsfresh_time else "",
                ]
            )
            csv_file.flush()

        print("-" * 120)
        print(
            f"  {'TOTALS':<40} {jax_warmup_total * 1000:8.2f}ms {jax_run_total * 1000:6.2f}ms {tsfresh_total * 1000:8.2f}ms"
        )

    # Summary
    print("\n" + "=" * 120)
    print("INDIVIDUAL BENCHMARK SUMMARY")
    print("=" * 120)
    print(f"  Features tested: {len(INDIVIDUAL_FEATURE_CONFIGS)}")
    print(f"  Successful comparisons: {n_success}")
    print("\n  With compile time (single-run workflow):")
    print(f"    JAX faster: {n_jax_faster_with_compile}/{n_success}")
    if jax_warmup_total > 0:
        print(f"    Overall speedup: {tsfresh_total / jax_warmup_total:.1f}x")
    print("\n  Without compile time (repeated-run workflow):")
    print(f"    JAX faster: {n_jax_faster_without_compile}/{n_success}")
    if jax_run_total > 0:
        print(f"    Overall speedup: {tsfresh_total / jax_run_total:.1f}x")

    # Top 10 slowest JAX features (by warmup time, as that's the bottleneck)
    print("\n" + "-" * 80)
    print("TOP 10 SLOWEST JAX FEATURES (including compile time)")
    print("-" * 80)
    sorted_by_warmup = sorted(
        [r for r in results if r["jax_warmup"] is not None],
        key=lambda x: x["jax_warmup"],
        reverse=True,
    )
    for i, r in enumerate(sorted_by_warmup[:10], 1):
        pct = r["jax_warmup"] / jax_warmup_total * 100 if jax_warmup_total > 0 else 0
        print(f"  {i:2}. {r['feature']:<35} {r['jax_warmup'] * 1000:8.2f}ms ({pct:5.1f}%)")

    # Top 10 where tsfresh is faster (with compile time - single run scenario)
    print("\n" + "-" * 80)
    print("FEATURES WHERE TSFRESH IS FASTER (with compile time)")
    print("-" * 80)
    tsfresh_faster_with = [
        r
        for r in results
        if r["speedup_with_compile"] is not None and r["speedup_with_compile"] < 1
    ]
    if tsfresh_faster_with:
        tsfresh_faster_with.sort(key=lambda x: x["speedup_with_compile"])
        for i, r in enumerate(tsfresh_faster_with[:10], 1):
            print(f"  {i:2}. {r['feature']:<35} JAX {1 / r['speedup_with_compile']:.1f}x slower")
    else:
        print("  None - JAX is faster for all features!")

    # Top 10 where tsfresh is faster (without compile time - repeated runs)
    print("\n" + "-" * 80)
    print("FEATURES WHERE TSFRESH IS FASTER (without compile time)")
    print("-" * 80)
    tsfresh_faster_without = [
        r
        for r in results
        if r["speedup_without_compile"] is not None and r["speedup_without_compile"] < 1
    ]
    if tsfresh_faster_without:
        tsfresh_faster_without.sort(key=lambda x: x["speedup_without_compile"])
        for i, r in enumerate(tsfresh_faster_without[:10], 1):
            print(f"  {i:2}. {r['feature']:<35} JAX {1 / r['speedup_without_compile']:.1f}x slower")
    else:
        print("  None - JAX is faster for all features!")

    return results


def time_all_features_jax(x_jax, parallel=False, use_jit_wrapper=False, use_all=False):
    """Time extracting all features at once using JAX.

    Args:
        x_jax: Input data
        parallel: If True, use ThreadPoolExecutor (CPU only)
        use_jit_wrapper: If True, wrap all feature extraction in a single JIT call (GPU optimal)
        use_all: If True, use all features from JAX_COMPREHENSIVE_FC_PARAMETERS

    Returns:
        Tuple of (warmup_time, run_time, n_features)
    """
    feature_calls = []

    if use_all:
        # Use all features from JAX_COMPREHENSIVE_FC_PARAMETERS with ALL permutations
        for feature_name, param_list in JAX_COMPREHENSIVE_FC_PARAMETERS.items():
            if feature_name not in ALL_FEATURE_FUNCTIONS:
                continue
            func = ALL_FEATURE_FUNCTIONS[feature_name]
            if param_list is None:
                feature_calls.append((feature_name, func, {}))
            else:
                # Iterate ALL parameter permutations, not just the first one
                for params in param_list:
                    feature_calls.append((feature_name, func, params))
    else:
        # Use minimal feature subset for faster benchmarking
        for feature_name, kwargs in MINIMAL_BATCH_FEATURES.items():
            if feature_name not in ALL_FEATURE_FUNCTIONS:
                continue
            func = ALL_FEATURE_FUNCTIONS[feature_name]
            feature_calls.append((feature_name, func, kwargs or {}))

    def extract_all_sequential():
        results = {}
        for fname, func, kwargs in feature_calls:
            results[fname] = func(x_jax, **kwargs)
        jax.block_until_ready(list(results.values()))
        return results

    def extract_all_parallel():
        def run_feature(item):
            fname, func, kwargs = item
            result = func(x_jax, **kwargs)
            jax.block_until_ready(result)
            return fname, result

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = dict(executor.map(run_feature, feature_calls))
        return results

    @jax.jit
    def extract_all_jit(x):
        return {fname: func(x, **kwargs) for fname, func, kwargs in feature_calls}

    def extract_jit_wrapper():
        results = extract_all_jit(x_jax)
        jax.block_until_ready(list(results.values()))
        return results

    if use_jit_wrapper:
        extract_fn = extract_jit_wrapper
    elif parallel:
        extract_fn = extract_all_parallel
    else:
        extract_fn = extract_all_sequential

    # Warmup
    t0_warmup = time.perf_counter()
    _ = extract_fn()
    warmup_time = time.perf_counter() - t0_warmup

    # Single run
    t0 = time.perf_counter()
    _ = extract_fn()
    run_time = time.perf_counter() - t0

    return warmup_time, run_time, len(feature_calls)


def time_all_features_tsfresh(
    n_timesteps: int,
    n_batches: int,
    use_all: bool,
):
    """Time extracting features using tsfresh extract_features API."""
    # Parameter name mappings from JAX to tsfresh
    param_remap = {
        "number_cwt_peaks": {"max_width": "n"},
        "agg_linear_trend": {"chunk_size": "chunk_len"},
        "range_count": {"min_val": "min", "max_val": "max"},
    }

    # JAX to tsfresh feature name mappings
    name_remap = {
        "has_variance_larger_than_standard_deviation": "variance_larger_than_standard_deviation",
    }

    if use_all:
        fc_parameters = EfficientFCParameters()
        # Exclude features not implemented in JAX
        fc_parameters.pop("query_similarity_count", None)
    else:
        fc_parameters = {}
        # Use minimal feature subset
        for feature_name, kwargs in MINIMAL_BATCH_FEATURES.items():
            tsfresh_name = name_remap.get(feature_name, feature_name)

            if kwargs is None:
                fc_parameters[tsfresh_name] = None
            else:
                params = kwargs.copy()
                # Apply parameter name remapping
                if feature_name in param_remap:
                    for jax_key, tsfresh_key in param_remap[feature_name].items():
                        if jax_key in params:
                            params[tsfresh_key] = params.pop(jax_key)
                fc_parameters[tsfresh_name] = [params]

    # Count total feature calls (including all permutations)
    n_features = sum(len(params) if params is not None else 1 for params in fc_parameters.values())

    # Create DataFrame
    np.random.seed(42)
    data_rows = []
    for batch_id in range(n_batches):
        values = np.random.randn(n_timesteps)
        for t_idx, val in enumerate(values):
            data_rows.append({"id": batch_id, "time": t_idx, "value": val})

    df = pd.DataFrame(data_rows)

    # Single timed run
    n_jobs = os.cpu_count() or 1
    t0 = time.perf_counter()
    _ = extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_parameters,
        disable_progressbar=True,
        n_jobs=n_jobs,
    )
    run_time = time.perf_counter() - t0

    return run_time, n_features


def run_batch_benchmark(
    n_timesteps: int,
    n_batches: int,
    use_gpu: bool,
    use_all: bool,
):
    """Benchmark extracting all features at once (batch mode)."""
    print("\n" + "=" * 100)
    print("BATCH FEATURE EXTRACTION BENCHMARK")
    print("=" * 100)

    device = jax.devices()[0]
    n_states = 2  # Pendulum has 2 states
    total_series = n_batches * n_states

    feature_set = "all (EfficientFCParameters)" if use_all else "minimal (41 features)"

    print(f"\nDevice: {device}")
    print(
        f"Data: {n_timesteps} timesteps, {n_batches} batches x {n_states} states = {total_series} series"
    )
    print(f"Feature set: {feature_set}")
    print(f"tsfresh n_jobs={os.cpu_count()}")

    # Create test data from pendulum ODE
    x_jax = generate_sample(n_timesteps, n_batches)

    print("Timing tsfresh bulk extraction...")
    tsfresh_time, tsfresh_n_features = time_all_features_tsfresh(n_timesteps, total_series, use_all)

    if use_gpu:
        print("\nTiming JAX bulk extraction (parallel on GPU)...")
        jax_warmup, jax_time, jax_n_features = time_all_features_jax(
            x_jax, parallel=True, use_jit_wrapper=False, use_all=use_all
        )

        print("\n" + "-" * 80)
        print("BATCH EXTRACTION RESULTS (GPU MODE)")
        print("-" * 80)
        print(
            f"  JAX warmup ({jax_n_features} features):".ljust(45) + f"{jax_warmup * 1000:10.2f}ms"
        )
        print("  JAX post-warmup:".ljust(45) + f"{jax_time * 1000:10.2f}ms")
        print(
            f"  tsfresh ({tsfresh_n_features} features):".ljust(45)
            + f"{tsfresh_time * 1000:10.2f}ms"
        )
        print("-" * 80)

        if jax_time > 0:
            print(f"  {'Speedup (post-warmup):':45} {tsfresh_time / jax_time:10.1f}x")

        print(
            f"\n  Per-series: JAX {jax_time / total_series * 1e6:.2f}us vs tsfresh {tsfresh_time / total_series * 1e6:.2f}us"
        )

    else:
        print("\nTiming JAX bulk extraction (parallel)...")
        jax_warmup_par, jax_time_par, jax_n_features = time_all_features_jax(
            x_jax, parallel=True, use_all=use_all
        )

        print("\n" + "-" * 80)
        print("BATCH EXTRACTION RESULTS (CPU MODE)")
        print("-" * 80)
        print(
            f"  JAX parallel warmup ({jax_n_features} features):".ljust(45)
            + f"{jax_warmup_par * 1000:10.2f}ms"
        )
        print("  JAX parallel post-warmup:".ljust(45) + f"{jax_time_par * 1000:10.2f}ms")
        print(
            f"  tsfresh ({tsfresh_n_features} features):".ljust(45)
            + f"{tsfresh_time * 1000:10.2f}ms"
        )
        print("-" * 80)

        if jax_time_par > 0:
            print(f"  {'JAX parallel speedup:':45} {tsfresh_time / jax_time_par:10.1f}x")

        print("\n  Per-series timing:")
        print(f"    JAX parallel:   {jax_time_par / total_series * 1e6:.2f}us/series")
        print(f"    tsfresh:        {tsfresh_time / total_series * 1e6:.2f}us/series")


def main():
    print("=" * 100)
    print("JAX vs TSFRESH FEATURE TIMING BENCHMARK")
    print("=" * 100)

    # Parse arguments
    individual_only = "--individual-only" in sys.argv
    batch_only = "--batch-only" in sys.argv
    use_gpu = USE_GPU  # Already parsed at module level
    use_all = USE_ALL_FEATURES  # Already parsed at module level

    # Parse --batches
    n_batches = 10000
    for arg in sys.argv:
        if arg.startswith("--batches="):
            n_batches = int(arg.split("=")[1])

    n_timesteps = 10000
    n_states = 2  # Pendulum always has 2 states

    print("\nConfiguration:")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Batches: {n_batches}")
    print(f"  States: {n_states} (pendulum)")
    print(f"  Total series: {n_batches * n_states}")
    print(f"  CPU cores: {os.cpu_count()}")
    print(f"  GPU mode: {use_gpu}")
    print(f"  All features: {use_all}")
    print(f"  Multiprocessing: {USE_MULTIPROCESSING}")
    print(f"  Individual benchmark: {not batch_only}")
    print(f"  Batch benchmark: {not individual_only}")

    if not batch_only:
        run_individual_benchmark(n_timesteps, n_batches)

    if not individual_only:
        run_batch_benchmark(n_timesteps, n_batches, use_gpu, use_all)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
