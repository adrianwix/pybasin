# pyright: basic
"""
Benchmark to compare PyTorch vs tsfresh feature extraction timing.

This script times each feature in both PyTorch and tsfresh to compare performance.
Unlike JAX, PyTorch uses eager execution without JIT compilation overhead.

Usage:
    uv run python benchmarks/benchmark_torch_features_timing.py --individual-only
    uv run python benchmarks/benchmark_torch_features_timing.py --batch-only --comprehensive
    uv run python benchmarks/benchmark_torch_features_timing.py --batch-only --gpu
    uv run python benchmarks/benchmark_torch_features_timing.py --batch-only --batches=10000
    uv run python benchmarks/benchmark_torch_features_timing.py --batch-only --real-data
"""

import os
import sys
import time
import warnings
from datetime import datetime

# Add parent directory to path for imports when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
import torch
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import feature_calculators as fc

from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE, PendulumParams
from pybasin.sampler import GridSampler
from pybasin.solvers import JaxSolver
from pybasin.ts_torch.settings import (
    ALL_FEATURE_FUNCTIONS,
    TORCH_COMPREHENSIVE_FC_PARAMETERS,
    TORCH_GPU_FC_PARAMETERS,
)
from pybasin.ts_torch.torch_feature_processors import (
    count_features,
    extract_features_gpu,
    extract_features_gpu_batched,
    extract_features_parallel,
    extract_features_sequential,
)

# Suppress pandas FutureWarning from tsfresh
warnings.filterwarnings("ignore", category=FutureWarning, module="tsfresh")

# Parse --gpu and --cpu-only flags
USE_GPU = "--gpu" in sys.argv
CPU_ONLY = "--cpu-only" in sys.argv

# Number of timing runs for benchmarks
N_RUNS = 1


# =============================================================================
# DATA GENERATION
# =============================================================================
def generate_sample(
    n_timesteps: int,
    n_batches: int,
    n_states: int,
    real_data: bool = False,
    time_steady: float = 950.0,
    seed: int = 42,
) -> torch.Tensor:
    """Generate sample data for benchmarking.

    Args:
        n_timesteps: Number of time steps per series
        n_batches: Number of batches (trajectories)
        n_states: Number of state variables
        real_data: If True, generate real pendulum ODE trajectories.
                   If False, generate random Gaussian noise.
        time_steady: Time threshold for filtering transients (only used with real_data)
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (n_timesteps, n_batches, n_states)
    """
    np.random.seed(seed)

    if not real_data:
        # Generate random Gaussian noise
        x_np = np.random.randn(n_timesteps, n_batches, n_states).astype(np.float32)
        return torch.from_numpy(x_np)

    # Generate real pendulum ODE trajectories
    print(f"  Generating {n_batches} real pendulum trajectories...")

    # Define pendulum parameters
    params: PendulumParams = {"alpha": 0.1, "T": 0.5, "K": 1.0}
    ode_system = PendulumJaxODE(params)

    # Create sampler with pendulum-appropriate limits
    sampler = GridSampler(
        min_limits=[-np.pi + np.arcsin(params["T"] / params["K"]), -10.0],
        max_limits=[np.pi + np.arcsin(params["T"] / params["K"]), 10.0],
        device="cpu",
    )

    # Sample initial conditions
    y0 = sampler.sample(n_batches)

    # Create solver - compute n_steps to get desired n_timesteps after filtering
    # We need extra steps to account for transient filtering
    total_time = 1000.0
    # Calculate how many steps we need in total to get n_timesteps after filtering
    steady_fraction = (total_time - time_steady) / total_time
    total_steps = int(n_timesteps / steady_fraction) + 10  # Add buffer

    solver = JaxSolver(
        t_span=(0, total_time),
        t_steps=total_steps,
        device="cpu",
        rtol=1e-8,
        atol=1e-6,
        cache_dir=None,
    )

    # Integrate ODE system
    t, y = solver.integrate(ode_system, y0)

    # Filter to steady state portion (t >= time_steady)
    # t has shape (n_steps,), y has shape (n_steps, n_batches, n_states)
    steady_mask = t >= time_steady
    y_steady = y[steady_mask, :, :]

    # Trim or pad to exactly n_timesteps
    current_timesteps = y_steady.shape[0]
    if current_timesteps >= n_timesteps:
        y_steady = y_steady[:n_timesteps, :, :]
    else:
        # Pad with last value if needed (shouldn't happen with proper buffer)
        pad_size = n_timesteps - current_timesteps
        last_val = y_steady[-1:, :, :].expand(pad_size, -1, -1)
        y_steady = torch.cat([y_steady, last_val], dim=0)

    # Limit states if needed
    if n_states < y_steady.shape[2]:
        y_steady = y_steady[:, :, :n_states]

    print(f"  Generated trajectory shape: {y_steady.shape}")
    return y_steady.float()


# =============================================================================
# MINIMAL FEATURE SET FOR FAST BATCH BENCHMARKS
# =============================================================================
MINIMAL_BATCH_FEATURES: dict[str, dict | None] = {
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
    "abs_energy": None,
    "kurtosis": None,
    "skewness": None,
    "variation_coefficient": None,
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "mean_change": None,
    "mean_second_derivative_central": None,
    "count_above_mean": None,
    "count_below_mean": None,
    "has_duplicate": None,
    "has_duplicate_max": None,
    "has_duplicate_min": None,
    "has_variance_larger_than_standard_deviation": None,
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_maximum": None,
    "last_location_of_minimum": None,
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    "fourier_entropy": {"bins": 10},
    "cid_ce": {"normalize": True},
    "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
    "percentage_of_reoccurring_values_to_all_values": None,
    "sum_of_reoccurring_data_points": None,
    "sum_of_reoccurring_values": None,
    "ratio_value_number_to_time_series_length": None,
    "benford_correlation": None,
    "mean_n_absolute_max": {"number_of_maxima": 1},
    "ratio_beyond_r_sigma": {"r": 1.0},
    "symmetry_looking": {"r": 0.1},
}


# =============================================================================
# FEATURE CONFIGURATIONS FOR INDIVIDUAL BENCHMARKS
# =============================================================================
# Each entry: (torch_func_name, torch_kwargs, tsfresh_func_name, tsfresh_kwargs)
# We use ONE representative parameter set per feature

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
    (
        "time_reversal_asymmetry_statistic",
        {"lag": 1},
        "time_reversal_asymmetry_statistic",
        {"lag": 1},
    ),
    ("c3", {"lag": 1}, "c3", {"lag": 1}),
    ("cid_ce", {"normalize": True}, "cid_ce", {"normalize": True}),
    ("symmetry_looking", {"r": 0.1}, "symmetry_looking", {"param": [{"r": 0.1}]}),
    ("large_standard_deviation", {"r": 0.25}, "large_standard_deviation", {"r": 0.25}),
    ("quantile", {"q": 0.5}, "quantile", {"q": 0.5}),
    ("autocorrelation", {"lag": 1}, "autocorrelation", {"lag": 1}),
    (
        "agg_autocorrelation",
        {"f_agg": "mean", "maxlag": 40},
        "agg_autocorrelation",
        {"param": [{"f_agg": "mean", "maxlag": 40}]},
    ),
    ("partial_autocorrelation", {"lag": 1}, "partial_autocorrelation", {"param": [{"lag": 1}]}),
    ("number_cwt_peaks", {"max_width": 5}, "number_cwt_peaks", {"n": 5}),
    ("number_peaks", {"n": 3}, "number_peaks", {"n": 3}),
    ("binned_entropy", {"max_bins": 10}, "binned_entropy", {"max_bins": 10}),
    ("fourier_entropy", {"bins": 10}, "fourier_entropy", {"bins": 10}),
    (
        "permutation_entropy",
        {"tau": 1, "dimension": 3},
        "permutation_entropy",
        {"tau": 1, "dimension": 3},
    ),
    ("lempel_ziv_complexity", {"bins": 2}, "lempel_ziv_complexity", {"bins": 2}),
    ("index_mass_quantile", {"q": 0.5}, "index_mass_quantile", {"param": [{"q": 0.5}]}),
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
    (
        "change_quantiles",
        {"ql": 0.0, "qh": 0.5, "isabs": False, "f_agg": "mean"},
        "change_quantiles",
        {"ql": 0.0, "qh": 0.5, "isabs": False, "f_agg": "mean"},
    ),
    ("count_above", {"t": 0}, "count_above", {"t": 0}),
    ("count_below", {"t": 0}, "count_below", {"t": 0}),
    ("number_crossing_m", {"m": 0}, "number_crossing_m", {"m": 0}),
    (
        "energy_ratio_by_chunks",
        {"num_segments": 10, "segment_focus": 0},
        "energy_ratio_by_chunks",
        {"param": [{"num_segments": 10, "segment_focus": 0}]},
    ),
    ("ratio_beyond_r_sigma", {"r": 1.0}, "ratio_beyond_r_sigma", {"r": 1.0}),
    ("value_count", {"value": 0}, "value_count", {"value": 0}),
    ("range_count", {"min_val": -1, "max_val": 1}, "range_count", {"min": -1, "max": 1}),
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


def time_torch_feature(func, x_torch, kwargs):
    """Time a PyTorch feature function."""
    try:
        # Warmup
        _ = func(x_torch, **kwargs)
        if USE_GPU:
            torch.cuda.synchronize()

        # Time
        times = []
        for _ in range(N_RUNS):
            start = time.perf_counter()
            _ = func(x_torch, **kwargs)
            if USE_GPU:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        return np.mean(times), None
    except Exception as e:
        return None, str(e)


def time_tsfresh_feature(func_name, x_np, kwargs, n_batches):
    """Time a tsfresh feature function over n_batches calls."""
    if not hasattr(fc, func_name):
        return None, f"No tsfresh function: {func_name}"

    tsfresh_fn = getattr(fc, func_name)

    try:
        # Test call
        if "param" in kwargs:
            list(tsfresh_fn(x_np, kwargs["param"]))
        else:
            tsfresh_fn(x_np, **kwargs)

        # Time n_batches calls
        times = []
        for _ in range(N_RUNS):
            start = time.perf_counter()
            for _ in range(n_batches):
                if "param" in kwargs:
                    list(tsfresh_fn(x_np, kwargs["param"]))
                else:
                    tsfresh_fn(x_np, **kwargs)
            times.append(time.perf_counter() - start)

        return np.mean(times), None
    except Exception as e:
        return None, str(e)


def time_all_features_torch(x_torch, use_comprehensive=False, use_gpu_friendly=False):
    """Time extracting all features at once using PyTorch.

    Args:
        x_torch: Input tensor
        use_comprehensive: If True, use all features from TORCH_COMPREHENSIVE_FC_PARAMETERS
        use_gpu_friendly: If True, use GPU-optimized subset (excludes slow features)

    Returns:
        Tuple of (mean_run_time, n_features)
    """
    if use_gpu_friendly:
        fc_params = TORCH_GPU_FC_PARAMETERS
    elif use_comprehensive:
        fc_params = TORCH_COMPREHENSIVE_FC_PARAMETERS
    else:
        fc_params = MINIMAL_BATCH_FEATURES  # type: ignore[assignment]

    # Time
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        _ = extract_features_sequential(x_torch, fc_params)
        times.append(time.perf_counter() - start)

    n_features = count_features(fc_params)
    return np.mean(times), n_features


def time_all_features_torch_parallel(
    x_torch, use_comprehensive=False, use_gpu_friendly=False, n_workers=None
):
    """Time extracting all features using PyTorch with batch-level parallelism.

    Uses multiprocessing to truly parallelize across CPU cores.

    Args:
        x_torch: Input tensor of shape (n_timesteps, n_batches, n_states)
        use_comprehensive: If True, use all features from TORCH_COMPREHENSIVE_FC_PARAMETERS
        use_gpu_friendly: If True, use GPU-optimized subset (excludes slow features)
        n_workers: Number of worker threads (default: cpu_count)

    Returns:
        Tuple of (mean_run_time, n_features)
    """
    if use_gpu_friendly:
        fc_params = TORCH_GPU_FC_PARAMETERS
    elif use_comprehensive:
        fc_params = TORCH_COMPREHENSIVE_FC_PARAMETERS
    else:
        fc_params = MINIMAL_BATCH_FEATURES  # type: ignore[assignment]

    # Time
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        _ = extract_features_parallel(x_torch, fc_params, n_workers)
        times.append(time.perf_counter() - start)

    n_features = count_features(fc_params)
    return np.mean(times), n_features


def time_all_features_torch_gpu(x_torch, use_comprehensive=False, use_gpu_friendly=True):
    """Time extracting all features using PyTorch on GPU.

    GPU naturally parallelizes across all batches without chunking.
    The key is to keep data on GPU and avoid CPU-GPU transfers.

    Args:
        x_torch: Input tensor (will be moved to GPU)
        use_comprehensive: If True, use all features from TORCH_COMPREHENSIVE_FC_PARAMETERS
        use_gpu_friendly: If True, use GPU-optimized subset (excludes slow features)

    Returns:
        Tuple of (mean_run_time, n_features)
    """
    if not torch.cuda.is_available():
        return None, 0

    # Select feature parameters based on mode
    if use_gpu_friendly:
        fc_params = TORCH_GPU_FC_PARAMETERS
    elif use_comprehensive:
        fc_params = TORCH_COMPREHENSIVE_FC_PARAMETERS
    else:
        fc_params = MINIMAL_BATCH_FEATURES  # type: ignore[assignment]

    # Warmup
    _ = extract_features_gpu(x_torch, fc_params, use_gpu_friendly=False)

    # Time
    times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = extract_features_gpu(x_torch, fc_params, use_gpu_friendly=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    n_features = count_features(fc_params)
    return np.mean(times), n_features


def time_all_features_torch_gpu_batched(x_torch, use_comprehensive=False, use_gpu_friendly=True):
    """Time extracting all features using PyTorch on GPU with batched operations.

    Uses batched calculators that group parameterized features together
    (e.g., all autocorrelation lags computed in one FFT call).

    Args:
        x_torch: Input tensor (will be moved to GPU)
        use_comprehensive: If True, use all features from TORCH_COMPREHENSIVE_FC_PARAMETERS
        use_gpu_friendly: If True, use GPU-optimized subset (excludes slow features)

    Returns:
        Tuple of (mean_run_time, n_features)
    """
    if not torch.cuda.is_available():
        return None, 0

    # Select feature parameters based on mode
    if use_gpu_friendly:
        fc_params = TORCH_GPU_FC_PARAMETERS
    elif use_comprehensive:
        fc_params = TORCH_COMPREHENSIVE_FC_PARAMETERS
    else:
        fc_params = MINIMAL_BATCH_FEATURES  # type: ignore[assignment]

    # Warmup
    # print("    Warming up batched GPU extraction...")
    # _ = extract_features_gpu_batched(x_torch, fc_params)
    # torch.cuda.synchronize()

    # Time
    times = []
    for _ in range(N_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = extract_features_gpu_batched(x_torch, fc_params)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    n_features = count_features(fc_params)
    return np.mean(times), n_features


def time_all_features_tsfresh(
    n_timesteps: int,
    n_batches: int,
    use_comprehensive: bool = False,
    use_gpu_friendly: bool = False,
):
    """Time extracting features using tsfresh extract_features API.

    Note: When use_comprehensive=True, we use EfficientFCParameters instead of
    ComprehensiveFCParameters because PyTorch does not implement approximate_entropy
    and sample_entropy (marked as high_comp_cost in tsfresh).

    When use_gpu_friendly=True, we use EfficientFCParameters but exclude
    permutation_entropy which is slow on GPU.
    """
    if use_gpu_friendly:
        fc_parameters = dict(EfficientFCParameters())
        if "permutation_entropy" in fc_parameters:
            del fc_parameters["permutation_entropy"]
        n_features = sum(len(v) if v else 1 for v in fc_parameters.values())
    elif use_comprehensive:
        fc_parameters = EfficientFCParameters()
        n_features = sum(len(v) if v else 1 for v in fc_parameters.values())
    else:
        param_remap = {
            "number_cwt_peaks": {"max_width": "n"},
            "agg_linear_trend": {"chunk_size": "chunk_len"},
            "range_count": {"min_val": "min", "max_val": "max"},
        }
        name_remap = {
            "has_variance_larger_than_standard_deviation": "variance_larger_than_standard_deviation",
        }
        fc_parameters = {}
        for feature_name, kwargs in MINIMAL_BATCH_FEATURES.items():
            tsfresh_name = name_remap.get(feature_name, feature_name)
            if kwargs is None:
                fc_parameters[tsfresh_name] = None
            else:
                remapped = {}
                for k, v in kwargs.items():
                    if feature_name in param_remap and k in param_remap[feature_name]:
                        remapped[param_remap[feature_name][k]] = v
                    else:
                        remapped[k] = v
                fc_parameters[tsfresh_name] = [remapped]
        n_features = len(fc_parameters)

    # Create DataFrame
    np.random.seed(42)
    data_rows = []
    for batch_id in range(n_batches):
        values = np.random.randn(n_timesteps)
        for t_idx, val in enumerate(values):
            data_rows.append({"id": batch_id, "time": t_idx, "value": val})

    df = pd.DataFrame(data_rows)

    # Time
    n_jobs = os.cpu_count() or 1
    times = []
    for _ in range(N_RUNS):
        start = time.perf_counter()
        _ = extract_features(
            df,
            column_id="id",
            column_sort="time",
            default_fc_parameters=fc_parameters,
            disable_progressbar=True,
            n_jobs=n_jobs,
        )
        times.append(time.perf_counter() - start)

    return np.mean(times), n_features


def run_individual_benchmark(n_timesteps: int, n_batches: int, n_states: int):
    """Run individual feature benchmarks comparing PyTorch vs tsfresh."""
    print("\n" + "=" * 100)
    print("INDIVIDUAL FEATURE BENCHMARK")
    print("=" * 100)

    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    total_series = n_batches * n_states

    print(f"\nDevice: {device}")
    print(
        f"Data: {n_timesteps} timesteps, {n_batches} batches x {n_states} states = {total_series} series"
    )
    print(
        f"PyTorch processes all {total_series} series at once, tsfresh runs {total_series}x per feature"
    )
    print(f"Timing: {N_RUNS} runs per feature\n")

    # Create test data
    np.random.seed(42)
    x_torch = torch.from_numpy(
        np.random.randn(n_timesteps, n_batches, n_states).astype(np.float32)
    ).to(device)
    x_np = np.random.randn(n_timesteps)  # Single series for tsfresh

    print("-" * 100)
    print(f"{'Feature':<45} {'PyTorch':>12} {'tsfresh':>12} {'Speedup':>12} {'Status'}")
    print("-" * 100)

    results = []
    torch_total = 0.0
    tsfresh_total = 0.0
    n_success = 0
    n_torch_faster = 0
    n_tsfresh_faster = 0

    for torch_name, torch_kwargs, tsfresh_name, tsfresh_kwargs in INDIVIDUAL_FEATURE_CONFIGS:
        # Get PyTorch function
        if torch_name not in ALL_FEATURE_FUNCTIONS:
            print(f"  {torch_name:<45} {'SKIP':>12} {'':>12} {'':>12} PyTorch func not found")
            continue

        torch_func = ALL_FEATURE_FUNCTIONS[torch_name]

        # Time PyTorch
        torch_time, torch_err = time_torch_feature(torch_func, x_torch, torch_kwargs)

        # Time tsfresh
        tsfresh_time, tsfresh_err = time_tsfresh_feature(
            tsfresh_name, x_np, tsfresh_kwargs, total_series
        )

        # Format results
        if torch_time is not None:
            torch_str = f"{torch_time * 1000:8.2f}ms"
            torch_total += torch_time
        else:
            torch_str = "ERROR"

        if tsfresh_time is not None:
            tsfresh_str = f"{tsfresh_time * 1000:8.2f}ms"
            tsfresh_total += tsfresh_time
        else:
            tsfresh_str = "ERROR"

        if torch_time is not None and tsfresh_time is not None:
            speedup = tsfresh_time / torch_time
            if speedup >= 1:
                speedup_str = f"{speedup:8.1f}x"
                n_torch_faster += 1
            else:
                speedup_str = f"{1 / speedup:7.1f}x slower"
                n_tsfresh_faster += 1
            status = "OK"
            n_success += 1
        else:
            speedup_str = "N/A"
            status = torch_err or tsfresh_err or "ERROR"
            status = status[:20] if len(status) > 20 else status

        # Display name with params
        display_name = torch_name
        if torch_kwargs:
            param_str = ", ".join(f"{k}={v}" for k, v in list(torch_kwargs.items())[:2])
            display_name = f"{torch_name}({param_str})"
        display_name = display_name[:44]

        print(f"  {display_name:<45} {torch_str:>12} {tsfresh_str:>12} {speedup_str:>12} {status}")

        results.append(
            {
                "feature": torch_name,
                "torch_kwargs": torch_kwargs,
                "torch_time": torch_time,
                "tsfresh_time": tsfresh_time,
                "speedup": tsfresh_time / torch_time if torch_time and tsfresh_time else None,
            }
        )

    print("-" * 100)
    print("\nSUMMARY:")
    print(f"  Features tested: {n_success}/{len(INDIVIDUAL_FEATURE_CONFIGS)}")
    print(f"  PyTorch faster: {n_torch_faster}")
    print(f"  tsfresh faster: {n_tsfresh_faster}")
    if torch_total > 0 and tsfresh_total > 0:
        print(f"\n  Total PyTorch time: {torch_total * 1000:.2f}ms")
        print(f"  Total tsfresh time: {tsfresh_total * 1000:.2f}ms")
        print(f"  Overall speedup: {tsfresh_total / torch_total:.1f}x")

    # Show features where tsfresh is faster (torch is slower)
    slower_features = [
        r
        for r in results
        if r["torch_time"] is not None and r["speedup"] is not None and r["speedup"] < 1
    ]
    if slower_features:
        print("\n  Features where PyTorch is slower than tsfresh:")
        sorted_slower = sorted(slower_features, key=lambda r: r["speedup"])
        for r in sorted_slower:
            slowdown = 1 / r["speedup"]
            print(
                f"    {r['feature']:<40} {r['torch_time'] * 1000:8.2f}ms  ({slowdown:.1f}x slower)"
            )


def run_batch_benchmark(
    n_timesteps: int,
    n_batches: int,
    n_states: int,
    use_gpu: bool,
    use_comprehensive: bool = False,
    gpu_only: bool = False,
    gpu_friendly: bool = False,
    cpu_only: bool = False,
    real_data: bool = False,
):
    """Benchmark extracting all features at once (batch mode)."""
    print("\n" + "=" * 100)
    print("BATCH FEATURE EXTRACTION BENCHMARK")
    print("=" * 100)

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    total_series = n_batches * n_states

    if gpu_friendly:
        feature_set = "TORCH_GPU_FC_PARAMETERS (GPU-optimized subset)"
    elif use_comprehensive:
        feature_set = "Comprehensive (EfficientFCParameters for tsfresh, TORCH_COMPREHENSIVE_FC_PARAMETERS for PyTorch)"
    else:
        feature_set = f"MINIMAL_BATCH_FEATURES ({len(MINIMAL_BATCH_FEATURES)} features)"

    data_source = "Real pendulum ODE trajectories" if real_data else "Random Gaussian noise"

    print(f"\nDevice: {device}")
    print(
        f"Data: {n_timesteps} timesteps, {n_batches} batches x {n_states} states = {total_series} series"
    )
    print(f"Data source: {data_source}")
    print(f"Feature set: {feature_set}")

    # Create test data
    x_torch = generate_sample(
        n_timesteps=n_timesteps,
        n_batches=n_batches,
        n_states=n_states,
        real_data=real_data,
    ).to(device)

    tsfresh_time, tsfresh_n_features = 0.0, 0

    torch_time, torch_n_features = 0.0, 0
    torch_parallel_time = 0.0

    if not gpu_only:
        print("\nTiming PyTorch sequential extraction...")
        torch_time, torch_n_features = time_all_features_torch(
            x_torch, use_comprehensive=use_comprehensive, use_gpu_friendly=gpu_friendly
        )

        print("Timing PyTorch parallel extraction (CPU multiprocessing)...")
        torch_parallel_time, _ = time_all_features_torch_parallel(
            x_torch, use_comprehensive=use_comprehensive, use_gpu_friendly=gpu_friendly
        )

    # GPU timing - standard and batched versions
    torch_gpu_time = None
    torch_gpu_batched_time = None
    torch_gpu_n_features = 0
    if torch.cuda.is_available() and not cpu_only:
        print("Timing PyTorch GPU extraction (standard)...")
        torch_gpu_time, torch_gpu_n_features = time_all_features_torch_gpu(
            x_torch, use_comprehensive=use_comprehensive, use_gpu_friendly=gpu_friendly
        )

        print("Timing PyTorch GPU extraction (batched operations)...")
        torch_gpu_batched_time, _ = time_all_features_torch_gpu_batched(
            x_torch, use_comprehensive=use_comprehensive, use_gpu_friendly=gpu_friendly
        )

    # tsfresh timing
    if not gpu_only:
        print("Timing tsfresh bulk extraction...")
        tsfresh_time, tsfresh_n_features = time_all_features_tsfresh(
            n_timesteps, total_series, use_comprehensive, use_gpu_friendly=gpu_friendly
        )

    print("\n" + "-" * 100)
    print("RESULTS")
    print("-" * 100)
    if not gpu_only:
        print(f"  PyTorch sequential CPU: {torch_time * 1000:.2f}ms ({torch_n_features} features)")
        print(
            f"  PyTorch parallel CPU ({os.cpu_count()} workers): {torch_parallel_time * 1000:.2f}ms ({torch_n_features} features)"
        )
    if torch_gpu_time is not None:
        gpu_n_features = torch_n_features if torch_n_features > 0 else torch_gpu_n_features
        print(f"  PyTorch GPU (cuda): {torch_gpu_time * 1000:.2f}ms ({gpu_n_features} features)")
    if torch_gpu_batched_time is not None:
        gpu_n_features = torch_n_features if torch_n_features > 0 else torch_gpu_n_features
        print(
            f"  PyTorch GPU batched (cuda): {torch_gpu_batched_time * 1000:.2f}ms ({gpu_n_features} features)"
        )

    if tsfresh_time > 0:
        print(
            f"  tsfresh ({os.cpu_count()} jobs): {tsfresh_time * 1000:.2f}ms ({tsfresh_n_features} features)"
        )

    # Speedups vs tsfresh (always show these)
    print("\n  Speedups vs tsfresh:")
    if tsfresh_time > 0:
        if torch_time > 0:
            speedup_seq = tsfresh_time / torch_time
            print(f"    PyTorch sequential: {speedup_seq:.2f}x faster")
        if torch_parallel_time > 0:
            speedup_par = tsfresh_time / torch_parallel_time
            print(f"    PyTorch parallel: {speedup_par:.2f}x faster")
        if torch_gpu_time is not None:
            speedup_gpu = tsfresh_time / torch_gpu_time
            print(f"    PyTorch GPU: {speedup_gpu:.2f}x faster")
        if torch_gpu_batched_time is not None:
            speedup_gpu_batched = tsfresh_time / torch_gpu_batched_time
            print(f"    PyTorch GPU batched: {speedup_gpu_batched:.2f}x faster")

    # Internal speedups
    print("\n  Internal speedups:")
    if not gpu_only and torch_time > 0 and torch_parallel_time > 0:
        speedup_parallel = torch_time / torch_parallel_time
        print(f"    Parallel vs sequential: {speedup_parallel:.2f}x")
    if torch_gpu_time is not None and torch_parallel_time > 0:
        speedup_gpu = torch_gpu_time / torch_parallel_time
        print(f"    GPU vs parallel CPU: {speedup_gpu:.2f}x")
    if torch_gpu_batched_time is not None and torch_gpu_time is not None:
        speedup_batched = torch_gpu_time / torch_gpu_batched_time
        print(f"    GPU batched vs GPU standard: {speedup_batched:.2f}x")

    print("\n" + "=" * 100)

    # Save results to CSV
    if not gpu_only:
        save_batch_result(
            "pytorch",
            "sequential",
            "cpu",
            n_timesteps,
            n_batches,
            n_states,
            torch_n_features,
            float(torch_time * 1000),
        )
        save_batch_result(
            "pytorch",
            "parallel",
            "cpu",
            n_timesteps,
            n_batches,
            n_states,
            torch_n_features,
            float(torch_parallel_time * 1000),
        )
    if torch_gpu_time is not None:
        save_batch_result(
            "pytorch",
            "gpu",
            "cuda",
            n_timesteps,
            n_batches,
            n_states,
            gpu_n_features,
            float(torch_gpu_time * 1000),
        )
    if torch_gpu_batched_time is not None:
        save_batch_result(
            "pytorch",
            "gpu_batched",
            "cuda",
            n_timesteps,
            n_batches,
            n_states,
            gpu_n_features,
            float(torch_gpu_batched_time * 1000),
        )
    if tsfresh_time > 0:
        save_batch_result(
            "tsfresh",
            "parallel",
            "cpu",
            n_timesteps,
            n_batches,
            n_states,
            tsfresh_n_features,
            float(tsfresh_time * 1000),
        )


def save_batch_result(
    backend: str,
    mode: str,
    device: str,
    n_timesteps: int,
    n_batches: int,
    n_states: int,
    n_features: int,
    time_ms: float,
) -> None:
    """Save a single batch benchmark result to CSV file."""
    results_file = os.path.join(os.path.dirname(__file__), "batch_benchmark_results.csv")
    file_exists = os.path.exists(results_file)

    python_version = sys.version.split()[0]
    is_free_threaded = hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled()  # type: ignore[attr-defined]
    gil_status = "free-threaded" if is_free_threaded else "with-gil"
    timestamp = datetime.now().isoformat()
    n_workers = os.cpu_count() or 1

    with open(results_file, "a") as f:
        if not file_exists:
            f.write(
                "timestamp,python_version,gil_status,backend,mode,device,n_timesteps,n_batches,n_states,n_features,time_ms,n_workers\n"
            )
        f.write(
            f"{timestamp},{python_version},{gil_status},{backend},{mode},{device},{n_timesteps},{n_batches},{n_states},{n_features},{time_ms:.2f},{n_workers}\n"
        )

    print(f"  Saved {backend} {mode}: {time_ms:.2f}ms")


def main():
    print("=" * 100)
    print("PYTORCH vs TSFRESH FEATURE TIMING BENCHMARK")
    print("=" * 100)

    # Parse arguments
    individual_only = "--individual-only" in sys.argv
    use_gpu = USE_GPU
    use_comprehensive = "--comprehensive" in sys.argv
    gpu_only = "--gpu-only" in sys.argv
    gpu_friendly = "--gpu-friendly" in sys.argv
    cpu_only = CPU_ONLY
    real_data = "--real-data" in sys.argv

    # Parse --batches and --states
    n_batches = 1000
    n_states = 1
    for arg in sys.argv:
        if arg.startswith("--batches="):
            n_batches = int(arg.split("=")[1])
        if arg.startswith("--states="):
            n_states = int(arg.split("=")[1])

    n_timesteps = 200

    print("\nConfiguration:")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Batches: {n_batches}")
    print(f"  States: {n_states}")
    print(f"  Total series: {n_batches * n_states}")
    print(f"  GPU mode: {use_gpu}")
    print(f"  GPU only: {gpu_only}")
    print(f"  CPU only: {cpu_only}")
    print(f"  GPU friendly subset: {gpu_friendly}")
    print(f"  Comprehensive: {use_comprehensive}")
    print(f"  Real data (pendulum): {real_data}")
    print(f"  Batch benchmark: {not individual_only}")

    batch_only = "--batch-only" in sys.argv

    if individual_only or (not batch_only):
        run_individual_benchmark(n_timesteps, n_batches, n_states)

    if not individual_only:
        run_batch_benchmark(
            n_timesteps,
            n_batches,
            n_states,
            use_gpu,
            use_comprehensive,
            gpu_only,
            gpu_friendly,
            cpu_only,
            real_data,
        )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
