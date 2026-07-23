"""``TorchFeatureExtractor`` configuration presets.

Every name defined here is a *feature-name string* keyed configuration
(`FCParameters`/`MINIMAL_FEATURE_NAMES`/preset dicts below) — this module never imports a
calculator function. ``TorchFeatureExtractor`` resolves a feature name to its actual
``Feature`` class via ``SUPPORTED_FEATURES`` (`torch_feature_extractor.py`), not via
anything in this file.
"""

from collections.abc import Mapping
from typing import Any

FCParameters = Mapping[str, list[dict[str, Any]] | None]

# =============================================================================
# TSFRESH-COMPATIBLE CONFIGURATION (partial - will be extended)
# =============================================================================

TORCH_COMPREHENSIVE_FC_PARAMETERS: FCParameters = {
    # Minimal features (no parameters)
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
    # Simple statistics
    "abs_energy": None,
    "kurtosis": None,
    "skewness": None,
    "quantile": [{"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
    "variation_coefficient": None,
    # Change/difference
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "mean_change": None,
    "mean_second_derivative_central": None,
    # Counting
    "count_above_mean": None,
    "count_below_mean": None,
    "count_above": [{"t": 0}],
    "count_below": [{"t": 0}],
    # Boolean
    "has_duplicate": None,
    "has_duplicate_max": None,
    "has_duplicate_min": None,
    "has_variance_larger_than_standard_deviation": None,
    "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
    # Location
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,
    "last_location_of_maximum": None,
    "last_location_of_minimum": None,
    "index_mass_quantile": [{"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
    # Streak/pattern
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    "number_crossing_m": [{"m": 0}, {"m": -1}, {"m": 1}],
    "number_cwt_peaks": [{"n": 1}, {"n": 5}],
    "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
    # Autocorrelation
    "autocorrelation": [{"lag": lag} for lag in range(10)],
    "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
    "partial_autocorrelation": [{"lag": lag} for lag in range(10)],
    # Entropy/complexity
    "binned_entropy": [{"max_bins": 10}],
    "fourier_entropy": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "permutation_entropy": [{"tau": 1, "dimension": x} for x in [3, 4, 5, 6, 7]],
    "lempel_ziv_complexity": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "cid_ce": [{"normalize": True}, {"normalize": False}],
    # Frequency domain
    "fft_coefficient": [
        {"coeff": k, "attr": a} for a in ["real", "imag", "abs", "angle"] for k in range(100)
    ],
    "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
    "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
    "cwt_coefficients": [
        {"widths": (w,), "coeff": coeff, "w": w} for w in [2, 5, 10, 20] for coeff in range(15)
    ],
    # Trend/regression
    "linear_trend": [
        {"attr": "pvalue"},
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
        {"attr": "stderr"},
    ],
    "agg_linear_trend": [
        {"attr": attr, "chunk_size": i, "f_agg": f}
        for attr in ["rvalue", "intercept", "slope", "stderr"]
        for i in [5, 10, 50]
        for f in ["max", "min", "mean", "var"]
    ],
    "augmented_dickey_fuller": [
        {"attr": "teststat"},
        {"attr": "pvalue"},
        {"attr": "usedlag"},
    ],
    "ar_coefficient": [{"coeff": coeff, "k": 10} for coeff in range(11)],
    "linear_trend_timewise": [
        {"attr": "pvalue"},
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
        {"attr": "stderr"},
    ],
    # Reoccurrence
    "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
    "percentage_of_reoccurring_values_to_all_values": None,
    "sum_of_reoccurring_data_points": None,
    "sum_of_reoccurring_values": None,
    "ratio_value_number_to_time_series_length": None,
    # Advanced
    "benford_correlation": None,
    "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
    "c3": [{"lag": lag} for lag in range(1, 4)],
    "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
    "change_quantiles": [
        {"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
        for ql in [0.0, 0.2, 0.4, 0.6, 0.8]
        for qh in [0.2, 0.4, 0.6, 0.8, 1.0]
        for b in [False, True]
        for f in ["mean", "var"]
        if ql < qh
    ],
    "count_value": [{"value": value} for value in [0, 1, -1]],
    "count_in_range": [
        {"min_val": -1, "max_val": 1},
        {"min_val": -1e12, "max_val": 0},
        {"min_val": 0, "max_val": 1e12},
    ],
    "friedrich_coefficients": [{"coeff": coeff, "m": 3, "r": 30} for coeff in range(4)],
    "max_langevin_fixed_point": [{"m": 3, "r": 30}],
    "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": i} for i in range(10)],
    "ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]],
    "mean_n_absolute_max": [{"number_of_maxima": n} for n in [3, 5, 7]],
    # Multi-output/refactor candidates: calculators may return ``(B, S, K)``.
    "lyapunov_r": [
        {"emb_dim": 10, "lag": 1, "trajectory_len": 20, "tau": 1.0},
    ],
    "lyapunov_e": [
        {"emb_dim": 10, "matrix_dim": 4, "min_nb": 8, "min_tsep": 0, "tau": 1.0},
    ],
    "correlation_dimension": [
        {"emb_dim": 4, "lag": 1, "n_rvals": 50},
    ],
}

# Custom features (PyTorch-only, not in tsfresh)
TORCH_CUSTOM_FC_PARAMETERS: FCParameters = {
    "delta": None,
    "log_delta": None,
}

# Minimal feature names
MINIMAL_FEATURE_NAMES: list[str] = [
    # "sum_values",
    "median",
    "mean",
    # "length",
    "standard_deviation",
    "variance",
    "root_mean_square",
    "maximum",
    "absolute_maximum",
    "minimum",
    "delta",
    "log_delta",
]

# Minimal feature configuration (equivalent to tsfresh MinimalFCParameters + custom features)
TORCH_MINIMAL_FC_PARAMETERS: FCParameters = dict.fromkeys(MINIMAL_FEATURE_NAMES)

# Default configuration: minimal features
DEFAULT_TORCH_FC_PARAMETERS: FCParameters = {**TORCH_MINIMAL_FC_PARAMETERS}

# =============================================================================
# DYNAMICAL SYSTEM CLASSIFICATION FEATURES
# =============================================================================
# Features optimized for hierarchical attractor type classification:
# - variance: distinguishes fixed points (very low) from oscillating attractors
# - amplitude: distinguishes limit cycles with different oscillation sizes
# - mean: distinguishes fixed points/attractors at different locations
# - linear_trend (slope): detects drifting/rotating solutions
# - autocorrelation_periodicity: detects periodic vs chaotic behavior

DYNAMICAL_SYSTEM_FC_PARAMETERS: FCParameters = {
    "variance": None,
    "amplitude": None,
    "mean": None,
    "linear_trend": [{"attr": "slope"}],
    "autocorrelation_periodicity": [
        {"output": "strength"},
        {"output": "period"},
    ],
    "spectral_frequency_ratio": None,
}
