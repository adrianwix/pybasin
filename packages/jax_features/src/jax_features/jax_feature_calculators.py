# pyright: basic
"""JAX-based feature calculators for time series analysis.

This module provides GPU-accelerated feature calculators using JAX, implementing
a comprehensive set of tsfresh features. All functions are designed to work with
batched inputs and leverage JAX's vectorization and JIT compilation.

The calculators operate on time series data with shape (N, B, S) where:

- N: number of time steps
- B: batch size (number of trajectories)
- S: number of state variables

Each calculator returns features with shape (B, S).

Features implemented:

- MinimalFCParameters (10 features): sum_values, median, mean, length,
  standard_deviation, variance, root_mean_square, maximum, absolute_maximum, minimum

- Simple Statistics: abs_energy, kurtosis, skewness, quantile, variation_coefficient

- Change/Difference: absolute_sum_of_changes, mean_abs_change, mean_change,
  mean_second_derivative_central

- Counting: count_above, count_above_mean, count_below, count_below_mean

- Boolean: has_duplicate, has_duplicate_max, has_duplicate_min,
  has_variance_larger_than_standard_deviation, has_large_standard_deviation

- Location: first_location_of_maximum, first_location_of_minimum,
  last_location_of_maximum, last_location_of_minimum, index_mass_quantile

- Streak/Pattern: longest_strike_above_mean, longest_strike_below_mean,
  number_crossing_m, number_peaks, number_cwt_peaks

- Autocorrelation: autocorrelation, partial_autocorrelation, agg_autocorrelation

- Entropy/Complexity: permutation_entropy, binned_entropy, fourier_entropy,
  lempel_ziv_complexity, cid_ce

- Frequency Domain: fft_coefficient, fft_aggregated, spkt_welch_density, cwt_coefficients

- Trend/Regression: linear_trend, linear_trend_timewise, agg_linear_trend,
  ar_coefficient, augmented_dickey_fuller

- Reoccurrence: percentage_of_reoccurring_datapoints_to_all_datapoints,
  percentage_of_reoccurring_values_to_all_values, sum_of_reoccurring_data_points,
  sum_of_reoccurring_values, ratio_value_number_to_time_series_length

- Other Advanced: benford_correlation, c3, change_quantiles, energy_ratio_by_chunks,
  friedrich_coefficients, max_langevin_fixed_point, matrix_profile,
  mean_n_absolute_max, range_count, ratio_beyond_r_sigma, symmetry_looking,
  time_reversal_asymmetry_statistic, value_count

- Custom: delta, log_delta
"""

from collections.abc import Callable, Mapping
from math import factorial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, lax

from pybasin.feature_extractors.utils import format_feature_name

# =============================================================================
# MINIMAL FEATURES (tsfresh MinimalFCParameters - 10 features)
# =============================================================================


def sum_values(x: Array) -> Array:
    """Calculate the sum over the time series values."""
    return jnp.sum(x, axis=0)


def median(x: Array) -> Array:
    """Calculate the median of the time series."""
    return jnp.median(x, axis=0)


def mean(x: Array) -> Array:
    """Calculate the mean of the time series."""
    return jnp.mean(x, axis=0)


def length(x: Array) -> Array:
    """Calculate the length of the time series (number of time steps N).

    Optimized: Uses zeros + scalar addition to avoid multiplication.
    """
    n = x.shape[0]
    # zeros + scalar is faster than ones * scalar
    return jnp.zeros(x.shape[1:], dtype=jnp.float32) + n


def standard_deviation(x: Array) -> Array:
    """Calculate the standard deviation of the time series."""
    return jnp.std(x, axis=0)


def variance(x: Array) -> Array:
    """Calculate the variance of the time series."""
    return jnp.var(x, axis=0)


def root_mean_square(x: Array) -> Array:
    """Calculate the root mean square (RMS) of the time series."""
    return jnp.sqrt(jnp.mean(jnp.square(x), axis=0))


def maximum(x: Array) -> Array:
    """Calculate the maximum value of the time series."""
    return jnp.max(x, axis=0)


def absolute_maximum(x: Array) -> Array:
    """Calculate the maximum absolute value of the time series."""
    return jnp.max(jnp.abs(x), axis=0)


def minimum(x: Array) -> Array:
    """Calculate the minimum value of the time series."""
    return jnp.min(x, axis=0)


# =============================================================================
# SIMPLE STATISTICS (5 features)
# =============================================================================


def abs_energy(x: Array) -> Array:
    """Calculate the absolute energy (sum of squared values)."""
    return jnp.sum(jnp.square(x), axis=0)


def kurtosis(x: Array) -> Array:
    """Calculate the kurtosis (Fisher's definition, bias-corrected)."""
    n = x.shape[0]
    m = jnp.mean(x, axis=0)
    m4 = jnp.mean(jnp.power(x - m, 4), axis=0)
    m2 = jnp.var(x, axis=0)
    excess_kurt = (m4 / (m2**2 + 1e-12)) - 3.0
    correction = ((n - 1) / ((n - 2) * (n - 3) + 1e-12)) * ((n + 1) * excess_kurt + 6)
    return correction


def skewness(x: Array) -> Array:
    """Calculate the skewness (Fisher's definition, bias-corrected)."""
    n = x.shape[0]
    m = jnp.mean(x, axis=0)
    m3 = jnp.mean(jnp.power(x - m, 3), axis=0)
    m2 = jnp.var(x, axis=0)
    g1 = m3 / (jnp.power(m2, 1.5) + 1e-12)
    correction = jnp.sqrt(n * (n - 1)) / (n - 2 + 1e-12)
    return correction * g1


def quantile(x: Array, q: float) -> Array:
    """Calculate the q-quantile of the time series."""
    return jnp.quantile(x, q, axis=0)


def variation_coefficient(x: Array) -> Array:
    """Calculate the coefficient of variation (std / mean)."""
    return jnp.std(x, axis=0) / (jnp.abs(jnp.mean(x, axis=0)) + 1e-12)


# =============================================================================
# CHANGE/DIFFERENCE BASED (4 features)
# =============================================================================


def absolute_sum_of_changes(x: Array) -> Array:
    """Calculate the sum of absolute consecutive changes."""
    return jnp.sum(jnp.abs(jnp.diff(x, axis=0)), axis=0)


def mean_abs_change(x: Array) -> Array:
    """Calculate the mean of absolute consecutive changes."""
    return jnp.mean(jnp.abs(jnp.diff(x, axis=0)), axis=0)


def mean_change(x: Array) -> Array:
    """Calculate the mean of consecutive changes.

    Optimized: Uses O(1) formula (x[-1] - x[0]) / (n-1) instead of computing all diffs.
    Pre-computes 1/(n-1) to use multiplication instead of division.
    """
    n = x.shape[0]
    inv_n_minus_1 = 1.0 / (n - 1)
    return (x[-1] - x[0]) * inv_n_minus_1


def mean_second_derivative_central(x: Array) -> Array:
    """Calculate the mean of central approximation of second derivative.

    Optimized: Uses O(1) formula based on telescoping sum.
    The sum of (x[i+1] - 2*x[i] + x[i-1]) for i=1..n-2 telescopes to:
    (x[-1] - x[-2] - x[1] + x[0]) when summed, divided by (n-2) for mean.
    Pre-computes divisor to use multiplication.
    """
    n = x.shape[0]
    inv_2n_minus_4 = 1.0 / (2 * (n - 2))
    return (x[-1] - x[-2] - x[1] + x[0]) * inv_2n_minus_4


# =============================================================================
# COUNTING FEATURES (4 features)
# =============================================================================


def count_above(x: Array, t: float) -> Array:
    """Calculate the percentage of values above threshold t."""
    return jnp.mean(x > t, axis=0)


def count_above_mean(x: Array) -> Array:
    """Calculate the number of values above the mean."""
    m = jnp.mean(x, axis=0, keepdims=True)
    return jnp.sum(x > m, axis=0).astype(jnp.float32)


def count_below(x: Array, t: float) -> Array:
    """Calculate the percentage of values below threshold t."""
    return jnp.mean(x < t, axis=0)


def count_below_mean(x: Array) -> Array:
    """Calculate the number of values below the mean."""
    m = jnp.mean(x, axis=0, keepdims=True)
    return jnp.sum(x < m, axis=0).astype(jnp.float32)


# =============================================================================
# BOOLEAN FEATURES (5 features)
# =============================================================================


def has_duplicate(x: Array) -> Array:
    """Check if any value occurs more than once (returns 1.0 or 0.0).

    Optimized: Uses sorting to detect duplicates - this is already efficient.
    The sort is O(n log n) which is optimal for duplicate detection.
    """
    sorted_vals = jnp.sort(x, axis=0)
    # Any zero diff means consecutive equal values (duplicate)
    has_dup = jnp.any(sorted_vals[1:] == sorted_vals[:-1], axis=0)
    return has_dup.astype(jnp.float32)


def has_duplicate_max(x: Array) -> Array:
    """Check if maximum value occurs more than once."""
    max_val = jnp.max(x, axis=0, keepdims=True)
    return (jnp.sum(x == max_val, axis=0) > 1).astype(jnp.float32)


def has_duplicate_min(x: Array) -> Array:
    """Check if minimum value occurs more than once."""
    min_val = jnp.min(x, axis=0, keepdims=True)
    return (jnp.sum(x == min_val, axis=0) > 1).astype(jnp.float32)


def has_variance_larger_than_standard_deviation(x: Array) -> Array:
    """Check if variance > standard deviation (equivalent to std > 1)."""
    return (jnp.std(x, axis=0) > 1.0).astype(jnp.float32)


def has_large_standard_deviation(x: Array, r: float = 0.25) -> Array:
    """Check if std > r * (max - min)."""
    std = jnp.std(x, axis=0)
    range_val = jnp.max(x, axis=0) - jnp.min(x, axis=0)
    return (std > r * range_val).astype(jnp.float32)


# =============================================================================
# LOCATION FEATURES (5 features)
# =============================================================================


def first_location_of_maximum(x: Array) -> Array:
    """Calculate the relative first location of maximum value."""
    n = x.shape[0]
    return jnp.argmax(x, axis=0).astype(jnp.float32) / n


def first_location_of_minimum(x: Array) -> Array:
    """Calculate the relative first location of minimum value."""
    n = x.shape[0]
    return jnp.argmin(x, axis=0).astype(jnp.float32) / n


def last_location_of_maximum(x: Array) -> Array:
    """Calculate the relative last location of maximum value."""
    n = x.shape[0]
    x_reversed = jnp.flip(x, axis=0)
    return 1.0 - jnp.argmax(x_reversed, axis=0).astype(jnp.float32) / n


def last_location_of_minimum(x: Array) -> Array:
    """Calculate the relative last location of minimum value.

    Optimized: Uses negative indexing trick to find last occurrence.
    """
    n = x.shape[0]
    # Find minimum value
    min_val = jnp.min(x, axis=0, keepdims=True)
    # Create mask where value equals minimum
    is_min = x == min_val
    # Multiply by indices and take max to get last index
    indices = jnp.arange(n).reshape(-1, 1, 1)
    last_idx = jnp.max(jnp.where(is_min, indices, -1), axis=0)
    return last_idx.astype(jnp.float32) / n


def index_mass_quantile(x: Array, q: float) -> Array:
    """Calculate the relative index where q% of mass lies to the left."""
    abs_x = jnp.abs(x)
    total = jnp.sum(abs_x, axis=0, keepdims=True)
    cumsum = jnp.cumsum(abs_x, axis=0)
    n = x.shape[0]
    threshold = q * total
    indices = jnp.argmax(cumsum >= threshold, axis=0)
    return indices.astype(jnp.float32) / n


# =============================================================================
# STREAK/PATTERN FEATURES (5 features)
# =============================================================================


def _longest_consecutive_run(mask: Array) -> Array:
    """Calculate longest consecutive run of True values.

    Efficient vectorized implementation without explicit loops.
    """
    mask_int = mask.astype(jnp.int32)
    n = mask.shape[0]

    # For each True position, compute distance to previous False
    # This gives run length at each position
    false_positions = jnp.where(mask_int == 0, jnp.arange(n).reshape(-1, 1, 1), -1)
    last_false = jnp.maximum.accumulate(false_positions, axis=0)
    positions = jnp.arange(n).reshape(-1, 1, 1) * jnp.ones_like(mask_int)
    run_lengths = jnp.where(mask_int == 1, positions - last_false, 0)

    return jnp.max(run_lengths, axis=0).astype(jnp.float32)


def longest_strike_above_mean(x: Array) -> Array:
    """Calculate the longest consecutive subsequence above mean."""
    m = jnp.mean(x, axis=0, keepdims=True)
    above = x > m
    return _longest_consecutive_run(above)


def longest_strike_below_mean(x: Array) -> Array:
    """Calculate the longest consecutive subsequence below mean."""
    m = jnp.mean(x, axis=0, keepdims=True)
    below = x < m
    return _longest_consecutive_run(below)


def number_crossing_m(x: Array, m: float) -> Array:
    """Calculate number of crossings of level m."""
    shifted = x - m
    signs = jnp.sign(shifted)
    crossings = jnp.abs(jnp.diff(signs, axis=0)) > 0
    return jnp.sum(crossings, axis=0).astype(jnp.float32)


def number_peaks(x: Array, n: int) -> Array:
    """Calculate number of peaks with support n."""

    def is_peak(arr: Array, support: int) -> Array:
        def check_peak(i: int, result: Array) -> Array:
            left = jnp.roll(arr, i, axis=0)
            right = jnp.roll(arr, -i, axis=0)
            return result & (arr > left) & (arr > right)

        result = jnp.ones(arr.shape, dtype=jnp.bool_)
        result = lax.fori_loop(1, support + 1, check_peak, result)

        # Use mask instead of dynamic slicing for JIT compatibility
        indices = jnp.arange(arr.shape[0]).reshape(-1, 1, 1)
        edge_mask = (indices >= support) & (indices < arr.shape[0] - support)
        result = result & edge_mask
        return result

    peaks = is_peak(x, n)
    return jnp.sum(peaks, axis=0).astype(jnp.float32)


def number_cwt_peaks(x: Array, max_width: int = 5) -> Array:
    """Calculate number of peaks using continuous wavelet transform approach.

    Simplified version using multi-scale peak detection.
    """

    def accumulate_peaks(width: int, total: Array) -> Array:
        peaks = number_peaks(x, width)
        return total + peaks

    total_peaks = lax.fori_loop(
        1, max_width + 1, accumulate_peaks, jnp.zeros(x.shape[1:], dtype=jnp.float32)
    )
    return total_peaks / max_width


# =============================================================================
# AUTOCORRELATION FEATURES (3 features)
# =============================================================================


def autocorrelation(x: Array, lag: int) -> Array:
    """Calculate autocorrelation at specified lag."""
    n = x.shape[0]
    if lag >= n:
        return jnp.zeros(x.shape[1:])
    if lag == 0:
        return jnp.ones(x.shape[1:])

    m = jnp.mean(x, axis=0, keepdims=True)
    x_centered = x - m
    var = jnp.var(x, axis=0)

    autocov = jnp.mean(x_centered[:-lag] * x_centered[lag:], axis=0)
    return autocov / (var + 1e-12)


def partial_autocorrelation(x: Array, lag: int) -> Array:
    """Calculate partial autocorrelation at specified lag using Durbin-Levinson."""
    if lag == 0:
        return jnp.ones(x.shape[1:])

    acf = jnp.stack([autocorrelation(x, i) for i in range(lag + 1)], axis=0)

    if lag == 1:
        return acf[1]

    def durbin_levinson_step(k: int, phi: Array) -> Array:
        j_range = jnp.arange(1, lag + 1)
        mask = j_range < k

        phi_vals = phi[k - 1, j_range]
        phi_prev = jnp.where(mask.reshape(-1, *([1] * (phi.ndim - 2))), phi_vals, 0.0)

        acf_rev_indices = jnp.clip(k - j_range, 0, lag)
        acf_rev = jnp.take(acf, acf_rev_indices, axis=0)
        acf_rev = jnp.where(mask.reshape(-1, *([1] * (acf.ndim - 1))), acf_rev, 0.0)

        acf_fwd = jnp.take(acf, jnp.clip(j_range, 0, lag), axis=0)
        acf_fwd = jnp.where(mask.reshape(-1, *([1] * (acf.ndim - 1))), acf_fwd, 0.0)

        acf_k = lax.dynamic_slice(acf, (k, 0, 0), (1, acf.shape[1], acf.shape[2]))
        acf_k = acf_k.reshape(acf.shape[1:])

        num = acf_k - jnp.sum(phi_prev * acf_rev, axis=0, keepdims=False)
        denom = 1 - jnp.sum(phi_prev * acf_fwd, axis=0, keepdims=False)

        phi_kk = num / (denom + 1e-12)
        phi = phi.at[k, k].set(phi_kk)

        phi_k_minus_1 = phi[k - 1, j_range]
        phi_k_minus_1_reverse = phi[k - 1, jnp.clip(k - j_range, 0, lag)]

        phi_k_j = phi_k_minus_1 - phi_kk.reshape(*([1] * 1), *phi_kk.shape) * phi_k_minus_1_reverse
        phi_k_j = jnp.where(mask.reshape(-1, *([1] * (phi.ndim - 2))), phi_k_j, phi[k, j_range])

        phi = phi.at[k, j_range].set(phi_k_j)

        return phi

    phi = jnp.zeros((lag + 1, lag + 1) + x.shape[1:])
    phi = phi.at[1, 1].set(acf[1])
    phi = lax.fori_loop(2, lag + 1, durbin_levinson_step, phi)

    return phi[lag, lag]


def agg_autocorrelation(x: Array, maxlag: int = 40, f_agg: str = "mean") -> Array:
    """Calculate aggregated autocorrelation statistic.

    JIT-compatible: uses FFT-based autocorrelation which is fully vectorized.
    """
    n = x.shape[0]
    m = jnp.mean(x, axis=0, keepdims=True)
    x_centered = x - m
    var = jnp.var(x, axis=0)

    fft_size = 2 * n
    fft_x = jnp.fft.rfft(x_centered, n=fft_size, axis=0)
    acf_full = jnp.fft.irfft(fft_x * jnp.conj(fft_x), n=fft_size, axis=0)[:n] / n

    acf_normalized = acf_full / (var + 1e-12)

    effective_maxlag = jnp.minimum(maxlag, n - 1)
    acf_values = acf_normalized[1 : maxlag + 1]

    lags = jnp.arange(1, maxlag + 1)
    valid_mask = (lags <= effective_maxlag).reshape(-1, 1, 1)
    acf_values = jnp.where(valid_mask, acf_values, 0.0)
    n_valid = jnp.sum(lags <= effective_maxlag)

    if f_agg == "mean":
        return jnp.sum(acf_values, axis=0) / jnp.maximum(n_valid, 1)
    elif f_agg == "var":
        mean_acf = jnp.sum(acf_values, axis=0) / jnp.maximum(n_valid, 1)
        sq_diff = jnp.where(valid_mask, (acf_values - mean_acf) ** 2, 0.0)
        return jnp.sum(sq_diff, axis=0) / jnp.maximum(n_valid, 1)
    elif f_agg == "std":
        mean_acf = jnp.sum(acf_values, axis=0) / jnp.maximum(n_valid, 1)
        sq_diff = jnp.where(valid_mask, (acf_values - mean_acf) ** 2, 0.0)
        return jnp.sqrt(jnp.sum(sq_diff, axis=0) / jnp.maximum(n_valid, 1))
    elif f_agg == "median":
        sorted_acf = jnp.sort(acf_values, axis=0)
        mid_idx = maxlag // 2
        return sorted_acf[mid_idx]
    else:
        return jnp.sum(acf_values, axis=0) / jnp.maximum(n_valid, 1)


# =============================================================================
# ENTROPY/COMPLEXITY FEATURES (5 features, excluding sample/approximate entropy)
# =============================================================================


def permutation_entropy(x: Array, tau: int = 1, dimension: int = 3) -> Array:
    """Calculate permutation entropy.

    Vectorized: Uses matrix operations to compute pattern indices for all
    positions and batches simultaneously, with one-hot counting.
    """
    n = x.shape[0]
    n_patterns = n - (dimension - 1) * tau

    if n_patterns <= 0:
        return jnp.zeros(x.shape[1:])

    max_patterns = factorial(dimension)

    # Build all pattern windows at once using advanced indexing
    # pattern_starts: (n_patterns,), offsets: (dimension,)
    pattern_starts = jnp.arange(n_patterns)
    offsets = jnp.arange(dimension) * tau
    # indices: (n_patterns, dimension)
    indices = pattern_starts[:, None] + offsets[None, :]
    # patterns: (n_patterns, dimension, B, S)
    patterns = x[indices]

    # Compute ranks along dimension axis (axis=1)
    # ranks: (n_patterns, dimension, B, S)
    ranks = jnp.argsort(jnp.argsort(patterns, axis=1), axis=1)

    # Convert ranks to pattern indices using factorial number system
    multipliers = jnp.array([factorial(dimension - 1 - i) for i in range(dimension)])
    # pattern_indices: (n_patterns, B, S)
    pattern_indices = jnp.sum(ranks * multipliers[None, :, None, None], axis=1).astype(jnp.int32)

    # Count patterns using one-hot encoding
    # patterns_range: (max_patterns, 1, 1, 1)
    patterns_range = jnp.arange(max_patterns).reshape(max_patterns, 1, 1, 1)
    # pattern_indices_expanded: (1, n_patterns, B, S)
    pattern_indices_expanded = pattern_indices.reshape(1, n_patterns, x.shape[1], x.shape[2])
    # counts: (max_patterns, B, S)
    counts = jnp.sum(pattern_indices_expanded == patterns_range, axis=1)

    # Compute entropy
    probs = counts / n_patterns
    log_probs = jnp.where(probs > 0, jnp.log(probs), 0.0)
    entropy = -jnp.sum(probs * log_probs, axis=0)  # (B, S)

    # Normalize by maximum entropy
    max_entropy = jnp.log(jnp.array(max_patterns, dtype=jnp.float32))
    return entropy / max_entropy


def binned_entropy(x: Array, max_bins: int = 10) -> Array:
    """Calculate binned entropy of the time series.

    Vectorized: Uses one-hot encoding for bin counting across all batches simultaneously.
    """
    n = x.shape[0]
    min_val = jnp.min(x, axis=0, keepdims=True)
    max_val = jnp.max(x, axis=0, keepdims=True)
    range_val = max_val - min_val + 1e-12

    normalized = (x - min_val) / range_val
    bin_indices = jnp.floor(normalized * max_bins).astype(jnp.int32)
    bin_indices = jnp.clip(bin_indices, 0, max_bins - 1)  # shape (N, B, S)

    # One-hot encode bins and sum along time axis to get counts
    # bins_range: (max_bins, 1, 1, 1), bin_indices: (1, N, B, S)
    bins_range = jnp.arange(max_bins).reshape(max_bins, 1, 1, 1)
    bin_indices_expanded = bin_indices.reshape(1, n, x.shape[1], x.shape[2])
    counts = jnp.sum(bin_indices_expanded == bins_range, axis=1)  # shape (max_bins, B, S)

    # Compute probabilities and entropy
    probs = counts / n  # shape (max_bins, B, S)
    # Use where to avoid log(0): if prob > 0, use prob * log(prob), else 0
    log_probs = jnp.where(probs > 0, jnp.log(probs), 0.0)
    entropy = -jnp.sum(probs * log_probs, axis=0)  # shape (B, S)

    return entropy


def fourier_entropy(x: Array, bins: int = 10) -> Array:
    """Calculate entropy of the power spectral density."""
    fft_vals = jnp.fft.rfft(x, axis=0)
    psd = jnp.abs(fft_vals) ** 2
    psd_normalized = psd / (jnp.sum(psd, axis=0, keepdims=True) + 1e-12)

    psd_normalized = jnp.where(psd_normalized > 0, psd_normalized, 1e-12)
    return -jnp.sum(psd_normalized * jnp.log(psd_normalized), axis=0)


def lempel_ziv_complexity(x: Array, bins: int = 2) -> Array:
    """Calculate Lempel-Ziv complexity estimate.

    JIT-compatible: uses a simplified approximation based on run-length encoding.
    Counts transitions in the binary sequence as a proxy for complexity.
    """
    n = x.shape[0]
    med = jnp.median(x, axis=0, keepdims=True)
    binary = (x > med).astype(jnp.int32)

    transitions = jnp.sum(jnp.abs(jnp.diff(binary, axis=0)), axis=0)

    c_approx = transitions + 1

    b_n = n / (jnp.log(jnp.float32(n)) + 1e-12)
    return c_approx / b_n


def cid_ce(x: Array, normalize: bool = True) -> Array:
    """Calculate complexity estimate based on consecutive differences."""
    diff = jnp.diff(x, axis=0)
    ce = jnp.sqrt(jnp.sum(jnp.square(diff), axis=0))

    if normalize:
        std = jnp.std(x, axis=0)
        return ce / (std + 1e-12)
    return ce


# =============================================================================
# FREQUENCY DOMAIN FEATURES (4 features)
# =============================================================================


def fft_coefficient(x: Array, coeff: int = 0, attr: str = "abs") -> Array:
    """Calculate FFT coefficient attributes."""
    fft_vals = jnp.fft.rfft(x, axis=0)

    coeff_val = jnp.where(
        coeff >= fft_vals.shape[0], 0.0, fft_vals[jnp.minimum(coeff, fft_vals.shape[0] - 1)]
    )

    # Compute all possible results
    results = {
        "abs": jnp.abs(coeff_val),
        "real": jnp.real(coeff_val),
        "imag": jnp.imag(coeff_val),
        "angle": jnp.angle(coeff_val),
    }

    return results.get(attr, jnp.abs(coeff_val))


def fft_aggregated(x: Array, aggtype: str = "centroid") -> Array:
    """Calculate aggregated FFT spectrum statistics."""
    fft_vals = jnp.fft.rfft(x, axis=0)
    spectrum = jnp.abs(fft_vals)

    n_freqs = spectrum.shape[0]
    freqs = jnp.arange(n_freqs).reshape(-1, 1, 1)

    total = jnp.sum(spectrum, axis=0) + 1e-12

    # Compute all needed intermediate values
    centroid = jnp.sum(freqs * spectrum, axis=0) / total
    var = jnp.sum(((freqs - centroid) ** 2) * spectrum, axis=0) / total

    # Precompute all possible results
    results = {
        "centroid": centroid,
        "variance": var,
        "skew": jnp.sum(((freqs - centroid) ** 3) * spectrum, axis=0)
        / (total * (var**1.5 + 1e-12)),
        "kurtosis": jnp.sum(((freqs - centroid) ** 4) * spectrum, axis=0)
        / (total * (var**2 + 1e-12)),
    }

    return results.get(aggtype, centroid)


def spkt_welch_density(x: Array, coeff: int = 0) -> Array:
    """Estimate power spectral density using Welch's method (simplified)."""
    n = x.shape[0]
    segment_len = min(256, n)

    fft_vals = jnp.fft.rfft(x[:segment_len], axis=0)
    psd = jnp.abs(fft_vals) ** 2 / segment_len

    if coeff >= psd.shape[0]:
        return jnp.zeros(x.shape[1:])

    return psd[coeff]


def cwt_coefficients(x: Array, widths: tuple[int, ...] = (2,), coeff: int = 0, w: int = 2) -> Array:
    """Calculate continuous wavelet transform coefficients using Ricker wavelet.

    This matches tsfresh's cwt_coefficients interface:
    - widths: tuple of scale values to compute CWT for
    - coeff: coefficient index to extract from the convolution result
    - w: which width from the widths tuple to use for the result

    Note: This implementation uses a direct Ricker wavelet convolution which differs
    from tsfresh's pywt.cwt in normalization. Results have the same sign but different
    scaling. This is acceptable for feature extraction where relative patterns matter.

    :param x: Input array of shape (N, B, S).
    :param widths: Tuple of wavelet width (scale) parameters.
    :param coeff: Coefficient index to extract.
    :param w: Which width from widths to use (must be in widths).
    :return: Array of shape (B, S) with the CWT coefficient.
    """
    n = x.shape[0]

    if w not in widths:
        return jnp.zeros(x.shape[1:])

    width = w

    t = jnp.arange(-width * 4, width * 4 + 1, dtype=jnp.float32)
    amplitude = 2 / (jnp.sqrt(3 * width) * (jnp.pi**0.25))
    wavelet = amplitude * (1 - (t / width) ** 2) * jnp.exp(-0.5 * (t / width) ** 2)

    pad_size = len(t) // 2
    x_padded = jnp.pad(x, ((pad_size, pad_size), (0, 0), (0, 0)), mode="reflect")

    kernel = wavelet[::-1].reshape(1, 1, -1)
    x_conv = x_padded.transpose(1, 2, 0).reshape(-1, 1, x_padded.shape[0])
    conv_out = jax.lax.conv_general_dilated(x_conv, kernel, window_strides=(1,), padding="VALID")
    result = conv_out.reshape(x.shape[1], x.shape[2], n).transpose(2, 0, 1)

    if coeff >= n:
        return jnp.zeros(x.shape[1:])

    return result[coeff]


# =============================================================================
# TREND/REGRESSION FEATURES (5 features)
# =============================================================================


def linear_trend(x: Array, attr: str = "slope") -> Array:
    """Calculate linear least-squares regression attributes."""
    n = x.shape[0]
    t = jnp.arange(n, dtype=jnp.float32).reshape(-1, 1, 1)

    t_mean = jnp.mean(t)
    x_mean = jnp.mean(x, axis=0, keepdims=True)

    ss_tt = jnp.sum((t - t_mean) ** 2)
    ss_tx = jnp.sum((t - t_mean) * (x - x_mean), axis=0)

    slope = ss_tx / (ss_tt + 1e-12)
    intercept = x_mean.squeeze(0) - slope * t_mean

    # Compute all attributes
    ss_xx = jnp.sum((x - x_mean) ** 2, axis=0)
    r = ss_tx / (jnp.sqrt(ss_tt * ss_xx) + 1e-12)
    y_pred = intercept + slope * t
    residuals = x - y_pred
    mse = jnp.sum(residuals**2, axis=0) / (n - 2 + 1e-12)
    stderr = jnp.sqrt(mse / (ss_tt + 1e-12))

    results = {
        "slope": slope,
        "intercept": intercept,
        "rvalue": r,
        "pvalue": 1 - jnp.abs(r),
        "stderr": stderr,
    }

    return results.get(attr, slope)


def linear_trend_timewise(x: Array, attr: str = "slope") -> Array:
    """Calculate linear trend (same as linear_trend for our use case)."""
    return linear_trend(x, attr)


def agg_linear_trend(
    x: Array, chunk_size: int = 10, f_agg: str = "mean", attr: str = "slope"
) -> Array:
    """Calculate linear trend on aggregated chunks."""
    n = x.shape[0]
    n_chunks = n // chunk_size

    if n_chunks < 2:
        return linear_trend(x, attr)

    chunks = x[: n_chunks * chunk_size].reshape(n_chunks, chunk_size, x.shape[1], x.shape[2])

    # Precompute all aggregations
    agg_functions = {
        "mean": lambda: jnp.mean(chunks, axis=1),
        "var": lambda: jnp.var(chunks, axis=1),
        "std": lambda: jnp.std(chunks, axis=1),
        "min": lambda: jnp.min(chunks, axis=1),
        "max": lambda: jnp.max(chunks, axis=1),
        "median": lambda: jnp.median(chunks, axis=1),
    }

    agg_chunks = agg_functions.get(f_agg, agg_functions["mean"])()

    return linear_trend(agg_chunks, attr)

    # Precompute all aggregations
    agg_functions = {
        "mean": lambda: jnp.mean(chunks, axis=1),
        "var": lambda: jnp.var(chunks, axis=1),
        "std": lambda: jnp.std(chunks, axis=1),
        "min": lambda: jnp.min(chunks, axis=1),
        "max": lambda: jnp.max(chunks, axis=1),
        "median": lambda: jnp.median(chunks, axis=1),
    }

    agg_chunks = agg_functions.get(f_agg, agg_functions["mean"])()

    return linear_trend(agg_chunks, attr)


def ar_coefficient(x: Array, k: int = 1, coeff: int = 0) -> Array:
    """Calculate autoregressive AR(k) coefficient using Yule-Walker equations."""
    if k < 1:
        return jnp.zeros(x.shape[1:])

    acf = jnp.stack([autocorrelation(x, i) for i in range(k + 1)], axis=0)

    # Build correlation matrix using broadcasting
    i_idx, j_idx = jnp.meshgrid(jnp.arange(k), jnp.arange(k), indexing="ij")
    corr_matrix = acf[jnp.abs(i_idx - j_idx)]  # Shape: (k, k, B, S)

    r = acf[1 : k + 1]  # Shape: (k, B, S)

    # Vectorize over batch and state dimensions
    def solve_for_one(corr_mat: Array, r_vec: Array) -> Array:
        r_reg = corr_mat + 1e-6 * jnp.eye(k)
        coeffs = jnp.linalg.solve(r_reg, r_vec)
        coeff_val = coeffs.at[coeff].get()
        zero = jnp.zeros_like(coeff_val)
        return jnp.asarray(jnp.where(coeff < k, coeff_val, zero))

    # Apply vmap over both batch and state dimensions
    solve_batch = jax.vmap(jax.vmap(solve_for_one, in_axes=(2, 1)), in_axes=(2, 1))
    result = solve_batch(corr_matrix, r)

    return result


def augmented_dickey_fuller(x: Array, attr: str = "teststat") -> Array:
    """Calculate Augmented Dickey-Fuller test statistic (simplified)."""
    n = x.shape[0]

    diff_x = jnp.diff(x, axis=0)
    x_lag = x[:-1]

    x_lag_mean = jnp.mean(x_lag, axis=0, keepdims=True)
    diff_mean = jnp.mean(diff_x, axis=0, keepdims=True)

    ss_xx = jnp.sum((x_lag - x_lag_mean) ** 2, axis=0)
    ss_xy = jnp.sum((x_lag - x_lag_mean) * (diff_x - diff_mean), axis=0)

    gamma = ss_xy / (ss_xx + 1e-12)

    y_pred = diff_mean + gamma * (x_lag - x_lag_mean)
    residuals = diff_x - y_pred
    se_gamma = jnp.sqrt(jnp.sum(residuals**2, axis=0) / ((n - 2) * ss_xx + 1e-12))

    test_stat = gamma / (se_gamma + 1e-12)

    results = {
        "teststat": test_stat,
        "pvalue": 1 - jnp.abs(jnp.tanh(test_stat)),
        "usedlag": jnp.ones(x.shape[1:]),
    }

    return results.get(attr, test_stat)


# =============================================================================
# REOCCURRENCE FEATURES (5 features)
# =============================================================================


def percentage_of_reoccurring_datapoints_to_all_datapoints(x: Array) -> Array:
    """Calculate percentage of non-unique data points.

    Vectorized: Uses sorting to count unique values without Python loops.
    """
    n = x.shape[0]
    # Sort along time axis
    sorted_vals = jnp.sort(x, axis=0)
    # Count unique values: 1 + number of non-zero diffs
    n_unique = 1 + jnp.sum(jnp.diff(sorted_vals, axis=0) != 0, axis=0)
    return 1.0 - n_unique / n


def percentage_of_reoccurring_values_to_all_values(x: Array) -> Array:
    """Calculate percentage of values occurring more than once.

    Vectorized: Uses sorting to count unique values that reoccur.
    This returns the fraction of unique values that appear more than once.
    """
    sorted_vals = jnp.sort(x, axis=0)
    # is_same: 1 if current equals previous
    is_same = sorted_vals[1:] == sorted_vals[:-1]  # (N-1, B, S)

    # Count unique values
    n_unique = 1.0 + jnp.sum(~is_same, axis=0)  # (B, S)

    # Count unique values that have duplicates:
    # A unique value has duplicates if is_same[i] is True (equals next)
    # But we only count once per run, so count where is_same AND NOT prev_same
    # Simpler: count runs of duplicates by detecting starts of runs
    prev_not_same = jnp.concatenate(
        [jnp.ones((1,) + x.shape[1:], dtype=jnp.bool_), ~is_same[:-1]], axis=0
    )  # (N-1, B, S)
    # Start of dup run: is_same AND (prev was not same OR first)
    dup_run_starts = is_same & prev_not_same
    n_reoccurring_unique = jnp.sum(dup_run_starts, axis=0).astype(jnp.float32)

    return n_reoccurring_unique / n_unique


def sum_of_reoccurring_data_points(x: Array) -> Array:
    """Calculate sum of all data points occurring more than once.

    Vectorized: Uses sorting and masking to sum duplicate values.
    """
    sorted_vals = jnp.sort(x, axis=0)
    # is_equal[i] = True if sorted[i+1] == sorted[i]
    is_equal = sorted_vals[1:] == sorted_vals[:-1]  # (N-1, B, S)

    # A value is duplicate if it equals prev OR equals next
    # equals_prev: (N, B, S) - first row is False
    equals_prev = jnp.concatenate(
        [jnp.zeros((1,) + x.shape[1:], dtype=jnp.bool_), is_equal], axis=0
    )
    # equals_next: (N, B, S) - last row is False
    equals_next = jnp.concatenate(
        [is_equal, jnp.zeros((1,) + x.shape[1:], dtype=jnp.bool_)], axis=0
    )
    # Value is part of duplicate group if either condition is true
    is_dup = equals_prev | equals_next

    return jnp.sum(jnp.where(is_dup, sorted_vals, 0.0), axis=0)


def sum_of_reoccurring_values(x: Array) -> Array:
    """Calculate sum of unique values occurring more than once.

    Vectorized: Sums only one occurrence of each value that appears multiple times.
    """
    sorted_vals = jnp.sort(x, axis=0)
    # is_new: True if this value differs from previous (or is first)
    is_new = jnp.concatenate(
        [jnp.ones((1,) + x.shape[1:], dtype=jnp.bool_), sorted_vals[1:] != sorted_vals[:-1]],
        axis=0,
    )  # (N, B, S)
    # has_next_dup: True if this value equals next value
    has_next_dup = jnp.concatenate(
        [sorted_vals[:-1] == sorted_vals[1:], jnp.zeros((1,) + x.shape[1:], dtype=jnp.bool_)],
        axis=0,
    )  # (N, B, S)
    # First occurrence of a value that has duplicates
    is_first_of_dup = is_new & has_next_dup

    return jnp.sum(jnp.where(is_first_of_dup, sorted_vals, 0.0), axis=0)


def ratio_value_number_to_time_series_length(x: Array) -> Array:
    """Calculate ratio of unique values to time series length.

    Vectorized: Uses sorting and direct comparison for efficiency.
    """
    n = x.shape[0]
    sorted_vals = jnp.sort(x, axis=0)
    # Count unique: 1 (first element) + number of transitions
    n_unique = 1 + jnp.sum(sorted_vals[1:] != sorted_vals[:-1], axis=0)
    return n_unique / n


# =============================================================================
# OTHER ADVANCED FEATURES (13 features)
# =============================================================================


def benford_correlation(x: Array) -> Array:
    """Calculate correlation with Benford's Law distribution.

    Vectorized: Computes first digit distribution and correlation without Python loops.
    """
    # Benford's law distribution for digits 1-9
    benford_dist = jnp.log10(1 + 1 / jnp.arange(1, 10))  # shape (9,)
    ben_mean = jnp.mean(benford_dist)
    ben_centered = benford_dist - ben_mean
    ben_var = jnp.sum(ben_centered**2)

    # Get absolute values, replace zeros with 1 to avoid log issues
    vals = jnp.abs(x)
    vals = jnp.where(vals > 0, vals, 1.0)

    # Extract first digit: floor(val / 10^floor(log10(val)))
    log_vals = jnp.floor(jnp.log10(vals + 1e-12))
    first_digits = jnp.floor(vals / (10**log_vals)).astype(jnp.int32)
    first_digits = jnp.clip(first_digits, 1, 9)  # shape (N, B, S)

    # Count occurrences of each digit 1-9 for each (B, S) combination
    # Use one-hot encoding and sum along time axis
    n = x.shape[0]
    digits_range = jnp.arange(1, 10).reshape(9, 1, 1, 1)  # shape (9, 1, 1, 1)
    first_digits_expanded = first_digits.reshape(1, n, x.shape[1], x.shape[2])  # shape (1, N, B, S)
    counts = jnp.sum(first_digits_expanded == digits_range, axis=1)  # shape (9, B, S)

    # Normalize to get observed distribution
    observed = counts / (n + 1e-12)  # shape (9, B, S)

    # Compute correlation for each (B, S)
    obs_mean = jnp.mean(observed, axis=0, keepdims=True)  # shape (1, B, S)
    obs_centered = observed - obs_mean  # shape (9, B, S)

    # Pearson correlation
    num = jnp.sum(obs_centered * ben_centered.reshape(9, 1, 1), axis=0)  # shape (B, S)
    obs_var = jnp.sum(obs_centered**2, axis=0)  # shape (B, S)
    denom = jnp.sqrt(obs_var * ben_var + 1e-12)

    return num / denom


def c3(x: Array, lag: int) -> Array:
    """Calculate c3 statistic for non-linearity measure."""
    n = x.shape[0]
    if 2 * lag >= n:
        return jnp.zeros(x.shape[1:])

    return jnp.mean(x[2 * lag :] * x[lag:-lag] * x[: -2 * lag], axis=0)


def change_quantiles(
    x: Array, ql: float, qh: float, isabs: bool = True, f_agg: str = "mean"
) -> Array:
    """Calculate aggregated change within quantile corridor."""
    q_low = jnp.quantile(x, ql, axis=0, keepdims=True)
    q_high = jnp.quantile(x, qh, axis=0, keepdims=True)

    mask = (x >= q_low) & (x <= q_high)

    diff = jnp.diff(x, axis=0)
    diff = jnp.where(isabs, jnp.abs(diff), diff)

    mask_diff = mask[:-1] & mask[1:]

    masked_diff = jnp.where(mask_diff, diff, jnp.nan)

    agg_functions = {
        "mean": jnp.nanmean(masked_diff, axis=0),
        "var": jnp.nanvar(masked_diff, axis=0),
        "std": jnp.nanstd(masked_diff, axis=0),
    }

    return agg_functions.get(f_agg, jnp.nanmean(masked_diff, axis=0))


def energy_ratio_by_chunks(x: Array, num_segments: int = 10, segment_focus: int = 0) -> Array:
    """Calculate energy ratio of a specific segment."""
    n = x.shape[0]
    segment_len = n // num_segments

    if segment_len < 1:
        return jnp.zeros(x.shape[1:])

    total_energy = jnp.sum(x**2, axis=0)

    start = segment_focus * segment_len
    end = start + segment_len

    if start >= n:
        return jnp.zeros(x.shape[1:])

    segment_energy = jnp.sum(x[start:end] ** 2, axis=0)

    return segment_energy / (total_energy + 1e-12)


def friedrich_coefficients(x: Array, m: int = 3, r: float = 30.0, coeff: int = 0) -> Array:
    """Calculate Friedrich coefficients (simplified polynomial fit)."""
    delta_x = jnp.diff(x, axis=0)
    x_vals = x[:-1]

    def compute_coeff(x_v: Array, dx: Array) -> Array:
        # Build design matrix using broadcasting
        powers = jnp.arange(m + 1)
        design_matrix = x_v[:, None] ** powers[None, :]

        xtx = design_matrix.T @ design_matrix + 1e-6 * jnp.eye(m + 1)
        xty = design_matrix.T @ dx
        coeffs = jnp.linalg.solve(xtx, xty)

        coeff_val = coeffs.at[coeff].get()
        zero = jnp.zeros_like(coeff_val)
        return jnp.asarray(jnp.where(coeff <= m, coeff_val, zero))

    # Vectorize over batch and state dimensions
    compute_batch = jax.vmap(jax.vmap(compute_coeff, in_axes=(1, 1)), in_axes=(1, 1))
    result = compute_batch(x_vals, delta_x)

    return result


def max_langevin_fixed_point(x: Array, r: float = 3, m: int = 30) -> Array:
    """Calculate maximum fixed point of Langevin dynamics (simplified).

    JIT-compatible: uses argmax instead of dynamic jnp.where.
    """
    coeffs = jnp.stack([friedrich_coefficients(x, m=3, r=r, coeff=i) for i in range(4)], axis=0)

    a0, a1, a2, a3 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

    x_range = jnp.linspace(-r, r, m).reshape(-1, 1, 1)

    h_x = a0 + a1 * x_range + a2 * x_range**2 + a3 * x_range**3

    zero_crossings = jnp.diff(jnp.sign(h_x), axis=0)

    has_crossing = jnp.abs(zero_crossings) > 0

    indices = jnp.arange(m - 1).reshape(-1, 1, 1)
    weighted_indices = jnp.where(has_crossing, indices, -1)
    max_crossing_idx = jnp.max(weighted_indices, axis=0)

    x_range_flat = jnp.linspace(-r, r, m)
    max_crossing_idx_clipped = jnp.clip(max_crossing_idx, 0, m - 2)
    result = x_range_flat[max_crossing_idx_clipped]

    any_crossing = jnp.any(has_crossing, axis=0)
    result = jnp.where(any_crossing, result, 0.0)

    return result


def matrix_profile(x: Array, m: int = 10, feature: str = "min") -> Array:
    """Calculate matrix profile features (simplified using distance profile)."""
    n = x.shape[0]

    if m >= n:
        return jnp.zeros(x.shape[1:])

    n_subsequences = n - m + 1

    def compute_for_series(series: Array) -> Array:
        # Extract all subsequences at once using advanced indexing
        indices = jnp.arange(n_subsequences)[:, None] + jnp.arange(m)[None, :]
        subsequences = series[indices]  # (n_subsequences, m)

        # Normalize all subsequences
        means = jnp.mean(subsequences, axis=1, keepdims=True)
        stds = jnp.std(subsequences, axis=1, keepdims=True) + 1e-12
        subseqs_norm = (subsequences - means) / stds

        # Compute pairwise distances with exclusion zone
        def compute_min_dist(i: int) -> Array:
            # Compute distances to all other subsequences
            diffs = subseqs_norm - subseqs_norm[i : i + 1, :]
            dists = jnp.sqrt(jnp.sum(diffs**2, axis=1))

            # Mask out self and exclusion zone
            mask = jnp.abs(jnp.arange(n_subsequences) - i) >= m // 2
            valid_dists = jnp.where(mask, dists, jnp.inf)
            return jnp.min(valid_dists)

        # Vectorize over all subsequences
        distances = jax.vmap(lambda i: compute_min_dist(i))(jnp.arange(n_subsequences))
        finite_distances = jnp.where(jnp.isfinite(distances), distances, 0.0)

        # Use lax.switch for efficient branch selection (only computes selected feature)
        feature_map = {"min": 0, "max": 1, "mean": 2, "std": 3, "median": 4, "25": 5, "75": 6}
        index = feature_map.get(feature, 7)

        branches = [
            lambda d: jnp.min(d),
            lambda d: jnp.max(d),
            lambda d: jnp.mean(d),
            lambda d: jnp.std(d),
            lambda d: jnp.median(d),
            lambda d: jnp.percentile(d, 25),
            lambda d: jnp.percentile(d, 75),
            lambda d: jnp.zeros_like(d[0]),
        ]

        return lax.switch(index, branches, finite_distances)

    # Vectorize over batch and state dimensions
    compute_batch = jax.vmap(jax.vmap(compute_for_series, in_axes=1), in_axes=1)
    result = compute_batch(x)

    return result


def mean_n_absolute_max(x: Array, number_of_maxima: int = 1) -> Array:
    """Calculate mean of n absolute maximum values.

    Optimized: Uses lax.top_k for partial sorting instead of full sort.
    """
    abs_x = jnp.abs(x)
    # Transpose to (B, S, N) for top_k, then transpose back
    abs_x_t = jnp.moveaxis(abs_x, 0, -1)  # (B, S, N)
    # top_k returns (values, indices) for top k along last axis
    top_k_values, _ = lax.top_k(abs_x_t, number_of_maxima)  # (B, S, number_of_maxima)
    return jnp.mean(top_k_values, axis=-1)  # (B, S)


def range_count(x: Array, min_val: float, max_val: float) -> Array:
    """Count values in range [min_val, max_val)."""
    return jnp.sum((x >= min_val) & (x < max_val), axis=0).astype(jnp.float32)


def ratio_beyond_r_sigma(x: Array, r: float = 1.0) -> Array:
    """Calculate ratio of values beyond r * sigma from mean."""
    m = jnp.mean(x, axis=0, keepdims=True)
    std = jnp.std(x, axis=0, keepdims=True)

    beyond = jnp.abs(x - m) > r * std
    return jnp.mean(beyond, axis=0)


def symmetry_looking(x: Array, r: float = 0.1) -> Array:
    """Check if distribution looks symmetric.

    Optimized: Reuse min/max computation for range.
    """
    m = jnp.mean(x, axis=0)
    med = jnp.median(x, axis=0)
    max_val = jnp.max(x, axis=0)
    min_val = jnp.min(x, axis=0)
    range_val = max_val - min_val

    return (jnp.abs(m - med) < r * range_val).astype(jnp.float32)


def time_reversal_asymmetry_statistic(x: Array, lag: int) -> Array:
    """Calculate time reversal asymmetry statistic."""
    n = x.shape[0]
    if 2 * lag >= n:
        return jnp.zeros(x.shape[1:])

    x_sq_lag = x[lag:] ** 2
    x_lag = x[:-lag]

    x_sq_2lag = x[2 * lag :] ** 2
    x_2lag = x[: -2 * lag]

    return jnp.mean(x_sq_lag[:-lag] * x_2lag - x_sq_2lag * x_lag[:-lag], axis=0)


def value_count(x: Array, value: float) -> Array:
    """Count occurrences of a specific value."""
    return jnp.sum(x == value, axis=0).astype(jnp.float32)


# =============================================================================
# CUSTOM FEATURES (2 features)
# =============================================================================


def delta(x: Array) -> Array:
    """Calculate the absolute difference between maximum and mean.

    This feature captures the spread of values around the mean and is useful
    for distinguishing between different dynamical behaviors:
    - Near-constant signals: delta ≈ 0
    - Oscillating signals: delta > 0
    """
    return jnp.abs(jnp.max(x, axis=0) - jnp.mean(x, axis=0))


def log_delta(x: Array) -> Array:
    """Calculate log(delta + epsilon) for improved feature space separation.

    Applies logarithmic transformation to the delta feature, which can
    linearize exponential ranges and improve classification performance
    when values span multiple orders of magnitude.
    """
    eps = 1e-12  # Small epsilon to avoid log(0)
    return jnp.log(delta(x) + eps)


# =============================================================================
# FEATURE FUNCTIONS DICTIONARY
# =============================================================================

# All feature functions (no lambdas, direct function references)
# This is the single source of truth for all available feature calculators
ALL_FEATURE_FUNCTIONS: dict[str, Callable[..., Array]] = {
    # Minimal features (tsfresh MinimalFCParameters - 10 features)
    "sum_values": sum_values,
    "median": median,
    "mean": mean,
    "length": length,
    "standard_deviation": standard_deviation,
    "variance": variance,
    "root_mean_square": root_mean_square,
    "maximum": maximum,
    "absolute_maximum": absolute_maximum,
    "minimum": minimum,
    # Simple Statistics
    "abs_energy": abs_energy,
    "kurtosis": kurtosis,
    "skewness": skewness,
    "quantile": quantile,
    "variation_coefficient": variation_coefficient,
    # Change/Difference
    "absolute_sum_of_changes": absolute_sum_of_changes,
    "mean_abs_change": mean_abs_change,
    "mean_change": mean_change,
    "mean_second_derivative_central": mean_second_derivative_central,
    # Counting
    "count_above": count_above,
    "count_above_mean": count_above_mean,
    "count_below": count_below,
    "count_below_mean": count_below_mean,
    # Boolean
    "has_duplicate": has_duplicate,
    "has_duplicate_max": has_duplicate_max,
    "has_duplicate_min": has_duplicate_min,
    "has_variance_larger_than_standard_deviation": has_variance_larger_than_standard_deviation,
    "large_standard_deviation": has_large_standard_deviation,
    # Location
    "first_location_of_maximum": first_location_of_maximum,
    "first_location_of_minimum": first_location_of_minimum,
    "last_location_of_maximum": last_location_of_maximum,
    "last_location_of_minimum": last_location_of_minimum,
    "index_mass_quantile": index_mass_quantile,
    # Streak/Pattern
    "longest_strike_above_mean": longest_strike_above_mean,
    "longest_strike_below_mean": longest_strike_below_mean,
    "number_crossing_m": number_crossing_m,
    "number_peaks": number_peaks,
    "number_cwt_peaks": number_cwt_peaks,
    # Autocorrelation
    "autocorrelation": autocorrelation,
    "partial_autocorrelation": partial_autocorrelation,
    "agg_autocorrelation": agg_autocorrelation,
    # Entropy/Complexity (excluding approximate_entropy, sample_entropy)
    "permutation_entropy": permutation_entropy,
    "binned_entropy": binned_entropy,
    "fourier_entropy": fourier_entropy,
    "lempel_ziv_complexity": lempel_ziv_complexity,
    "cid_ce": cid_ce,
    # Frequency Domain
    "fft_coefficient": fft_coefficient,
    "fft_aggregated": fft_aggregated,
    "spkt_welch_density": spkt_welch_density,
    "cwt_coefficients": cwt_coefficients,
    # Trend/Regression
    "linear_trend": linear_trend,
    "linear_trend_timewise": linear_trend_timewise,
    "agg_linear_trend": agg_linear_trend,
    "ar_coefficient": ar_coefficient,
    "augmented_dickey_fuller": augmented_dickey_fuller,
    # Reoccurrence
    "percentage_of_reoccurring_datapoints_to_all_datapoints": percentage_of_reoccurring_datapoints_to_all_datapoints,
    "percentage_of_reoccurring_values_to_all_values": percentage_of_reoccurring_values_to_all_values,
    "sum_of_reoccurring_data_points": sum_of_reoccurring_data_points,
    "sum_of_reoccurring_values": sum_of_reoccurring_values,
    "ratio_value_number_to_time_series_length": ratio_value_number_to_time_series_length,
    # Advanced
    "benford_correlation": benford_correlation,
    "c3": c3,
    "change_quantiles": change_quantiles,
    "energy_ratio_by_chunks": energy_ratio_by_chunks,
    "friedrich_coefficients": friedrich_coefficients,
    "max_langevin_fixed_point": max_langevin_fixed_point,
    # "matrix_profile": matrix_profile,
    "mean_n_absolute_max": mean_n_absolute_max,
    "range_count": range_count,
    "ratio_beyond_r_sigma": ratio_beyond_r_sigma,
    "symmetry_looking": symmetry_looking,
    "time_reversal_asymmetry_statistic": time_reversal_asymmetry_statistic,
    "value_count": value_count,
    # Custom (JAX-only)
    "delta": delta,
    "log_delta": log_delta,
}

# =============================================================================
# TSFRESH-COMPATIBLE CONFIGURATION
# =============================================================================

# Type alias for feature configuration parameters
# Using Mapping for covariance (allows dict[str, None] to be passed)
FCParameters = Mapping[str, list[dict[str, Any]] | None]

# JAX Comprehensive Feature Configuration (tsfresh-compatible format)
# Format: {"feature_name": None} for no params, {"feature_name": [{"param": val}, ...]} for parameterized
# Excludes: approximate_entropy, sample_entropy (not implemented in JAX)
# JIT-incompatible features listed separately in JAX_NON_JIT_FC_PARAMETERS
JAX_COMPREHENSIVE_FC_PARAMETERS: FCParameters = {
    # No-parameter features (None means no parameters)
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
    "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
    "percentage_of_reoccurring_values_to_all_values": None,
    "sum_of_reoccurring_data_points": None,
    "sum_of_reoccurring_values": None,
    "ratio_value_number_to_time_series_length": None,
    "benford_correlation": None,
    # Parameterized features (matching tsfresh ComprehensiveFCParameters)
    "time_reversal_asymmetry_statistic": [{"lag": lag} for lag in range(1, 4)],
    "c3": [{"lag": lag} for lag in range(1, 4)],
    "cid_ce": [{"normalize": True}, {"normalize": False}],
    "symmetry_looking": [{"r": r * 0.05} for r in range(20)],
    "large_standard_deviation": [{"r": r * 0.05} for r in range(1, 20)],
    "quantile": [{"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
    "autocorrelation": [{"lag": lag} for lag in range(10)],
    "agg_autocorrelation": [{"f_agg": s, "maxlag": 40} for s in ["mean", "median", "var"]],
    "partial_autocorrelation": [{"lag": lag} for lag in range(10)],
    "number_cwt_peaks": [{"max_width": n} for n in [1, 5]],
    "number_peaks": [{"n": n} for n in [1, 3, 5, 10, 50]],
    "binned_entropy": [{"max_bins": 10}],
    "index_mass_quantile": [{"q": q} for q in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]],
    "cwt_coefficients": [
        {"widths": (w,), "coeff": coeff, "w": w} for w in [2, 5, 10, 20] for coeff in range(15)
    ],
    "spkt_welch_density": [{"coeff": coeff} for coeff in [2, 5, 8]],
    "ar_coefficient": [{"coeff": coeff, "k": 10} for coeff in range(11)],
    "change_quantiles": [
        {"ql": ql, "qh": qh, "isabs": b, "f_agg": f}
        for ql in [0.0, 0.2, 0.4, 0.6, 0.8]
        for qh in [0.2, 0.4, 0.6, 0.8, 1.0]
        for b in [False, True]
        for f in ["mean", "var"]
        if ql < qh
    ],
    "fft_coefficient": [
        {"coeff": k, "attr": a} for a in ["real", "imag", "abs", "angle"] for k in range(100)
    ],
    "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance", "skew", "kurtosis"]],
    "value_count": [{"value": value} for value in [0, 1, -1]],
    "range_count": [
        {"min_val": -1, "max_val": 1},
        {"min_val": -1e12, "max_val": 0},
        {"min_val": 0, "max_val": 1e12},
    ],
    "friedrich_coefficients": [{"coeff": coeff, "m": 3, "r": 30} for coeff in range(4)],
    "max_langevin_fixed_point": [{"m": 3, "r": 30}],
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
    "number_crossing_m": [{"m": 0}, {"m": -1}, {"m": 1}],
    "energy_ratio_by_chunks": [{"num_segments": 10, "segment_focus": i} for i in range(10)],
    "ratio_beyond_r_sigma": [{"r": x} for x in [0.5, 1, 1.5, 2, 2.5, 3, 5, 6, 7, 10]],
    "linear_trend_timewise": [
        {"attr": "pvalue"},
        {"attr": "rvalue"},
        {"attr": "intercept"},
        {"attr": "slope"},
        {"attr": "stderr"},
    ],
    "count_above": [{"t": 0}],
    "count_below": [{"t": 0}],
    "lempel_ziv_complexity": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "fourier_entropy": [{"bins": x} for x in [2, 3, 5, 10, 100]],
    "permutation_entropy": [{"tau": 1, "dimension": x} for x in [3, 4, 5, 6, 7]],
    # "matrix_profile": [{"feature": f} for f in ["min", "max", "mean", "median", "25", "75"]],
    "mean_n_absolute_max": [{"number_of_maxima": 7}],
}


def extract_features(
    x: Array,
    fc_parameters: FCParameters,
) -> dict[str, Array]:
    """Extract features from time series using tsfresh-compatible configuration.

    This function processes all B trajectories at once (vectorized) and returns
    a dictionary mapping feature names to result arrays.

    Warning:
        Using JAX_COMPREHENSIVE_FC_PARAMETERS may cause very long JIT compile times
        (~40 minutes). Use JAX_MINIMAL_FC_PARAMETERS or a custom subset for faster
        compilation.

    ```python
    import jax.numpy as jnp
    from jax_features.jax_feature_calculators import (
        extract_features,
        JAX_MINIMAL_FC_PARAMETERS,
    )

    x = jnp.ones((100, 5, 2))  # 100 timesteps, 5 trajectories, 2 states
    features = extract_features(x, JAX_MINIMAL_FC_PARAMETERS)
    features["mean"].shape  # (5, 2)
    ```

    :param x: Input array with shape (N, B, S) where N is number of time steps,
        B is batch size (number of trajectories), and S is number of state variables.
    :param fc_parameters: Feature configuration dictionary in tsfresh format.
        Keys are feature names (strings), values are None (no params) or list of param dicts.
    :return: Dictionary mapping feature names to Arrays with shape (B, S).
        Feature names follow tsfresh convention: "feature__param1_val1_param2_val2".
    """
    results: dict[str, Array] = {}

    for feature_name, param_list in fc_parameters.items():
        if feature_name not in ALL_FEATURE_FUNCTIONS:
            continue

        func = ALL_FEATURE_FUNCTIONS[feature_name]

        if param_list is None:
            output_name = format_feature_name(feature_name)
            results[output_name] = func(x)
        else:
            for params in param_list:
                output_name = format_feature_name(feature_name, params)
                results[output_name] = func(x, **params)

    return results


def get_feature_names_from_config(
    fc_parameters: FCParameters,
) -> list[str]:
    """Get list of feature names that would be extracted with given configuration.

    Used by the InteractivePlotter to display the features.

    :param fc_parameters: Feature configuration dictionary.
    :return: List of feature names in tsfresh naming convention.
    """
    names: list[str] = []

    for feature_name, param_list in fc_parameters.items():
        if feature_name not in ALL_FEATURE_FUNCTIONS:
            continue

        if param_list is None:
            names.append(format_feature_name(feature_name))
        else:
            for params in param_list:
                names.append(format_feature_name(feature_name, params))

    return names


# =============================================================================
# SIMPLE API (for JaxFeatureExtractor)
# =============================================================================

# Minimal feature names (matching Torch minimal = 10 features)
MINIMAL_FEATURE_NAMES: list[str] = [
    "median",
    "mean",
    "standard_deviation",
    "variance",
    "root_mean_square",
    "maximum",
    "absolute_maximum",
    "minimum",
    "delta",
    "log_delta",
]

# JAX Minimal Feature Configuration (FCParameters format)
JAX_MINIMAL_FC_PARAMETERS: FCParameters = {
    "median": None,
    "mean": None,
    "standard_deviation": None,
    "variance": None,
    "root_mean_square": None,
    "maximum": None,
    "absolute_maximum": None,
    "minimum": None,
    "delta": None,
    "log_delta": None,
}
