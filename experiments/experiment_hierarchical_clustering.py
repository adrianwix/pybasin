# pyright: basic
"""Experiment: Hierarchical clustering for basin stability.

This script runs hierarchical clustering experiments on dynamical systems to compute
basin stability using an unsupervised two-stage classification approach. It supports
pendulum, Duffing oscillator, friction, and Lorenz systems, and compares results
against expected basin stability values from supervised methods.

Two-Stage Classification Approach:
1. First stage: Classify trajectories by attractor TYPE (fixed point, limit cycle, chaos, unbounded)
2. Second stage: Sub-classify within each type using specialized features

This approach addresses the limitation of statistical features alone, which struggle
to distinguish different n-period limit cycles (e.g., period-2 vs period-3 cycles).

Stage 1 - Attractor Type Classification:
- Fixed Point (FP): Very low variance in steady state
- Limit Cycle (LC): Periodic oscillations with consistent autocorrelation
- Chaos: Aperiodic with positive Lyapunov-like behavior
- Unbounded: Divergent trajectories (handled separately)

Stage 2 - Specialized Sub-classifiers:
- FP: Cluster by steady-state values (mean of final states)
- LC: Use spectral features (dominant frequency, number of peaks per period)
  to distinguish different n-period cycles
- Chaos: Use statistical features for potential sub-attractors

Usage:
    python experiment_hierarchical_clustering.py [OPTIONS]

Arguments:
    --system {pendulum,duffing,friction,lorenz,all}
        System to run experiment on (default: all)
    --device {cpu,gpu}
        Device for computation (default: cpu)
    --n-jobs INT
        Number of CPU workers for parallel processing (default: None = auto)
    --fp-variance-threshold FLOAT
        Variance threshold for fixed point detection (default: 1e-6)
    --lc-periodicity-threshold FLOAT
        Periodicity threshold for limit cycle detection (default: 0.5)
    --quiet
        Disable logging except for final comparison across all systems

Examples:
    # Run all systems with verbose output
    python experiment_hierarchical_clustering.py

    # Run single system quietly
    python experiment_hierarchical_clustering.py --system pendulum --quiet

    # Customize thresholds
    python experiment_hierarchical_clustering.py --fp-variance-threshold 1e-5 --lc-periodicity-threshold 0.6
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.cluster import HDBSCAN  # type: ignore[attr-defined]
from tstorch.torch_feature_extractor import TorchFeatureExtractor

from case_studies.duffing_oscillator.setup_duffing_oscillator_system import (
    setup_duffing_oscillator_system,
)
from case_studies.friction.setup_friction_system import setup_friction_system
from case_studies.lorenz.setup_lorenz_system import setup_lorenz_system
from case_studies.pendulum.setup_pendulum_system import setup_pendulum_system
from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer
from pybasin.solution import Solution


def detect_unbounded(y: np.ndarray) -> np.ndarray:
    """Detect unbounded trajectories.

    Args:
        y: Array of shape (n_timesteps, n_batches, n_states)

    Returns:
        Boolean mask of shape (n_batches,) where True = unbounded
    """
    return ~np.all(np.isfinite(y), axis=(0, 2))


def detect_drifting_dimensions(
    y: np.ndarray, n_steady: int = 100, monotonic_threshold: float = 0.9
) -> tuple[list[int], np.ndarray]:
    """Detect dimensions with linear drift (rotating variables).

    For rotating solutions (e.g., pendulum rotating), the angle variable
    increases monotonically while velocity oscillates. This function
    identifies such drifting dimensions by checking for monotonicity
    (consistent sign of derivative).

    Args:
        y: Array of shape (n_timesteps, n_batches, n_states)
        n_steady: Number of time points for analysis
        monotonic_threshold: Fraction of same-sign derivatives to classify as monotonic

    Returns:
        Tuple of:
        - List of drifting dimension indices
        - Boolean mask of shape (n_batches,) indicating trajectories with drift
    """
    y_steady = y[-n_steady:]
    n_dims = y_steady.shape[-1]
    n_batches = y_steady.shape[1]

    drifting_dims: list[int] = []
    has_drift = np.zeros(n_batches, dtype=bool)

    for dim in range(n_dims):
        signal = y_steady[:, :, dim]  # (time, batches)

        # Check for monotonicity: derivative has consistent sign
        diff = np.diff(signal, axis=0)  # (time-1, batches)

        # Fraction of positive steps per trajectory
        pos_frac = np.mean(diff > 0, axis=0)

        # Monotonic if mostly one sign (> threshold or < 1-threshold)
        is_monotonic = (pos_frac > monotonic_threshold) | (pos_frac < (1 - monotonic_threshold))

        # Also require significant net change (not just noise)
        net_change = np.abs(signal[-1, :] - signal[0, :])
        has_significant_change = net_change > 1.0  # More than 1 radian/unit change

        # Mark trajectories with drift in this dimension
        drift_in_dim = is_monotonic & has_significant_change
        has_drift |= drift_in_dim

        # Dimension is globally drifting if majority of trajectories show drift
        if np.mean(drift_in_dim) > 0.5:
            drifting_dims.append(dim)

    return drifting_dims, has_drift


def compute_steady_state_variance(
    y: np.ndarray, n_steady: int = 100, exclude_dims: list[int] | None = None
) -> np.ndarray:
    """Compute variance in steady state for each trajectory.

    Args:
        y: Array of shape (n_timesteps, n_batches, n_states)
        n_steady: Number of time points to use for steady state
        exclude_dims: Dimensions to exclude from variance computation

    Returns:
        Array of shape (n_batches,) with mean variance across states
    """
    y_steady = y[-n_steady:]

    if exclude_dims:
        dims_to_use = [d for d in range(y_steady.shape[-1]) if d not in exclude_dims]
        if len(dims_to_use) == 0:
            dims_to_use = list(range(y_steady.shape[-1]))
        y_steady = y_steady[:, :, dims_to_use]

    var_per_state = np.var(y_steady, axis=0)
    return np.mean(var_per_state, axis=1)


def compute_autocorrelation_periodicity(
    y: np.ndarray, n_steady: int = 100, state_idx: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute autocorrelation-based periodicity measure.

    Returns the lag of the first significant autocorrelation peak (indicating period)
    and the peak height (indicating periodicity strength).

    Args:
        y: Array of shape (n_timesteps, n_batches, n_states)
        n_steady: Number of time points for analysis
        state_idx: Which state variable to analyze

    Returns:
        Tuple of (period_estimate, periodicity_strength) arrays of shape (n_batches,)
    """
    y_steady = y[-n_steady:, :, state_idx]
    n_batches = y_steady.shape[1]

    periods = np.zeros(n_batches)
    strengths = np.zeros(n_batches)

    for i in range(n_batches):
        sig = y_steady[:, i]
        sig = sig - np.mean(sig)
        if np.std(sig) < 1e-10:
            periods[i] = 0
            strengths[i] = 0
            continue

        sig = sig / np.std(sig)
        autocorr = np.correlate(sig, sig, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        autocorr = autocorr / autocorr[0]

        peaks, properties = signal.find_peaks(autocorr, height=0.3, distance=2)
        if len(peaks) > 0:
            periods[i] = peaks[0]
            strengths[i] = properties["peak_heights"][0]
        else:
            periods[i] = 0
            strengths[i] = 0

    return periods, strengths


def compute_spectral_features(
    y: np.ndarray, dt: float, n_steady: int = 100, state_idx: int = 0
) -> dict[str, np.ndarray]:
    """Compute spectral features for limit cycle characterization.

    Args:
        y: Array of shape (n_timesteps, n_batches, n_states)
        dt: Time step
        n_steady: Number of time points for analysis
        state_idx: Which state variable to analyze

    Returns:
        Dictionary with spectral features for each batch
    """
    y_steady = y[-n_steady:, :, state_idx]
    n_batches = y_steady.shape[1]
    n_points = y_steady.shape[0]

    features = {
        "dominant_freq": np.zeros(n_batches),
        "n_significant_freqs": np.zeros(n_batches),
        "spectral_entropy": np.zeros(n_batches),
        "freq_ratio_2nd_1st": np.zeros(n_batches),
    }

    freqs = fftfreq(n_points, dt)
    positive_mask = freqs > 0

    for i in range(n_batches):
        sig = y_steady[:, i]
        sig = sig - np.mean(sig)

        if np.std(sig) < 1e-10:
            continue

        fft_vals = np.abs(fft(sig))
        fft_positive = fft_vals[positive_mask]
        freqs_positive = freqs[positive_mask]

        power = fft_positive**2
        power_norm = power / np.sum(power) if np.sum(power) > 0 else power

        dom_idx = np.argmax(power)
        features["dominant_freq"][i] = freqs_positive[dom_idx]

        threshold = 0.1 * np.max(power)
        features["n_significant_freqs"][i] = np.sum(power > threshold)

        power_prob = power_norm + 1e-10
        features["spectral_entropy"][i] = -np.sum(power_prob * np.log(power_prob))

        sorted_indices = np.argsort(power)[::-1]
        if len(sorted_indices) >= 2 and power[sorted_indices[0]] > 0:
            features["freq_ratio_2nd_1st"][i] = (
                freqs_positive[sorted_indices[1]] / freqs_positive[sorted_indices[0]]
            )

    return features


def count_peaks_per_period(y: np.ndarray, n_steady: int = 100, state_idx: int = 0) -> np.ndarray:
    """Count number of local maxima per estimated period.

    This helps distinguish period-1, period-2, period-3 cycles.

    Args:
        y: Array of shape (n_timesteps, n_batches, n_states)
        n_steady: Number of time points for analysis
        state_idx: Which state variable to analyze

    Returns:
        Array of shape (n_batches,) with peaks per period estimate
    """
    y_steady = y[-n_steady:, :, state_idx]
    n_batches = y_steady.shape[1]

    peaks_per_period = np.zeros(n_batches)

    for i in range(n_batches):
        sig = y_steady[:, i]
        if np.std(sig) < 1e-10:
            peaks_per_period[i] = 0
            continue

        peaks, _ = signal.find_peaks(sig)
        if len(peaks) < 2:
            peaks_per_period[i] = 1
            continue

        peak_diffs = np.diff(peaks)
        median_period = np.median(peak_diffs)

        n_peaks = len(peaks)
        total_time = len(sig)
        estimated_n_periods = total_time / median_period if median_period > 0 else 1
        peaks_per_period[i] = n_peaks / estimated_n_periods if estimated_n_periods > 0 else n_peaks

    return peaks_per_period


def classify_attractor_type(
    y: np.ndarray,
    dt: float,
    n_steady: int = 100,
    fp_variance_threshold: float = 1e-6,
    lc_periodicity_threshold: float = 0.5,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Classify each trajectory by attractor type.

    Handles rotating solutions (e.g., pendulum) by:
    1. Detecting drifting dimensions (angle variables that increase linearly)
    2. Using only bounded dimensions for periodicity analysis
    3. Classifying trajectories with drift + bounded velocity as limit cycles

    Args:
        y: Array of shape (n_timesteps, n_batches, n_states)
        dt: Time step
        n_steady: Number of time points for steady state analysis
        fp_variance_threshold: Variance threshold for fixed point detection
        lc_periodicity_threshold: Periodicity strength threshold for LC detection

    Returns:
        Tuple of:
        - labels: Array of shape (n_batches,) with type codes:
          -1 = unbounded, 0 = fixed point, 1 = limit cycle, 2 = chaos/other
        - features: Dictionary of computed features for sub-classification
    """
    n_batches = y.shape[1]
    n_dims = y.shape[2]
    labels = np.zeros(n_batches, dtype=int)

    # Step 1: Detect and filter out unbounded trajectories FIRST
    unbounded_mask = detect_unbounded(y)
    labels[unbounded_mask] = -1
    bounded_mask = ~unbounded_mask

    # Filter to only bounded trajectories for all subsequent processing
    y_bounded = y[:, bounded_mask, :]
    n_bounded = y_bounded.shape[1]

    if n_bounded == 0:
        # All trajectories are unbounded
        return labels, {}

    # Step 2: Detect drifting dimensions (rotating variables like pendulum angle)
    drifting_dims, has_drift = detect_drifting_dimensions(y_bounded, n_steady)

    # Determine which dimension to use for periodicity analysis
    bounded_dims_list = [d for d in range(n_dims) if d not in drifting_dims]
    analysis_dim = bounded_dims_list[0] if bounded_dims_list else 0

    # Step 3: Compute features on bounded trajectories only
    variance = compute_steady_state_variance(y_bounded, n_steady, exclude_dims=drifting_dims)
    periods, periodicity = compute_autocorrelation_periodicity(
        y_bounded, n_steady, state_idx=analysis_dim
    )
    spectral = compute_spectral_features(y_bounded, dt, n_steady, state_idx=analysis_dim)
    peaks_per_period = count_peaks_per_period(y_bounded, n_steady, state_idx=analysis_dim)

    y_steady = y_bounded[-n_steady:]
    steady_mean = np.mean(y_steady, axis=0)

    y_steady_analysis = y_steady[:, :, analysis_dim]
    amplitude = np.max(y_steady_analysis, axis=0) - np.min(y_steady_analysis, axis=0)
    signal_mean = np.mean(y_steady_analysis, axis=0)

    # Step 4: Classify bounded trajectories
    fp_mask_bounded = variance < fp_variance_threshold

    # For limit cycles: require periodicity AND moderate variance
    # High variance + periodicity = chaos (e.g., Lorenz butterfly)
    chaos_variance_threshold = 5.0  # High variance suggests chaos, not LC
    lc_periodic_mask = (
        ~fp_mask_bounded
        & (periodicity > lc_periodicity_threshold)
        & (variance < chaos_variance_threshold)  # Exclude high-variance pseudo-periodic
    )
    lc_rotating_mask = ~fp_mask_bounded & has_drift & (variance > fp_variance_threshold)
    lc_mask_bounded = lc_periodic_mask | lc_rotating_mask
    chaos_mask_bounded = ~fp_mask_bounded & ~lc_mask_bounded

    # Map back to original indices
    bounded_indices = np.where(bounded_mask)[0]
    labels[bounded_indices[fp_mask_bounded]] = 0
    labels[bounded_indices[lc_mask_bounded]] = 1
    labels[bounded_indices[chaos_mask_bounded]] = 2

    # Create full-size feature arrays (with NaN for unbounded)
    features = {
        "variance": np.full(n_batches, np.nan),
        "period_estimate": np.full(n_batches, np.nan),
        "periodicity_strength": np.full(n_batches, np.nan),
        "dominant_freq": np.full(n_batches, np.nan),
        "n_significant_freqs": np.full(n_batches, np.nan),
        "spectral_entropy": np.full(n_batches, np.nan),
        "freq_ratio": np.full(n_batches, np.nan),
        "peaks_per_period": np.full(n_batches, np.nan),
        "steady_mean": np.full((n_batches, n_dims), np.nan),
        "amplitude": np.full(n_batches, np.nan),
        "signal_mean": np.full(n_batches, np.nan),
        "drifting_dims": drifting_dims,
        "has_drift": np.full(n_batches, False),
        "analysis_dim": analysis_dim,
    }

    # Fill in values for bounded trajectories
    features["variance"][bounded_mask] = variance
    features["period_estimate"][bounded_mask] = periods
    features["periodicity_strength"][bounded_mask] = periodicity
    features["dominant_freq"][bounded_mask] = spectral["dominant_freq"]
    features["n_significant_freqs"][bounded_mask] = spectral["n_significant_freqs"]
    features["spectral_entropy"][bounded_mask] = spectral["spectral_entropy"]
    features["freq_ratio"][bounded_mask] = spectral["freq_ratio_2nd_1st"]
    features["peaks_per_period"][bounded_mask] = peaks_per_period
    features["steady_mean"][bounded_mask] = steady_mean
    features["amplitude"][bounded_mask] = amplitude
    features["signal_mean"][bounded_mask] = signal_mean
    features["has_drift"][bounded_mask] = has_drift

    return labels, features


def sub_classify_fixed_points(
    indices: np.ndarray,
    features: dict[str, np.ndarray],
    single_cluster_range_threshold: float = 0.01,
) -> np.ndarray:
    """Sub-classify fixed points by their steady-state values.

    Args:
        indices: Indices of trajectories classified as fixed points
        features: Feature dictionary from classify_attractor_type
        single_cluster_range_threshold: If range (max-min) of steady_mean values
            is below this threshold for all dimensions, treat as single cluster

    Returns:
        Sub-cluster labels for the fixed point trajectories
    """
    if len(indices) == 0:
        return np.array([], dtype=int)

    steady_vals = features["steady_mean"][indices].copy()
    drifting_dims = features.get("drifting_dims", [])

    dims_to_use = [d for d in range(steady_vals.shape[1]) if d not in drifting_dims]

    if len(dims_to_use) == 0:
        return np.zeros(len(indices), dtype=int)

    steady_vals = steady_vals[:, dims_to_use]

    data_range = np.max(steady_vals, axis=0) - np.min(steady_vals, axis=0)
    if np.all(data_range < single_cluster_range_threshold):
        return np.zeros(len(indices), dtype=int)

    scaler = StandardScaler()
    steady_scaled = scaler.fit_transform(steady_vals)

    min_cluster_size = max(50, len(indices) // 10)
    clusterer = HDBSCANClusterer(
        hdbscan=HDBSCAN(min_cluster_size=min_cluster_size),  # type: ignore[call-arg]
        auto_tune=False,
        assign_noise=True,
    )
    labels = clusterer.fit_predict(steady_scaled)

    return labels


def sub_classify_limit_cycles(
    indices: np.ndarray,
    features: dict[str, np.ndarray],
    amp_cv_threshold: float = 0.1,
    mean_range_threshold: float = 0.05,
) -> np.ndarray:
    """Sub-classify limit cycles using hierarchical period-based approach.

    Two-level hierarchical clustering:
    1. First level: Group by period-n (freq_ratio rounded to nearest integer)
    2. Second level: Within each period group, cluster by amplitude and mean

    Args:
        indices: Indices of trajectories classified as limit cycles
        features: Feature dictionary from classify_attractor_type
        amp_cv_threshold: Min coefficient of variation for amplitude to cluster
        mean_range_threshold: Min range for signal_mean to cluster

    Returns:
        Sub-cluster labels for the limit cycle trajectories
    """
    if len(indices) == 0:
        return np.array([], dtype=int)

    freq_ratios = features["freq_ratio"][indices]
    amplitudes = features["amplitude"][indices]
    signal_means = features["signal_mean"][indices]

    finite_mask = np.isfinite(freq_ratios) & np.isfinite(amplitudes) & np.isfinite(signal_means)
    if not np.any(finite_mask):
        return np.zeros(len(indices), dtype=int)

    period_n = np.round(freq_ratios).astype(int)
    period_n = np.clip(period_n, 1, 10)

    unique_periods = np.unique(period_n[finite_mask])
    labels = np.zeros(len(indices), dtype=int)
    current_label = 0

    for period in unique_periods:
        period_mask = (period_n == period) & finite_mask
        period_indices = np.where(period_mask)[0]

        if len(period_indices) == 0:
            continue

        period_amps = amplitudes[period_mask]
        period_means = signal_means[period_mask]

        amp_cv = np.std(period_amps) / (np.mean(period_amps) + 1e-10)
        mean_range = np.max(period_means) - np.min(period_means)

        amp_varies = amp_cv > amp_cv_threshold
        mean_varies = mean_range > mean_range_threshold

        if not amp_varies and not mean_varies:
            labels[period_mask] = current_label
            current_label += 1
        else:
            if amp_varies and mean_varies:
                sub_features = np.column_stack([period_amps, period_means])
            elif amp_varies:
                sub_features = period_amps.reshape(-1, 1)
            else:
                sub_features = period_means.reshape(-1, 1)

            sub_labels = _cluster_1d_or_2d(sub_features, len(period_indices))

            for sub_label in np.unique(sub_labels):
                sub_mask = sub_labels == sub_label
                full_mask = np.zeros(len(indices), dtype=bool)
                full_mask[period_indices[sub_mask]] = True
                labels[full_mask] = current_label
                current_label += 1

    labels[~finite_mask] = 0

    return labels


def _cluster_1d_or_2d(data: np.ndarray, n_samples: int) -> np.ndarray:
    """Cluster 1D or 2D data using appropriate method.

    For 1D data: detect natural gaps/modes
    For 2D data: use HDBSCAN
    """
    if data.shape[1] == 1:
        return _cluster_1d_with_gaps(data.ravel())
    else:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data)

        min_cluster_size = max(15, n_samples // 20)
        use_auto_tune = n_samples >= 500
        clusterer = HDBSCANClusterer(
            hdbscan=HDBSCAN(min_cluster_size=min_cluster_size),  # type: ignore[call-arg]
            auto_tune=use_auto_tune,
            assign_noise=True,
        )
        return clusterer.fit_predict(scaled)


def _cluster_1d_with_gaps(
    data: np.ndarray, min_gap_ratio: float = 5.0, max_clusters: int = 5
) -> np.ndarray:
    """Cluster 1D data by detecting the largest natural gap(s).

    Finds gaps in sorted data that are significantly larger than typical spacing.
    Only creates up to max_clusters to avoid over-splitting.
    """
    n = len(data)
    if n < 4:
        return np.zeros(n, dtype=int)

    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]

    gaps = np.diff(sorted_data)

    if np.max(gaps) < 1e-6:
        return np.zeros(n, dtype=int)

    gap_threshold = np.median(gaps) * min_gap_ratio
    gap_threshold = max(gap_threshold, np.max(gaps) * 0.3)

    significant_gap_indices = np.where(gaps > gap_threshold)[0]

    if len(significant_gap_indices) == 0:
        return np.zeros(n, dtype=int)

    if len(significant_gap_indices) >= max_clusters:
        gap_values = gaps[significant_gap_indices]
        top_gaps = np.argsort(gap_values)[-(max_clusters - 1) :]
        significant_gap_indices = sorted(significant_gap_indices[top_gaps])

    labels_sorted = np.zeros(n, dtype=int)
    current_label = 0
    for i in range(n - 1):
        labels_sorted[i] = current_label
        if i in significant_gap_indices:
            current_label += 1
    labels_sorted[n - 1] = current_label

    labels = np.zeros(n, dtype=int)
    labels[sorted_indices] = labels_sorted

    return labels


def sub_classify_chaos(
    indices: np.ndarray,
    y: np.ndarray,
    features: dict[str, np.ndarray] | None = None,
    device: Literal["cpu", "gpu"] = "cpu",
    n_jobs: int | None = None,
) -> np.ndarray:
    """Sub-classify chaotic trajectories using steady-state spatial mean.

    For systems like Lorenz with multiple chaotic attractors (butterfly wings),
    the spatial mean distinguishes them. Falls back to statistical features
    if spatial clustering fails.

    Args:
        indices: Indices of trajectories classified as chaotic
        y: Full trajectory array
        features: Pre-computed features dict with 'steady_mean'
        device: Device for feature extraction
        n_jobs: Number of workers

    Returns:
        Sub-cluster labels for chaotic trajectories
    """
    if len(indices) == 0:
        return np.array([], dtype=int)

    # Try spatial clustering first (using steady-state mean position)
    if features is not None and "steady_mean" in features:
        steady_vals = features["steady_mean"][indices]

        # Remove NaN values
        finite_mask = np.all(np.isfinite(steady_vals), axis=1)
        if np.sum(finite_mask) > 1:
            steady_clean = steady_vals[finite_mask]

            scaler = StandardScaler()
            steady_scaled = scaler.fit_transform(steady_clean)

            clusterer = HDBSCANClusterer(auto_tune=True, assign_noise=True)
            labels_clean = clusterer.fit_predict(steady_scaled)

            labels = np.zeros(len(indices), dtype=int)
            labels[finite_mask] = labels_clean
            return labels

    # Fallback to statistical features
    y_chaos = y[:, indices, :]

    extractor = TorchFeatureExtractor(
        time_steady=0.9 * y.shape[0],
        features="minimal",
        normalize=True,
        device=device,
        n_jobs=n_jobs,
    )

    time_arr = torch.linspace(0, 1, y_chaos.shape[0])
    ics = torch.zeros(len(indices), y_chaos.shape[2])
    y_tensor = torch.from_numpy(y_chaos).float()
    solution = Solution(initial_condition=ics, time=time_arr, y=y_tensor)

    features_extracted = extractor.extract_features(solution).numpy()

    clusterer = HDBSCANClusterer(auto_tune=True, assign_noise=True)
    labels = clusterer.fit_predict(features_extracted)

    return labels


def hierarchical_clustering(
    y: np.ndarray,
    dt: float,
    device: Literal["cpu", "gpu"] = "cpu",
    n_jobs: int | None = None,
    fp_variance_threshold: float = 1e-6,
    lc_periodicity_threshold: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Perform hierarchical clustering with specialized sub-classifiers.

    Args:
        y: Trajectory array of shape (n_timesteps, n_batches, n_states)
        dt: Time step
        device: Device for feature extraction
        n_jobs: Number of workers
        fp_variance_threshold: Threshold for FP detection
        lc_periodicity_threshold: Threshold for LC detection

    Returns:
        Tuple of (final_labels, info_dict)
    """
    n_batches = y.shape[1]
    logging.info("\n   Stage 1: Attractor Type Classification")
    logging.info("   " + "-" * 40)

    type_labels, features = classify_attractor_type(
        y,
        dt,
        fp_variance_threshold=fp_variance_threshold,
        lc_periodicity_threshold=lc_periodicity_threshold,
    )

    n_unbounded = np.sum(type_labels == -1)
    n_fp = np.sum(type_labels == 0)
    n_lc = np.sum(type_labels == 1)
    n_chaos = np.sum(type_labels == 2)

    logging.info(f"   Unbounded: {n_unbounded} ({n_unbounded / n_batches:.4f})")
    logging.info(f"   Fixed Points: {n_fp} ({n_fp / n_batches:.4f})")
    logging.info(f"   Limit Cycles: {n_lc} ({n_lc / n_batches:.4f})")
    logging.info(f"   Chaos/Other: {n_chaos} ({n_chaos / n_batches:.4f})")

    logging.info("\n   Stage 2: Sub-classification within each type")
    logging.info("   " + "-" * 40)

    final_labels = np.full(n_batches, -1, dtype=int)
    current_label = 0
    label_info = {}

    if n_unbounded > 0:
        unbounded_indices = np.where(type_labels == -1)[0]
        final_labels[unbounded_indices] = current_label
        label_info[current_label] = {"type": "unbounded", "count": n_unbounded}
        logging.info(f"   Unbounded → Cluster {current_label}")
        current_label += 1

    if n_fp > 0:
        fp_indices = np.where(type_labels == 0)[0]
        fp_sub_labels = sub_classify_fixed_points(fp_indices, features)
        n_fp_clusters = len(np.unique(fp_sub_labels))
        logging.info(f"   Fixed Points → {n_fp_clusters} sub-cluster(s)")

        for sub_label in np.unique(fp_sub_labels):
            mask = fp_sub_labels == sub_label
            final_labels[fp_indices[mask]] = current_label
            count = np.sum(mask)

            steady_mean_cluster = features["steady_mean"][fp_indices[mask]]
            median_steady = np.median(steady_mean_cluster, axis=0)
            std_steady = np.std(steady_mean_cluster, axis=0)
            label_info[current_label] = {
                "type": "FP",
                "sub_label": sub_label,
                "count": count,
                "median_steady_mean": median_steady.tolist(),
                "std_steady_mean": std_steady.tolist(),
            }
            steady_str = ", ".join([f"{v:.3f}" for v in median_steady])
            std_str = ", ".join([f"{v:.3f}" for v in std_steady])
            logging.info(
                f"     FP sub-cluster {sub_label} → Cluster {current_label} "
                f"({count} samples, steady=[{steady_str}], std=[{std_str}])"
            )
            current_label += 1

    if n_lc > 0:
        lc_indices = np.where(type_labels == 1)[0]
        lc_sub_labels = sub_classify_limit_cycles(lc_indices, features)
        n_lc_clusters = len(np.unique(lc_sub_labels))
        logging.info(f"   Limit Cycles → {n_lc_clusters} sub-cluster(s)")

        for sub_label in np.unique(lc_sub_labels):
            mask = lc_sub_labels == sub_label
            final_labels[lc_indices[mask]] = current_label
            count = np.sum(mask)

            freq_ratio = np.median(features["freq_ratio"][lc_indices[mask]])
            amplitude = np.median(features["amplitude"][lc_indices[mask]])
            sig_mean = np.median(features["signal_mean"][lc_indices[mask]])
            label_info[current_label] = {
                "type": "LC",
                "sub_label": sub_label,
                "count": count,
                "median_freq_ratio": freq_ratio,
                "median_amplitude": amplitude,
                "median_mean": sig_mean,
            }
            logging.info(
                f"     LC sub-cluster {sub_label} → Cluster {current_label} "
                f"({count} samples, freq_ratio={freq_ratio:.2f}, amp={amplitude:.2f}, mean={sig_mean:.3f})"
            )
            current_label += 1

    if n_chaos > 0:
        chaos_indices = np.where(type_labels == 2)[0]
        chaos_sub_labels = sub_classify_chaos(chaos_indices, y, features, device, n_jobs)
        n_chaos_clusters = len(np.unique(chaos_sub_labels))
        logging.info(f"   Chaos/Other → {n_chaos_clusters} sub-cluster(s)")

        for sub_label in np.unique(chaos_sub_labels):
            mask = chaos_sub_labels == sub_label
            final_labels[chaos_indices[mask]] = current_label
            count = np.sum(mask)
            label_info[current_label] = {"type": "chaos", "sub_label": sub_label, "count": count}
            logging.info(
                f"     Chaos sub-cluster {sub_label} → Cluster {current_label} ({count} samples)"
            )
            current_label += 1

    info = {
        "type_labels": type_labels,
        "features": features,
        "label_info": label_info,
        "n_total_clusters": current_label,
    }

    return final_labels, info


def run_system_experiment(
    system_name: str,
    setup_fn,
    device: Literal["cpu", "gpu"] = "cpu",
    n_jobs: int | None = None,
    fp_variance_threshold: float = 1e-6,
    lc_periodicity_threshold: float = 0.5,
):
    """Run hierarchical clustering experiment on a single system."""
    logging.info("\n" + "=" * 80)
    logging.info(f"{system_name.upper()} System - Hierarchical Clustering")
    logging.info("=" * 80)

    logging.info("\n1. Setup")
    logging.info("-" * 40)
    setup = setup_fn()
    n = setup["n"]
    ode_system = setup["ode_system"]
    sampler = setup["sampler"]
    solver = setup["solver"]

    logging.info(f"   System: {system_name}")
    logging.info(f"   Samples: {n}")

    logging.info("\n2. Sampling and Integration")
    logging.info("-" * 40)
    t_start = time.perf_counter()
    initial_conditions = sampler.sample(n)
    logging.info(f"   Initial conditions: {initial_conditions.shape}")

    time_arr, y_arr = solver.integrate(ode_system, initial_conditions)
    t_integrate = time.perf_counter() - t_start

    y_np = y_arr.cpu().numpy() if hasattr(y_arr, "cpu") else np.array(y_arr)
    time_np = time_arr.cpu().numpy() if hasattr(time_arr, "cpu") else np.array(time_arr)
    dt = float(time_np[1] - time_np[0]) if len(time_np) > 1 else 0.01

    logging.info(f"   Solution shape: {y_np.shape}")
    logging.info(f"   Time step: {dt:.4f}")
    logging.info(f"   Integration time: {t_integrate:.3f}s")

    logging.info("\n3. Hierarchical Clustering")
    logging.info("-" * 40)
    t_cluster_start = time.perf_counter()

    final_labels, info = hierarchical_clustering(
        y_np,
        dt,
        device=device,
        n_jobs=n_jobs,
        fp_variance_threshold=fp_variance_threshold,
        lc_periodicity_threshold=lc_periodicity_threshold,
    )

    t_cluster = time.perf_counter() - t_cluster_start

    logging.info("\n4. Basin Stability Results")
    logging.info("-" * 40)

    unique_labels, counts = np.unique(final_labels, return_counts=True)
    bs_results = {}

    for label, count in zip(unique_labels, counts, strict=True):
        frac = count / n
        label_type = info["label_info"].get(label, {}).get("type", "unknown")
        bs_results[f"Cluster_{label}_{label_type}"] = frac
        logging.info(f"   Cluster {label} ({label_type}): {frac:.4f} ({count} samples)")

    logging.info("\n" + "=" * 80)
    logging.info("Summary")
    logging.info("=" * 80)
    logging.info(f"Total samples: {n}")
    logging.info(f"Total clusters found: {info['n_total_clusters']}")
    logging.info(f"Integration time: {t_integrate:.3f}s")
    logging.info(f"Clustering time: {t_cluster:.3f}s")

    return {
        "system_name": system_name,
        "n_samples": n,
        "n_clusters": info["n_total_clusters"],
        "basin_stability": bs_results,
        "label_info": info["label_info"],
        "times": {"integration": t_integrate, "clustering": t_cluster},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical clustering experiment for basin stability"
    )
    parser.add_argument(
        "--system",
        choices=["pendulum", "duffing", "friction", "lorenz", "all"],
        default="all",
        help="System to run (default: all)",
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Device")
    parser.add_argument("--n-jobs", type=int, default=None, help="CPU workers")
    parser.add_argument(
        "--fp-variance-threshold",
        type=float,
        default=1e-6,
        help="Variance threshold for fixed point detection",
    )
    parser.add_argument(
        "--lc-periodicity-threshold",
        type=float,
        default=0.5,
        help="Periodicity threshold for limit cycle detection",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable logging except for final comparison across all systems",
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.basicConfig(level=logging.CRITICAL, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    systems = {
        "pendulum": setup_pendulum_system,
        "duffing": setup_duffing_oscillator_system,
        "friction": setup_friction_system,
        "lorenz": setup_lorenz_system,
    }

    # Expected basin stability from supervised methods
    expected_bs = {
        "pendulum": [
            {"label": "FP", "basinStability": 0.152},
            {"label": "LC", "basinStability": 0.848},
        ],
        "duffing": [
            {"label": "period-1 LC y_1", "basinStability": 0.1908},
            {"label": "period-1 LC y_2", "basinStability": 0.4994},
            {"label": "period-2 LC y_3", "basinStability": 0.0278},
            {"label": "period-2 LC y_4", "basinStability": 0.022},
            {"label": "period-3 LC y_5", "basinStability": 0.26},
        ],
        "friction": [
            {"label": "FP", "basinStability": 0.304},
            {"label": "LC", "basinStability": 0.696},
        ],
        "lorenz": [
            {"label": "butterfly1", "basinStability": 0.0894},
            {"label": "butterfly2", "basinStability": 0.08745},
            {"label": "unbounded", "basinStability": 0.82315},
        ],
    }

    if args.system == "all":
        systems_to_run = list(systems.items())
    else:
        systems_to_run = [(args.system, systems[args.system])]

    results = []
    for system_name, setup_fn in systems_to_run:
        result = run_system_experiment(
            system_name=system_name,
            setup_fn=setup_fn,
            device=args.device,
            n_jobs=args.n_jobs,
            fp_variance_threshold=args.fp_variance_threshold,
            lc_periodicity_threshold=args.lc_periodicity_threshold,
        )
        results.append(result)

    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON ACROSS ALL SYSTEMS")
    print("=" * 80)

    for result in results:
        system_name = result["system_name"]
        print(f"\n{system_name.upper()}:")
        print(f"  Clusters found: {result['n_clusters']}")
        print("  Basin Stability:")
        for label, frac in sorted(result["basin_stability"].items()):
            print(f"    {label}: {frac:.4f}")

        # Compare with expected results
        if system_name in expected_bs:
            print("\n  Comparison with Expected (by type):")
            print("  " + "-" * 56)

            # Group actual results by type (extract type from cluster name like "Cluster_0_FP")
            actual_by_type: dict[str, list[tuple[str, float]]] = {}
            for cluster_id, bs in result["basin_stability"].items():
                # Extract type from name (last part after underscore)
                parts = cluster_id.split("_")
                cluster_type = parts[-1] if len(parts) > 1 else "unknown"
                if cluster_type not in actual_by_type:
                    actual_by_type[cluster_type] = []
                actual_by_type[cluster_type].append((cluster_id, bs))

            # Sort each type group by basin stability (descending)
            for t in actual_by_type:
                actual_by_type[t].sort(key=lambda x: x[1], reverse=True)

            # Group expected by type
            expected_by_type: dict[str, list[dict]] = {}
            for item in expected_bs[system_name]:
                label = item["label"]
                if "FP" in label or label == "FP":
                    exp_type = "FP"
                elif "LC" in label or label == "LC":
                    exp_type = "LC"
                elif "chaos" in label.lower() or "butterfly" in label.lower():
                    exp_type = "chaos"
                elif "unbounded" in label.lower():
                    exp_type = "unbounded"
                else:
                    exp_type = "other"
                if exp_type not in expected_by_type:
                    expected_by_type[exp_type] = []
                expected_by_type[exp_type].append(item)

            # Sort each expected type group by basin stability (descending)
            for t in expected_by_type:
                expected_by_type[t].sort(key=lambda x: x["basinStability"], reverse=True)

            total_diff = 0.0
            matched_count = 0
            # Compare by type
            for exp_type, expected_items in expected_by_type.items():
                actual_items = actual_by_type.get(exp_type, [])
                for i, expected_item in enumerate(expected_items):
                    expected = expected_item["basinStability"]
                    exp_label = expected_item["label"]
                    if i < len(actual_items):
                        cluster_id, actual = actual_items[i]
                        diff = actual - expected
                        total_diff += abs(diff)
                        matched_count += 1
                        match_symbol = "✓" if abs(diff) < 0.01 else "✗" if abs(diff) > 0.05 else "~"
                        print(
                            f"    {match_symbol} {cluster_id} → {exp_label}: "
                            f"Actual={actual:.4f}  Expected={expected:.4f}  Diff={diff:+.4f}"
                        )
                    else:
                        print(f"    ✗ MISSING → {exp_label}: Expected={expected:.4f}")
                        total_diff += expected

            # Report any unmatched actual clusters
            for act_type, actual_items in actual_by_type.items():
                expected_count = len(expected_by_type.get(act_type, []))
                for i, (cluster_id, actual) in enumerate(actual_items):
                    if i >= expected_count:
                        print(f"    ⚠ EXTRA {cluster_id}: Actual={actual:.4f}")
                        total_diff += actual

            print(f"  Total absolute difference: {total_diff:.4f}")
            n_actual = len(result["basin_stability"])
            n_expected = len(expected_bs[system_name])
            if n_actual != n_expected:
                print(f"  ⚠ Warning: {n_actual} clusters found, {n_expected} expected")


if __name__ == "__main__":
    main()
