# pyright: basic
"""Experiment: Analyze periodic features for Duffing attractor templates.

Extracts and displays the spectral/periodic features used for limit cycle
sub-classification on the 5 known Duffing attractors:
- y1: period-1 LC
- y2: period-1 LC
- y3: period-2 LC
- y4: period-2 LC
- y5: period-3 LC

This helps understand which features distinguish different n-period limit cycles.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from scipy.fft import fft, fftfreq

sys.path.insert(0, str(Path(__file__).parent.parent))

from case_studies.duffing_oscillator.duffing_jax_ode import DuffingJaxODE, DuffingParams
from pybasin.solvers.jax_solver import JaxSolver


def compute_steady_state_variance(y: np.ndarray, n_steady: int = 100) -> np.ndarray:
    """Compute variance in steady state for each trajectory."""
    y_steady = y[-n_steady:]
    var_per_state = np.var(y_steady, axis=0)
    return np.mean(var_per_state, axis=1)


def compute_autocorrelation_periodicity(
    y: np.ndarray, n_steady: int = 100, state_idx: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute autocorrelation-based periodicity measure."""
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
    """Compute spectral features for limit cycle characterization."""
    y_steady = y[-n_steady:, :, state_idx]
    n_batches = y_steady.shape[1]
    n_points = y_steady.shape[0]

    features = {
        "dominant_freq": np.zeros(n_batches),
        "n_significant_freqs": np.zeros(n_batches),
        "spectral_entropy": np.zeros(n_batches),
        "freq_ratio_2nd_1st": np.zeros(n_batches),
        "power_1st_harmonic": np.zeros(n_batches),
        "power_2nd_harmonic": np.zeros(n_batches),
        "power_3rd_harmonic": np.zeros(n_batches),
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
        total_power = np.sum(power)
        power_norm = power / total_power if total_power > 0 else power

        dom_idx = np.argmax(power)
        features["dominant_freq"][i] = freqs_positive[dom_idx]

        threshold = 0.1 * np.max(power)
        features["n_significant_freqs"][i] = np.sum(power > threshold)

        power_prob = power_norm + 1e-10
        features["spectral_entropy"][i] = -np.sum(power_prob * np.log(power_prob))

        sorted_indices = np.argsort(power)[::-1]
        if len(sorted_indices) >= 1:
            features["power_1st_harmonic"][i] = power[sorted_indices[0]] / total_power
        if len(sorted_indices) >= 2:
            features["power_2nd_harmonic"][i] = power[sorted_indices[1]] / total_power
            if power[sorted_indices[0]] > 0:
                features["freq_ratio_2nd_1st"][i] = (
                    freqs_positive[sorted_indices[1]] / freqs_positive[sorted_indices[0]]
                )
        if len(sorted_indices) >= 3:
            features["power_3rd_harmonic"][i] = power[sorted_indices[2]] / total_power

    return features


def count_peaks_per_period(
    y: np.ndarray, n_steady: int = 100, state_idx: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count number of local maxima and analyze peak patterns."""
    y_steady = y[-n_steady:, :, state_idx]
    n_batches = y_steady.shape[1]

    peaks_per_period = np.zeros(n_batches)
    total_peaks = np.zeros(n_batches)
    peak_height_std = np.zeros(n_batches)

    for i in range(n_batches):
        sig = y_steady[:, i]
        if np.std(sig) < 1e-10:
            continue

        peaks, properties = signal.find_peaks(sig, prominence=0.01)
        total_peaks[i] = len(peaks)

        if len(peaks) < 2:
            peaks_per_period[i] = 1
            continue

        peak_diffs = np.diff(peaks)
        median_period = np.median(peak_diffs)

        total_time = len(sig)
        estimated_n_periods = total_time / median_period if median_period > 0 else 1
        peaks_per_period[i] = (
            len(peaks) / estimated_n_periods if estimated_n_periods > 0 else len(peaks)
        )

        if len(peaks) > 1:
            peak_heights = sig[peaks]
            peak_height_std[i] = np.std(peak_heights)

    return peaks_per_period, total_peaks, peak_height_std


def compute_amplitude_features(
    y: np.ndarray, n_steady: int = 100, state_idx: int = 0
) -> dict[str, np.ndarray]:
    """Compute amplitude-based features."""
    y_steady = y[-n_steady:, :, state_idx]
    n_batches = y_steady.shape[1]

    features = {
        "amplitude": np.zeros(n_batches),
        "mean": np.zeros(n_batches),
        "std": np.zeros(n_batches),
        "max": np.zeros(n_batches),
        "min": np.zeros(n_batches),
    }

    for i in range(n_batches):
        sig = y_steady[:, i]
        features["amplitude"][i] = np.max(sig) - np.min(sig)
        features["mean"][i] = np.mean(sig)
        features["std"][i] = np.std(sig)
        features["max"][i] = np.max(sig)
        features["min"][i] = np.min(sig)

    return features


def main():
    print("=" * 80)
    print("PERIODIC FEATURES ANALYSIS FOR DUFFING ATTRACTORS")
    print("=" * 80)

    params: DuffingParams = {"delta": 0.08, "k3": 1, "A": 0.2}
    ode_system = DuffingJaxODE(params)

    template_ics = [
        [-0.21, 0.02],
        [1.05, 0.77],
        [-0.67, 0.02],
        [-0.46, 0.30],
        [-0.43, 0.12],
    ]

    template_labels = [
        "y1: period-1 LC",
        "y2: period-1 LC",
        "y3: period-2 LC",
        "y4: period-2 LC",
        "y5: period-3 LC",
    ]

    y0 = torch.tensor(template_ics, dtype=torch.float32)

    solver = JaxSolver(
        time_span=(0, 1000),
        n_steps=1000,
        device="cpu",
        cache_dir=None,
    )

    print("\n1. Integrating template initial conditions...")
    print("-" * 40)
    time_arr, y_arr = solver.integrate(ode_system, y0)

    y_np = y_arr.cpu().numpy()
    time_np = time_arr.cpu().numpy()
    dt = float(time_np[1] - time_np[0])

    print(f"   Solution shape: {y_np.shape}")
    print(f"   Time step: {dt:.4f}")

    n_steady = 200

    print("\n2. Computing features...")
    print("-" * 40)

    variance = compute_steady_state_variance(y_np, n_steady)
    periods, periodicity = compute_autocorrelation_periodicity(y_np, n_steady, state_idx=0)
    spectral = compute_spectral_features(y_np, dt, n_steady, state_idx=0)
    peaks_pp, total_peaks, peak_std = count_peaks_per_period(y_np, n_steady, state_idx=0)
    amplitude = compute_amplitude_features(y_np, n_steady, state_idx=0)

    print("\n3. Feature Values for Each Attractor")
    print("=" * 80)

    feature_table = {
        "Variance": variance,
        "Autocorr Period (lag)": periods,
        "Periodicity Strength": periodicity,
        "Dominant Frequency": spectral["dominant_freq"],
        "N Significant Freqs": spectral["n_significant_freqs"],
        "Spectral Entropy": spectral["spectral_entropy"],
        "Freq Ratio (2nd/1st)": spectral["freq_ratio_2nd_1st"],
        "Power 1st Harmonic": spectral["power_1st_harmonic"],
        "Power 2nd Harmonic": spectral["power_2nd_harmonic"],
        "Power 3rd Harmonic": spectral["power_3rd_harmonic"],
        "Peaks per Period": peaks_pp,
        "Total Peaks": total_peaks,
        "Peak Height Std": peak_std,
        "Amplitude": amplitude["amplitude"],
        "Mean": amplitude["mean"],
        "Std Dev": amplitude["std"],
        "Max": amplitude["max"],
        "Min": amplitude["min"],
    }

    header = f"{'Feature':<25}"
    for label in template_labels:
        header += f" | {label:>15}"
    print(header)
    print("-" * len(header))

    for feature_name, values in feature_table.items():
        row = f"{feature_name:<25}"
        for v in values:
            if abs(v) < 0.001 and v != 0:
                row += f" | {v:>15.2e}"
            else:
                row += f" | {v:>15.4f}"
        print(row)

    print("\n4. Key Observations for Clustering")
    print("=" * 80)

    print("\n   Features that distinguish period-1 from period-2/3:")
    print("   " + "-" * 50)

    p1_indices = [0, 1]
    p2_indices = [2, 3]
    p3_indices = [4]

    for fname, vals in feature_table.items():
        p1_mean = np.mean([vals[i] for i in p1_indices])
        p2_mean = np.mean([vals[i] for i in p2_indices])
        p3_mean = vals[p3_indices[0]]

        if p1_mean != 0:
            diff_p1_p2 = abs(p2_mean - p1_mean) / abs(p1_mean) * 100
            diff_p1_p3 = abs(p3_mean - p1_mean) / abs(p1_mean) * 100
        else:
            diff_p1_p2 = abs(p2_mean - p1_mean) * 100
            diff_p1_p3 = abs(p3_mean - p1_mean) * 100

        if diff_p1_p2 > 20 or diff_p1_p3 > 20:
            print(f"   {fname}: P1={p1_mean:.4f}, P2={p2_mean:.4f}, P3={p3_mean:.4f}")

    print("\n   Features that distinguish period-2 from period-3:")
    print("   " + "-" * 50)

    for fname, vals in feature_table.items():
        p2_mean = np.mean([vals[i] for i in p2_indices])
        p3_mean = vals[p3_indices[0]]

        if p2_mean != 0:
            diff = abs(p3_mean - p2_mean) / abs(p2_mean) * 100
        else:
            diff = abs(p3_mean - p2_mean) * 100

        if diff > 10:
            print(f"   {fname}: P2={p2_mean:.4f}, P3={p3_mean:.4f} (diff: {diff:.1f}%)")

    print("\n   Features that distinguish y1 from y2 (both period-1):")
    print("   " + "-" * 50)

    for fname, vals in feature_table.items():
        y1_val = vals[0]
        y2_val = vals[1]

        if y1_val != 0:
            diff = abs(y2_val - y1_val) / abs(y1_val) * 100
        else:
            diff = abs(y2_val - y1_val) * 100

        if diff > 5:
            print(f"   {fname}: y1={y1_val:.4f}, y2={y2_val:.4f} (diff: {diff:.1f}%)")

    print("\n   Features that distinguish y3 from y4 (both period-2):")
    print("   " + "-" * 50)

    for fname, vals in feature_table.items():
        y3_val = vals[2]
        y4_val = vals[3]

        if y3_val != 0:
            diff = abs(y4_val - y3_val) / abs(y3_val) * 100
        else:
            diff = abs(y4_val - y3_val) * 100

        if diff > 5:
            print(f"   {fname}: y3={y3_val:.4f}, y4={y4_val:.4f} (diff: {diff:.1f}%)")


if __name__ == "__main__":
    main()
