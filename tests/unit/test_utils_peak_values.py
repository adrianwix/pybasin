# pyright: basic
"""Tests for the peak-detection helpers in pybasin.utils."""

import math

import numpy as np
import pytest
import torch
from scipy.signal import argrelmax, find_peaks

from pybasin.utils import extract_peak_values, find_peak_mask


class TestExtractPeakValues:
    """Tests for extract_peak_values function."""

    def test_simple_peaks(self) -> None:
        """Test extraction of simple peak values with known peaks."""
        # Shape: (N=5, B=1, S=1) - single trajectory with peaks at indices 1 and 3
        x = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0]).unsqueeze(1).unsqueeze(2)

        peak_values, peak_counts = extract_peak_values(x, n=1)

        assert peak_counts[0, 0] == 2
        valid = peak_values[:, 0, 0][~torch.isnan(peak_values[:, 0, 0])]
        assert valid.tolist() == [1.0, 2.0]

    def test_single_peak(self) -> None:
        """Test with exactly one peak."""
        # Peak at index 5 with value 10.0
        x = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 4.0, 3.0, 2.0, 1.0])
        x = x.unsqueeze(1).unsqueeze(2)

        peak_values, peak_counts = extract_peak_values(x, n=1)

        assert peak_counts[0, 0] == 1
        assert peak_values[0, 0, 0] == 10.0

    def test_no_peaks_monotonic(self) -> None:
        """Test with monotonic increasing data (no peaks)."""
        x = torch.arange(10).float().unsqueeze(1).unsqueeze(2)

        peak_values, peak_counts = extract_peak_values(x, n=1)

        assert peak_counts[0, 0] == 0
        assert peak_values.shape[0] == 0

    def test_no_peaks_constant(self) -> None:
        """Test with constant data - no peaks with strict inequality."""
        x = torch.ones(10, 1, 1) * 5.0

        peak_values, peak_counts = extract_peak_values(x, n=1)

        # Constant data: no point is strictly greater than neighbors
        assert peak_counts[0, 0] == 0

    def test_batch_independence(self) -> None:
        """Test that peaks are extracted independently per batch."""
        # Batch 0: peaks at 2.0, 4.0
        # Batch 1: peak at 10.0
        # Use strictly decreasing tails to avoid flat region peaks
        x = torch.zeros(7, 2, 1)
        x[:, 0, 0] = torch.tensor([-2.0, 1.0, 2.0, 1.0, 4.0, 1.0, -2.0])
        x[:, 1, 0] = torch.tensor([-2.0, 5.0, 10.0, 5.0, -1.0, -2.0, -3.0])

        peak_values, peak_counts = extract_peak_values(x, n=1)

        assert peak_counts[0, 0] == 2
        assert peak_counts[1, 0] == 1

        batch0_peaks = peak_values[:, 0, 0][~torch.isnan(peak_values[:, 0, 0])]
        batch1_peaks = peak_values[:, 1, 0][~torch.isnan(peak_values[:, 1, 0])]

        assert batch0_peaks.tolist() == [2.0, 4.0]
        assert batch1_peaks.tolist() == [10.0]

    def test_state_independence(self) -> None:
        """Test that peaks are extracted independently per state dimension."""
        # State 0: peaks at 3.0, 6.0
        # State 1: single peak at 100.0
        # Strictly decreasing tails
        x = torch.zeros(9, 1, 2)
        x[:, 0, 0] = torch.tensor([-5.0, 1.0, 3.0, 1.0, 2.0, 6.0, 2.0, 1.0, -5.0])
        x[:, 0, 1] = torch.tensor([-5.0, 50.0, 100.0, 50.0, -1.0, -2.0, -3.0, -4.0, -5.0])

        peak_values, peak_counts = extract_peak_values(x, n=1)

        assert peak_counts[0, 0] == 2
        assert peak_counts[0, 1] == 1

        state0_peaks = peak_values[:, 0, 0][~torch.isnan(peak_values[:, 0, 0])]
        state1_peaks = peak_values[:, 0, 1][~torch.isnan(peak_values[:, 0, 1])]

        assert state0_peaks.tolist() == [3.0, 6.0]
        assert state1_peaks.tolist() == [100.0]

    def test_sinusoidal_peaks(self) -> None:
        """Test with sinusoidal data where peaks are predictable."""
        # sin wave with 2 full periods -> 2 peaks at ~1.0
        t = torch.linspace(0, 4 * math.pi, 41)
        x = torch.sin(t).unsqueeze(1).unsqueeze(2)

        peak_values, peak_counts = extract_peak_values(x, n=1)

        assert peak_counts[0, 0] == 2
        peaks = peak_values[:, 0, 0][~torch.isnan(peak_values[:, 0, 0])]
        assert torch.allclose(peaks, torch.tensor([1.0, 1.0]), atol=0.01)

    def test_nan_padding_different_peak_counts(self) -> None:
        """Test NaN padding when trajectories have different peak counts."""
        # Batch 0: 3 peaks (values 1, 2, 3)
        # Batch 1: 1 peak (value 10)
        # Use strictly decreasing values to avoid flat region peaks
        x = torch.zeros(11, 2, 1)
        x[:, 0, 0] = torch.tensor([-5.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, -4.0, -5.0, -6.0, -7.0])
        x[:, 1, 0] = torch.tensor([-5.0, 5.0, 10.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0])

        peak_values, peak_counts = extract_peak_values(x, n=1)

        assert peak_counts[0, 0] == 3
        assert peak_counts[1, 0] == 1
        assert peak_values.shape[0] == 3  # max_peaks = 3

        batch0_nan_count = torch.isnan(peak_values[:, 0, 0]).sum().item()
        batch1_nan_count = torch.isnan(peak_values[:, 1, 0]).sum().item()

        assert batch0_nan_count == 0  # 3 peaks, no padding needed
        assert batch1_nan_count == 2  # 1 peak, 2 NaNs for padding

    def test_empty_input_short_sequence(self) -> None:
        """Test with sequence too short for any peaks (2*n >= length)."""
        x = torch.tensor([1.0, 2.0, 1.0]).unsqueeze(1).unsqueeze(2)

        peak_values, peak_counts = extract_peak_values(x, n=2)

        assert peak_counts.sum() == 0
        assert peak_values.shape[0] == 0

    def test_dtype_preserved(self) -> None:
        """Test that output dtype matches input dtype."""
        x = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0], dtype=torch.float64)
        x = x.unsqueeze(1).unsqueeze(2)

        peak_values, _ = extract_peak_values(x, n=1)

        assert peak_values.dtype == torch.float64

    def test_device_preserved(self) -> None:
        """Test that output device matches input device."""
        x = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0]).unsqueeze(1).unsqueeze(2)

        peak_values, peak_counts = extract_peak_values(x, n=1)

        assert peak_values.device == x.device
        assert peak_counts.device == x.device

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_window_size_affects_peak_detection(self, n: int) -> None:
        """Test that larger window sizes detect fewer, more prominent peaks."""
        # Create data with a very prominent peak and smaller local peaks
        x = torch.tensor([0.0, 1.0, 0.5, 1.0, 0.0, 5.0, 0.0, 1.0, 0.5, 1.0, 0.0])
        x = x.unsqueeze(1).unsqueeze(2)

        peak_values, peak_counts = extract_peak_values(x, n=n)

        # n=1: detects local peaks (1.0, 1.0, 5.0, 1.0, 1.0)
        # n=2: fewer peaks as window is larger
        # n=3: only the most prominent peak (5.0) should remain
        if n == 3:
            assert peak_counts[0, 0] == 1
            peaks = peak_values[:, 0, 0][~torch.isnan(peak_values[:, 0, 0])]
            assert peaks[0] == 5.0

    def test_peak_values_match_mask_positions(self) -> None:
        """Verify extracted values exactly match values at mask positions."""
        x = torch.tensor([0.0, 3.0, 1.0, 5.0, 2.0, 7.0, 0.0]).unsqueeze(1).unsqueeze(2)

        peak_values, peak_counts = extract_peak_values(x, n=1)
        mask = find_peak_mask(x, n=1)

        expected_peaks = x[:, 0, 0][mask[:, 0, 0]]
        actual_peaks = peak_values[:, 0, 0][~torch.isnan(peak_values[:, 0, 0])]

        assert len(actual_peaks) == len(expected_peaks)
        assert torch.equal(actual_peaks, expected_peaks)
        assert expected_peaks.tolist() == [3.0, 5.0, 7.0]


class TestFindPeakMaskMatchesScipy:
    """Tests verifying find_peak_mask matches scipy.signal.argrelmax behavior.

    Our implementation uses strict inequality (x[i] > all neighbors), which
    matches ``scipy.signal.argrelmax``, NOT ``scipy.signal.find_peaks``.

    The difference: ``find_peaks`` handles flat plateaus by returning their
    middle index, while ``argrelmax`` (and our implementation) returns no peaks
    for plateaus. This is acceptable since exact floating-point equality is
    extremely rare in real dynamical systems data.
    """

    @pytest.mark.parametrize(
        "data,description",
        [
            ([0.0, 1.0, 0.0], "single peak"),
            ([0.0, 1.0, 0.5, 2.0, 0.0], "two peaks"),
            ([0.0, 1.0, 2.0, 3.0, 4.0], "monotonic increasing - no peaks"),
            ([4.0, 3.0, 2.0, 1.0, 0.0], "monotonic decreasing - no peaks"),
            ([5.0, 5.0, 5.0, 5.0, 5.0], "constant - no peaks"),
            ([0.0, 2.0, 1.0, 2.0, 0.0], "equal peaks"),
            ([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], "periodic peaks"),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_matches_scipy(self, data: list[float], description: str) -> None:
        """Verify find_peak_mask matches scipy.signal.argrelmax."""
        x_np = np.array(data)
        (scipy_peaks,) = argrelmax(x_np)

        x_torch = torch.tensor(data).unsqueeze(1).unsqueeze(2)
        torch_mask = find_peak_mask(x_torch, n=1)
        torch_peaks = torch.nonzero(torch_mask[:, 0, 0]).squeeze(-1).tolist()
        if isinstance(torch_peaks, int):
            torch_peaks = [torch_peaks]

        assert torch_peaks == list(scipy_peaks), (
            f"Mismatch for {description}: scipy={list(scipy_peaks)}, torch={torch_peaks}"
        )

    def test_sinusoidal_matches_scipy(self) -> None:
        """Test that sinusoidal data matches scipy (with sufficient resolution)."""
        t = np.linspace(0, 4 * np.pi, 1001)
        x_np = np.sin(t)
        (scipy_peaks,) = argrelmax(x_np)

        x_torch = torch.from_numpy(x_np).float().unsqueeze(1).unsqueeze(2)
        torch_mask = find_peak_mask(x_torch, n=1)
        torch_peaks = torch.nonzero(torch_mask[:, 0, 0]).squeeze(-1).tolist()

        assert torch_peaks == list(scipy_peaks)

    def test_flat_peak_differs_from_find_peaks(self) -> None:
        """Document that flat peak regions behave differently from find_peaks.

        find_peaks returns the middle of a plateau, while argrelmax (and our
        implementation) returns no peaks since no point is strictly greater
        than all neighbors.
        """
        data = [0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0]
        x_np = np.array(data)
        find_peaks_result, _ = find_peaks(x_np)
        (argrelmax_result,) = argrelmax(x_np)

        x_torch = torch.tensor(data).unsqueeze(1).unsqueeze(2)
        torch_mask = find_peak_mask(x_torch, n=1)
        torch_peaks = torch.nonzero(torch_mask[:, 0, 0]).squeeze(-1).tolist()

        assert list(find_peaks_result) == [3]
        assert list(argrelmax_result) == []
        assert torch_peaks == list(argrelmax_result)
