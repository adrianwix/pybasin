"""Tests for the pybasin::find_peaks custom op (CPU/scipy and CUDA/Triton dispatch)."""

import numpy as np
import pytest
import scipy.signal  # type: ignore[import-untyped]
import torch

from pybasin._ops.find_peaks import find_peaks


def _scipy_reference(s: np.ndarray, height: float) -> tuple[np.ndarray, np.ndarray]:
    peaks, props = scipy.signal.find_peaks(s, height=height, prominence=0)
    return peaks, props["prominences"]


CURVES = {
    "random": (np.random.default_rng(0).normal(size=200).astype(np.float32), 0.0),
    "monotonic_increasing": (np.linspace(0, 1, 50).astype(np.float32), 0.0),
    "below_threshold": (np.sin(np.linspace(0, 20, 100)).astype(np.float32), 5.0),
    "flat": (np.zeros(30, dtype=np.float32), 0.0),
    "single_element": (np.array([1.0], dtype=np.float32), 0.0),
    "two_elements": (np.array([1.0, 2.0], dtype=np.float32), 0.0),
}


@pytest.mark.parametrize("name", list(CURVES.keys()))
def test_cpu_matches_scipy(name: str) -> None:
    s, height = CURVES[name]
    peaks_ref, prom_ref = _scipy_reference(s, height)

    peaks, prominences = find_peaks(torch.from_numpy(s), height)  # type: ignore[arg-type]

    assert np.array_equal(peaks.numpy(), peaks_ref)
    assert np.allclose(prominences.numpy(), prom_ref, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("name", list(CURVES.keys()))
def test_cuda_matches_scipy(name: str) -> None:
    s, height = CURVES[name]
    peaks_ref, prom_ref = _scipy_reference(s, height)

    peaks, prominences = find_peaks(torch.from_numpy(s).cuda(), height)  # type: ignore[arg-type]

    assert np.array_equal(peaks.cpu().numpy(), peaks_ref)
    assert np.allclose(prominences.cpu().numpy(), prom_ref, atol=1e-3)


def test_zero_length_input() -> None:
    s = np.array([], dtype=np.float32)
    peaks, prominences = find_peaks(torch.from_numpy(s), 0.0)  # type: ignore[arg-type]
    assert peaks.numel() == 0
    assert prominences.numel() == 0
