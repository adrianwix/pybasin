"""Regression tests for the benchmark trajectory BST contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from .benchmark_utils import (
    TRAJECTORY_CACHE_LAYOUT,
    TRAJECTORY_CACHE_VERSION,
    BenchmarkSeriesData,
    filter_steady,
    load_or_build_cache,
)


def _data() -> BenchmarkSeriesData:
    time = torch.arange(5, dtype=torch.float32)
    y_bst = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    return BenchmarkSeriesData(
        time=time,
        y_bst=y_bst,
        case_tensors={"sample": y_bst},
        case_times={"sample": time},
        case_slices={"sample": (0, 2)},
        metadata={"num_trajectories": 2},
    )


def test_filter_steady_slices_time_axis_and_preserves_batch_state() -> None:
    data = _data()

    time, y_bst = filter_steady(data.time, data.y_bst, 2.0)

    assert time.shape == (3,)
    assert y_bst.shape == (2, 3, 3)
    torch.testing.assert_close(y_bst, data.y_bst[:, :, 2:])


def test_filter_steady_rejects_time_axis_mismatch() -> None:
    with pytest.raises(ValueError, match="Expected time shape"):
        filter_steady(torch.arange(5), torch.zeros(2, 3, 4), 2.0)


@pytest.mark.parametrize("cache_problem", ["version", "layout", "wrong_axis"])
def test_cache_rebuilds_stale_or_non_bst_payload(tmp_path: Path, cache_problem: str) -> None:
    path = tmp_path / "trajectories.pt"
    original = _data()
    payload = {
        "cache_version": TRAJECTORY_CACHE_VERSION,
        "layout": TRAJECTORY_CACHE_LAYOUT,
        "time": original.time,
        "y_bst": original.y_bst,
        "case_tensors": original.case_tensors,
        "case_times": original.case_times,
        "case_slices": original.case_slices,
        "metadata": original.metadata,
    }
    if cache_problem == "version":
        payload["cache_version"] = TRAJECTORY_CACHE_VERSION - 1
    elif cache_problem == "layout":
        payload["layout"] = "TSB"
    else:
        payload["y_bst"] = original.y_bst.permute(2, 0, 1)
        payload["case_tensors"] = {"sample": payload["y_bst"]}

    torch.save(payload, path)
    built = _data()
    calls = 0

    def builder() -> BenchmarkSeriesData:
        nonlocal calls
        calls += 1
        return built

    result = load_or_build_cache(path, builder)

    assert calls == 1
    assert result.y_bst.shape == (2, 3, 5)
    refreshed = torch.load(path, map_location="cpu", weights_only=False)
    assert refreshed["cache_version"] == TRAJECTORY_CACHE_VERSION
    assert refreshed["layout"] == TRAJECTORY_CACHE_LAYOUT


def test_cache_loads_current_bst_payload_without_rebuilding(tmp_path: Path) -> None:
    path = tmp_path / "trajectories.pt"
    original = _data()
    load_or_build_cache(path, lambda: original)

    def unexpected_builder() -> BenchmarkSeriesData:
        pytest.fail("valid cache should load without rebuilding")

    result = load_or_build_cache(path, unexpected_builder)

    assert result.y_bst.shape == (2, 3, 5)
