"""Accuracy, naming, compilation, and export tests for counting features."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from tstorch.calculators.torch_features_count import (
    CountAbove,
    CountAboveMean,
    CountAboveParams,
    CountBelow,
    CountBelowMean,
    CountBelowParams,
    CountInRange,
    CountInRangeParams,
    CountValue,
    CountValueParams,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["count"]}

_THRESHOLDS = [-2.0, -1.0, -0.25, 0.0, 0.25, 1.0, 2.0, 10.0]
_COUNT_ABOVE_PARAMS: list[CountAboveParams] = [{"t": threshold} for threshold in _THRESHOLDS]
_COUNT_BELOW_PARAMS: list[CountBelowParams] = [{"t": threshold} for threshold in _THRESHOLDS]
_COUNT_IN_RANGE_PARAMS: list[CountInRangeParams] = [
    {"min_val": -2.0, "max_val": 2.0},
    {"min_val": -1.0, "max_val": 1.0},
    {"min_val": 0.0, "max_val": 1.0},
    {"min_val": 1.0, "max_val": 2.0},
    {"min_val": -10.0, "max_val": 0.0},
    {"min_val": 0.0, "max_val": 10.0},
    {"min_val": 2.0, "max_val": 2.0},
    {"min_val": 3.0, "max_val": -3.0},
]
_COUNT_IN_RANGE_TSFRESH_PARAMS: list[dict[str, float]] = [
    {"min": param["min_val"], "max": param["max_val"]} for param in _COUNT_IN_RANGE_PARAMS
]
_COUNT_VALUE_PARAMS: list[CountValueParams] = [
    {"value": -2.0},
    {"value": -1.0},
    {"value": 0.0},
    {"value": 1.0},
    {"value": 2.0},
    {"value": 10.0},
    {"value": float("inf")},
    {"value": float("nan")},
]


def test_count_above_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(
        CountAbove, FEATURES["count_above"], canonical_trajectories, _COUNT_ABOVE_PARAMS
    )


def test_count_above_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("count_above", CountAbove, _COUNT_ABOVE_PARAMS[:2], y_bst)


def test_count_above_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("count_above", CountAbove, _COUNT_ABOVE_PARAMS[:2], y_bst)


def test_count_above_has_valid_names() -> None:
    assert_feature_names_are_valid("count_above", CountAbove, _COUNT_ABOVE_PARAMS)


def test_count_above_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        CountAbove([])


def test_count_above_mean_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        CountAboveMean, FEATURES["count_above_mean"], canonical_trajectories
    )


def test_count_above_mean_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("count_above_mean", CountAboveMean, None, y_bst)


def test_count_above_mean_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("count_above_mean", CountAboveMean, None, y_bst)


def test_count_above_mean_has_valid_names() -> None:
    assert_feature_names_are_valid("count_above_mean", CountAboveMean)


def test_count_below_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(
        CountBelow, FEATURES["count_below"], canonical_trajectories, _COUNT_BELOW_PARAMS
    )


def test_count_below_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("count_below", CountBelow, _COUNT_BELOW_PARAMS[:2], y_bst)


def test_count_below_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("count_below", CountBelow, _COUNT_BELOW_PARAMS[:2], y_bst)


def test_count_below_has_valid_names() -> None:
    assert_feature_names_are_valid("count_below", CountBelow, _COUNT_BELOW_PARAMS)


def test_count_below_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        CountBelow([])


def test_count_below_mean_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        CountBelowMean, FEATURES["count_below_mean"], canonical_trajectories
    )


def test_count_below_mean_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("count_below_mean", CountBelowMean, None, y_bst)


def test_count_below_mean_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("count_below_mean", CountBelowMean, None, y_bst)


def test_count_below_mean_has_valid_names() -> None:
    assert_feature_names_are_valid("count_below_mean", CountBelowMean)


def test_count_in_range_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        CountInRange,
        FEATURES["count_in_range"],
        canonical_trajectories,
        _COUNT_IN_RANGE_PARAMS,
        tsfresh_params=_COUNT_IN_RANGE_TSFRESH_PARAMS,
    )


def test_count_in_range_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "count_in_range", CountInRange, _COUNT_IN_RANGE_PARAMS[:2], y_bst
    )


def test_count_in_range_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "count_in_range", CountInRange, _COUNT_IN_RANGE_PARAMS[:2], y_bst
    )


def test_count_in_range_has_valid_names() -> None:
    assert_feature_names_are_valid("count_in_range", CountInRange, _COUNT_IN_RANGE_PARAMS)


def test_count_in_range_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        CountInRange([])


def test_count_value_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(
        CountValue, FEATURES["count_value"], canonical_trajectories, _COUNT_VALUE_PARAMS
    )


def test_count_value_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("count_value", CountValue, _COUNT_VALUE_PARAMS[:2], y_bst)


def test_count_value_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("count_value", CountValue, _COUNT_VALUE_PARAMS[:2], y_bst)


def test_count_value_has_valid_names() -> None:
    assert_feature_names_are_valid("count_value", CountValue, _COUNT_VALUE_PARAMS)


def test_count_value_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        CountValue([])


def test_count_boundary_and_nan_semantics() -> None:
    x = torch.tensor([[[-1.0, 0.0, 0.0, 1.0, float("nan"), 2.0]]])

    assert torch.equal(CountAbove([{"t": 0.0}])(x), torch.tensor([[[4.0 / 6.0]]]))
    assert torch.equal(CountBelow([{"t": 0.0}])(x), torch.tensor([[[3.0 / 6.0]]]))
    assert torch.equal(CountAboveMean()(x), torch.tensor([[[0.0]]]))
    assert torch.equal(CountBelowMean()(x), torch.tensor([[[0.0]]]))
    assert torch.equal(
        CountInRange([{"min_val": -1.0, "max_val": 1.0}])(x),
        torch.tensor([[[3.0]]]),
    )
    assert torch.equal(CountValue([{"value": float("nan")}])(x), torch.tensor([[[1.0]]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_count_features_match_cpu_on_gpu(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_cpu = next(iter(canonical_trajectories.case_tensors.values()))
    features = (
        CountAbove(_COUNT_ABOVE_PARAMS),
        CountAboveMean(),
        CountBelow(_COUNT_BELOW_PARAMS),
        CountBelowMean(),
        CountInRange(_COUNT_IN_RANGE_PARAMS),
        CountValue(_COUNT_VALUE_PARAMS),
    )
    for feature in features:
        cpu_result: Tensor = feature(y_cpu)
        gpu_result: Tensor = feature(y_cpu.cuda()).cpu()
        assert torch.allclose(cpu_result, gpu_result, atol=1e-5, equal_nan=True)
