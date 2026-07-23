"""Accuracy, naming, compilation, and export tests for pattern features."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from tstorch.calculators.torch_features_pattern import (
    LongestStrikeAboveMean,
    LongestStrikeBelowMean,
    NumberCrossingM,
    NumberCrossingMParams,
    NumberCwtPeaks,
    NumberCwtPeaksParams,
    NumberPeaks,
    NumberPeaksParams,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["pattern"]}

_NUMBER_CROSSING_M_PARAMS: list[NumberCrossingMParams] = [
    {"m": -10.0},
    {"m": -2.0},
    {"m": -1.0},
    {"m": -0.25},
    {"m": 0.0},
    {"m": 0.25},
    {"m": 1.0},
    {"m": 10.0},
]
_NUMBER_PEAKS_PARAMS: list[NumberPeaksParams] = [
    {"n": 1},
    {"n": 2},
    {"n": 3},
    {"n": 5},
    {"n": 10},
    {"n": 25},
    {"n": 50},
    {"n": 100},
]
_NUMBER_CWT_PEAKS_PARAMS: list[NumberCwtPeaksParams] = [
    {"n": 1},
    {"n": 2},
    {"n": 3},
    {"n": 4},
    {"n": 5},
    {"n": 6},
    {"n": 7},
    {"n": 8},
]


def test_longest_strike_above_mean_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        LongestStrikeAboveMean,
        FEATURES["longest_strike_above_mean"],
        canonical_trajectories,
    )


def test_longest_strike_above_mean_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "longest_strike_above_mean", LongestStrikeAboveMean, None, y_bst
    )


def test_longest_strike_above_mean_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("longest_strike_above_mean", LongestStrikeAboveMean, None, y_bst)


def test_longest_strike_above_mean_has_valid_names() -> None:
    assert_feature_names_are_valid("longest_strike_above_mean", LongestStrikeAboveMean)


def test_longest_strike_below_mean_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        LongestStrikeBelowMean,
        FEATURES["longest_strike_below_mean"],
        canonical_trajectories,
    )


def test_longest_strike_below_mean_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "longest_strike_below_mean", LongestStrikeBelowMean, None, y_bst
    )


def test_longest_strike_below_mean_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("longest_strike_below_mean", LongestStrikeBelowMean, None, y_bst)


def test_longest_strike_below_mean_has_valid_names() -> None:
    assert_feature_names_are_valid("longest_strike_below_mean", LongestStrikeBelowMean)


def test_number_crossing_m_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        NumberCrossingM,
        FEATURES["number_crossing_m"],
        canonical_trajectories,
        _NUMBER_CROSSING_M_PARAMS,
    )


def test_number_crossing_m_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "number_crossing_m", NumberCrossingM, _NUMBER_CROSSING_M_PARAMS[:2], y_bst
    )


def test_number_crossing_m_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "number_crossing_m", NumberCrossingM, _NUMBER_CROSSING_M_PARAMS[:2], y_bst
    )


def test_number_crossing_m_has_valid_names() -> None:
    assert_feature_names_are_valid("number_crossing_m", NumberCrossingM, _NUMBER_CROSSING_M_PARAMS)


def test_number_crossing_m_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        NumberCrossingM([])


def test_number_peaks_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        NumberPeaks,
        FEATURES["number_peaks"],
        canonical_trajectories,
        _NUMBER_PEAKS_PARAMS,
    )


def test_number_cwt_peaks_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        NumberCwtPeaks,
        FEATURES["number_cwt_peaks"],
        canonical_trajectories,
        _NUMBER_CWT_PEAKS_PARAMS,
        atol=3.0,
        rtol=0.0,
    )


def test_number_cwt_peaks_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "number_cwt_peaks", NumberCwtPeaks, _NUMBER_CWT_PEAKS_PARAMS[4:5], y_bst
    )


def test_number_cwt_peaks_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "number_cwt_peaks", NumberCwtPeaks, _NUMBER_CWT_PEAKS_PARAMS[4:5], y_bst
    )


def test_number_cwt_peaks_has_valid_names() -> None:
    assert_feature_names_are_valid("number_cwt_peaks", NumberCwtPeaks, _NUMBER_CWT_PEAKS_PARAMS)


def test_number_cwt_peaks_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        NumberCwtPeaks([])
    with pytest.raises(ValueError, match="positive"):
        NumberCwtPeaks([{"n": 0}])


def test_number_cwt_peaks_documents_roundoff_sensitive_strict_maximum() -> None:
    x = torch.ones((1, 1, 31), dtype=torch.float64)
    result = NumberCwtPeaks([{"n": 5}])(x)
    assert torch.equal(result, x.new_tensor([[[0.0]]]))


def test_number_peaks_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("number_peaks", NumberPeaks, _NUMBER_PEAKS_PARAMS[:2], y_bst)


def test_number_peaks_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("number_peaks", NumberPeaks, _NUMBER_PEAKS_PARAMS[:2], y_bst)


def test_number_peaks_has_valid_names() -> None:
    assert_feature_names_are_valid("number_peaks", NumberPeaks, _NUMBER_PEAKS_PARAMS)


def test_number_peaks_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        NumberPeaks([])
    with pytest.raises(ValueError, match="positive"):
        NumberPeaks([{"n": 0}])


def test_pattern_boundary_semantics() -> None:
    x = torch.tensor([[[0.0, 1.0, 0.0, 1.0, 0.0]]], dtype=torch.float64)

    assert torch.equal(LongestStrikeAboveMean()(x), x.new_tensor([[[1.0]]]))
    assert torch.equal(LongestStrikeBelowMean()(x), x.new_tensor([[[1.0]]]))
    assert torch.equal(NumberCrossingM([{"m": 0.0}])(x), x.new_tensor([[[4.0]]]))
    assert torch.equal(NumberPeaks([{"n": 1}, {"n": 2}])(x), x.new_tensor([[[2.0, 0.0]]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_pattern_features_match_cpu_on_gpu(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_cpu = next(iter(canonical_trajectories.case_tensors.values()))
    features = (
        LongestStrikeAboveMean(),
        LongestStrikeBelowMean(),
        NumberCrossingM(_NUMBER_CROSSING_M_PARAMS),
        NumberCwtPeaks(_NUMBER_CWT_PEAKS_PARAMS[:2]),
        NumberPeaks(_NUMBER_PEAKS_PARAMS),
    )
    for feature in features:
        cpu_result: Tensor = feature(y_cpu)
        gpu_result: Tensor = feature(y_cpu.cuda()).cpu()
        assert torch.allclose(cpu_result, gpu_result, atol=1e-5, rtol=1e-5, equal_nan=True)
