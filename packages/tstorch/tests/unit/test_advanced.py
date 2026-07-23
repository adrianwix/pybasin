"""Accuracy, naming, compilation, and export tests for advanced features."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import Tensor
from tstorch.calculators.torch_features_advanced import (
    C3,
    BenfordCorrelation,
    C3Params,
    EnergyRatioByChunks,
    EnergyRatioByChunksParams,
    TimeReversalAsymmetryStatistic,
    TimeReversalAsymmetryStatisticParams,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["advanced"]}

_LAGS = [1, 0, 2, 3, 5, 10, 499, 500]
_C3_PARAMS: list[C3Params] = [{"lag": lag} for lag in _LAGS]
_TIME_REVERSAL_PARAMS: list[TimeReversalAsymmetryStatisticParams] = [{"lag": lag} for lag in _LAGS]
_ENERGY_RATIO_PARAMS: list[EnergyRatioByChunksParams] = [
    {"num_segments": 10, "segment_focus": 0},
    {"num_segments": 1, "segment_focus": 0},
    {"num_segments": 2, "segment_focus": 0},
    {"num_segments": 2, "segment_focus": 1},
    {"num_segments": 3, "segment_focus": 0},
    {"num_segments": 3, "segment_focus": 1},
    {"num_segments": 3, "segment_focus": 2},
    {"num_segments": 10, "segment_focus": 9},
]


def test_benford_correlation_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        BenfordCorrelation,
        FEATURES["benford_correlation"],
        canonical_trajectories,
    )


def test_benford_correlation_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("benford_correlation", BenfordCorrelation, None, y_bst)


def test_benford_correlation_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("benford_correlation", BenfordCorrelation, None, y_bst)


def test_benford_correlation_has_valid_names() -> None:
    assert_feature_names_are_valid("benford_correlation", BenfordCorrelation)


def test_benford_correlation_is_nan_for_constant_digit_distribution() -> None:
    values = torch.arange(1.0, 10.0).repeat(2).reshape(1, 1, -1)
    assert torch.isnan(BenfordCorrelation()(values)).all()


def test_c3_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(
        C3,
        FEATURES["c3"],
        canonical_trajectories,
        _C3_PARAMS,
    )


def test_c3_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("c3", C3, _C3_PARAMS, y_bst)


def test_c3_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("c3", C3, _C3_PARAMS, y_bst)


def test_c3_has_valid_names() -> None:
    assert_feature_names_are_valid("c3", C3, _C3_PARAMS)


def test_c3_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        C3([])


def test_energy_ratio_by_chunks_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        EnergyRatioByChunks,
        FEATURES["energy_ratio_by_chunks"],
        canonical_trajectories,
        _ENERGY_RATIO_PARAMS,
    )


def test_energy_ratio_by_chunks_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "energy_ratio_by_chunks",
        EnergyRatioByChunks,
        _ENERGY_RATIO_PARAMS,
        y_bst,
    )


def test_energy_ratio_by_chunks_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "energy_ratio_by_chunks",
        EnergyRatioByChunks,
        _ENERGY_RATIO_PARAMS,
        y_bst,
    )


def test_energy_ratio_by_chunks_compiles_dynamic_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    feature = EnergyRatioByChunks(_ENERGY_RATIO_PARAMS)
    compile_feature = cast(
        Callable[..., Callable[[Tensor], Tensor]],
        vars(torch)["compile"],
    )
    compiled = compile_feature(feature, dynamic=True, fullgraph=True, backend="eager")
    compiled(y_bst)


def test_energy_ratio_by_chunks_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "energy_ratio_by_chunks", EnergyRatioByChunks, _ENERGY_RATIO_PARAMS
    )


def test_energy_ratio_by_chunks_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        EnergyRatioByChunks([])


def test_time_reversal_asymmetry_statistic_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        TimeReversalAsymmetryStatistic,
        FEATURES["time_reversal_asymmetry_statistic"],
        canonical_trajectories,
        _TIME_REVERSAL_PARAMS,
    )


def test_time_reversal_asymmetry_statistic_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "time_reversal_asymmetry_statistic",
        TimeReversalAsymmetryStatistic,
        _TIME_REVERSAL_PARAMS,
        y_bst,
    )


def test_time_reversal_asymmetry_statistic_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "time_reversal_asymmetry_statistic",
        TimeReversalAsymmetryStatistic,
        _TIME_REVERSAL_PARAMS,
        y_bst,
    )


def test_time_reversal_asymmetry_statistic_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "time_reversal_asymmetry_statistic",
        TimeReversalAsymmetryStatistic,
        _TIME_REVERSAL_PARAMS,
    )


def test_time_reversal_asymmetry_statistic_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        TimeReversalAsymmetryStatistic([])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_advanced_features_match_cpu_on_gpu(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_cpu = next(iter(canonical_trajectories.case_tensors.values()))
    features = [
        BenfordCorrelation(),
        C3(_C3_PARAMS),
        EnergyRatioByChunks(_ENERGY_RATIO_PARAMS),
        TimeReversalAsymmetryStatistic(_TIME_REVERSAL_PARAMS),
    ]
    for feature in features:
        cpu_result = feature(y_cpu)
        gpu_result = feature(y_cpu.cuda()).cpu()
        assert torch.allclose(cpu_result, gpu_result, atol=1e-5, equal_nan=True)
