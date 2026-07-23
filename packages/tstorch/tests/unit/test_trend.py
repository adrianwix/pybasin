"""Accuracy, naming, compilation, and export tests for trend features."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import Tensor
from tstorch.calculators.torch_features_trend import (
    AggLinearTrend,
    AggLinearTrendParams,
    ArCoefficient,
    ArCoefficientParams,
    AugmentedDickeyFuller,
    AugmentedDickeyFullerParams,
    LinearTrend,
    LinearTrendParams,
    LinearTrendTimewise,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["trend"]}

_LINEAR_TREND_PARAMS: list[LinearTrendParams] = [
    {"attr": "slope"},
    {"attr": "intercept"},
    {"attr": "rvalue"},
    {"attr": "pvalue"},
    {"attr": "stderr"},
]
_AGG_LINEAR_TREND_PARAMS: list[AggLinearTrendParams] = [
    {"chunk_size": 5, "f_agg": "mean", "attr": "slope"},
    {"chunk_size": 5, "f_agg": "var", "attr": "intercept"},
    {"chunk_size": 10, "f_agg": "min", "attr": "rvalue"},
    {"chunk_size": 10, "f_agg": "max", "attr": "pvalue"},
    {"chunk_size": 20, "f_agg": "mean", "attr": "stderr"},
    {"chunk_size": 50, "f_agg": "mean", "attr": "pvalue"},
    {"chunk_size": 100, "f_agg": "var", "attr": "slope"},
    {"chunk_size": 250, "f_agg": "max", "attr": "intercept"},
]
_AGG_LINEAR_TREND_TSFRESH_PARAMS = [
    {
        "chunk_len": param["chunk_size"],
        "f_agg": param["f_agg"],
        "attr": param["attr"],
    }
    for param in _AGG_LINEAR_TREND_PARAMS
]
_AR_COEFFICIENT_PARAMS: list[ArCoefficientParams] = [
    {"k": 3, "coeff": 0},
    {"k": 3, "coeff": 1},
    {"k": 3, "coeff": 2},
    {"k": 3, "coeff": 3},
    {"k": 5, "coeff": 0},
    {"k": 5, "coeff": 2},
    {"k": 5, "coeff": 5},
    {"k": 5, "coeff": 6},
]
_ADF_PARAMS: list[AugmentedDickeyFullerParams] = [
    {"attr": "teststat"},
    {"attr": "pvalue"},
    {"attr": "usedlag"},
    {"attr": "teststat", "autolag": "BIC"},
    {"attr": "pvalue", "autolag": "BIC"},
    {"attr": "usedlag", "autolag": "t-stat"},
    {"attr": "teststat", "autolag": "t-stat"},
    {"attr": "pvalue", "autolag": None},
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_linear_trend_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
    dtype: torch.dtype,
) -> None:
    assert_feature_matches_tsfresh(
        LinearTrend,
        FEATURES["linear_trend"],
        canonical_trajectories,
        _LINEAR_TREND_PARAMS,
        input_dtype=dtype,
    )


def test_linear_trend_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "linear_trend",
        LinearTrend,
        [{"attr": "slope"}],
        y_bst,
    )


def test_linear_trend_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "linear_trend",
        LinearTrend,
        [{"attr": "slope"}],
        y_bst,
    )


def test_linear_trend_has_valid_names() -> None:
    assert_feature_names_are_valid("linear_trend", LinearTrend, _LINEAR_TREND_PARAMS)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_linear_trend_timewise_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
    dtype: torch.dtype,
) -> None:
    assert_feature_matches_tsfresh(
        LinearTrendTimewise,
        FEATURES["linear_trend_timewise"],
        canonical_trajectories,
        _LINEAR_TREND_PARAMS,
        input_dtype=dtype,
    )


def test_linear_trend_timewise_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "linear_trend_timewise",
        LinearTrendTimewise,
        [{"attr": "slope"}],
        y_bst,
    )


def test_linear_trend_timewise_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "linear_trend_timewise",
        LinearTrendTimewise,
        [{"attr": "slope"}],
        y_bst,
    )


def test_linear_trend_timewise_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "linear_trend_timewise",
        LinearTrendTimewise,
        _LINEAR_TREND_PARAMS,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_agg_linear_trend_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
    dtype: torch.dtype,
) -> None:
    assert_feature_matches_tsfresh(
        AggLinearTrend,
        FEATURES["agg_linear_trend"],
        canonical_trajectories,
        _AGG_LINEAR_TREND_PARAMS,
        tsfresh_params=_AGG_LINEAR_TREND_TSFRESH_PARAMS,
        input_dtype=dtype,
        atol=6.0e-3 if dtype == torch.float32 else 1.0e-3,
    )


def test_agg_linear_trend_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "agg_linear_trend",
        AggLinearTrend,
        [{"chunk_size": 5, "f_agg": "mean", "attr": "slope"}],
        y_bst,
    )


def test_agg_linear_trend_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "agg_linear_trend",
        AggLinearTrend,
        [{"chunk_size": 5, "f_agg": "mean", "attr": "slope"}],
        y_bst,
    )


def test_agg_linear_trend_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "agg_linear_trend",
        AggLinearTrend,
        _AGG_LINEAR_TREND_PARAMS,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_ar_coefficient_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
    dtype: torch.dtype,
) -> None:
    assert_feature_matches_tsfresh(
        ArCoefficient,
        FEATURES["ar_coefficient"],
        canonical_trajectories,
        _AR_COEFFICIENT_PARAMS,
        input_dtype=dtype,
        atol=1.5e-1 if dtype == torch.float32 else 1.0e-3,
    )


def test_ar_coefficient_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "ar_coefficient",
        ArCoefficient,
        [{"k": 3, "coeff": 1}],
        y_bst,
    )


def test_ar_coefficient_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "ar_coefficient",
        ArCoefficient,
        [{"k": 3, "coeff": 1}],
        y_bst,
    )


def test_ar_coefficient_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "ar_coefficient",
        ArCoefficient,
        _AR_COEFFICIENT_PARAMS,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_augmented_dickey_fuller_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
    dtype: torch.dtype,
) -> None:
    assert_feature_matches_tsfresh(
        AugmentedDickeyFuller,
        FEATURES["augmented_dickey_fuller"],
        canonical_trajectories,
        _ADF_PARAMS,
        input_dtype=dtype,
        atol=2.0 if dtype == torch.float32 else 1.0e-3,
    )


def test_augmented_dickey_fuller_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "augmented_dickey_fuller",
        AugmentedDickeyFuller,
        [{"attr": "teststat"}],
        y_bst,
    )


def test_augmented_dickey_fuller_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "augmented_dickey_fuller",
        AugmentedDickeyFuller,
        [{"attr": "teststat"}],
        y_bst,
    )


def test_augmented_dickey_fuller_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "augmented_dickey_fuller",
        AugmentedDickeyFuller,
        _ADF_PARAMS,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_trend_cuda_matches_cpu(dtype: torch.dtype) -> None:
    x = torch.randn(
        4,
        3,
        128,
        generator=torch.Generator().manual_seed(123),
        dtype=dtype,
    )
    features: tuple[tuple[str, Callable[[Tensor], Tensor]], ...] = (
        ("linear_trend", LinearTrend(_LINEAR_TREND_PARAMS)),
        ("linear_trend_timewise", LinearTrendTimewise(_LINEAR_TREND_PARAMS)),
        ("agg_linear_trend", AggLinearTrend(_AGG_LINEAR_TREND_PARAMS)),
        ("ar_coefficient", ArCoefficient(_AR_COEFFICIENT_PARAMS)),
        ("augmented_dickey_fuller", AugmentedDickeyFuller(_ADF_PARAMS)),
    )
    tolerance = 3.0e-5 if dtype == torch.float32 else 1.0e-12
    for feature_name, feature in features:
        expected = feature(x)
        result = feature(x.cuda()).cpu()
        assert result.dtype == dtype, feature_name
        torch.testing.assert_close(
            result,
            expected,
            atol=tolerance,
            rtol=tolerance,
            equal_nan=True,
            msg=feature_name,
        )
