"""Numerical-accuracy and torch.compile(fullgraph=True)/torch.export safety tests for
`tstorch.calculators.torch_features_change`.
"""

from __future__ import annotations

from tstorch.calculators.torch_features_change import (
    AbsoluteSumOfChanges,
    ChangeQuantiles,
    ChangeQuantilesParams,
    MeanAbsChange,
    MeanChange,
    MeanSecondDerivativeCentral,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["change"]}


def test_absolute_sum_of_changes_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        AbsoluteSumOfChanges,
        FEATURES["absolute_sum_of_changes"],
        canonical_trajectories,
    )


def test_absolute_sum_of_changes_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("absolute_sum_of_changes", AbsoluteSumOfChanges, None, y_bst)


def test_absolute_sum_of_changes_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("absolute_sum_of_changes", AbsoluteSumOfChanges, None, y_bst)


def test_absolute_sum_of_changes_has_valid_names() -> None:
    assert_feature_names_are_valid("absolute_sum_of_changes", AbsoluteSumOfChanges)


def test_mean_abs_change_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(
        MeanAbsChange, FEATURES["mean_abs_change"], canonical_trajectories
    )


def test_mean_abs_change_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("mean_abs_change", MeanAbsChange, None, y_bst)


def test_mean_abs_change_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("mean_abs_change", MeanAbsChange, None, y_bst)


def test_mean_abs_change_has_valid_names() -> None:
    assert_feature_names_are_valid("mean_abs_change", MeanAbsChange)


def test_mean_change_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(MeanChange, FEATURES["mean_change"], canonical_trajectories)


def test_mean_change_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("mean_change", MeanChange, None, y_bst)


def test_mean_change_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("mean_change", MeanChange, None, y_bst)


def test_mean_change_has_valid_names() -> None:
    assert_feature_names_are_valid("mean_change", MeanChange)


def test_mean_second_derivative_central_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        MeanSecondDerivativeCentral,
        FEATURES["mean_second_derivative_central"],
        canonical_trajectories,
    )


def test_mean_second_derivative_central_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "mean_second_derivative_central", MeanSecondDerivativeCentral, None, y_bst
    )


def test_mean_second_derivative_central_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "mean_second_derivative_central", MeanSecondDerivativeCentral, None, y_bst
    )


def test_mean_second_derivative_central_has_valid_names() -> None:
    assert_feature_names_are_valid("mean_second_derivative_central", MeanSecondDerivativeCentral)


_CHANGE_QUANTILES_PARAMS: list[ChangeQuantilesParams] = [
    {"ql": 0.0, "qh": 1.0, "isabs": True, "f_agg": "mean"},
    {"ql": 0.0, "qh": 0.5, "isabs": True, "f_agg": "mean"},
    {"ql": 0.1, "qh": 0.9, "isabs": False, "f_agg": "mean"},
    {"ql": 0.2, "qh": 0.8, "isabs": True, "f_agg": "mean"},
    {"ql": 0.2, "qh": 0.8, "isabs": False, "f_agg": "mean"},
    {"ql": 0.2, "qh": 0.8, "isabs": True, "f_agg": "var"},
    {"ql": 0.4, "qh": 0.6, "isabs": False, "f_agg": "var"},
    {"ql": 0.7, "qh": 0.3, "isabs": True, "f_agg": "mean"},
]


def test_change_quantiles_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    params = _CHANGE_QUANTILES_PARAMS
    assert_feature_matches_tsfresh(
        ChangeQuantiles, FEATURES["change_quantiles"], canonical_trajectories, params
    )


def test_change_quantiles_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "change_quantiles", ChangeQuantiles, _CHANGE_QUANTILES_PARAMS[:2], y_bst
    )


def test_change_quantiles_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "change_quantiles", ChangeQuantiles, _CHANGE_QUANTILES_PARAMS[:2], y_bst
    )


def test_change_quantiles_has_valid_names() -> None:
    assert_feature_names_are_valid("change_quantiles", ChangeQuantiles, _CHANGE_QUANTILES_PARAMS)
