"""Numerical-accuracy and torch.compile(fullgraph=True)/torch.export safety tests for
`tstorch.calculators.torch_features_boolean`.
"""

from __future__ import annotations

from tstorch.calculators.torch_features_boolean import (
    HasDuplicate,
    HasDuplicateMax,
    HasDuplicateMin,
    HasVarianceLargerThanStandardDeviation,
    LargeStandardDeviation,
    LargeStandardDeviationParams,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["boolean"]}


def test_has_duplicate_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(HasDuplicate, FEATURES["has_duplicate"], canonical_trajectories)


def test_has_duplicate_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("has_duplicate", HasDuplicate, None, y_bst)


def test_has_duplicate_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("has_duplicate", HasDuplicate, None, y_bst)


def test_has_duplicate_has_valid_names() -> None:
    assert_feature_names_are_valid("has_duplicate", HasDuplicate)


def test_has_duplicate_max_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(
        HasDuplicateMax,
        FEATURES["has_duplicate_max"],
        canonical_trajectories,
    )


def test_has_duplicate_max_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("has_duplicate_max", HasDuplicateMax, None, y_bst)


def test_has_duplicate_max_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("has_duplicate_max", HasDuplicateMax, None, y_bst)


def test_has_duplicate_max_has_valid_names() -> None:
    assert_feature_names_are_valid("has_duplicate_max", HasDuplicateMax)


def test_has_duplicate_min_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(
        HasDuplicateMin,
        FEATURES["has_duplicate_min"],
        canonical_trajectories,
    )


def test_has_duplicate_min_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("has_duplicate_min", HasDuplicateMin, None, y_bst)


def test_has_duplicate_min_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("has_duplicate_min", HasDuplicateMin, None, y_bst)


def test_has_duplicate_min_has_valid_names() -> None:
    assert_feature_names_are_valid("has_duplicate_min", HasDuplicateMin)


def test_has_variance_larger_than_standard_deviation_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        HasVarianceLargerThanStandardDeviation,
        FEATURES["has_variance_larger_than_standard_deviation"],
        canonical_trajectories,
    )


def test_has_variance_larger_than_standard_deviation_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "has_variance_larger_than_standard_deviation",
        HasVarianceLargerThanStandardDeviation,
        None,
        y_bst,
    )


def test_has_variance_larger_than_standard_deviation_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "has_variance_larger_than_standard_deviation",
        HasVarianceLargerThanStandardDeviation,
        None,
        y_bst,
    )


def test_has_variance_larger_than_standard_deviation_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "has_variance_larger_than_standard_deviation", HasVarianceLargerThanStandardDeviation
    )


_LARGE_STANDARD_DEVIATION_RS = [0.01, 0.05, 0.1, 0.2, 0.25, 0.35, 0.5, 0.75]


def test_large_standard_deviation_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    params: list[LargeStandardDeviationParams] = [{"r": r} for r in _LARGE_STANDARD_DEVIATION_RS]
    assert_feature_matches_tsfresh(
        LargeStandardDeviation,
        FEATURES["large_standard_deviation"],
        canonical_trajectories,
        params,
    )


def test_large_standard_deviation_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "large_standard_deviation", LargeStandardDeviation, [{"r": 0.1}, {"r": 0.25}], y_bst
    )


def test_large_standard_deviation_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "large_standard_deviation", LargeStandardDeviation, [{"r": 0.1}, {"r": 0.25}], y_bst
    )


def test_large_standard_deviation_has_valid_names() -> None:
    params: list[LargeStandardDeviationParams] = [{"r": r} for r in _LARGE_STANDARD_DEVIATION_RS]
    assert_feature_names_are_valid("large_standard_deviation", LargeStandardDeviation, params)
