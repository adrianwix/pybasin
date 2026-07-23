"""Accuracy, naming, compilation, and export tests for location features."""

from __future__ import annotations

import pytest
import torch
from tstorch.calculators.torch_features_location import (
    FirstLocationOfMaximum,
    FirstLocationOfMinimum,
    IndexMassQuantile,
    IndexMassQuantileParams,
    LastLocationOfMaximum,
    LastLocationOfMinimum,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["location"]}

_INDEX_MASS_QUANTILE_PARAMS: list[IndexMassQuantileParams] = [
    {"q": q} for q in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9]
]


@pytest.mark.parametrize(
    ("feature_name", "feature_class"),
    [
        ("first_location_of_maximum", FirstLocationOfMaximum),
        ("first_location_of_minimum", FirstLocationOfMinimum),
        ("last_location_of_maximum", LastLocationOfMaximum),
        ("last_location_of_minimum", LastLocationOfMinimum),
    ],
)
def test_parameterless_location_feature_matches_tsfresh(
    feature_name: str,
    feature_class: type[
        FirstLocationOfMaximum
        | FirstLocationOfMinimum
        | LastLocationOfMaximum
        | LastLocationOfMinimum
    ],
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        feature_class,
        FEATURES[feature_name],
        canonical_trajectories,
    )


@pytest.mark.parametrize(
    ("feature_name", "feature_class"),
    [
        ("first_location_of_maximum", FirstLocationOfMaximum),
        ("first_location_of_minimum", FirstLocationOfMinimum),
        ("last_location_of_maximum", LastLocationOfMaximum),
        ("last_location_of_minimum", LastLocationOfMinimum),
    ],
)
def test_parameterless_location_feature_compiles_fullgraph(
    feature_name: str,
    feature_class: type[
        FirstLocationOfMaximum
        | FirstLocationOfMinimum
        | LastLocationOfMaximum
        | LastLocationOfMinimum
    ],
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(feature_name, feature_class, None, y_bst)


@pytest.mark.parametrize(
    ("feature_name", "feature_class"),
    [
        ("first_location_of_maximum", FirstLocationOfMaximum),
        ("first_location_of_minimum", FirstLocationOfMinimum),
        ("last_location_of_maximum", LastLocationOfMaximum),
        ("last_location_of_minimum", LastLocationOfMinimum),
    ],
)
def test_parameterless_location_feature_exports_cleanly(
    feature_name: str,
    feature_class: type[
        FirstLocationOfMaximum
        | FirstLocationOfMinimum
        | LastLocationOfMaximum
        | LastLocationOfMinimum
    ],
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(feature_name, feature_class, None, y_bst)


@pytest.mark.parametrize(
    ("feature_name", "feature_class"),
    [
        ("first_location_of_maximum", FirstLocationOfMaximum),
        ("first_location_of_minimum", FirstLocationOfMinimum),
        ("last_location_of_maximum", LastLocationOfMaximum),
        ("last_location_of_minimum", LastLocationOfMinimum),
    ],
)
def test_parameterless_location_feature_has_valid_names(
    feature_name: str,
    feature_class: type[
        FirstLocationOfMaximum
        | FirstLocationOfMinimum
        | LastLocationOfMaximum
        | LastLocationOfMinimum
    ],
) -> None:
    assert_feature_names_are_valid(feature_name, feature_class)


def test_index_mass_quantile_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        IndexMassQuantile,
        FEATURES["index_mass_quantile"],
        canonical_trajectories,
        _INDEX_MASS_QUANTILE_PARAMS,
        atol=5e-3,
    )


def test_index_mass_quantile_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "index_mass_quantile",
        IndexMassQuantile,
        _INDEX_MASS_QUANTILE_PARAMS,
        y_bst,
    )


def test_index_mass_quantile_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "index_mass_quantile",
        IndexMassQuantile,
        _INDEX_MASS_QUANTILE_PARAMS,
        y_bst,
    )


def test_index_mass_quantile_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "index_mass_quantile",
        IndexMassQuantile,
        _INDEX_MASS_QUANTILE_PARAMS,
    )


def test_index_mass_quantile_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        IndexMassQuantile([])


def test_last_locations_resolve_ties_from_the_end() -> None:
    x = torch.tensor([[[3.0, -2.0, 3.0, -2.0]]])

    maximum = LastLocationOfMaximum()(x)
    minimum = LastLocationOfMinimum()(x)

    torch.testing.assert_close(maximum, torch.tensor([[[0.75]]]))
    torch.testing.assert_close(minimum, torch.tensor([[[1.0]]]))


def test_index_mass_quantile_handles_zero_mass_and_q_one() -> None:
    feature = IndexMassQuantile([{"q": 0.5}, {"q": 1.0}])
    x = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 2.0]]])

    result = feature(x)

    torch.testing.assert_close(
        result,
        torch.tensor([[[float("nan"), float("nan")], [2.0 / 3.0, 1.0]]]),
        equal_nan=True,
    )
