"""Accuracy, naming, compilation, and export tests for reoccurrence features."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from tstorch.calculators.torch_features_reocurrance import (
    PercentageOfReoccurringDatapointsToAllDatapoints,
    PercentageOfReoccurringValuesToAllValues,
    RatioValueNumberToTimeSeriesLength,
    SumOfReoccurringDataPoints,
    SumOfReoccurringValues,
)
from tstorch.feature_factory import ParameterlessFeature

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["reocurrance"]}


def test_percentage_of_reoccurring_datapoints_to_all_datapoints_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        PercentageOfReoccurringDatapointsToAllDatapoints,
        FEATURES["percentage_of_reoccurring_datapoints_to_all_datapoints"],
        canonical_trajectories,
    )


def test_percentage_of_reoccurring_datapoints_to_all_datapoints_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        PercentageOfReoccurringDatapointsToAllDatapoints,
        None,
        y_bst,
    )


def test_percentage_of_reoccurring_datapoints_to_all_datapoints_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        PercentageOfReoccurringDatapointsToAllDatapoints,
        None,
        y_bst,
    )


def test_percentage_of_reoccurring_datapoints_to_all_datapoints_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "percentage_of_reoccurring_datapoints_to_all_datapoints",
        PercentageOfReoccurringDatapointsToAllDatapoints,
    )


def test_percentage_of_reoccurring_values_to_all_values_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        PercentageOfReoccurringValuesToAllValues,
        FEATURES["percentage_of_reoccurring_values_to_all_values"],
        canonical_trajectories,
    )


def test_percentage_of_reoccurring_values_to_all_values_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "percentage_of_reoccurring_values_to_all_values",
        PercentageOfReoccurringValuesToAllValues,
        None,
        y_bst,
    )


def test_percentage_of_reoccurring_values_to_all_values_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "percentage_of_reoccurring_values_to_all_values",
        PercentageOfReoccurringValuesToAllValues,
        None,
        y_bst,
    )


def test_percentage_of_reoccurring_values_to_all_values_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "percentage_of_reoccurring_values_to_all_values",
        PercentageOfReoccurringValuesToAllValues,
    )


def test_sum_of_reoccurring_data_points_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        SumOfReoccurringDataPoints,
        FEATURES["sum_of_reoccurring_data_points"],
        canonical_trajectories,
    )


def test_sum_of_reoccurring_data_points_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "sum_of_reoccurring_data_points", SumOfReoccurringDataPoints, None, y_bst
    )


def test_sum_of_reoccurring_data_points_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "sum_of_reoccurring_data_points", SumOfReoccurringDataPoints, None, y_bst
    )


def test_sum_of_reoccurring_data_points_has_valid_names() -> None:
    assert_feature_names_are_valid("sum_of_reoccurring_data_points", SumOfReoccurringDataPoints)


def test_sum_of_reoccurring_values_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        SumOfReoccurringValues,
        FEATURES["sum_of_reoccurring_values"],
        canonical_trajectories,
    )


def test_sum_of_reoccurring_values_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "sum_of_reoccurring_values", SumOfReoccurringValues, None, y_bst
    )


def test_sum_of_reoccurring_values_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("sum_of_reoccurring_values", SumOfReoccurringValues, None, y_bst)


def test_sum_of_reoccurring_values_has_valid_names() -> None:
    assert_feature_names_are_valid("sum_of_reoccurring_values", SumOfReoccurringValues)


def test_ratio_value_number_to_time_series_length_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        RatioValueNumberToTimeSeriesLength,
        FEATURES["ratio_value_number_to_time_series_length"],
        canonical_trajectories,
    )


def test_ratio_value_number_to_time_series_length_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "ratio_value_number_to_time_series_length",
        RatioValueNumberToTimeSeriesLength,
        None,
        y_bst,
    )


def test_ratio_value_number_to_time_series_length_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "ratio_value_number_to_time_series_length",
        RatioValueNumberToTimeSeriesLength,
        None,
        y_bst,
    )


def test_ratio_value_number_to_time_series_length_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "ratio_value_number_to_time_series_length", RatioValueNumberToTimeSeriesLength
    )


def test_reoccurrence_nan_and_multiplicity_semantics() -> None:
    x = torch.tensor([[[2.0, 2.0, 2.0, 1.0, float("nan"), float("nan")]]])

    torch.testing.assert_close(
        PercentageOfReoccurringDatapointsToAllDatapoints()(x),
        torch.tensor([[[0.5]]]),
    )
    torch.testing.assert_close(
        PercentageOfReoccurringValuesToAllValues()(x),
        torch.tensor([[[2.0 / 3.0]]]),
    )
    assert torch.isnan(SumOfReoccurringDataPoints()(x)).all()
    assert torch.isnan(SumOfReoccurringValues()(x)).all()
    torch.testing.assert_close(
        RatioValueNumberToTimeSeriesLength()(x),
        torch.tensor([[[0.5]]]),
    )


def test_reoccurrence_empty_series_semantics() -> None:
    x = torch.empty(2, 3, 0)

    assert torch.isnan(PercentageOfReoccurringDatapointsToAllDatapoints()(x)).all()
    assert torch.isnan(PercentageOfReoccurringValuesToAllValues()(x)).all()
    assert torch.equal(SumOfReoccurringDataPoints()(x), torch.zeros(2, 3, 1))
    assert torch.equal(SumOfReoccurringValues()(x), torch.zeros(2, 3, 1))
    assert torch.isnan(RatioValueNumberToTimeSeriesLength()(x)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_reoccurrence_features_match_cpu_on_gpu(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_cpu = next(iter(canonical_trajectories.case_tensors.values()))
    features: tuple[ParameterlessFeature, ...] = (
        PercentageOfReoccurringDatapointsToAllDatapoints(),
        PercentageOfReoccurringValuesToAllValues(),
        SumOfReoccurringDataPoints(),
        SumOfReoccurringValues(),
        RatioValueNumberToTimeSeriesLength(),
    )
    for feature in features:
        cpu_result: Tensor = feature(y_cpu)
        gpu_result: Tensor = feature(y_cpu.cuda()).cpu()
        torch.testing.assert_close(gpu_result, cpu_result, atol=1e-5, rtol=1e-5, equal_nan=True)
