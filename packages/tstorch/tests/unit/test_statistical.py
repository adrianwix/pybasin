"""Numerical-accuracy and torch.compile(fullgraph=True)/torch.export safety tests for
`tstorch.calculators.torch_features_statistical`. See the tstorch-feature-perf skill's
"Recording numerical accuracy" / "Recording torch.compile(fullgraph=True) safety"
sections for what each assertion helper records and why.
"""

from __future__ import annotations

import pytest
from tstorch.calculators.torch_features_statistical import (
    AbsEnergy,
    AbsoluteMaximum,
    Amplitude,
    Delta,
    Kurtosis,
    Length,
    LogDelta,
    Maximum,
    Mean,
    MeanNAbsoluteMax,
    MeanNAbsoluteMaxParams,
    Median,
    Minimum,
    Quantile,
    QuantileParams,
    RatioBeyondRSigma,
    RatioBeyondRSigmaParams,
    RootMeanSquare,
    Skewness,
    StandardDeviation,
    SumValues,
    SymmetryLooking,
    SymmetryLookingParams,
    Variance,
    VariationCoefficient,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["statistical"]}

# =============================================================================
# parameter-free features
# =============================================================================


def test_sum_values_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(SumValues, FEATURES["sum_values"], canonical_trajectories)


def test_sum_values_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("sum_values", SumValues, None, y_bst)


def test_sum_values_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("sum_values", SumValues, None, y_bst)


def test_sum_values_has_valid_names() -> None:
    assert_feature_names_are_valid("sum_values", SumValues)


def test_median_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(Median, FEATURES["median"], canonical_trajectories)


def test_median_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("median", Median, None, y_bst)


def test_median_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("median", Median, None, y_bst)


def test_median_has_valid_names() -> None:
    assert_feature_names_are_valid("median", Median)


def test_mean_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(Mean, FEATURES["mean"], canonical_trajectories)


def test_mean_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("mean", Mean, None, y_bst)


def test_mean_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("mean", Mean, None, y_bst)


def test_mean_has_valid_names() -> None:
    assert_feature_names_are_valid("mean", Mean)


def test_length_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(Length, FEATURES["length"], canonical_trajectories)


def test_length_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("length", Length, None, y_bst)


def test_length_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("length", Length, None, y_bst)


def test_length_has_valid_names() -> None:
    assert_feature_names_are_valid("length", Length)


def test_standard_deviation_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(
        StandardDeviation, FEATURES["standard_deviation"], canonical_trajectories
    )


def test_standard_deviation_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("standard_deviation", StandardDeviation, None, y_bst)


def test_standard_deviation_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("standard_deviation", StandardDeviation, None, y_bst)


def test_standard_deviation_has_valid_names() -> None:
    assert_feature_names_are_valid("standard_deviation", StandardDeviation)


def test_variance_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(Variance, FEATURES["variance"], canonical_trajectories)


def test_variance_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("variance", Variance, None, y_bst)


def test_variance_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("variance", Variance, None, y_bst)


def test_variance_has_valid_names() -> None:
    assert_feature_names_are_valid("variance", Variance)


def test_root_mean_square_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(
        RootMeanSquare, FEATURES["root_mean_square"], canonical_trajectories
    )


def test_root_mean_square_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("root_mean_square", RootMeanSquare, None, y_bst)


def test_root_mean_square_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("root_mean_square", RootMeanSquare, None, y_bst)


def test_root_mean_square_has_valid_names() -> None:
    assert_feature_names_are_valid("root_mean_square", RootMeanSquare)


def test_maximum_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(Maximum, FEATURES["maximum"], canonical_trajectories)


def test_maximum_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("maximum", Maximum, None, y_bst)


def test_maximum_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("maximum", Maximum, None, y_bst)


def test_maximum_has_valid_names() -> None:
    assert_feature_names_are_valid("maximum", Maximum)


def test_absolute_maximum_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(
        AbsoluteMaximum, FEATURES["absolute_maximum"], canonical_trajectories
    )


def test_absolute_maximum_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("absolute_maximum", AbsoluteMaximum, None, y_bst)


def test_absolute_maximum_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("absolute_maximum", AbsoluteMaximum, None, y_bst)


def test_absolute_maximum_has_valid_names() -> None:
    assert_feature_names_are_valid("absolute_maximum", AbsoluteMaximum)


def test_minimum_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(Minimum, FEATURES["minimum"], canonical_trajectories)


def test_minimum_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("minimum", Minimum, None, y_bst)


def test_minimum_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("minimum", Minimum, None, y_bst)


def test_minimum_has_valid_names() -> None:
    assert_feature_names_are_valid("minimum", Minimum)


def test_delta_shape_and_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    """No tsfresh equivalent (see feature_specs.py's `supports_tsfresh=False`) - shape only."""
    for y_bst in canonical_trajectories.case_tensors.values():
        result = Delta()(y_bst)
        expected_shape = (*y_bst.shape[:2], 1)
        assert result.shape == expected_shape, f"expected {expected_shape}, got {result.shape}"
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("delta", Delta, None, y_bst)


def test_delta_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("delta", Delta, None, y_bst)


def test_delta_has_valid_names() -> None:
    assert_feature_names_are_valid("delta", Delta)


def test_log_delta_shape_and_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    """No tsfresh equivalent (see feature_specs.py's `supports_tsfresh=False`) - shape only."""
    for y_bst in canonical_trajectories.case_tensors.values():
        result = LogDelta()(y_bst)
        expected_shape = (*y_bst.shape[:2], 1)
        assert result.shape == expected_shape, f"expected {expected_shape}, got {result.shape}"
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("log_delta", LogDelta, None, y_bst)


def test_log_delta_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("log_delta", LogDelta, None, y_bst)


def test_log_delta_has_valid_names() -> None:
    assert_feature_names_are_valid("log_delta", LogDelta)


def test_abs_energy_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(AbsEnergy, FEATURES["abs_energy"], canonical_trajectories)


def test_abs_energy_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("abs_energy", AbsEnergy, None, y_bst)


def test_abs_energy_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("abs_energy", AbsEnergy, None, y_bst)


def test_abs_energy_has_valid_names() -> None:
    assert_feature_names_are_valid("abs_energy", AbsEnergy)


def test_kurtosis_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(Kurtosis, FEATURES["kurtosis"], canonical_trajectories)


def test_kurtosis_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("kurtosis", Kurtosis, None, y_bst)


def test_kurtosis_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("kurtosis", Kurtosis, None, y_bst)


def test_kurtosis_has_valid_names() -> None:
    assert_feature_names_are_valid("kurtosis", Kurtosis)


def test_skewness_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    assert_feature_matches_tsfresh(Skewness, FEATURES["skewness"], canonical_trajectories)


def test_skewness_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("skewness", Skewness, None, y_bst)


def test_skewness_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("skewness", Skewness, None, y_bst)


def test_skewness_has_valid_names() -> None:
    assert_feature_names_are_valid("skewness", Skewness)


def test_variation_coefficient_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        VariationCoefficient, FEATURES["variation_coefficient"], canonical_trajectories
    )


def test_variation_coefficient_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("variation_coefficient", VariationCoefficient, None, y_bst)


def test_variation_coefficient_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("variation_coefficient", VariationCoefficient, None, y_bst)


def test_variation_coefficient_has_valid_names() -> None:
    assert_feature_names_are_valid("variation_coefficient", VariationCoefficient)


def test_amplitude_shape_and_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    """No tsfresh equivalent (see feature_specs.py's `supports_tsfresh=False`) - shape only."""
    for y_bst in canonical_trajectories.case_tensors.values():
        result = Amplitude()(y_bst)
        expected_shape = (*y_bst.shape[:2], 1)
        assert result.shape == expected_shape, f"expected {expected_shape}, got {result.shape}"
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("amplitude", Amplitude, None, y_bst)


def test_amplitude_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("amplitude", Amplitude, None, y_bst)


def test_amplitude_has_valid_names() -> None:
    assert_feature_names_are_valid("amplitude", Amplitude)


# =============================================================================
# quantile
# =============================================================================

# 8 distinct quantiles: both endpoints, the four quartile-ish values, and two
# off-lattice values that don't land exactly on an index.
_QUANTILE_QS = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]


def test_quantile_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    params: list[QuantileParams] = [{"q": q} for q in _QUANTILE_QS]
    assert_feature_matches_tsfresh(Quantile, FEATURES["quantile"], canonical_trajectories, params)


def test_quantile_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph("quantile", Quantile, [{"q": 0.1}, {"q": 0.5}], y_bst)


def test_quantile_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("quantile", Quantile, [{"q": 0.1}, {"q": 0.5}], y_bst)


def test_quantile_has_valid_names() -> None:
    params: list[QuantileParams] = [{"q": q} for q in _QUANTILE_QS]
    assert_feature_names_are_valid("quantile", Quantile, params)


# =============================================================================
# mean_n_absolute_max
# =============================================================================

# 8 distinct counts, including 1 (degenerate top-1) and values approaching/at the
# shortest cached case's T (the `number_of_maxima >= T` NaN boundary).
_MEAN_N_ABSOLUTE_MAX_NS = [1, 2, 3, 5, 8, 13, 21, 34]


def test_mean_n_absolute_max_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    params: list[MeanNAbsoluteMaxParams] = [
        {"number_of_maxima": n} for n in _MEAN_N_ABSOLUTE_MAX_NS
    ]
    assert_feature_matches_tsfresh(
        MeanNAbsoluteMax,
        FEATURES["mean_n_absolute_max"],
        canonical_trajectories,
        params,
    )


def test_mean_n_absolute_max_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "mean_n_absolute_max",
        MeanNAbsoluteMax,
        [{"number_of_maxima": 1}, {"number_of_maxima": 5}],
        y_bst,
    )


def test_mean_n_absolute_max_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "mean_n_absolute_max",
        MeanNAbsoluteMax,
        [{"number_of_maxima": 1}, {"number_of_maxima": 5}],
        y_bst,
    )


def test_mean_n_absolute_max_has_valid_names() -> None:
    params: list[MeanNAbsoluteMaxParams] = [
        {"number_of_maxima": n} for n in _MEAN_N_ABSOLUTE_MAX_NS
    ]
    assert_feature_names_are_valid("mean_n_absolute_max", MeanNAbsoluteMax, params)


# =============================================================================
# ratio_beyond_r_sigma
# =============================================================================

# 8 distinct r's spanning well inside 1 sigma to well beyond 2 sigma.
_RATIO_BEYOND_R_SIGMA_RS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]


def test_ratio_beyond_r_sigma_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    params: list[RatioBeyondRSigmaParams] = [{"r": r} for r in _RATIO_BEYOND_R_SIGMA_RS]
    assert_feature_matches_tsfresh(
        RatioBeyondRSigma,
        FEATURES["ratio_beyond_r_sigma"],
        canonical_trajectories,
        params,
        # tstorch computes in float64 while tsfresh computes in the input's native
        # float32 precision - a boundary value very close to r * sigma can legitimately
        # land on opposite sides in the two precisions (one full 1/T count difference).
        # See migration-logs/statistical.md.
        atol=3e-3,
    )


def test_ratio_beyond_r_sigma_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "ratio_beyond_r_sigma", RatioBeyondRSigma, [{"r": 1.0}, {"r": 2.0}], y_bst
    )


def test_ratio_beyond_r_sigma_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "ratio_beyond_r_sigma", RatioBeyondRSigma, [{"r": 1.0}, {"r": 2.0}], y_bst
    )


def test_ratio_beyond_r_sigma_has_valid_names() -> None:
    params: list[RatioBeyondRSigmaParams] = [{"r": r} for r in _RATIO_BEYOND_R_SIGMA_RS]
    assert_feature_names_are_valid("ratio_beyond_r_sigma", RatioBeyondRSigma, params)


# =============================================================================
# symmetry_looking
# =============================================================================

# 8 distinct tolerances spanning tight to loose.
_SYMMETRY_LOOKING_RS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75]


def test_symmetry_looking_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    params: list[SymmetryLookingParams] = [{"r": r} for r in _SYMMETRY_LOOKING_RS]
    assert_feature_matches_tsfresh(
        SymmetryLooking, FEATURES["symmetry_looking"], canonical_trajectories, params
    )


def test_symmetry_looking_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "symmetry_looking", SymmetryLooking, [{"r": 0.1}, {"r": 0.5}], y_bst
    )


def test_symmetry_looking_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "symmetry_looking", SymmetryLooking, [{"r": 0.1}, {"r": 0.5}], y_bst
    )


def test_symmetry_looking_has_valid_names() -> None:
    params: list[SymmetryLookingParams] = [{"r": r} for r in _SYMMETRY_LOOKING_RS]
    assert_feature_names_are_valid("symmetry_looking", SymmetryLooking, params)


# =============================================================================
# empty/default params
# =============================================================================

_EMPTY_PARAMS_FEATURE_CLASSES = [
    Quantile,
    MeanNAbsoluteMax,
    RatioBeyondRSigma,
    SymmetryLooking,
]


@pytest.mark.parametrize("feature_cls", _EMPTY_PARAMS_FEATURE_CLASSES)
def test_rejects_empty_params(feature_cls: type) -> None:
    with pytest.raises(ValueError, match="requires at least one param"):
        feature_cls([])


@pytest.mark.parametrize("feature_cls", _EMPTY_PARAMS_FEATURE_CLASSES)
def test_omitted_params_uses_tsfresh_matching_default(
    feature_cls: type, canonical_trajectories: BenchmarkSeriesData
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    with_default = feature_cls()
    assert len(with_default.params) > 0
    assert with_default(y_bst).shape == (*y_bst.shape[:2], len(with_default.params))
