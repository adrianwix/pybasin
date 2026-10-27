"""Numerical-accuracy and torch.compile(fullgraph=True)/torch.export safety tests for
`tstorch.calculators.torch_features_autocorrelation`. See the tstorch-feature-perf skill's
"Recording numerical accuracy" / "Recording torch.compile(fullgraph=True) safety" sections
for what each assertion helper records and why.
"""

from __future__ import annotations

import pytest
import torch
from tstorch.calculators.torch_features_autocorrelation import (
    AggAutocorrelation,
    AggAutocorrelationParams,
    Autocorrelation,
    AutocorrelationParams,
    AutocorrelationPeriodicity,
    AutocorrelationPeriodicityParams,
    PartialAutocorrelation,
    PartialAutocorrelationParams,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.feature_test_provider import construct_test_feature
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["autocorrelation"]}

# =============================================================================
# autocorrelation
# =============================================================================

# 8 distinct lags including 0 (degenerate-variance edge case, see
# migration-logs/autocorrelation.md) and several non-trivial values.
_AUTOCORRELATION_LAGS = [0, 1, 2, 3, 5, 8, 13, 21]


@pytest.mark.shared_provider
def test_autocorrelation_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    params: list[AutocorrelationParams] = [{"lag": lag} for lag in _AUTOCORRELATION_LAGS]
    assert_feature_matches_tsfresh(
        Autocorrelation, FEATURES["autocorrelation"], canonical_trajectories, params
    )


@pytest.mark.shared_provider
def test_autocorrelation_compiles_fullgraph(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "autocorrelation", Autocorrelation, [{"lag": 1}, {"lag": 5}], y_bst
    )


@pytest.mark.shared_provider
def test_autocorrelation_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "autocorrelation", Autocorrelation, [{"lag": 1}, {"lag": 5}], y_bst
    )


@pytest.mark.shared_provider
def test_autocorrelation_has_valid_names() -> None:
    params: list[AutocorrelationParams] = [{"lag": lag} for lag in _AUTOCORRELATION_LAGS]
    assert_feature_names_are_valid("autocorrelation", Autocorrelation, params)


# =============================================================================
# partial_autocorrelation
# =============================================================================

_PARTIAL_AUTOCORRELATION_LAGS = [0, 1, 2, 3, 4, 5, 6, 7]


@pytest.mark.shared_provider
def test_partial_autocorrelation_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    params: list[PartialAutocorrelationParams] = [
        {"lag": lag} for lag in _PARTIAL_AUTOCORRELATION_LAGS
    ]
    assert_feature_matches_tsfresh(
        PartialAutocorrelation,
        FEATURES["partial_autocorrelation"],
        canonical_trajectories,
        params,
    )


@pytest.mark.shared_provider
def test_partial_autocorrelation_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "partial_autocorrelation", PartialAutocorrelation, [{"lag": 1}, {"lag": 5}], y_bst
    )


@pytest.mark.shared_provider
def test_partial_autocorrelation_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "partial_autocorrelation", PartialAutocorrelation, [{"lag": 1}, {"lag": 5}], y_bst
    )


@pytest.mark.shared_provider
def test_partial_autocorrelation_has_valid_names() -> None:
    params: list[PartialAutocorrelationParams] = [
        {"lag": lag} for lag in _PARTIAL_AUTOCORRELATION_LAGS
    ]
    assert_feature_names_are_valid("partial_autocorrelation", PartialAutocorrelation, params)


# =============================================================================
# agg_autocorrelation
# =============================================================================

# 8 combos: every f_agg, several maxlags including 0 (degenerate early-return, see
# migration-logs/autocorrelation.md's "global_maxlag == 0" note).
_AGG_AUTOCORRELATION_PARAMS: list[AggAutocorrelationParams] = [
    {"f_agg": "mean", "maxlag": 0},
    {"f_agg": "mean", "maxlag": 1},
    {"f_agg": "mean", "maxlag": 5},
    {"f_agg": "mean", "maxlag": 20},
    {"f_agg": "median", "maxlag": 5},
    {"f_agg": "median", "maxlag": 20},
    {"f_agg": "var", "maxlag": 5},
    {"f_agg": "var", "maxlag": 20},
]


@pytest.mark.shared_provider
def test_agg_autocorrelation_matches_tsfresh(canonical_trajectories: BenchmarkSeriesData) -> None:
    params = _AGG_AUTOCORRELATION_PARAMS
    assert_feature_matches_tsfresh(
        AggAutocorrelation, FEATURES["agg_autocorrelation"], canonical_trajectories, params
    )


@pytest.mark.shared_provider
def test_agg_autocorrelation_direct_sum_matches_fft_path(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    """Cross-check direct-sum and FFT branches through the public Feature class.

    The first instance uses the direct-sum branch at its current upper boundary. The second
    includes one extra output column with ``maxlag + 1`` to force the FFT branch; the shared
    first column must match.
    """
    maxlag = 128
    for y_bst in canonical_trajectories.case_tensors.values():
        if y_bst.shape[2] <= maxlag + 1:
            continue  # feature clamps maxlag to T - 1; skip cases too short to straddle it
        params: list[AggAutocorrelationParams] = [{"maxlag": maxlag, "f_agg": "mean"}]
        forcing_fft_params: list[AggAutocorrelationParams] = [
            {"maxlag": maxlag, "f_agg": "mean"},
            {"maxlag": maxlag + 1, "f_agg": "mean"},
        ]

        direct = construct_test_feature(
            "agg_autocorrelation", params, lambda params=params: AggAutocorrelation(params)
        )(y_bst)
        fft = construct_test_feature(
            "agg_autocorrelation",
            forcing_fft_params,
            lambda forcing_fft_params=forcing_fft_params: AggAutocorrelation(forcing_fft_params),
        )(y_bst)[..., :1]

        torch.testing.assert_close(direct, fft, atol=1e-4, rtol=1e-4)


@pytest.mark.shared_provider
def test_agg_autocorrelation_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "agg_autocorrelation",
        AggAutocorrelation,
        [{"maxlag": 10, "f_agg": "mean"}, {"maxlag": 5, "f_agg": "var"}],
        y_bst,
    )


@pytest.mark.shared_provider
def test_agg_autocorrelation_exports_cleanly(canonical_trajectories: BenchmarkSeriesData) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "agg_autocorrelation",
        AggAutocorrelation,
        [{"maxlag": 10, "f_agg": "mean"}, {"maxlag": 5, "f_agg": "var"}],
        y_bst,
    )


@pytest.mark.shared_provider
def test_agg_autocorrelation_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "agg_autocorrelation", AggAutocorrelation, _AGG_AUTOCORRELATION_PARAMS
    )


# =============================================================================
# autocorrelation_periodicity - no tsfresh equivalent (FeatureSpec has
# supports_tsfresh=False), so no accuracy-vs-tsfresh test; instead a batching-consistency
# check (batched multi-param call vs. one call per param) plus compile/export safety.
# =============================================================================


@pytest.mark.shared_provider
def test_autocorrelation_periodicity_batches_consistently(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    params: list[AutocorrelationPeriodicityParams] = [
        {"min_lag": 2, "peak_threshold": 0.3, "output": "strength"},
        {"min_lag": 2, "peak_threshold": 0.3, "output": "period"},
        {"min_lag": 4, "peak_threshold": 0.1, "output": "strength"},
        {"min_lag": 4, "peak_threshold": 0.1, "output": "period"},
    ]
    for y_bst in canonical_trajectories.case_tensors.values():
        batched = construct_test_feature(
            "autocorrelation_periodicity", params, lambda: AutocorrelationPeriodicity(params)
        )(y_bst)
        expected_shape = (*y_bst.shape[:2], len(params))
        assert batched.shape == expected_shape, f"expected {expected_shape}, got {batched.shape}"

        separate = torch.cat(
            [
                construct_test_feature(
                    "autocorrelation_periodicity", [p], lambda p=p: AutocorrelationPeriodicity([p])
                )(y_bst)
                for p in params
            ],
            dim=-1,
        )
        assert torch.equal(batched, separate)


@pytest.mark.shared_provider
def test_autocorrelation_periodicity_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "autocorrelation_periodicity",
        AutocorrelationPeriodicity,
        [{"output": "strength"}, {"output": "period"}],
        y_bst,
    )


@pytest.mark.shared_provider
def test_autocorrelation_periodicity_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "autocorrelation_periodicity",
        AutocorrelationPeriodicity,
        [{"output": "strength"}, {"output": "period"}],
        y_bst,
    )


@pytest.mark.shared_provider
def test_autocorrelation_periodicity_has_valid_names() -> None:
    params: list[AutocorrelationPeriodicityParams] = [
        {"min_lag": 2, "peak_threshold": 0.3, "output": "strength"},
        {"min_lag": 2, "peak_threshold": 0.3, "output": "period"},
        {"min_lag": 4, "peak_threshold": 0.1, "output": "strength"},
        {"min_lag": 4, "peak_threshold": 0.1, "output": "period"},
    ]
    assert_feature_names_are_valid(
        "autocorrelation_periodicity", AutocorrelationPeriodicity, params
    )


# =============================================================================
# empty/default params
# =============================================================================

_EMPTY_PARAMS_FEATURE_CLASSES = [
    Autocorrelation,
    PartialAutocorrelation,
    AggAutocorrelation,
    AutocorrelationPeriodicity,
]


@pytest.mark.parametrize("feature_cls", _EMPTY_PARAMS_FEATURE_CLASSES)
@pytest.mark.native_only
def test_rejects_empty_params(feature_cls: type) -> None:
    with pytest.raises(ValueError, match="requires at least one param"):
        feature_cls([])


@pytest.mark.parametrize("feature_cls", _EMPTY_PARAMS_FEATURE_CLASSES)
@pytest.mark.native_only
def test_omitted_params_uses_tsfresh_matching_default(
    feature_cls: type, canonical_trajectories: BenchmarkSeriesData
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    with_default = feature_cls()
    assert len(with_default.params) > 0
    assert with_default(y_bst).shape == (*y_bst.shape[:2], len(with_default.params))
