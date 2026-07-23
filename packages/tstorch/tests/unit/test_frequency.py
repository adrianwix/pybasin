"""Accuracy, naming, compilation, and export tests for frequency features."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from tstorch.calculators.torch_features_frequency import (
    CwtCoefficients,
    CwtCoefficientsParams,
    FftAggregated,
    FftAggregatedParams,
    FftCoefficient,
    FftCoefficientParams,
    SpectralFrequencyRatio,
    SpktWelchDensity,
    SpktWelchDensityParams,
)

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
    assert_feature_matches_tsfresh,
    assert_feature_names_are_valid,
)

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["frequency"]}

_FFT_COEFFICIENT_PARAMS: list[FftCoefficientParams] = [
    {"coeff": 0, "attr": "real"},
    {"coeff": 1, "attr": "imag"},
    {"coeff": 2, "attr": "abs"},
    {"coeff": 3, "attr": "angle"},
    {"coeff": 8, "attr": "real"},
    {"coeff": 13, "attr": "imag"},
    {"coeff": 21, "attr": "abs"},
    {"coeff": 10_000, "attr": "angle"},
]
_FFT_AGGREGATED_PARAMS: list[FftAggregatedParams] = [
    {"aggtype": "centroid"},
    {"aggtype": "variance"},
    {"aggtype": "skew"},
    {"aggtype": "kurtosis"},
]
_SPKT_WELCH_DENSITY_PARAMS: list[SpktWelchDensityParams] = [
    {"coeff": 0},
    {"coeff": 1},
    {"coeff": 2},
    {"coeff": 5},
    {"coeff": 8},
    {"coeff": 13},
    {"coeff": 21},
    {"coeff": 10_000},
]
_CWT_COEFFICIENTS_PARAMS: list[CwtCoefficientsParams] = [
    {"widths": (1, 2), "coeff": 0, "w": 1},
    {"widths": (1, 2), "coeff": 1, "w": 2},
    {"widths": (2, 5), "coeff": 2, "w": 2},
    {"widths": (2, 5), "coeff": 5, "w": 5},
    {"widths": (5, 10), "coeff": 8, "w": 5},
    {"widths": (5, 10), "coeff": 13, "w": 10},
    {"widths": (10, 20), "coeff": 21, "w": 20},
    {"widths": (2,), "coeff": 10_000, "w": 2},
]


def test_fft_coefficient_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        FftCoefficient,
        FEATURES["fft_coefficient"],
        canonical_trajectories,
        _FFT_COEFFICIENT_PARAMS,
        atol=0.5,
        rtol=1e-3,
    )


def test_fft_coefficient_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "fft_coefficient", FftCoefficient, _FFT_COEFFICIENT_PARAMS[:4], y_bst
    )


def test_fft_coefficient_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "fft_coefficient", FftCoefficient, _FFT_COEFFICIENT_PARAMS[:4], y_bst
    )


def test_fft_coefficient_has_valid_names() -> None:
    assert_feature_names_are_valid("fft_coefficient", FftCoefficient, _FFT_COEFFICIENT_PARAMS)


def test_fft_coefficient_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        FftCoefficient([])


def test_fft_aggregated_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        FftAggregated,
        FEATURES["fft_aggregated"],
        canonical_trajectories,
        _FFT_AGGREGATED_PARAMS,
        rtol=2e-3,
    )


def test_fft_aggregated_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "fft_aggregated", FftAggregated, _FFT_AGGREGATED_PARAMS, y_bst
    )


def test_fft_aggregated_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("fft_aggregated", FftAggregated, _FFT_AGGREGATED_PARAMS, y_bst)


def test_fft_aggregated_has_valid_names() -> None:
    assert_feature_names_are_valid("fft_aggregated", FftAggregated, _FFT_AGGREGATED_PARAMS)


def test_fft_aggregated_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        FftAggregated([])


def test_spkt_welch_density_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        SpktWelchDensity,
        FEATURES["spkt_welch_density"],
        canonical_trajectories,
        _SPKT_WELCH_DENSITY_PARAMS,
    )


def test_spkt_welch_density_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "spkt_welch_density",
        SpktWelchDensity,
        _SPKT_WELCH_DENSITY_PARAMS[:4],
        y_bst,
    )


def test_spkt_welch_density_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "spkt_welch_density",
        SpktWelchDensity,
        _SPKT_WELCH_DENSITY_PARAMS[:4],
        y_bst,
    )


def test_spkt_welch_density_has_valid_names() -> None:
    assert_feature_names_are_valid(
        "spkt_welch_density", SpktWelchDensity, _SPKT_WELCH_DENSITY_PARAMS
    )


def test_spkt_welch_density_rejects_empty_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        SpktWelchDensity([])


def test_cwt_coefficients_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(
        CwtCoefficients,
        FEATURES["cwt_coefficients"],
        canonical_trajectories,
        _CWT_COEFFICIENTS_PARAMS,
        atol=1e-5,
        rtol=1e-3,
    )


def test_cwt_coefficients_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "cwt_coefficients",
        CwtCoefficients,
        _CWT_COEFFICIENTS_PARAMS[:4],
        y_bst,
    )


def test_cwt_coefficients_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "cwt_coefficients",
        CwtCoefficients,
        _CWT_COEFFICIENTS_PARAMS[:4],
        y_bst,
    )


def test_cwt_coefficients_has_valid_names() -> None:
    assert_feature_names_are_valid("cwt_coefficients", CwtCoefficients, _CWT_COEFFICIENTS_PARAMS)


def test_cwt_coefficients_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="at least one param"):
        CwtCoefficients([])
    with pytest.raises(ValueError, match="requires each `w`"):
        CwtCoefficients([{"widths": (2, 5), "coeff": 1, "w": 10}])


def test_spectral_frequency_ratio_compiles_fullgraph(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "spectral_frequency_ratio", SpectralFrequencyRatio, None, y_bst
    )


def test_spectral_frequency_ratio_exports_cleanly(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly("spectral_frequency_ratio", SpectralFrequencyRatio, None, y_bst)


def test_spectral_frequency_ratio_has_valid_names() -> None:
    assert_feature_names_are_valid("spectral_frequency_ratio", SpectralFrequencyRatio)


def test_spectral_frequency_ratio_semantics() -> None:
    time = torch.arange(120, dtype=torch.float64)
    period_one = torch.sin(2 * torch.pi * time / 20)
    harmonic = 0.5 * torch.sin(2 * torch.pi * time / 10)
    x = (period_one + harmonic).view(1, 1, -1)

    result = SpectralFrequencyRatio()(x)

    torch.testing.assert_close(result, torch.tensor([[[2.0]]], dtype=x.dtype))
    assert torch.equal(
        SpectralFrequencyRatio()(torch.ones(2, 3, 32)),
        torch.zeros(2, 3, 1),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_frequency_features_match_cpu_on_gpu(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    y_cpu = next(iter(canonical_trajectories.case_tensors.values()))
    features = (
        FftCoefficient(_FFT_COEFFICIENT_PARAMS),
        FftAggregated(_FFT_AGGREGATED_PARAMS),
        SpktWelchDensity(_SPKT_WELCH_DENSITY_PARAMS),
        CwtCoefficients(_CWT_COEFFICIENTS_PARAMS),
        SpectralFrequencyRatio(),
    )
    for feature in features:
        cpu_result: Tensor = feature(y_cpu)
        gpu_result: Tensor = feature(y_cpu.cuda()).cpu()
        atol = 0.5 if isinstance(feature, FftCoefficient) else 1e-5
        rtol = 1e-3 if isinstance(feature, FftCoefficient) else 1e-5
        # Spectral projections reduce long products in a backend-dependent order.
        if isinstance(feature, (SpktWelchDensity, CwtCoefficients)):
            atol = 1e-3
            rtol = 1e-3
        torch.testing.assert_close(gpu_result, cpu_result, atol=atol, rtol=rtol, equal_nan=True)
