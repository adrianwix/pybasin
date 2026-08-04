# pyright: basic
"""Tests for `TorchFeatureExtractorAdapter`.

`tstorch.TorchFeatureExtractor` only returns raw features; this adapter composes that raw
extraction with `tstorch.preprocessing` imputation and an optional standalone
`StandardScaler`, driven by `impute_method` and `normalize`.
"""

import pytest
import torch

from pybasin.feature_extractors.tstorch_feature_extractor import TorchFeatureExtractorAdapter
from pybasin.solution import Solution


def _make_solution(y: torch.Tensor) -> Solution:
    n_steps = y.shape[-1]
    time = torch.linspace(0, 1, n_steps)
    ic = y[:, :, 0]
    return Solution(initial_condition=ic, time=time, y=y.contiguous())


class TestTorchFeatureExtractorAdapter:
    def test_feature_names_forwarded(self) -> None:
        y = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(3, 2, 2)
        solution = _make_solution(y)

        extractor = TorchFeatureExtractorAdapter(
            time_steady=0.0, normalize=False, use_compile=False
        )
        features = extractor.extract_features(solution)

        assert features.shape == (3, 20)
        assert extractor.feature_names == [
            f"state_{state_idx}__{name}"
            for state_idx in range(2)
            for name in (
                "median",
                "mean",
                "standard_deviation",
                "variance",
                "root_mean_square",
                "maximum",
                "absolute_maximum",
                "minimum",
                "delta",
                "log_delta",
            )
        ]

    def test_normalize_false_returns_raw_statistics(self) -> None:
        x = torch.tensor([1.0, -3.0, 2.0, 5.0, 0.0])
        y = x.view(1, 1, -1)
        solution = _make_solution(y)

        extractor = TorchFeatureExtractorAdapter(
            time_steady=0.0, normalize=False, use_compile=False
        )
        features = extractor.extract_features(solution)
        values = dict(zip(extractor.feature_names, features[0].tolist(), strict=True))

        assert values["state_0__mean"] == pytest.approx(x.mean().item())

    def test_normalize_default_zscores_features(self) -> None:
        y = torch.randn(50, 1, 20)
        solution = _make_solution(y)

        extractor = TorchFeatureExtractorAdapter(time_steady=0.0, use_compile=False)
        features = extractor.extract_features(solution)

        assert torch.allclose(features.mean(dim=0), torch.zeros(features.shape[1]), atol=1e-5)
        assert torch.allclose(
            features.std(dim=0, correction=0), torch.ones(features.shape[1]), atol=1e-5
        )

    def test_normalize_freezes_stats_across_calls(self) -> None:
        y1 = torch.randn(50, 1, 20)
        y2 = torch.randn(5, 1, 20) * 100 + 1000

        extractor = TorchFeatureExtractorAdapter(time_steady=0.0, use_compile=False)
        assert extractor._scaler is not None
        extractor.extract_features(_make_solution(y1))
        mean_after_first_call = extractor._scaler.state.mean.clone()

        extractor.extract_features(_make_solution(y2))

        assert torch.equal(extractor._scaler.state.mean, mean_after_first_call)

    def test_impute_extreme_replaces_non_finite(self) -> None:
        y = torch.zeros(1, 1, 5)  # constant series -> zero variance features
        solution = _make_solution(y)

        extractor = TorchFeatureExtractorAdapter(
            time_steady=0.0, normalize=False, impute_method="extreme", use_compile=False
        )
        features = extractor.extract_features(solution)

        assert torch.isfinite(features).all()

    def test_impute_statistical_replaces_non_finite(self) -> None:
        y = torch.zeros(1, 1, 5)
        solution = _make_solution(y)

        extractor = TorchFeatureExtractorAdapter(
            time_steady=0.0, normalize=False, impute_method="statistical", use_compile=False
        )
        features = extractor.extract_features(solution)

        assert torch.isfinite(features).all()

    def test_device_and_return_type(self) -> None:
        y = torch.randn(4, 1, 10)
        solution = _make_solution(y)

        extractor = TorchFeatureExtractorAdapter(time_steady=0.0, use_compile=False)
        features = extractor.extract_features(solution)

        assert isinstance(features, torch.Tensor)
        assert features.device == y.device
