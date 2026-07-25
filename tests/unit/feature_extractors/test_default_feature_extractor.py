# pyright: basic
"""Tests for DefaultFeatureExtractor's fixed statistical feature set."""

import math

import pytest
import torch

from pybasin.feature_extractors.default_feature_extractor import DefaultFeatureExtractor
from pybasin.solution import Solution


def _make_solution(y: torch.Tensor) -> Solution:
    n_steps = y.shape[-1]
    time = torch.linspace(0, 1, n_steps)
    ic = y[:, :, 0]
    return Solution(initial_condition=ic, time=time, y=y.contiguous())


class TestDefaultFeatureExtractor:
    def test_feature_names(self) -> None:
        y = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(3, 2, 2)
        solution = _make_solution(y)

        extractor = DefaultFeatureExtractor(time_steady=0.0)
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

    def test_values_match_manual_computation(self) -> None:
        # Single batch, single state, odd-length series with a known peak.
        x = torch.tensor([1.0, -3.0, 2.0, 5.0, 0.0])
        y = x.view(1, 1, -1)
        solution = _make_solution(y)

        extractor = DefaultFeatureExtractor(time_steady=0.0)
        features = extractor.extract_features(solution)
        values = dict(zip(extractor.feature_names, features[0].tolist(), strict=True))

        expected_mean = x.mean().item()
        expected_median = 1.0  # sorted: [-3, 0, 1, 2, 5] -> middle element
        expected_std = x.std(correction=0).item()
        expected_var = x.var(correction=0).item()
        expected_rms = math.sqrt((x**2).mean().item())
        expected_delta = abs(x.max().item() - expected_mean)

        assert values["state_0__mean"] == pytest.approx(expected_mean)
        assert values["state_0__median"] == pytest.approx(expected_median)
        assert values["state_0__standard_deviation"] == pytest.approx(expected_std)
        assert values["state_0__variance"] == pytest.approx(expected_var)
        assert values["state_0__root_mean_square"] == pytest.approx(expected_rms)
        assert values["state_0__maximum"] == pytest.approx(5.0)
        assert values["state_0__absolute_maximum"] == pytest.approx(5.0)
        assert values["state_0__minimum"] == pytest.approx(-3.0)
        assert values["state_0__delta"] == pytest.approx(expected_delta)
        assert values["state_0__log_delta"] == pytest.approx(math.log(expected_delta + 1e-10))

    def test_median_matches_numpy_for_even_length(self) -> None:
        # numpy.median averages the two middle values; torch.median does not.
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        y = x.view(1, 1, -1)
        solution = _make_solution(y)

        extractor = DefaultFeatureExtractor(time_steady=0.0)
        features = extractor.extract_features(solution)

        assert features[0, extractor.feature_names.index("state_0__median")].item() == 2.5
