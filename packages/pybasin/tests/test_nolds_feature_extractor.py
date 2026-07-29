# pyright: basic
"""Tests for NoldsFeatureExtractor."""

from typing import Any

import numpy as np
import pytest
import torch

nolds = pytest.importorskip("nolds")

from pybasin.feature_extractors.nolds_feature_extractor import (  # noqa: E402
    NOLDS_DEFAULT_FC_PARAMETERS,
    NoldsFeatureExtractor,
    _get_nolds_feature_functions,
)
from pybasin.solution import Solution  # noqa: E402


@pytest.fixture
def random_walk_data() -> np.ndarray:
    """Generate random walk time series (float32 for consistency with torch)."""
    np.random.seed(42)
    n_points = 1000
    steps = np.random.randn(n_points)
    return np.cumsum(steps).astype(np.float32)


@pytest.fixture
def sine_wave_data() -> np.ndarray:
    """Generate simple sine wave time series (float32 for consistency with torch)."""
    t = np.linspace(0, 10 * np.pi, 1000, dtype=np.float32)
    return np.sin(t).astype(np.float32)


@pytest.fixture
def noisy_sine_data() -> np.ndarray:
    """Generate noisy sine wave time series (float32 for consistency with torch)."""
    np.random.seed(42)
    t = np.linspace(0, 10 * np.pi, 1000)
    return (np.sin(t) + 0.1 * np.random.randn(len(t))).astype(np.float32)


@pytest.fixture
def solution_from_random_walk(random_walk_data: np.ndarray) -> Solution:
    """Create a Solution object from random walk data."""
    n_steps = len(random_walk_data)
    n_batch = 1
    n_states = 1

    y = torch.tensor(random_walk_data, dtype=torch.float32).reshape(n_batch, n_states, n_steps)
    time = torch.linspace(0, 10, n_steps)
    ic = y[:, :, 0]

    return Solution(initial_condition=ic, time=time, y=y)


@pytest.fixture
def solution_multi_batch() -> Solution:
    """Create a Solution with multiple batches and states."""
    n_steps = 500
    n_batch = 3
    n_states = 2

    np.random.seed(42)
    y_np = np.cumsum(np.random.randn(n_batch, n_states, n_steps), axis=-1).astype(np.float32)

    y = torch.tensor(y_np)
    time = torch.linspace(0, 10, n_steps)
    ic = y[:, :, 0]

    return Solution(initial_condition=ic, time=time, y=y)


class TestNoldsFeatureFunctions:
    """Tests for the nolds feature function mapping."""

    def test_get_nolds_feature_functions_returns_all_features(self) -> None:
        funcs = _get_nolds_feature_functions()

        expected_features = [
            "lyap_r",
            "lyap_e",
            "sampen",
            "hurst_rs",
            "corr_dim",
            "dfa",
            "mfhurst_b",
            "mfhurst_dm",
        ]

        for feature in expected_features:
            assert feature in funcs, f"Missing feature: {feature}"

    def test_feature_functions_are_callable(self) -> None:
        funcs = _get_nolds_feature_functions()

        for name, func in funcs.items():
            assert callable(func), f"Feature {name} is not callable"


class TestNoldsExtractorMatchesDirectCalls:
    """Tests that extractor results match direct nolds function calls.

    Note:
        - lyap_r uses fit="poly" for determinism since default RANSAC is random.
        - time_steady=-1.0 ensures no time filtering is applied so data matches.
    """

    def test_lyap_r_matches_direct_call(self, random_walk_data: np.ndarray) -> None:
        params: dict[str, Any] = {"fit": "poly"}
        direct_result = nolds.lyap_r(random_walk_data, **params)

        n_steps = len(random_walk_data)
        y = torch.tensor(random_walk_data, dtype=torch.float32).reshape(1, 1, n_steps)
        time = torch.linspace(0, 10, n_steps)
        ic = y[:, :, 0]
        solution = Solution(initial_condition=ic, time=time, y=y)

        extractor = NoldsFeatureExtractor(
            time_steady=-1.0,
            features={"lyap_r": [params]},
            n_jobs=1,
        )
        features = extractor.extract_features(solution)

        assert features.shape == (1, 1)
        np.testing.assert_allclose(features[0, 0].item(), direct_result, rtol=1e-5)

    def test_lyap_r_with_params_matches_direct_call(self, random_walk_data: np.ndarray) -> None:
        params: dict[str, Any] = {"emb_dim": 5, "lag": 2, "fit": "poly"}
        direct_result = nolds.lyap_r(random_walk_data, **params)

        n_steps = len(random_walk_data)
        y = torch.tensor(random_walk_data, dtype=torch.float32).reshape(1, 1, n_steps)
        time = torch.linspace(0, 10, n_steps)
        ic = y[:, :, 0]
        solution = Solution(initial_condition=ic, time=time, y=y)

        extractor = NoldsFeatureExtractor(
            time_steady=-1.0,
            features={"lyap_r": [params]},
            n_jobs=1,
        )
        features = extractor.extract_features(solution)

        assert features.shape == (1, 1)
        np.testing.assert_allclose(features[0, 0].item(), direct_result, rtol=1e-5)

    def test_corr_dim_matches_direct_call(self, noisy_sine_data: np.ndarray) -> None:
        params: dict[str, Any] = {"emb_dim": 5}
        direct_result = nolds.corr_dim(noisy_sine_data, **params)

        n_steps = len(noisy_sine_data)
        y = torch.tensor(noisy_sine_data, dtype=torch.float32).reshape(1, 1, n_steps)
        time = torch.linspace(0, 10, n_steps)
        ic = y[:, :, 0]
        solution = Solution(initial_condition=ic, time=time, y=y)

        extractor = NoldsFeatureExtractor(
            time_steady=-1.0,
            features={"corr_dim": [params]},
            n_jobs=1,
        )
        features = extractor.extract_features(solution)

        assert features.shape == (1, 1)
        np.testing.assert_allclose(features[0, 0].item(), direct_result, rtol=1e-5)

    def test_sampen_matches_direct_call(self, sine_wave_data: np.ndarray) -> None:
        direct_result = nolds.sampen(sine_wave_data)

        n_steps = len(sine_wave_data)
        y = torch.tensor(sine_wave_data, dtype=torch.float32).reshape(1, 1, n_steps)
        time = torch.linspace(0, 10, n_steps)
        ic = y[:, :, 0]
        solution = Solution(initial_condition=ic, time=time, y=y)

        extractor = NoldsFeatureExtractor(
            time_steady=-1.0,
            features={"sampen": None},
            n_jobs=1,
        )
        features = extractor.extract_features(solution)

        assert features.shape == (1, 1)
        np.testing.assert_allclose(features[0, 0].item(), direct_result, rtol=1e-5)

    def test_hurst_rs_matches_direct_call(self, random_walk_data: np.ndarray) -> None:
        direct_result = nolds.hurst_rs(random_walk_data)

        n_steps = len(random_walk_data)
        y = torch.tensor(random_walk_data, dtype=torch.float32).reshape(1, 1, n_steps)
        time = torch.linspace(0, 10, n_steps)
        ic = y[:, :, 0]
        solution = Solution(initial_condition=ic, time=time, y=y)

        extractor = NoldsFeatureExtractor(
            time_steady=-1.0,
            features={"hurst_rs": None},
            n_jobs=1,
        )
        features = extractor.extract_features(solution)

        assert features.shape == (1, 1)
        np.testing.assert_allclose(features[0, 0].item(), direct_result, rtol=1e-5)

    def test_dfa_matches_direct_call(self, random_walk_data: np.ndarray) -> None:
        direct_result = nolds.dfa(random_walk_data)

        n_steps = len(random_walk_data)
        y = torch.tensor(random_walk_data, dtype=torch.float32).reshape(1, 1, n_steps)
        time = torch.linspace(0, 10, n_steps)
        ic = y[:, :, 0]
        solution = Solution(initial_condition=ic, time=time, y=y)

        extractor = NoldsFeatureExtractor(
            time_steady=-1.0,
            features={"dfa": None},
            n_jobs=1,
        )
        features = extractor.extract_features(solution)

        assert features.shape == (1, 1)
        np.testing.assert_allclose(features[0, 0].item(), direct_result, rtol=1e-5)


class TestNoldsExtractorConfiguration:
    """Tests for extractor configuration options."""

    def test_default_features(self, solution_from_random_walk: Solution) -> None:
        extractor = NoldsFeatureExtractor(time_steady=0.0, n_jobs=1)
        features = extractor.extract_features(solution_from_random_walk)

        assert "lyap_r" in NOLDS_DEFAULT_FC_PARAMETERS
        assert "corr_dim" in NOLDS_DEFAULT_FC_PARAMETERS
        assert features.shape[1] == 2

    def test_features_per_state(self, solution_multi_batch: Solution) -> None:
        extractor = NoldsFeatureExtractor(
            time_steady=0.0,
            features=None,
            features_per_state={
                0: {"lyap_r": None},
                1: {"hurst_rs": None},
            },
            n_jobs=1,
        )
        features = extractor.extract_features(solution_multi_batch)

        assert features.shape == (3, 2)
        assert "state_0__lyap_r" in extractor.feature_names
        assert "state_1__hurst_rs" in extractor.feature_names

    def test_multiple_params_same_feature(self, solution_from_random_walk: Solution) -> None:
        extractor = NoldsFeatureExtractor(
            time_steady=0.0,
            features={
                "lyap_r": [
                    {"emb_dim": 5},
                    {"emb_dim": 10},
                ],
            },
            n_jobs=1,
        )
        features = extractor.extract_features(solution_from_random_walk)

        assert features.shape[1] == 2
        assert "state_0__lyap_r__emb_dim=5" in extractor.feature_names
        assert "state_0__lyap_r__emb_dim=10" in extractor.feature_names

    def test_skip_state_with_none(self, solution_multi_batch: Solution) -> None:
        extractor = NoldsFeatureExtractor(
            time_steady=0.0,
            features={"lyap_r": None},
            features_per_state={
                1: None,
            },
            n_jobs=1,
        )
        features = extractor.extract_features(solution_multi_batch)

        assert features.shape == (3, 1)
        assert "state_0__lyap_r" in extractor.feature_names
        assert "state_1__lyap_r" not in extractor.feature_names

    def test_time_steady_filtering(self) -> None:
        n_steps = 100
        y = torch.arange(n_steps, dtype=torch.float32).reshape(1, 1, n_steps)
        time = torch.linspace(0, 10, n_steps)
        ic = y[:, :, 0]
        solution = Solution(initial_condition=ic, time=time, y=y)

        extractor = NoldsFeatureExtractor(
            time_steady=5.0,
            features={"hurst_rs": None},
            n_jobs=1,
        )
        features = extractor.extract_features(solution)

        assert features.shape == (1, 1)


class TestNoldsExtractorFeatureNames:
    """Tests for feature name generation."""

    def test_feature_names_before_extraction_raises(self) -> None:
        extractor = NoldsFeatureExtractor(time_steady=0.0)

        with pytest.raises(RuntimeError, match="Feature names not available"):
            _ = extractor.feature_names

    def test_feature_names_format(self, solution_from_random_walk: Solution) -> None:
        extractor = NoldsFeatureExtractor(
            time_steady=0.0,
            features={"lyap_r": [{"emb_dim": 10}]},
            n_jobs=1,
        )
        extractor.extract_features(solution_from_random_walk)

        names = extractor.feature_names
        assert len(names) == 1
        assert names[0] == "state_0__lyap_r__emb_dim=10"

    def test_feature_names_no_params(self, solution_from_random_walk: Solution) -> None:
        extractor = NoldsFeatureExtractor(
            time_steady=0.0,
            features={"lyap_r": None},
            n_jobs=1,
        )
        extractor.extract_features(solution_from_random_walk)

        names = extractor.feature_names
        assert len(names) == 1
        assert names[0] == "state_0__lyap_r"


class TestNoldsExtractorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_feature_name_raises(self, solution_from_random_walk: Solution) -> None:
        extractor = NoldsFeatureExtractor(
            time_steady=0.0,
            features={"invalid_feature": None},
            n_jobs=1,
        )

        with pytest.raises(ValueError, match="Unknown feature"):
            extractor.extract_features(solution_from_random_walk)

    def test_empty_features_returns_empty_tensor(self, solution_from_random_walk: Solution) -> None:
        extractor = NoldsFeatureExtractor(
            time_steady=0.0,
            features=None,
            features_per_state={},
            n_jobs=1,
        )

        extractor.features = None

        features = extractor.extract_features(solution_from_random_walk)
        assert features.shape == (1, 0)

    def test_handles_nan_from_failed_computation(self) -> None:
        y = torch.zeros(1, 1, 10)
        time = torch.linspace(0, 1, 10)
        ic = y[:, :, 0]
        solution = Solution(initial_condition=ic, time=time, y=y)

        extractor = NoldsFeatureExtractor(
            time_steady=0.0,
            features={"corr_dim": [{"emb_dim": 5}]},
            n_jobs=1,
        )
        features = extractor.extract_features(solution)

        assert features.shape == (1, 1)
        assert torch.isfinite(features).all()
