# pyright: reportPrivateUsage=false

import pytest
import torch

from pybasin.feature_extractors.feature_extractor import (
    FeatureExtractor as BaseFeatureExtractor,
)
from pybasin.feature_extractors.utils import to_snake_case
from pybasin.solution import Solution


class MeanFeatureExtractor(BaseFeatureExtractor):
    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered: torch.Tensor = self.filter_time(solution)
        features: torch.Tensor = y_filtered.mean(dim=-1)
        self._num_features = features.shape[1]
        return features


class CustomNamedExtractor(BaseFeatureExtractor):
    def __init__(self, time_steady: float = 0.0):
        super().__init__(time_steady=time_steady)
        self._feature_names = ["custom_1", "custom_2", "custom_3"]

    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered: torch.Tensor = self.filter_time(solution)
        features: torch.Tensor = y_filtered.mean(dim=-1)
        return torch.cat([features, torch.ones(features.shape[0], 1)], dim=1)


class SynchronizationFeatureExtractor(BaseFeatureExtractor):
    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered: torch.Tensor = self.filter_time(solution)
        features: torch.Tensor = y_filtered.mean(dim=-1)
        self._num_features = features.shape[1]
        return features


class FeatureExtractorTest(BaseFeatureExtractor):
    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered: torch.Tensor = self.filter_time(solution)
        features: torch.Tensor = y_filtered.mean(dim=-1)
        self._num_features = features.shape[1]
        return features


@pytest.fixture
def test_solution() -> Solution:
    """Create a test solution with standard dimensions."""
    n_steps: int = 100
    n_batch: int = 5
    n_states: int = 2
    ic: torch.Tensor = torch.randn(n_batch, n_states)
    time: torch.Tensor = torch.linspace(0, 10, n_steps)
    y: torch.Tensor = torch.randn(n_batch, n_states, n_steps)
    return Solution(initial_condition=ic, time=time, y=y)


@pytest.fixture
def test_solution_3_states() -> Solution:
    """Create a test solution with 3 state variables."""
    n_steps: int = 100
    n_batch: int = 5
    n_states: int = 3
    ic: torch.Tensor = torch.randn(n_batch, n_states)
    time: torch.Tensor = torch.linspace(0, 10, n_steps)
    y: torch.Tensor = torch.randn(n_batch, n_states, n_steps)
    return Solution(initial_condition=ic, time=time, y=y)


def test_feature_extractor_time_filtering(test_solution: Solution) -> None:
    time_steady: float = 5.0
    extractor: MeanFeatureExtractor = MeanFeatureExtractor(time_steady=time_steady)
    y_filtered: torch.Tensor = extractor.filter_time(test_solution)

    expected_steps: int = int((test_solution.time >= time_steady).sum().item())

    assert y_filtered.shape[-1] == expected_steps
    assert y_filtered.shape[:2] == (5, 2)


def test_feature_extractor_extract(test_solution: Solution) -> None:
    extractor: MeanFeatureExtractor = MeanFeatureExtractor(time_steady=0)
    features: torch.Tensor = extractor.extract_features(test_solution)

    assert features.shape == (5, 2)


def test_feature_names_default_generation(test_solution: Solution) -> None:
    extractor: MeanFeatureExtractor = MeanFeatureExtractor(time_steady=0)
    _features: torch.Tensor = extractor.extract_features(test_solution)

    feature_names: list[str] = extractor.feature_names
    assert feature_names == ["mean_1", "mean_2"]
    assert len(feature_names) == 2


def test_feature_names_removes_feature_extractor_suffix(test_solution_3_states: Solution) -> None:
    extractor: SynchronizationFeatureExtractor = SynchronizationFeatureExtractor(time_steady=0)
    _features: torch.Tensor = extractor.extract_features(test_solution_3_states)

    feature_names: list[str] = extractor.feature_names
    assert feature_names == ["synchronization_1", "synchronization_2", "synchronization_3"]
    assert len(feature_names) == 3


def test_feature_names_custom_override(test_solution: Solution) -> None:
    extractor: CustomNamedExtractor = CustomNamedExtractor(time_steady=0)
    _features: torch.Tensor = extractor.extract_features(test_solution)

    feature_names: list[str] = extractor.feature_names
    assert feature_names == ["custom_1", "custom_2", "custom_3"]
    assert len(feature_names) == 3


def test_feature_names_only_feature_extractor_class(test_solution: Solution) -> None:
    """Test class that would produce just 'feature' after stripping suffix."""
    extractor: FeatureExtractorTest = FeatureExtractorTest(time_steady=0)
    _features: torch.Tensor = extractor.extract_features(test_solution)

    feature_names: list[str] = extractor.feature_names
    assert feature_names == ["feature_extractor_test_1", "feature_extractor_test_2"]
    assert len(feature_names) == 2


def test_snake_case_conversion() -> None:
    assert to_snake_case("MeanFeatureExtractor") == "mean_feature_extractor"
    assert to_snake_case("SynchronizationFeatureExtractor") == "synchronization_feature_extractor"
    assert to_snake_case("FeatureExtractor") == "feature_extractor"
    assert to_snake_case("ABC") == "abc"
    assert to_snake_case("MyFEFeatureExtractor") == "my_fe_feature_extractor"
