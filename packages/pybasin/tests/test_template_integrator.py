# pyright: basic
import numpy as np
import pytest
import torch
from sklearn.feature_selection import VarianceThreshold  # type: ignore[import-untyped]

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.feature_selector import DefaultFeatureSelector, SklearnFeatureSelectorAdapter
from pybasin.solution import Solution
from pybasin.template_integrator import TemplateIntegrator


class _StaticFeatureExtractor(FeatureExtractor):
    """Feature extractor that ignores the solution and returns a fixed matrix."""

    def __init__(self, features: torch.Tensor):
        self._features = features

    def extract_features(self, solution: Solution) -> torch.Tensor:
        return self._features

    @property
    def feature_names(self) -> list[str]:
        return [f"f{i}" for i in range(self._features.shape[1])]


def _make_integrator_with_solution(
    features: torch.Tensor,
) -> tuple[TemplateIntegrator, _StaticFeatureExtractor]:
    integrator = TemplateIntegrator(
        template_y0=[[1.0]] * features.shape[0],
        labels=[f"label{i}" for i in range(features.shape[0])],
        ode_params={},
    )
    integrator.solution = Solution(
        initial_condition=torch.zeros(features.shape[0], 1),
        time=torch.zeros(1),
        y=torch.zeros(features.shape[0], 1, 1),
    )
    return integrator, _StaticFeatureExtractor(features)


def _make_features() -> np.ndarray:
    rng = np.random.default_rng(2)
    n = 30
    base = rng.standard_normal(n)
    X = np.column_stack([
        np.zeros(n),
        base,
        base + rng.standard_normal(n) * 0.01,
        rng.standard_normal(n),
    ]).astype(np.float32)
    return X


class TestGetTrainingDataFeatureSelectorDispatch:
    def test_device_native_selector_keeps_tensor_type(self):
        features = torch.from_numpy(_make_features())
        integrator, extractor = _make_integrator_with_solution(features)
        selector = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        selector.fit(features)

        X_train, y_train = integrator.get_training_data(extractor, feature_selector=selector)

        assert isinstance(X_train, torch.Tensor)
        assert X_train.device == features.device
        assert X_train.dtype == features.dtype
        assert y_train == integrator.labels

    def test_sklearn_selector_adapter(self):
        features = torch.from_numpy(_make_features())
        integrator, extractor = _make_integrator_with_solution(features)
        selector = SklearnFeatureSelectorAdapter(VarianceThreshold(threshold=0.01))
        selector.fit(features)

        X_train, _ = integrator.get_training_data(extractor, feature_selector=selector)

        assert isinstance(X_train, torch.Tensor)
        assert X_train.shape[1] == 3, "zero-variance feature dropped"

    def test_cuda_tensor_stays_on_device(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA GPU")

        features = torch.from_numpy(_make_features()).cuda()
        integrator, extractor = _make_integrator_with_solution(features)
        selector = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        selector.fit(features)

        X_train, _ = integrator.get_training_data(extractor, feature_selector=selector)

        assert isinstance(X_train, torch.Tensor)
        assert X_train.device.type == "cuda"
        assert X_train.dtype == torch.float32
