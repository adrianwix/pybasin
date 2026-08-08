# pyright: basic
import numpy as np
import pytest
import torch
from sklearn.base import clone  # type: ignore[import-untyped]

from pybasin.feature_selector import DefaultFeatureSelector


def _make_features() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    n_samples = 300
    base = torch.randn(n_samples, generator=generator)
    return torch.column_stack([
        torch.zeros(n_samples),
        base,
        base + torch.randn(n_samples, generator=generator) * 0.01,
        torch.randn(n_samples, generator=generator),
        torch.randn(n_samples, generator=generator),
    ])


class TestParameters:
    def test_defaults(self):
        selector = DefaultFeatureSelector()
        assert selector.variance_threshold == 0.01
        assert selector.correlation_threshold == 0.9
        assert selector.min_features == 3

    def test_clone_roundtrip(self):
        selector = DefaultFeatureSelector(variance_threshold=0.05)
        cloned = clone(selector)
        assert cloned.variance_threshold == 0.05

    def test_set_params_changes_fitted_stage(self):
        selector = DefaultFeatureSelector()
        selector.set_params(variance_threshold=0.5)
        selector.fit(_make_features())
        assert selector.variance_selector_.threshold == 0.5


class TestSelection:
    def test_two_stage_filtering(self):
        X = _make_features()
        selector = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        result = selector.fit_transform(X)
        support = selector.get_support()

        assert support.shape == (5,)
        assert not support[0]
        assert support[1] != support[2]
        assert support[3] and support[4]
        assert result.shape[1] == int(support.sum())

    def test_get_support_indices(self):
        selector = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        selector.fit(_make_features())
        np.testing.assert_array_equal(
            np.flatnonzero(selector.get_support()), selector.get_support(indices=True)
        )

    def test_transform_can_use_a_different_dtype(self):
        X = _make_features()
        selector = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2).fit(X)
        result = selector.transform(X.to(torch.float64))
        assert result.dtype == torch.float64


def test_cuda_tensor_stays_on_device():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")

    X = _make_features().cuda()
    result = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2).fit_transform(X)

    assert result.device.type == "cuda"
    assert result.dtype == X.dtype
