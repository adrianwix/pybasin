# pyright: basic
import numpy as np
import pytest
import torch
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.feature_selection import VarianceThreshold

from pybasin.feature_selector import VarianceThresholdSelector


def _make_features() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    n_samples = 200
    return torch.column_stack([
        torch.zeros(n_samples),
        torch.randn(n_samples, generator=generator) * 0.05,
        torch.randn(n_samples, generator=generator),
        torch.randn(n_samples, generator=generator) * 10,
    ])


class TestParameters:
    def test_defaults(self):
        selector = VarianceThresholdSelector()
        assert selector.threshold == 0.01
        assert not hasattr(selector, "support_")

    def test_clone_roundtrip(self):
        selector = VarianceThresholdSelector(threshold=0.05)
        cloned = clone(selector)
        assert cloned.threshold == 0.05


class TestSelection:
    def test_support_matches_sklearn_for_finite_dense_input(self):
        X = _make_features()
        selector = VarianceThresholdSelector(threshold=0.01).fit(X)
        reference = VarianceThreshold(threshold=0.01).fit(X.numpy())

        np.testing.assert_array_equal(selector.get_support(), reference.get_support())

    def test_transform_matches_support(self):
        X = _make_features()
        selector = VarianceThresholdSelector(threshold=0.01)
        result = selector.fit_transform(X)

        expected = X[:, torch.from_numpy(selector.get_support())]
        torch.testing.assert_close(result, expected)
        assert result.device == X.device
        assert result.dtype == X.dtype

    def test_stores_population_variances(self):
        X = _make_features()
        selector = VarianceThresholdSelector().fit(X)
        torch.testing.assert_close(selector.variances_, torch.var(X, dim=0, correction=0))

    def test_raises_when_no_feature_survives(self):
        with pytest.raises(ValueError, match="variance threshold"):
            VarianceThresholdSelector(threshold=0.01).fit(torch.zeros(50, 3))

    def test_rejects_negative_threshold(self):
        with pytest.raises(ValueError, match="non-negative"):
            VarianceThresholdSelector(threshold=-0.1).fit(_make_features())

    def test_get_support_indices(self):
        selector = VarianceThresholdSelector().fit(_make_features())
        np.testing.assert_array_equal(
            np.flatnonzero(selector.get_support()), selector.get_support(indices=True)
        )

    def test_requires_torch_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            VarianceThresholdSelector().fit(_make_features().numpy())  # type: ignore[arg-type]


def test_cuda_tensor_stays_on_device():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")

    X = _make_features().cuda()
    result = VarianceThresholdSelector().fit_transform(X)

    assert result.device.type == "cuda"
    assert result.dtype == X.dtype
