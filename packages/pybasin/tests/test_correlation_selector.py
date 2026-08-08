# pyright: basic
import numpy as np
import pytest
import torch
from sklearn.base import clone  # type: ignore[import-untyped]

from pybasin.feature_selector import CorrelationSelector


def _make_correlated_features() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    n_samples = 500
    base = torch.randn(n_samples, generator=generator)
    first = base + torch.randn(n_samples, generator=generator) * 0.01
    second = base + torch.randn(n_samples, generator=generator) * 0.01
    third = torch.randn(n_samples, generator=generator)
    fourth = torch.randn(n_samples, generator=generator)
    return torch.column_stack([first, second, third, fourth])


class TestParameters:
    def test_defaults(self):
        selector = CorrelationSelector()
        assert selector.threshold == 0.9
        assert selector.min_features == 3
        assert not hasattr(selector, "support_")

    def test_clone_roundtrip(self):
        selector = CorrelationSelector(threshold=0.8, min_features=2)
        cloned = clone(selector)
        assert cloned.threshold == 0.8
        assert cloned.min_features == 2


class TestSelection:
    def test_removes_one_of_correlated_pair(self):
        X = _make_correlated_features()
        selector = CorrelationSelector(threshold=0.95, min_features=2)
        result = selector.fit_transform(X)
        support = selector.get_support()

        assert result.shape[1] == 3
        assert support.sum() == 3
        assert support[2]
        assert support[3]

    def test_keeps_all_when_below_threshold(self):
        generator = torch.Generator().manual_seed(1)
        X = torch.randn(200, 4, generator=generator)
        result = CorrelationSelector(threshold=0.9, min_features=2).fit_transform(X)
        assert result.shape[1] == 4

    def test_min_features_respected(self):
        generator = torch.Generator().manual_seed(3)
        base = torch.randn(300, generator=generator)
        X = torch.column_stack([
            base + torch.randn(300, generator=generator) * 0.001,
            base + torch.randn(300, generator=generator) * 0.001,
            base + torch.randn(300, generator=generator) * 0.001,
        ])
        result = CorrelationSelector(threshold=0.5, min_features=2).fit_transform(X)
        assert result.shape[1] == 2

    def test_keeps_all_when_input_has_fewer_than_minimum(self):
        X = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        selector = CorrelationSelector(threshold=0.5, min_features=3)
        result = selector.fit_transform(X)
        assert result.shape[1] == 2
        assert selector.get_support().all()

    def test_transform_matches_support(self):
        X = _make_correlated_features()
        selector = CorrelationSelector(threshold=0.95, min_features=2)
        result = selector.fit_transform(X)
        expected = X[:, torch.from_numpy(selector.get_support())]
        torch.testing.assert_close(result, expected)

    def test_get_support_indices(self):
        selector = CorrelationSelector(threshold=0.95, min_features=2).fit(
            _make_correlated_features()
        )
        np.testing.assert_array_equal(
            np.flatnonzero(selector.get_support()), selector.get_support(indices=True)
        )

    def test_rejects_invalid_minimum(self):
        with pytest.raises(ValueError, match="at least 1"):
            CorrelationSelector(min_features=0).fit(_make_correlated_features())

    def test_requires_torch_tensor(self):
        with pytest.raises(TypeError, match="torch.Tensor"):
            CorrelationSelector().fit(_make_correlated_features().numpy())  # type: ignore[arg-type]


def test_cuda_tensor_stays_on_device():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")

    X = _make_correlated_features().cuda()
    result = CorrelationSelector(threshold=0.95, min_features=2).fit_transform(X)

    assert result.device.type == "cuda"
    assert result.dtype == X.dtype
