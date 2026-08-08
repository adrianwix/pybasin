# pyright: basic
import numpy as np
import torch
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.feature_selection import VarianceThreshold

from pybasin.feature_selector import SklearnFeatureSelectorAdapter


def _make_features() -> torch.Tensor:
    return torch.tensor([
        [0.0, 1.0, 1.0],
        [0.0, 2.0, 1.0],
        [0.0, 3.0, 1.0],
    ])


def test_adapter_fits_sklearn_selector_and_returns_tensor():
    X = _make_features()
    adapter = SklearnFeatureSelectorAdapter(VarianceThreshold())
    result = adapter.fit_transform(X)

    assert isinstance(result, torch.Tensor)
    assert result.device == X.device
    assert result.dtype == X.dtype
    torch.testing.assert_close(result, X[:, 1:2])
    np.testing.assert_array_equal(adapter.get_support(), [False, True, False])


def test_transform_uses_torch_mask_after_fit():
    adapter = SklearnFeatureSelectorAdapter(VarianceThreshold()).fit(_make_features())
    transformed = adapter.transform(_make_features().to(torch.float64))
    assert transformed.dtype == torch.float64


def test_clone_roundtrip():
    adapter = SklearnFeatureSelectorAdapter(VarianceThreshold(threshold=0.5))
    cloned = clone(adapter)
    assert isinstance(cloned.selector, VarianceThreshold)
    assert cloned.selector.get_params()["threshold"] == 0.5
