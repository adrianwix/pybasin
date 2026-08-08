"""Adapter for sklearn feature selectors."""

from typing import Any, Self

import numpy as np
import torch

from pybasin.feature_selector.base import FeatureSelector
from pybasin.protocols import SklearnFeatureSelectorProtocol


class SklearnFeatureSelectorAdapter(FeatureSelector):
    """Adapt an sklearn feature selector to the Torch selector contract.

    Fitting transfers the feature matrix to CPU and lets the wrapped selector
    compute its support mask. Transformations use that mask directly in Torch,
    including when the input tensor is on a GPU.

    :ivar selector: Wrapped sklearn selector with ``fit()`` and ``get_support()``.
    """

    def __init__(self, selector: SklearnFeatureSelectorProtocol):
        """Initialize the adapter.

        :param selector: Sklearn selector to fit on a NumPy array.
        """
        self.selector = selector

    def fit(self, X: torch.Tensor, y: Any = None) -> Self:
        """Fit the wrapped selector and store its selected-column mask.

        :param X: Training tensor of shape ``(n_samples, n_features)``.
        :param y: Optional targets. Torch targets are transferred to CPU before
            they are forwarded to the wrapped selector.
        :return: Fitted adapter.
        """
        self._prepare_fit(X)
        X_numpy = X.detach().cpu().numpy()
        y_numpy = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
        self.selector.fit(X_numpy, y_numpy)
        support = np.asarray(self.selector.get_support(indices=False), dtype=bool)
        self._set_support(support)
        return self
