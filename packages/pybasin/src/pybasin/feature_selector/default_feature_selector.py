"""Default feature selector."""

from typing import Any, Self

import numpy as np
import torch

from pybasin.feature_selector.base import FeatureSelector
from pybasin.feature_selector.correlation_selector import CorrelationSelector
from pybasin.feature_selector.variance_threshold_selector import VarianceThresholdSelector


class DefaultFeatureSelector(FeatureSelector):
    """Apply variance thresholding followed by correlation filtering.

    :ivar variance_threshold: Minimum population variance required to keep a
        feature.
    :ivar correlation_threshold: Absolute correlation threshold that triggers
        feature removal.
    :ivar min_features: Minimum number of features retained by correlation
        filtering.
    :ivar variance_selector_: Fitted variance stage.
    :ivar correlation_selector_: Fitted correlation stage.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.9,
        min_features: int = 3,
    ):
        """Initialize the selector.

        :param variance_threshold: Minimum population variance required to keep
            a feature.
        :param correlation_threshold: Absolute correlation threshold that
            triggers feature removal.
        :param min_features: Minimum number of features retained by correlation
            filtering.
        """
        self.variance_threshold: float = variance_threshold
        self.correlation_threshold: float = correlation_threshold
        self.min_features: int = min_features

    def fit(self, X: torch.Tensor, y: Any = None) -> Self:
        """Fit the variance and correlation stages.

        :param X: Training tensor of shape ``(n_samples, n_features)``.
        :param y: Ignored.
        :return: Fitted selector.
        """
        self._prepare_fit(X)
        self.variance_selector_ = VarianceThresholdSelector(self.variance_threshold)
        variance_filtered = self.variance_selector_.fit_transform(X)

        self.correlation_selector_ = CorrelationSelector(
            threshold=self.correlation_threshold,
            min_features=self.min_features,
        )
        self.correlation_selector_.fit(variance_filtered)

        variance_support = self.variance_selector_.get_support()
        correlation_support = self.correlation_selector_.get_support()
        variance_indices = np.flatnonzero(variance_support)
        combined_support = np.zeros(self.n_features_in_, dtype=bool)
        combined_support[variance_indices[correlation_support]] = True
        self._set_support(combined_support)
        return self
