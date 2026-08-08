"""Feature selector for removing highly correlated features.

The selector uses the mean absolute correlation ranking from Kuhn and Johnson
(2013), "Applied Predictive Modeling", Chapter 3.
"""

from typing import Any, Self

import torch

from pybasin.feature_selector.base import FeatureSelector


class CorrelationSelector(FeatureSelector):
    """Remove features with excessive pairwise absolute correlation.

    When a pair exceeds ``threshold``, the feature with the higher mean
    absolute correlation against all remaining features is removed. Mean
    correlations are recalculated over the remaining features after each
    removal.

    :ivar threshold: Correlation threshold that triggers a removal decision.
    :ivar min_features: Minimum number of features retained.
    """

    def __init__(self, threshold: float = 0.9, min_features: int = 3):
        """Initialize the selector.

        :param threshold: Absolute correlation threshold that triggers removal.
        :param min_features: Minimum number of features to retain.
        """
        self.threshold: float = threshold
        self.min_features: int = min_features

    def fit(self, X: torch.Tensor, y: Any = None) -> Self:
        """Compute which features to retain.

        :param X: Training tensor of shape ``(n_samples, n_features)``.
        :param y: Ignored.
        :return: Fitted selector.
        :raises ValueError: If ``threshold`` is negative or ``min_features`` is
            less than one.
        """
        self._prepare_fit(X)
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative.")
        if self.min_features < 1:
            raise ValueError("min_features must be at least 1.")

        if self.n_features_in_ <= self.min_features:
            self._set_support(torch.ones(self.n_features_in_, dtype=torch.bool, device=X.device))
            return self

        correlation = torch.corrcoef(X.T).abs()
        correlation.fill_diagonal_(0.0)
        remaining = list(range(self.n_features_in_))

        while len(remaining) > self.min_features:
            indices = torch.tensor(remaining, device=X.device)
            reduced_correlation = correlation[indices][:, indices]
            max_correlation = float(torch.max(reduced_correlation))
            if max_correlation <= self.threshold:
                break

            first, second = divmod(int(torch.argmax(reduced_correlation)), len(remaining))
            first_mean = float(torch.mean(reduced_correlation[first]))
            second_mean = float(torch.mean(reduced_correlation[second]))
            drop = first if first_mean >= second_mean else second
            remaining.pop(drop)

        support = torch.zeros(self.n_features_in_, dtype=torch.bool, device=X.device)
        support[remaining] = True
        self._set_support(support)
        return self
