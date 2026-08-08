"""Feature selector for removing low-variance features."""

from typing import Any, Self

import torch

from pybasin.feature_selector.base import FeatureSelector


class VarianceThresholdSelector(FeatureSelector):
    """Keep features whose population variance exceeds a threshold.

    :ivar threshold: Minimum variance required to keep a feature.
    :ivar variances_: Per-feature population variances, stored on the fit
        tensor's device.
    """

    def __init__(self, threshold: float = 0.01):
        """Initialize the selector.

        :param threshold: Minimum population variance required to keep a feature.
        """
        self.threshold: float = threshold

    def fit(self, X: torch.Tensor, y: Any = None) -> Self:
        """Compute per-feature population variance and the support mask.

        :param X: Training tensor of shape ``(n_samples, n_features)``.
        :param y: Ignored.
        :return: Fitted selector.
        :raises ValueError: If ``threshold`` is negative or no feature survives.
        """
        self._prepare_fit(X)
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative.")

        self.variances_ = torch.var(X, dim=0, correction=0)
        support = self.variances_ > self.threshold
        if not bool(torch.any(support)):
            raise ValueError(f"No feature in X meets the variance threshold {self.threshold:.5f}")

        self._set_support(support)
        return self
