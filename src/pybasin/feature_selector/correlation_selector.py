"""Scikit-learn transformer for removing highly correlated features.

Uses the mean absolute correlation ranking from Kuhn & Johnson (2013),
"Applied Predictive Modeling", Chapter 3. When two features exceed the
correlation threshold, the one with higher mean absolute correlation
across all remaining features is dropped (more globally redundant).
"""

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator

from pybasin._sklearn_compat import ArrayTransformerMixin
from pybasin.feature_selector._device_backend import (
    Array,
    DeviceBackend,
    TArray,
    from_native,
    resolve_backend,
    to_native,
)


class CorrelationSelector(BaseEstimator, ArrayTransformerMixin):
    """Scikit-learn transformer to remove highly correlated features.

    When a pair of features has absolute correlation above ``threshold``,
    the feature with the higher mean absolute correlation against all
    remaining features is removed. This follows the algorithm described
    by Kuhn & Johnson (2013) and implemented in R's
    ``caret::findCorrelation()``.

    The mean absolute correlation is recomputed after each removal so
    that subsequent decisions reflect the current feature set.

    Runs on GPU via cupy when ``X`` is a CUDA tensor (``backend="auto"``, the
    default); runs on CPU via numpy otherwise. See ``backend``/``fallback``
    for execution-policy overrides.

    :ivar threshold: Correlation threshold. Feature pairs with absolute
        correlation above this value trigger a removal decision.
    :ivar min_features: Minimum number of features to keep.
    :ivar support_: Boolean mask of selected features (set after ``fit``).
    :ivar n_features_in_: Number of input features (set after ``fit``).
    """

    backend: Literal["auto", "cpu", "gpu"]
    fallback: Literal["raise", "cpu"]

    def __init__(
        self,
        threshold: float = 0.9,
        min_features: int = 3,
        *,
        backend: Literal["auto", "cpu", "gpu"] = "auto",
        fallback: Literal["raise", "cpu"] = "raise",
    ):
        """
        :param threshold: Correlation threshold. Feature pairs with absolute
            correlation above this value trigger a removal decision.
        :param min_features: Minimum number of features to keep.
        :param backend: ``"auto"`` (default) infers CPU/GPU from ``X``'s device
            at :meth:`fit` time. ``"cpu"``/``"gpu"`` force that backend.
        :param fallback: ``"raise"`` (default) errors if ``backend="auto"``
            resolves to GPU but cupy isn't installed. ``"cpu"`` falls back
            silently in that case instead.
        """
        self.threshold: float = threshold
        self.min_features: int = min_features
        self.backend = backend
        self.fallback = fallback

    def fit(self, X: Array, y: Any = None) -> "CorrelationSelector":
        """Compute which features to keep using mean absolute correlation ranking.

        For each pair exceeding the threshold, the feature with the higher
        mean absolute correlation across all remaining features is dropped.
        The correlation statistics are recomputed after each removal.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Not used, present for API consistency.
        :return: Fitted transformer.
        """
        backend = resolve_backend(X, requested=self.backend, fallback=self.fallback)
        self.backend_ = backend.name
        self.fallback_reason_ = backend.fallback_reason
        self._backend = backend

        xp = backend.xp
        X_native = to_native(X, backend=backend)
        self.n_features_in_ = X_native.shape[1]

        if self.n_features_in_ <= self.min_features:
            self.support_ = np.ones(self.n_features_in_, dtype=bool)
            return self

        corr_matrix = xp.abs(xp.corrcoef(X_native.T))
        xp.fill_diagonal(corr_matrix, 0.0)

        remaining: list[int] = list(range(self.n_features_in_))

        while len(remaining) > self.min_features:
            idx = xp.asarray(remaining)
            sub_corr = corr_matrix[xp.ix_(idx, idx)]
            max_corr: float = float(xp.max(sub_corr))

            if max_corr <= self.threshold:
                break

            i_local, j_local = divmod(int(xp.argmax(sub_corr)), len(remaining))
            mean_corr_i: float = float(xp.mean(sub_corr[i_local]))  # type: ignore[arg-type]
            mean_corr_j: float = float(xp.mean(sub_corr[j_local]))  # type: ignore[arg-type]

            drop_local: int = i_local if mean_corr_i >= mean_corr_j else j_local
            remaining.pop(drop_local)

        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[remaining] = True

        return self

    def transform(self, X: TArray) -> TArray:
        """Remove correlated features.

        :param X: Input data of shape (n_samples, n_features).
        :return: Data with correlated features removed, shape (n_samples, n_features_out),
            mirroring ``X``'s type/device.
        """
        backend: DeviceBackend = self._backend
        X_native = to_native(X, backend=backend)
        mask_native = backend.xp.asarray(self.support_)
        return from_native(X_native[:, mask_native], like=X, backend=backend)

    def fit_transform(self, X: TArray, y: Any = None, **fit_params: Any) -> TArray:
        """Fit to ``X`` then remove correlated features in one call.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Not used, present for API consistency.
        :param fit_params: Not used, present for API consistency with ``TransformerMixin``.
        :return: Data with correlated features removed, mirroring ``X``'s type/device.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get a mask or indices of selected features.

        :param indices: If True, return feature indices. Otherwise, return boolean mask.
        :return: Boolean mask or integer indices of selected features (host ``np.ndarray``).
        """
        if indices:
            return np.where(self.support_)[0]
        return self.support_
