"""Scikit-learn transformer for removing low-variance features.

A backend-aware equivalent of ``sklearn.feature_selection.VarianceThreshold``:
functionally identical (features with variance strictly above ``threshold``
are kept), but computes on whichever device ``X`` already lives on instead of
forcing a host round trip, so it can be chained with GPU-native selectors.
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
    to_numpy,
)


class VarianceThresholdSelector(BaseEstimator, ArrayTransformerMixin):
    """Removes features whose variance does not exceed ``threshold``.

    Runs on GPU via cupy when ``X`` is a CUDA tensor (``backend="auto"``, the
    default); runs on CPU via numpy otherwise. See ``backend``/``fallback``
    for execution-policy overrides.

    :ivar threshold: Minimum variance required to keep a feature. Features
        with variance not strictly greater than this value are removed.
    :ivar support_: Boolean mask of selected features (set after ``fit``).
    :ivar n_features_in_: Number of input features (set after ``fit``).
    """

    backend: Literal["auto", "cpu", "gpu"]
    fallback: Literal["raise", "cpu"]

    def __init__(
        self,
        threshold: float = 0.01,
        *,
        backend: Literal["auto", "cpu", "gpu"] = "auto",
        fallback: Literal["raise", "cpu"] = "raise",
    ):
        """
        :param threshold: Minimum variance required to keep a feature.
        :param backend: ``"auto"`` (default) infers CPU/GPU from ``X``'s device
            at :meth:`fit` time. ``"cpu"``/``"gpu"`` force that backend.
        :param fallback: ``"raise"`` (default) errors if ``backend="auto"``
            resolves to GPU but cupy isn't installed. ``"cpu"`` falls back
            silently in that case instead.
        """
        self.threshold: float = threshold
        self.backend = backend
        self.fallback = fallback

    def fit(self, X: Array, y: Any = None) -> "VarianceThresholdSelector":
        """Compute per-feature variance and the resulting selection mask.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Not used, present for API consistency.
        :return: Fitted transformer.
        :raises ValueError: If no feature's variance exceeds ``threshold``.
        """
        backend = resolve_backend(X, requested=self.backend, fallback=self.fallback)
        self.backend_ = backend.name
        self.fallback_reason_ = backend.fallback_reason
        self._backend = backend

        X_native = to_native(X, backend=backend)
        self.n_features_in_ = X_native.shape[1]

        variances = backend.xp.var(X_native, axis=0)
        self.support_ = to_numpy(variances, backend=backend) > self.threshold

        if not bool(self.support_.any()):
            raise ValueError(f"No feature in X meets the variance threshold {self.threshold:.5f}")

        return self

    def transform(self, X: TArray) -> TArray:
        """Remove low-variance features.

        :param X: Input data of shape (n_samples, n_features).
        :return: Data with low-variance features removed, shape (n_samples, n_features_out),
            mirroring ``X``'s type/device.
        """
        backend: DeviceBackend = self._backend
        X_native = to_native(X, backend=backend)
        mask_native = backend.xp.asarray(self.support_)
        return from_native(X_native[:, mask_native], like=X, backend=backend)

    def fit_transform(self, X: TArray, y: Any = None, **fit_params: Any) -> TArray:
        """Fit to ``X`` then remove low-variance features in one call.

        :param X: Training data of shape (n_samples, n_features).
        :param y: Not used, present for API consistency.
        :param fit_params: Not used, present for API consistency with ``TransformerMixin``.
        :return: Data with low-variance features removed, mirroring ``X``'s type/device.
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
