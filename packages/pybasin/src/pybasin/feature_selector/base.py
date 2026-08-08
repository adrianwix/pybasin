"""Base class for feature selectors."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Self

import numpy as np
import torch
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]


class FeatureSelector(BaseEstimator, ABC):
    """Base class for selectors that filter columns from a Torch tensor.

    Subclasses compute a boolean support mask in :meth:`fit`. The shared
    transformation applies that mask to the feature columns of its input.

    :ivar support_: NumPy boolean mask of selected features, set after fitting.
    :ivar n_features_in_: Number of input features seen during fitting.
    """

    tensor_native: ClassVar[Literal[True]] = True

    @abstractmethod
    def fit(self, X: torch.Tensor, y: Any = None) -> Self:
        """Compute the feature support mask.

        :param X: Training tensor of shape ``(n_samples, n_features)``.
        :param y: Optional target data.
        :return: Fitted selector.
        """
        ...

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Select the fitted feature columns from ``X``.

        :param X: Input tensor of shape ``(n_samples, n_features)``.
        :return: Tensor containing the selected columns on ``X``'s device.
        :raises RuntimeError: If the selector has not been fitted.
        :raises TypeError: If ``X`` is not a Torch tensor.
        :raises ValueError: If ``X`` is not two-dimensional or its feature count
            differs from the fitted feature count.
        """
        self._validate_tensor(X)
        if not hasattr(self, "support_"):
            raise RuntimeError(f"{type(self).__name__} must be fit() before transform().")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} was fitted "
                f"with {self.n_features_in_} features."
            )

        support = torch.as_tensor(self.support_, dtype=torch.bool, device=X.device)
        return X[:, support]

    def fit_transform(self, X: torch.Tensor, y: Any = None) -> torch.Tensor:
        """Fit to ``X`` and return its selected columns.

        :param X: Training tensor of shape ``(n_samples, n_features)``.
        :param y: Optional target data forwarded to :meth:`fit`.
        :return: Filtered training tensor on ``X``'s device.
        """
        return self.fit(X, y).transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Return the fitted support mask or selected feature indices.

        :param indices: Return integer indices instead of a boolean mask.
        :return: Host NumPy array containing the mask or indices.
        :raises RuntimeError: If the selector has not been fitted.
        """
        if not hasattr(self, "support_"):
            raise RuntimeError(f"{type(self).__name__} must be fit() before get_support().")
        if indices:
            return np.flatnonzero(self.support_)
        return self.support_.copy()

    def _prepare_fit(self, X: torch.Tensor) -> None:
        self._validate_tensor(X)
        if not torch.is_floating_point(X):
            raise TypeError("X must have a floating-point dtype.")
        self.n_features_in_ = int(X.shape[1])

    def _set_support(self, support: torch.Tensor | np.ndarray) -> None:
        if isinstance(support, torch.Tensor):
            support = support.detach().to(device="cpu", dtype=torch.bool).numpy()
        support_array = np.asarray(support, dtype=bool)
        if support_array.shape != (self.n_features_in_,):
            raise ValueError(
                "Support mask must contain one value for each input feature; "
                f"expected {(self.n_features_in_,)}, got {support_array.shape}."
            )
        self.support_ = support_array.copy()

    @staticmethod
    def _validate_tensor(X: object) -> None:
        if not isinstance(X, torch.Tensor):
            raise TypeError(f"X must be a torch.Tensor, got {type(X).__name__}.")
        if X.ndim != 2:
            raise ValueError(f"X must be two-dimensional, got shape {tuple(X.shape)}.")
