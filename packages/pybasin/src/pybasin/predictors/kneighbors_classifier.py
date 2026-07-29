from typing import Any, Literal

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin  # type: ignore[import-untyped]

from pybasin.predictors._device_backend import (
    DeviceBackend,
    require_supported,
    resolve_backend,
    to_native,
    to_numpy,
)
from pybasin.utils import DisplayNameMixin


class KNeighborsClassifier(DisplayNameMixin, ClassifierMixin, BaseEstimator):
    """K-nearest-neighbors classifier for supervised basin stability analysis.

    Runs on GPU via cuML when the training data ``X`` passed to :meth:`fit` is a
    CUDA tensor (``backend="auto"``, the default); runs on CPU via sklearn
    otherwise. ``fit()`` resolves and owns the backend; ``predict()`` only
    converts its own input and reuses what ``fit()`` built.
    """

    backend: Literal["auto", "cpu", "gpu"]
    fallback: Literal["raise", "cpu"]

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        metric: str = "euclidean",
        algorithm: Literal["auto", "brute", "kd_tree", "ball_tree"] = "auto",
        backend: Literal["auto", "cpu", "gpu"] = "auto",
        fallback: Literal["raise", "cpu"] = "raise",
        cpu_options: dict[str, Any] | None = None,
        gpu_options: dict[str, Any] | None = None,
    ):
        """
        :param n_neighbors: Number of neighbors to use.
        :param weights: Weight function used in prediction (``"uniform"`` or
            ``"distance"``).
        :param metric: Distance metric.
        :param algorithm: Neighbor-search algorithm. ``"auto"`` means "let each
            backend pick its own best" (sklearn's own auto-selection on CPU,
            cuML's only supported ``"brute"`` on GPU) rather than a literal
            forward. ``"kd_tree"``/``"ball_tree"`` are CPU-only; requesting them
            on a resolved GPU backend raises :class:`UnsupportedConfigurationError`.
        :param backend: ``"auto"`` (default) infers CPU/GPU from ``X``'s device
            at :meth:`fit` time. ``"cpu"``/``"gpu"`` force that backend.
        :param fallback: ``"raise"`` (default) errors if ``backend="auto"``
            resolves to GPU but cuML isn't installed. ``"cpu"`` falls back
            silently in that case instead.
        :param cpu_options: Extra keyword arguments forwarded to
            ``sklearn.neighbors.KNeighborsClassifier`` only (e.g. ``leaf_size``,
            ``n_jobs``).
        :param gpu_options: Extra keyword arguments forwarded to
            ``cuml.neighbors.KNeighborsClassifier`` only (e.g. ``algo_params``).
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.algorithm = algorithm
        self.backend = backend
        self.fallback = fallback
        self.cpu_options = cpu_options
        self.gpu_options = gpu_options

    def fit(self, X: np.ndarray | torch.Tensor, y: Any) -> "KNeighborsClassifier":
        """Resolve the backend, encode labels, and fit the native estimator (once).

        :param X: Training feature array.
        :param y: Training labels (string or numeric).
        :return: self.
        """
        backend = resolve_backend(X, requested=self.backend, fallback=self.fallback)
        self.backend_ = backend.name
        self.fallback_reason_ = backend.fallback_reason
        self.classes_, y_encoded = np.unique(np.asarray(y), return_inverse=True)
        self.resolved_params_ = self._canonical_parameters()
        self.estimator_ = self._create_estimator(backend)

        X_native = to_native(X, backend=backend)
        y_native = to_native(y_encoded, backend=backend)
        self.estimator_.fit(X_native, y_native)
        self._backend = backend
        return self

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Predict labels, decoding back to the original label vocabulary.

        :param X: Feature array to classify.
        :return: Host ``np.ndarray`` of original (string or numeric) labels.
        :raises RuntimeError: If called before :meth:`fit`.
        """
        if not hasattr(self, "estimator_"):
            raise RuntimeError(f"{type(self).__name__} must be fit() before predict().")

        pred_encoded = self.estimator_.predict(to_native(X, backend=self._backend))
        pred_encoded_np = to_numpy(pred_encoded, backend=self._backend)
        return self.classes_[pred_encoded_np.astype(np.int64)]

    def _canonical_parameters(self) -> dict[str, Any]:
        return {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "metric": self.metric,
            "algorithm": self.algorithm,
        }

    def _create_estimator(self, backend: DeviceBackend) -> Any:
        common = {"n_neighbors": self.n_neighbors, "weights": self.weights, "metric": self.metric}
        if not backend.is_gpu:
            options = {"algorithm": self.algorithm, **(self.cpu_options or {})}
            return backend.kneighbors_classifier_cls(**common, **options)

        require_supported(
            self.algorithm,
            gpu_supported={"auto", "brute"},
            param_name="algorithm",
            backend=backend,
        )
        options = {"algorithm": "brute", **(self.gpu_options or {})}
        return backend.kneighbors_classifier_cls(**common, **options)
