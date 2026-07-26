from typing import Any, Literal

from sklearn.base import BaseEstimator, ClusterMixin  # type: ignore[import-untyped]

from pybasin.predictors._device_backend import (
    DeviceBackend,
    assign_noise_to_clusters,
    from_native,
    require_supported,
    resolve_backend,
    to_native,
)
from pybasin.utils import DisplayNameMixin


class DBSCANClusterer(DisplayNameMixin, ClusterMixin, BaseEstimator):
    """DBSCAN clustering for basin stability analysis (unsupervised learning).

    Runs on GPU via cuML when ``X`` is a CUDA tensor (``backend="auto"``, the
    default); runs on CPU via sklearn otherwise. See ``backend``/``fallback``
    for execution-policy overrides. To auto-tune ``eps``, wrap this
    estimator in :class:`pybasin.predictors.tuned_clusterer.TunedClusterer` with
    :class:`pybasin.predictors.tuned_clusterer.EpsilonProminenceSearch`.
    """

    backend: Literal["auto", "cpu", "gpu"]
    fallback: Literal["raise", "cpu"]

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 10,
        *,
        metric: str = "euclidean",
        assign_noise: bool = False,
        n_neighbors: int = 5,
        backend: Literal["auto", "cpu", "gpu"] = "auto",
        fallback: Literal["raise", "cpu"] = "raise",
        cpu_options: dict[str, Any] | None = None,
        gpu_options: dict[str, Any] | None = None,
    ):
        """
        Initialize DBSCAN clusterer.

        :param eps: Maximum distance between two samples for one to be
            considered a neighbor of the other.
        :param min_samples: Number of samples in a neighborhood for a point to
            be considered a core point.
        :param metric: Distance metric. GPU backend only supports
            ``"euclidean"``/``"cosine"``/``"precomputed"``; requesting any
            other metric on a resolved GPU backend raises
            :class:`UnsupportedConfigurationError`.
        :param assign_noise: Whether to assign noise points (label ``-1``) to
            the nearest cluster via KNN majority vote.
        :param n_neighbors: Number of neighbors used for noise assignment. Only
            used when ``assign_noise=True``.
        :param backend: ``"auto"`` (default) infers CPU/GPU from ``X``'s device.
            ``"cpu"``/``"gpu"`` force that backend.
        :param fallback: ``"raise"`` (default) errors if ``backend="auto"``
            resolves to GPU but cuML isn't installed. ``"cpu"`` falls back
            silently in that case instead.
        :param cpu_options: Extra keyword arguments forwarded to
            ``sklearn.cluster.DBSCAN`` only.
        :param gpu_options: Extra keyword arguments forwarded to
            ``cuml.cluster.DBSCAN`` only.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.assign_noise = assign_noise
        self.n_neighbors = n_neighbors
        self.backend = backend
        self.fallback = fallback
        self.cpu_options = cpu_options
        self.gpu_options = gpu_options

    def fit_predict(self, X: Any, y: Any = None) -> Any:
        """
        Fit and predict labels using DBSCAN clustering.

        :param X: Feature array of shape ``(n_samples, n_features)``.
        :param y: Ignored (present for sklearn API compatibility).
        :return: Cluster labels (``-1`` for noise unless ``assign_noise=True``),
            mirroring ``X``'s type/device.
        """
        backend = resolve_backend(X, requested=self.backend, fallback=self.fallback)
        self.backend_ = backend.name
        self.fallback_reason_ = backend.fallback_reason
        self.resolved_params_ = self._canonical_parameters()
        self.estimator_ = self._create_estimator(backend)

        X_native = to_native(X, backend=backend)
        labels = self.estimator_.fit_predict(X_native)

        if self.assign_noise:
            labels = assign_noise_to_clusters(
                X_native, labels, n_neighbors=self.n_neighbors, backend=backend
            )

        return from_native(labels, like=X, backend=backend)

    def _canonical_parameters(self) -> dict[str, Any]:
        return {"eps": self.eps, "min_samples": self.min_samples, "metric": self.metric}

    def _create_estimator(self, backend: DeviceBackend) -> Any:
        common = {"eps": self.eps, "min_samples": self.min_samples}
        if not backend.is_gpu:
            options = {"metric": self.metric, **(self.cpu_options or {})}
            return backend.dbscan_cls(**common, **options)

        require_supported(
            self.metric,
            gpu_supported={"euclidean", "cosine", "precomputed"},
            param_name="metric",
            backend=backend,
        )
        return backend.dbscan_cls(**common, **(self.gpu_options or {}))
