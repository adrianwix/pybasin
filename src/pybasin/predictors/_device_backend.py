"""Shared CPU/GPU backend resolution for clustering and classification predictors.

Every predictor in ``pybasin.predictors`` that has a CPU (sklearn) and GPU (cuML)
implementation resolves its backend through :func:`resolve_backend` and converts
data at its boundary through :func:`to_native`/:func:`from_native`/:func:`to_numpy`.
These are the only device-crossing call sites for this feature -- no clusterer or
classifier does its own ``.to(device)``/``.cpu()``/``.numpy()`` internally.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import numpy as np
import torch
from sklearn.cluster import DBSCAN, HDBSCAN  # type: ignore[attr-defined]
from sklearn.metrics import silhouette_samples, silhouette_score  # type: ignore[import-untyped]
from sklearn.metrics.pairwise import euclidean_distances  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors  # type: ignore[import-untyped]

BackendName = Literal["cpu", "gpu"]


@dataclass(frozen=True)
class DeviceBackend:
    """A resolved set of backend primitives (array module, estimator classes, metrics).

    :ivar name: ``"cpu"`` or ``"gpu"``.
    :ivar is_gpu: Whether this backend runs on GPU (cuML/cupy).
    :ivar fallback_reason: Why a fallback occurred, or ``None`` if the resolved
        backend is exactly what was requested.
    :ivar xp: ``numpy`` or ``cupy``.
    """

    name: BackendName
    is_gpu: bool
    fallback_reason: str | None
    xp: ModuleType
    nearest_neighbors_cls: type
    kneighbors_classifier_cls: type
    silhouette_score: Callable[..., Any]
    silhouette_samples: Callable[..., Any]
    pairwise_distances: Callable[..., Any]
    hdbscan_cls: type
    dbscan_cls: type


class UnsupportedConfigurationError(ValueError):
    """A parameter value is only valid on a backend other than the one resolved for this call."""


def require_supported(
    value: str, *, gpu_supported: set[str], param_name: str, backend: DeviceBackend
) -> None:
    """Raise :class:`UnsupportedConfigurationError` if ``value`` isn't valid on a GPU backend.

    Shared by every backend-divergent parameter check (KNN's ``algorithm``,
    HDBSCAN/DBSCAN's ``metric``) so each predictor doesn't hand-roll its own.
    """
    if backend.is_gpu and value not in gpu_supported:
        raise UnsupportedConfigurationError(
            f"{param_name}={value!r} is not supported on the GPU backend "
            f"(supported: {sorted(gpu_supported)}). Pass a supported value, "
            f"or use backend='cpu'/fallback='cpu' to run on CPU instead."
        )


def _device_kind(X: Any) -> str:
    """Return 'cuda', 'cpu', or another torch device-type string for a tensor/array."""
    if isinstance(X, torch.Tensor):
        return X.device.type
    if type(X).__module__.startswith("cupy"):
        return "cuda"
    return "cpu"


def _cpu_backend(fallback_reason: str | None = None) -> DeviceBackend:
    return DeviceBackend(
        name="cpu",
        is_gpu=False,
        fallback_reason=fallback_reason,
        xp=np,
        nearest_neighbors_cls=NearestNeighbors,
        kneighbors_classifier_cls=KNeighborsClassifier,
        silhouette_score=silhouette_score,  # type: ignore[arg-type]
        silhouette_samples=silhouette_samples,  # type: ignore[arg-type]
        pairwise_distances=euclidean_distances,
        hdbscan_cls=HDBSCAN,  # type: ignore[arg-type]
        dbscan_cls=DBSCAN,
    )


def _gpu_backend() -> DeviceBackend:
    try:
        import cuml.cluster
        import cuml.cluster.hdbscan
        import cuml.metrics
        import cuml.metrics.cluster
        import cuml.neighbors
        import cupy
    except ImportError as e:
        raise ImportError(
            "GPU backend requested but cuML is not installed. "
            "Install RAPIDS cuML, see https://docs.rapids.ai/install"
        ) from e

    return DeviceBackend(
        name="gpu",
        is_gpu=True,
        fallback_reason=None,
        xp=cupy,
        nearest_neighbors_cls=cuml.neighbors.NearestNeighbors,
        kneighbors_classifier_cls=cuml.neighbors.KNeighborsClassifier,
        silhouette_score=cuml.metrics.cluster.silhouette_score,
        silhouette_samples=cuml.metrics.cluster.silhouette_samples,
        pairwise_distances=cuml.metrics.pairwise_distances,
        hdbscan_cls=cuml.cluster.hdbscan.HDBSCAN,
        dbscan_cls=cuml.cluster.DBSCAN,
    )


def resolve_backend(
    X: Any,
    *,
    requested: Literal["auto", "cpu", "gpu"] = "auto",
    fallback: Literal["raise", "cpu"] = "raise",
) -> DeviceBackend:
    """Resolve which backend (CPU/sklearn or GPU/cuML) a call against ``X`` should use.

    ``requested="auto"`` (the default) infers the backend from ``X``'s device:
    a CPU tensor or plain ``numpy``/``cupy`` array resolves accordingly; a CUDA
    tensor resolves to GPU, raising ``ImportError`` if cuML isn't installed
    unless ``fallback="cpu"``. Any other device (``mps``, ``xpu``, ...) always
    warns and falls back to CPU, regardless of ``fallback``.

    ``requested="cpu"``/``"gpu"`` force that backend; an explicit ``"gpu"``
    request that cuML can't satisfy always raises, regardless of ``fallback``.
    """
    if requested == "cpu":
        return _cpu_backend()
    if requested == "gpu":
        return _gpu_backend()

    kind = _device_kind(X)
    if kind == "cuda":
        try:
            return _gpu_backend()
        except ImportError:
            if fallback == "cpu":
                return _cpu_backend(fallback_reason="cuML not installed; fell back to CPU")
            raise
    if kind == "cpu":
        return _cpu_backend()

    warnings.warn(
        f"No GPU backend for device {kind!r}; falling back to CPU.",
        stacklevel=2,
    )
    return _cpu_backend(fallback_reason=f"unsupported device {kind!r}; fell back to CPU")


def to_native(X: Any, *, backend: DeviceBackend) -> Any:
    """Convert ``X`` to the array type ``backend`` operates on (numpy or cupy)."""
    if backend.is_gpu:
        cp = backend.xp
        if isinstance(X, torch.Tensor):
            if X.device.type == "cuda":
                return cp.from_dlpack(X)
            return cp.asarray(X.detach().cpu().numpy())
        if type(X).__module__.startswith("cupy"):
            return X
        return cp.asarray(X)

    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    if type(X).__module__.startswith("cupy"):
        return backend.xp.asnumpy(X)
    return X


def to_numpy(x: Any, *, backend: DeviceBackend) -> np.ndarray:
    """Unconditionally land ``x`` on host as a ``numpy`` array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if type(x).__module__.startswith("cupy"):
        return backend.xp.asnumpy(x)
    return np.asarray(x)


def from_native(labels: Any, *, like: Any, backend: DeviceBackend) -> Any:
    """Convert ``labels`` (produced by ``backend``) to mirror ``like``'s type/device."""
    if isinstance(like, torch.Tensor):
        if type(labels).__module__.startswith("cupy"):
            native_tensor = torch.from_dlpack(labels)
        else:
            native_tensor = torch.from_numpy(np.asarray(labels))  # type: ignore[arg-type]
        return native_tensor.to(device=like.device)

    if type(like).__module__.startswith("cupy"):
        if type(labels).__module__.startswith("cupy"):
            return labels
        return backend.xp.asarray(labels)

    return to_numpy(labels, backend=backend)


def standardize(x: Any, xp: ModuleType) -> Any:
    """Z-score normalize ``x`` along axis 0, using whichever array module ``xp`` is."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std_safe = xp.where(std == 0, 1.0, std)
    return (x - mean) / std_safe


def assign_noise_to_clusters(
    features: Any, labels: Any, *, n_neighbors: int = 5, backend: DeviceBackend
) -> Any:
    """Assign noise points (label ``-1``) to their nearest cluster via KNN majority vote.

    Shared by ``HDBSCANClusterer``/``DBSCANClusterer`` so the noise-reassignment logic
    is written once, parameterized by ``backend``, instead of duplicated per clusterer.
    """
    xp = backend.xp
    labels_updated = labels.copy()
    noise_mask = labels == -1

    if not bool(xp.any(noise_mask)):
        return labels_updated

    labeled_mask = ~noise_mask
    labeled_features = features[labeled_mask]
    labeled_labels = labels[labeled_mask]

    if labeled_features.shape[0] == 0:
        return labels_updated

    noise_features = features[noise_mask]
    k_actual = min(n_neighbors, labeled_features.shape[0])
    nn = backend.nearest_neighbors_cls(n_neighbors=k_actual)
    nn.fit(labeled_features)
    _, indices = nn.kneighbors(noise_features)
    indices = xp.asarray(indices)

    # Each noise point's k neighbor labels are compared only against each other (a (k, k)
    # pairwise equality matrix per point), not against the full set of cluster labels, so
    # memory is O(n_noise * k^2) and does not grow with the number of clusters. Ties break
    # toward the smallest unique-label id, matching argmax's first-match behavior over the
    # (ascending-sorted) unique labels.
    unique_labels, inverse = xp.unique(labeled_labels, return_inverse=True)
    neighbor_ids = inverse[indices]
    matches = neighbor_ids[:, :, None] == neighbor_ids[:, None, :]
    vote_counts = matches.sum(axis=2)
    max_count = vote_counts.max(axis=1, keepdims=True)
    is_max = vote_counts == max_count
    sentinel = xp.iinfo(neighbor_ids.dtype).max
    candidate_ids = xp.where(is_max, neighbor_ids, sentinel)
    majority_labels = unique_labels[candidate_ids.min(axis=1)]

    noise_indices = xp.where(noise_mask)[0]
    labels_updated[noise_indices] = majority_labels

    return labels_updated
