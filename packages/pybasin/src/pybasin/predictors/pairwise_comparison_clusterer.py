"""Nearest-leader sequential clustering (pybasin's port of Attractors.jl's ``GroupViaPairwiseComparison``)."""

from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import torch
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from torch import Tensor
from torch._higher_order_ops.scan import scan  # type: ignore[import-untyped]

from pybasin._device_backend import Array, TArray
from pybasin._sklearn_compat import ArrayClusterMixin
from pybasin.utils import DisplayNameMixin

Metric = Literal["minkowski"] | Callable[[Tensor, Tensor], Tensor]


def hausdorff_metric(a: Tensor, b: Tensor) -> Tensor:
    """Hausdorff distance between two point sets, for matrix-valued (trajectory) features.

    :param a: Point set of shape ``(M, D)``.
    :param b: Point set of shape ``(N, D)``, ``D`` matching ``a``.
    :return: Scalar 0-dimensional tensor: the Hausdorff distance between ``a`` and ``b``,
        ``max(max_i min_j d(a_i, b_j), max_j min_i d(a_i, b_j))``.
    """
    d = torch.cdist(a, b)
    return torch.maximum(d.min(dim=1).values.max(), d.min(dim=0).values.max())


def _rescale_columns_01(X: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Min-max rescale each column of ``X`` into ``[0, 1]``; constant columns become ``0``.

    :param X: Feature matrix of shape ``(n_samples, n_features)``.
    :return: ``(X_scaled, column_min, column_range)`` where ``column_range`` has already
        had zero entries replaced by ``1`` (the denominator actually used), so
        ``X_scaled == (X - column_min) / column_range`` exactly.
    """
    column_min = X.amin(dim=0)
    column_max = X.amax(dim=0)
    column_range = column_max - column_min
    safe_range = torch.where(column_range == 0, torch.ones_like(column_range), column_range)
    X_scaled = (X - column_min) / safe_range
    return X_scaled, column_min, safe_range


def _row_distances(
    x_i: Tensor, representatives: Tensor, *, use_cdist: bool, p: float, metric_fn: Any
) -> Tensor:
    """Distance from one sample to every representative slot (including unwritten ones).

    :param x_i: A single sample, shape ``(*feature_shape,)``.
    :param representatives: All representative slots, shape ``(capacity, *feature_shape)``.
    :param use_cdist: If ``True``, compute Minkowski distances with one batched
        :func:`torch.cdist` call. If ``False``, call ``metric_fn(x_i, representatives[j])``
        once per slot ``j`` (``capacity`` is a Python ``int``, so this loop unrolls at trace
        time and is compiled like any other fixed-length sequence of ops).
    :param p: Minkowski exponent, used only when ``use_cdist`` is ``True``.
    :param metric_fn: Callable ``(a, b) -> Tensor`` (scalar), used only when ``use_cdist``
        is ``False``.
    :return: Distances of shape ``(capacity,)``.
    """
    if use_cdist:
        return torch.cdist(x_i.unsqueeze(0), representatives, p=p).squeeze(0)
    capacity = representatives.shape[0]
    return torch.stack([metric_fn(x_i, representatives[j]) for j in range(capacity)])


def _pairwise_group_scan(
    X: Tensor,
    idx: Tensor,
    threshold: Tensor,
    capacity: int,
    p: float,
    use_cdist: bool,
    metric_fn: Any,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Leader-clustering kernel: one ``torch._higher_order_ops.scan`` pass at a fixed capacity.

    Carries a fixed-size table of up to ``capacity`` group representatives; each step
    compares the current sample against every representative slot (slots at or beyond the
    current group count are masked to ``+inf``), joins the nearest one if within
    ``threshold``, else appends a new representative. If more than ``capacity`` groups are
    needed, ``overflowed`` comes back ``True`` and the caller retries at a larger capacity;
    ``capacity`` only bounds how many groups this call can represent, never the semantics.

    :param X: Feature batch, shape ``(N, *feature_shape)`` — ``(N, F)`` for vector features,
        ``(N, M, D)`` for matrix-valued (point-set) features.
    :param idx: ``arange(N)``, on the same device as ``X``.
    :param threshold: 0-dimensional tensor, same dtype/device as ``X``.
    :param capacity: Maximum number of groups this call can represent.
    :param p: Minkowski exponent (used only when ``use_cdist`` is ``True``).
    :param use_cdist: Whether to use the batched Minkowski fast path or ``metric_fn``.
    :param metric_fn: Custom metric callable, used only when ``use_cdist`` is ``False``.
    :return: ``(labels, representative_indices, n_groups, overflowed)`` — ``labels`` has
        shape ``(N,)``; ``representative_indices`` has shape ``(capacity,)``, valid in
        ``[0, n_groups)``; ``n_groups`` and ``overflowed`` are 0-dimensional tensors.
    """
    feature_shape = X.shape[1:]
    representatives0 = X.new_zeros((capacity, *feature_shape))
    representative_indices0 = torch.zeros(capacity, dtype=torch.int64, device=X.device)
    n_groups0 = torch.zeros((), dtype=torch.int64, device=X.device)
    overflowed0 = torch.zeros((), dtype=torch.bool, device=X.device)
    slot_iota = torch.arange(capacity, device=X.device)

    def combine_fn(
        carry: tuple[Tensor, Tensor, Tensor, Tensor], x_slice: tuple[Tensor, Tensor]
    ) -> tuple[tuple[Tensor, Tensor, Tensor, Tensor], Tensor]:
        representatives, representative_indices, n_groups, overflowed = carry
        x_i, idx_i = x_slice

        raw = _row_distances(x_i, representatives, use_cdist=use_cdist, p=p, metric_fn=metric_fn)
        distances = torch.where(slot_iota < n_groups, raw, torch.full_like(raw, float("inf")))

        nearest = torch.argmin(distances)
        # distances[nearest] would index by a data-dependent scalar, which scan's
        # tracing can't specialize ("Could not extract specialized integer from
        # data-dependent expression"); distances.min() is the same value by definition
        # of argmin, with no indexing operation at all.
        is_new_group = distances.min() > threshold

        write_slot = torch.clamp(n_groups, max=capacity - 1)
        write_mask = is_new_group & (slot_iota == write_slot)
        feature_ndim = representatives.ndim - 1
        write_mask_expanded = write_mask.view(capacity, *([1] * feature_ndim))

        new_representatives = torch.where(write_mask_expanded, x_i.unsqueeze(0), representatives)
        new_representative_indices = torch.where(write_mask, idx_i, representative_indices)
        new_overflowed = overflowed | (is_new_group & (n_groups >= capacity))
        new_n_groups = torch.clamp(n_groups + is_new_group.to(torch.int64), max=capacity)
        label_i = torch.where(is_new_group, n_groups, nearest)

        new_carry = (new_representatives, new_representative_indices, new_n_groups, new_overflowed)
        return new_carry, label_i

    init = (representatives0, representative_indices0, n_groups0, overflowed0)
    final_carry, labels = scan(combine_fn, init, (X, idx))
    _representatives, representative_indices, n_groups, overflowed = final_carry
    return labels, representative_indices, n_groups, overflowed


# A single process may exercise many distinct (dtype, use_cdist, capacity) combinations
# against this one compiled function -- e.g. float32 and float64 inputs, "minkowski" and a
# custom metric, several capacities from the retry loop -- each a genuinely different
# specialization Dynamo must compile separately. torch's default recompile_limit (8) is
# tuned for functions with one or two stable call shapes and raises
# `FailOnRecompileLimitHit` under `fullgraph=True` well before that many legitimate
# specializations accumulate.
torch._dynamo.config.recompile_limit = 64  # pyright: ignore[reportPrivateUsage]

_compiled_pairwise_group_scan = torch.compile(  # pyright: ignore[reportUnknownMemberType]
    _pairwise_group_scan, fullgraph=True, dynamic=True
)


def _pairwise_group_loop(  # pyright: ignore[reportUnusedFunction]
    X: Tensor, threshold: float, *, use_cdist: bool, p: float, metric_fn: Any
) -> tuple[Tensor, Tensor, int]:
    """Leader-clustering kernel: plain per-row Python loop, exactly the reference algorithm.

    Correct on both CPU and CUDA tensors, for any metric, with no compilation involved, at
    the cost of one Python-level loop iteration per sample. Serves as the correctness
    oracle :func:`_pairwise_group_scan` is validated against, and is a plain, portable
    implementation any metric can fall back to if a future torch version regresses
    ``scan``'s support for this class's usage pattern.

    :param X: Feature batch, shape ``(N, *feature_shape)``.
    :param threshold: Maximum distance to join an existing group.
    :param use_cdist: Whether to use the batched Minkowski fast path or ``metric_fn``.
    :param p: Minkowski exponent (used only when ``use_cdist`` is ``True``).
    :param metric_fn: Custom metric callable, used only when ``use_cdist`` is ``False``.
    :return: ``(labels, representative_indices, n_groups)`` — ``labels`` has shape ``(N,)``,
        ``representative_indices`` has shape ``(n_groups,)``.
    """
    n = X.shape[0]
    labels = torch.empty(n, dtype=torch.int64, device=X.device)
    representative_indices: list[int] = [0]
    labels[0] = 0

    for i in range(1, n):
        representatives = X[representative_indices]
        if use_cdist:
            distances = torch.cdist(X[i : i + 1], representatives, p=p).squeeze(0)
        else:
            distances = torch.stack([metric_fn(X[i], rep) for rep in representatives])
        nearest = int(torch.argmin(distances))
        if float(distances[nearest]) > threshold:
            labels[i] = len(representative_indices)
            representative_indices.append(i)
        else:
            labels[i] = nearest

    rep_idx_tensor = torch.tensor(representative_indices, dtype=torch.int64, device=X.device)
    return labels, rep_idx_tensor, len(representative_indices)


class PairwiseComparisonClusterer(DisplayNameMixin, ArrayClusterMixin, BaseEstimator):
    """Nearest-leader sequential clustering for basin stability analysis (unsupervised learning).

    Each sample, in input order, is compared against the current group representatives
    only: it joins the nearest representative if that distance is at most ``threshold``,
    and otherwise becomes a new representative itself. Representatives are fixed once
    created (no centroid update, no merging), so the partition depends on input order.

    This is Attractors.jl's ``GroupViaPairwiseComparison`` (introduced in v1.12,
    reproduced here against v1.38.4, https://doi.org/10.5281/zenodo.20729175). It
    corresponds to the nearest-leader variant of the Leader algorithm in Hartigan (1975),
    *Clustering Algorithms*, Wiley, §3.2-3.4: Hartigan's base rule assigns a sample to the
    *first* leader within the threshold, and §3.4 proposes assigning it to the *closest*
    leader, which is the rule used here. Cost is ``O(N * K)`` metric evaluations and
    ``O(N + K)`` memory for ``N`` samples and ``K`` groups, compared with the ``O(N^2)``
    distance matrix of density-based clustering.

    Runs entirely in plain ``torch`` on whatever device ``X`` is already on; there is no
    optional GPU library to resolve, unlike :class:`~pybasin.predictors.dbscan_clusterer.DBSCANClusterer`.
    """

    def __init__(
        self,
        threshold: float = 0.1,
        *,
        metric: Metric = "minkowski",
        p: float = 2.0,
        rescale_features: bool = True,
        initial_group_capacity: int = 8,
    ):
        """
        :param threshold: Maximum distance between a sample and a group representative for
            the sample to join that group; strictly greater than ``threshold`` starts a new
            group. Equality joins the existing group.
        :param metric: ``"minkowski"`` (default) computes
            ``(sum(|a - b|^p))^(1/p)`` via a single batched :func:`torch.cdist` call. Any
            other callable ``metric(a, b) -> Tensor`` (scalar) is called once per group
            representative, exactly as in Attractors.jl — this is how matrix-valued
            (point-set) features are supported, e.g. :func:`hausdorff_metric`.
        :param p: Minkowski exponent; only meaningful when ``metric="minkowski"``.
            ``p=2.0`` is Euclidean, ``p=1.0`` is Manhattan.
        :param rescale_features: If ``True`` (default), min-max rescale each feature column
            into ``[0, 1]`` over the whole batch before grouping; a constant column becomes
            ``0`` rather than dividing by zero. Must be ``False`` for matrix-valued features
            (``X.ndim == 3``) — per-column min-max has no coherent meaning on a point set.
        :param initial_group_capacity: Initial number of representative slots to allocate.
            Doubled and retried (capped at ``n_samples``) if more groups are found than fit;
            a performance hint only, never changes the output labels.
        """
        self.threshold = threshold
        self.metric = metric
        self.p = p
        self.rescale_features = rescale_features
        self.initial_group_capacity = initial_group_capacity

    def fit_predict(self, X: TArray, y: Any = None) -> TArray:
        """
        Group features via sequential leader clustering.

        :param X: Feature array of shape ``(n_samples, n_features)`` for
            ``metric="minkowski"`` or a vector-valued custom metric, or
            ``(n_samples, M, D)`` (one size-``M`` point set per sample) for a matrix-valued
            custom metric such as :func:`hausdorff_metric`.
        :param y: Ignored (present for sklearn API compatibility).
        :return: Zero-based, contiguous group labels, one per sample. Never ``-1`` — every
            sample is assigned to some group. Mirrors ``X``'s type/device.
        :raises ValueError: If ``X`` isn't 2- or 3-dimensional, if ``rescale_features=True``
            with 3-dimensional (matrix-valued) ``X``, if ``metric`` is neither
            ``"minkowski"`` nor callable, or if ``initial_group_capacity < 1``.
        """
        if self.initial_group_capacity < 1:
            raise ValueError(
                f"initial_group_capacity must be >= 1, got {self.initial_group_capacity}"
            )
        if self.metric == "minkowski":
            use_cdist = True
            metric_fn = None
        elif callable(self.metric):
            use_cdist = False
            metric_fn = self.metric
        else:
            raise ValueError(
                f"metric must be 'minkowski' or a callable metric(a, b) -> Tensor, "
                f"got {self.metric!r}"
            )

        X_tensor = X if isinstance(X, Tensor) else torch.from_numpy(X)  # pyright: ignore[reportUnknownMemberType]
        if X_tensor.ndim not in (2, 3):
            raise ValueError(
                "X must be 2D (n_samples, n_features) or 3D (n_samples, M, D) for "
                f"matrix-valued features, got shape {tuple(X_tensor.shape)}"
            )
        if self.rescale_features and X_tensor.ndim != 2:
            raise ValueError(
                "rescale_features=True requires 2D vector-valued X; matrix-valued features "
                "(X.ndim == 3) have no coherent per-column min-max. Pass rescale_features=False."
            )

        self.resolved_params_ = self._canonical_parameters()

        n = X_tensor.shape[0]
        if n == 0:
            self.n_groups_ = 0
            self.representative_indices_ = np.empty(0, dtype=np.int64)
            self.representatives_ = self._mirror(X_tensor.new_zeros((0, *X_tensor.shape[1:])), X)
            empty_labels = torch.empty(0, dtype=torch.int64, device=X_tensor.device)
            return cast(TArray, self._mirror(empty_labels, X))

        working = X_tensor
        if self.rescale_features:
            working, column_min, column_range = _rescale_columns_01(X_tensor)
            self.rescale_min_ = self._mirror(column_min, X)
            self.rescale_range_ = self._mirror(column_range, X)

        labels, representative_indices, n_groups = self._run_kernel(working, use_cdist, metric_fn)

        self.n_groups_ = int(n_groups)
        rep_idx = representative_indices[: self.n_groups_]
        self.representative_indices_ = rep_idx.detach().cpu().numpy().astype(np.int64)
        self.representatives_ = self._mirror(X_tensor[rep_idx], X)

        return cast(TArray, self._mirror(labels, X))

    def _run_kernel(
        self, working: Tensor, use_cdist: bool, metric_fn: Any
    ) -> tuple[Tensor, Tensor, int]:
        n = working.shape[0]
        capacity = min(self.initial_group_capacity, n)
        idx = torch.arange(n, dtype=torch.int64, device=working.device)
        threshold_t = torch.as_tensor(self.threshold, dtype=working.dtype, device=working.device)

        while True:
            labels, rep_idx, n_groups, overflowed = _compiled_pairwise_group_scan(
                working, idx, threshold_t, capacity, self.p, use_cdist, metric_fn
            )
            if not bool(overflowed) or capacity >= n:
                return labels, rep_idx, int(n_groups)
            capacity = min(capacity * 2, n)

    def _mirror(self, value: Tensor, like: Array) -> Array:
        """Convert ``value`` to mirror ``like``'s concrete type/device."""
        if isinstance(like, Tensor):
            return value.to(device=like.device)
        return value.detach().cpu().numpy()

    def _canonical_parameters(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "metric": self.metric,
            "p": self.p,
            "rescale_features": self.rescale_features,
            "initial_group_capacity": self.initial_group_capacity,
        }
