"""Auto-tuning meta-estimator for backend-neutral clusterers.

:class:`TunedClusterer` wraps a backend-neutral base estimator (e.g.
``HDBSCANClusterer``, ``DBSCANClusterer``) and a search strategy that decides which
hyperparameters to try and how to pick a winner.
"""

from typing import Any, Literal, Protocol, cast

import numpy as np
import torch
from sklearn.base import BaseEstimator, clone  # type: ignore[import-untyped]

from pybasin._ops.find_peaks import find_peaks
from pybasin._sklearn_compat import ArrayClusterMixin
from pybasin.predictors._device_backend import (
    Array,
    DeviceBackend,
    TArray,
    from_native,
    resolve_backend,
    to_native,
)
from pybasin.utils import DisplayNameMixin


class SearchStrategy(Protocol):
    """A strategy that picks the best hyperparameters for a clusterer against ``X``."""

    def find_best_params(self, X: Any, estimator: Any) -> dict[str, Any]: ...


class TunedClusterer(DisplayNameMixin, ArrayClusterMixin, BaseEstimator):
    """Wraps a backend-neutral clusterer, tuning its hyperparameters via a search strategy."""

    def __init__(self, estimator: Any, search: SearchStrategy):
        """
        :param estimator: A backend-neutral clusterer (e.g. ``HDBSCANClusterer``,
            ``DBSCANClusterer``) whose hyperparameters will be tuned.
        :param search: A search strategy implementing ``find_best_params(X, estimator)``.
        """
        self.estimator = estimator
        self.search = search

    def fit_predict(self, X: TArray, y: Any = None) -> TArray:
        """Tune ``estimator``'s hyperparameters against ``X`` and fit the winner.

        :param X: Feature array to cluster.
        :param y: Ignored (present for sklearn API compatibility).
        :return: Cluster labels from the tuned estimator.
        """
        best_params = self.search.find_best_params(X, self.estimator)
        self.estimator_ = clone(self.estimator).set_params(**best_params)
        self.best_params_ = best_params
        return self.estimator_.fit_predict(X)


def _resolve_for(estimator: Any, X: Any) -> DeviceBackend:
    requested = cast(Literal["auto", "cpu", "gpu"], getattr(estimator, "backend", "auto"))
    fallback = cast(Literal["raise", "cpu"], getattr(estimator, "fallback", "raise"))
    return resolve_backend(X, requested=requested, fallback=fallback)


def _to_torch(arr: Any, backend: DeviceBackend) -> torch.Tensor:
    if backend.is_gpu:
        return torch.from_dlpack(arr)
    return torch.from_numpy(np.asarray(arr))  # type: ignore[arg-type]


class MinClusterSizeSearch:
    """Auto-tunes HDBSCAN's ``min_cluster_size``: five candidate sizes, silhouette argmax.

    Each candidate requires a full HDBSCAN fit plus a silhouette computation, so this
    search's cost scales with the dataset size. The search subsample size is
    ``clip(tune_sample_fraction * N, min_tune_sample_size, max_tune_sample_size)``: a
    fraction of the actual dataset, bounded below so tiny datasets still get a usable
    search set and above so huge datasets don't blow the search cost back up.

    !!! warning "Subsampling does not reliably reproduce the full-search answer"
        ``min_cluster_size`` is an absolute count with no natural way to rescale across
        sample sizes, unlike DBSCAN's ``eps`` (a distance threshold that stays
        meaningful regardless of how many points were drawn from the same feature
        distribution). Agreement between a subsampled search and the unsampled
        (``tune_sample_fraction=1.0``) answer is not monotonic in subsample size --
        a larger subsample is not guaranteed to be a more faithful approximation of the
        full search than a smaller one. cuML's ``cluster_persistence_`` is not a usable
        substitute scoring signal for this search either: it returns exactly ``1.0`` for
        every cluster in every candidate regardless of ``min_cluster_size``, so it
        carries no discriminating information. The defaults below keep the search fast;
        they are not a guarantee of matching a full search's answer. Set
        ``tune_sample_fraction=1.0`` to disable subsampling and pay the full cost when
        clustering quality matters more than search speed.

    :param tune_sample_fraction: Fraction of the full dataset to use for the search,
        before clamping to ``[min_tune_sample_size, max_tune_sample_size]``.
    :param min_tune_sample_size: Floor on the search subsample size, so small datasets
        aren't subsampled down further than necessary.
    :param max_tune_sample_size: Ceiling on the search subsample size, so huge datasets
        don't erase the speed benefit of subsampling at all.
    """

    _FRACTIONS = (0.005, 0.01, 0.02, 0.03, 0.05)

    def __init__(
        self,
        tune_sample_fraction: float = 0.1,
        min_tune_sample_size: int = 2_000,
        max_tune_sample_size: int = 20_000,
    ):
        self.tune_sample_fraction = tune_sample_fraction
        self.min_tune_sample_size = min_tune_sample_size
        self.max_tune_sample_size = max_tune_sample_size

    def find_best_params(self, X: Array, estimator: Any) -> dict[str, Any]:
        backend = _resolve_for(estimator, X)
        xp = backend.xp
        X_native_full = to_native(X, backend=backend)
        n_samples_full = X_native_full.shape[0]

        tune_sample_size = int(
            min(
                self.max_tune_sample_size,
                max(self.min_tune_sample_size, self.tune_sample_fraction * n_samples_full),
            )
        )

        X_search = X
        X_native_search = X_native_full
        if n_samples_full > tune_sample_size:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n_samples_full, size=tune_sample_size, replace=False)
            X_native_search = X_native_full[xp.asarray(idx)]
            X_search = from_native(X_native_search, like=X, backend=backend)

        n_samples_search = X_native_search.shape[0]

        # Candidates are evaluated on the (possibly subsampled) search set but reported
        # as fractions of the FULL dataset, since min_cluster_size is an absolute count
        # that the winner is later applied to the full dataset -- reporting a count
        # sized for the search subset would silently change the targeted fraction.
        min_sizes_full = [max(10, int(f * n_samples_full)) for f in self._FRACTIONS]
        min_sizes_search = [max(10, int(f * n_samples_search)) for f in self._FRACTIONS]

        best_score = -1.0
        best_params: dict[str, Any] = {
            "min_cluster_size": min_sizes_full[0],
            "min_samples": min(10, min_sizes_full[0] // 5),
        }

        for size_full, size_search in zip(min_sizes_full, min_sizes_search, strict=True):
            search_params = {
                "min_cluster_size": size_search,
                "min_samples": min(10, size_search // 5),
            }
            candidate = clone(estimator).set_params(**search_params)
            labels = to_native(candidate.fit_predict(X_search), backend=backend)

            mask = labels != -1
            unique_labels = xp.unique(labels[mask])
            if len(unique_labels) < 2 or int(xp.sum(mask)) <= 1:
                continue

            score = float(backend.silhouette_score(X_native_search[mask], labels[mask]))
            if score > best_score:
                best_score = score
                best_params = {
                    "min_cluster_size": size_full,
                    "min_samples": min(10, size_full // 5),
                }

        return best_params


class EpsilonProminenceSearch:
    """Auto-tunes DBSCAN's ``eps``: ``n_eps_grid`` candidates, ``find_peaks``-based selection.

    Replicates the epsilon search from the MATLAB `bSTAB
    <https://github.com/TUHH-DYN/bSTAB>`_ toolbox's ``classify_solution.m``
    unsupervised branch: sweep an epsilon grid derived from the feature ranges,
    record the minimum per-sample silhouette score for each candidate, and pick
    the most prominent peak in that curve (falling back to the global maximum
    if no peak clears ``min_peak_height``).

    The search subsample size is ``clip(tune_sample_fraction * N, min_tune_sample_size,
    max_tune_sample_size)``. Unlike :class:`MinClusterSizeSearch`'s ``min_cluster_size``,
    ``eps`` is a distance threshold, not an absolute count, so it stays meaningful
    regardless of how many points were subsampled from the same feature distribution --
    subsampling this search is on much safer footing, and too small a subsample can
    still pick an ``eps`` that badly over-segments the result.

    !!! warning "Subsample safety does not guarantee the true attractor count"
        A large enough subsample keeps the chosen ``eps`` consistent with what an
        unsampled search would pick, but that is a separate question from whether the
        peak-finding selection recovers the true number of attractors. Both depend on
        how separable the feature space is for a given system, and this search can
        still over-segment a dataset with several attractors that are close together in
        feature space even when subsampling itself is not the cause.

    Reference:
        Stender, M., Hoffmann, N. bSTAB: an open-source software for computing
        the basin stability of multi-stable dynamical systems. Nonlinear Dyn
        107, 1451-1468 (2022). https://doi.org/10.1007/s11071-021-06786-5

    :param n_eps_grid: Number of epsilon candidates to evaluate.
    :param tune_sample_fraction: Fraction of the full dataset to use for the search,
        before clamping to ``[min_tune_sample_size, max_tune_sample_size]``.
    :param min_tune_sample_size: Floor on the search subsample size, so small datasets
        aren't subsampled down further than necessary.
    :param max_tune_sample_size: Ceiling on the search subsample size, so huge datasets
        don't erase the speed benefit of subsampling at all.
    :param min_peak_height: Minimum silhouette peak height for the peak finder.
    """

    def __init__(
        self,
        n_eps_grid: int = 200,
        tune_sample_fraction: float = 0.1,
        min_tune_sample_size: int = 5_000,
        max_tune_sample_size: int = 20_000,
        min_peak_height: float = 0.9,
    ):
        self.n_eps_grid = n_eps_grid
        self.tune_sample_fraction = tune_sample_fraction
        self.min_tune_sample_size = min_tune_sample_size
        self.max_tune_sample_size = max_tune_sample_size
        self.min_peak_height = min_peak_height

    def find_best_params(self, X: Any, estimator: Any) -> dict[str, Any]:
        backend = _resolve_for(estimator, X)
        xp = backend.xp
        features = to_native(X, backend=backend)
        min_samples: int = estimator.min_samples
        default_eps: float = float(getattr(estimator, "eps", 0.5))

        n_samples = features.shape[0]
        tune_sample_size = int(
            min(
                self.max_tune_sample_size,
                max(self.min_tune_sample_size, self.tune_sample_fraction * n_samples),
            )
        )
        if n_samples > tune_sample_size:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n_samples, size=tune_sample_size, replace=False)
            features = features[xp.asarray(idx)]

        distance_matrix = backend.pairwise_distances(features)

        feat_ranges = xp.max(features, axis=0) - xp.min(features, axis=0)
        min_range = float(xp.min(feat_ranges))

        if min_range <= 0:
            return {"eps": default_eps}

        eps_grid = xp.linspace(min_range / self.n_eps_grid, min_range, self.n_eps_grid)
        s_grid = xp.zeros(self.n_eps_grid)

        for i in range(self.n_eps_grid):
            eps_candidate = float(eps_grid[i])
            labels = backend.dbscan_cls(
                eps=eps_candidate, min_samples=min_samples, metric="precomputed"
            ).fit_predict(distance_matrix)

            valid_mask = labels != -1
            n_valid = int(xp.sum(valid_mask))
            n_clusters = len(xp.unique(labels[valid_mask])) if n_valid > 0 else 0

            if n_clusters >= 2 and n_valid > n_clusters:
                valid_idx = xp.where(valid_mask)[0]
                if backend.is_gpu:
                    # cuML's silhouette_samples has no "precomputed" metric (unlike
                    # sklearn's), so recompute from raw features on the GPU path.
                    s = backend.silhouette_samples(
                        features[valid_idx], labels[valid_mask], metric="euclidean"
                    )
                else:
                    s = backend.silhouette_samples(
                        distance_matrix[xp.ix_(valid_idx, valid_idx)],
                        labels[valid_mask],
                        metric="precomputed",
                    )
                s_grid[i] = float(xp.min(s))

        s_grid[0] = 0.0
        s_grid[-1] = 0.0

        s_grid_tensor = _to_torch(s_grid, backend)
        peaks, prominences = find_peaks(s_grid_tensor, float(self.min_peak_height))

        if peaks.numel() > 0:
            best_peak_idx = int(peaks[int(prominences.argmax().item())].item())
            return {"eps": float(eps_grid[best_peak_idx])}

        best_idx = int(xp.argmax(s_grid))
        return {"eps": float(eps_grid[best_idx])}
