"""Tests for TunedClusterer, MinClusterSizeSearch, and EpsilonProminenceSearch."""

import numpy as np
from sklearn.cluster import HDBSCAN  # type: ignore[attr-defined]
from sklearn.metrics import (
    silhouette_score as sklearn_silhouette_score,  # type: ignore[import-untyped]
)

from pybasin.predictors.dbscan_clusterer import DBSCANClusterer
from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer
from pybasin.predictors.tuned_clusterer import (
    EpsilonProminenceSearch,
    MinClusterSizeSearch,
    TunedClusterer,
)


def make_blobs(n_per_cluster: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=(0, 0), scale=0.2, size=(n_per_cluster, 2))
    cluster_b = rng.normal(loc=(10, 10), scale=0.2, size=(n_per_cluster, 2))
    return np.vstack([cluster_a, cluster_b])


def sklearn_hdbscan_fit_predict(
    X: np.ndarray, *, min_cluster_size: int, min_samples: int
) -> np.ndarray:
    """Real sklearn.cluster.HDBSCAN, wrapped for a byte-for-byte legacy-parity check.

    sklearn.cluster.HDBSCAN lacks complete stubs (see hdbscan_clusterer.py's own
    `# type: ignore[attr-defined]`); the return annotation is the one place that
    needs to swallow the resulting Unknown type.
    """
    return HDBSCAN(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        min_cluster_size=min_cluster_size, min_samples=min_samples
    ).fit_predict(X)


def sklearn_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    return float(sklearn_silhouette_score(X, labels))


class TestMinClusterSizeSearchLegacyParity:
    def test_matches_former_auto_tune_true_behavior(self):
        X = make_blobs()

        tuned = TunedClusterer(estimator=HDBSCANClusterer(), search=MinClusterSizeSearch())
        tuned_labels = tuned.fit_predict(X)

        # Replicates the exact control flow HDBSCANClusterer.auto_tune=True used to run.
        n_samples = len(X)
        min_sizes = [
            max(10, int(0.005 * n_samples)),
            max(25, int(0.01 * n_samples)),
            max(50, int(0.02 * n_samples)),
            max(100, int(0.03 * n_samples)),
            max(150, int(0.05 * n_samples)),
        ]

        best_score = -1.0
        best_size = min_sizes[0]
        for size in min_sizes:
            labels = sklearn_hdbscan_fit_predict(
                X, min_cluster_size=size, min_samples=min(10, size // 5)
            )
            mask = labels != -1
            if len(np.unique(labels[mask])) >= 2 and mask.sum() > 1:
                score = sklearn_silhouette(X[mask], labels[mask])
                if score > best_score:
                    best_score = score
                    best_size = size

        assert tuned.best_params_["min_cluster_size"] == best_size
        expected_labels = sklearn_hdbscan_fit_predict(
            X, min_cluster_size=best_size, min_samples=min(10, best_size // 5)
        )
        np.testing.assert_array_equal(tuned_labels, expected_labels)


class TestEpsilonProminenceSearchLegacyParity:
    def test_finds_two_clusters(self):
        X = make_blobs()
        tuned = TunedClusterer(estimator=DBSCANClusterer(), search=EpsilonProminenceSearch())
        labels = tuned.fit_predict(X)
        assert len(set(labels[labels != -1])) == 2
        assert "eps" in tuned.best_params_


class TestTunedClustererState:
    def test_exposes_best_params_and_fitted_estimator(self):
        X = make_blobs()
        tuned = TunedClusterer(
            estimator=HDBSCANClusterer(assign_noise=True), search=MinClusterSizeSearch()
        )
        tuned.fit_predict(X)
        assert hasattr(tuned, "best_params_")
        assert hasattr(tuned, "estimator_")
        assert -1 not in tuned.estimator_.fit_predict(X)
