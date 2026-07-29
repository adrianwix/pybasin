"""Tests for HDBSCANClusterer."""

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.cluster import HDBSCAN  # type: ignore[attr-defined]

import pybasin.predictors.hdbscan_clusterer as hdbscan_module
from pybasin.predictors._device_backend import (
    DeviceBackend,
    UnsupportedConfigurationError,
)
from pybasin.predictors._device_backend import resolve_backend as real_resolve_backend
from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer


def make_blobs(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cluster_a = rng.normal(loc=(0, 0), scale=0.1, size=(50, 2))
    cluster_b = rng.normal(loc=(10, 10), scale=0.1, size=(50, 2))
    return np.vstack([cluster_a, cluster_b])


def fake_resolve_backend_gpu(
    X: Any,
    *,
    requested: Literal["auto", "cpu", "gpu"] = "auto",
    fallback: Literal["raise", "cpu"] = "raise",
) -> DeviceBackend:
    backend = real_resolve_backend(X, requested=requested, fallback=fallback)
    return replace(backend, name="gpu", is_gpu=True)


class TestInitIsAssignmentOnly:
    def test_defaults(self):
        clusterer = HDBSCANClusterer()
        assert clusterer.min_cluster_size == 50
        assert clusterer.min_samples == 10
        assert clusterer.metric == "euclidean"
        assert clusterer.assign_noise is False
        assert clusterer.backend == "auto"
        assert clusterer.fallback == "raise"
        assert not hasattr(clusterer, "estimator_")

    def test_get_set_params_roundtrip(self):
        clusterer = HDBSCANClusterer(min_cluster_size=20)
        params = clusterer.get_params()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        assert params["min_cluster_size"] == 20
        clusterer.set_params(min_cluster_size=30)  # pyright: ignore[reportUnknownMemberType]
        assert clusterer.min_cluster_size == 30

    def test_clone_roundtrip(self):
        clusterer = HDBSCANClusterer(min_cluster_size=15, assign_noise=True)
        cloned = clone(clusterer)
        assert cloned.min_cluster_size == 15
        assert cloned.assign_noise is True
        assert cloned is not clusterer


class TestFitPredictCpu:
    def test_finds_two_clusters(self):
        X = make_blobs()
        clusterer = HDBSCANClusterer(min_cluster_size=10)
        labels = clusterer.fit_predict(X)
        assert clusterer.backend_ == "cpu"
        assert len(labels) == 100
        assert len(set(labels[labels != -1])) == 2

    def test_matches_plain_sklearn(self):
        X = make_blobs()
        ours = HDBSCANClusterer(min_cluster_size=10).fit_predict(X)
        reference = HDBSCAN(min_cluster_size=10).fit_predict(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            X
        )
        np.testing.assert_array_equal(ours, reference)  # pyright: ignore[reportUnknownArgumentType]

    def test_assign_noise_removes_minus_one(self):
        X = make_blobs()
        X_with_outlier = np.vstack([X, [[500.0, 500.0]]])
        clusterer = HDBSCANClusterer(min_cluster_size=10, assign_noise=True)
        labels = clusterer.fit_predict(X_with_outlier)
        assert -1 not in labels


class TestMetricValidation:
    def test_cpu_backend_allows_any_metric(self):
        X = make_blobs()
        HDBSCANClusterer(min_cluster_size=10, metric="manhattan").fit_predict(X)

    def test_gpu_backend_rejects_non_euclidean_metric(self, monkeypatch: pytest.MonkeyPatch):
        X = make_blobs()
        monkeypatch.setattr(hdbscan_module, "resolve_backend", fake_resolve_backend_gpu)

        with pytest.raises(UnsupportedConfigurationError, match="metric"):
            HDBSCANClusterer(min_cluster_size=10, metric="manhattan").fit_predict(X)

    def test_gpu_backend_allows_euclidean_metric(self, monkeypatch: pytest.MonkeyPatch):
        X = make_blobs()
        monkeypatch.setattr(hdbscan_module, "resolve_backend", fake_resolve_backend_gpu)

        labels = HDBSCANClusterer(min_cluster_size=10, metric="euclidean").fit_predict(X)
        assert len(labels) == 100
