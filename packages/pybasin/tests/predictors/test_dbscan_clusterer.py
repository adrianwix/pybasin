"""Tests for DBSCANClusterer."""

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.cluster import DBSCAN

import pybasin.predictors.dbscan_clusterer as dbscan_module
from pybasin.predictors._device_backend import DeviceBackend, UnsupportedConfigurationError
from pybasin.predictors._device_backend import resolve_backend as real_resolve_backend
from pybasin.predictors.dbscan_clusterer import DBSCANClusterer


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
        clusterer = DBSCANClusterer()
        assert clusterer.eps == 0.5
        assert clusterer.min_samples == 10
        assert clusterer.metric == "euclidean"
        assert not hasattr(clusterer, "estimator_")

    def test_clone_roundtrip(self):
        clusterer = DBSCANClusterer(eps=1.5, assign_noise=True)
        cloned = clone(clusterer)
        assert cloned.eps == 1.5
        assert cloned.assign_noise is True


class TestFitPredictCpu:
    def test_matches_plain_sklearn(self):
        X = make_blobs()
        ours = DBSCANClusterer(eps=1.0, min_samples=5).fit_predict(X)
        reference = DBSCAN(eps=1.0, min_samples=5).fit_predict(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            X
        )
        np.testing.assert_array_equal(ours, reference)  # pyright: ignore[reportUnknownArgumentType]

    def test_assign_noise_removes_minus_one(self):
        X = make_blobs()
        X_with_outlier = np.vstack([X, [[500.0, 500.0]]])
        clusterer = DBSCANClusterer(eps=1.0, min_samples=5, assign_noise=True)
        labels = clusterer.fit_predict(X_with_outlier)
        assert -1 not in labels


class TestMetricValidation:
    def test_gpu_backend_rejects_unsupported_metric(self, monkeypatch: pytest.MonkeyPatch):
        X = make_blobs()
        monkeypatch.setattr(dbscan_module, "resolve_backend", fake_resolve_backend_gpu)

        with pytest.raises(UnsupportedConfigurationError, match="metric"):
            DBSCANClusterer(eps=1.0, metric="manhattan").fit_predict(X)

    def test_gpu_backend_allows_precomputed_metric_family(self, monkeypatch: pytest.MonkeyPatch):
        X = make_blobs()
        monkeypatch.setattr(dbscan_module, "resolve_backend", fake_resolve_backend_gpu)

        labels = DBSCANClusterer(eps=1.0, metric="cosine").fit_predict(X)
        assert len(labels) == 100
