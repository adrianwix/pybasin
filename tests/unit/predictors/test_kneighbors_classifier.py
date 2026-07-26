"""Tests for the pybasin KNeighborsClassifier wrapper."""

from dataclasses import replace
from typing import Any, Literal

import numpy as np
import pytest
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier

import pybasin.predictors.kneighbors_classifier as knn_module
from pybasin.predictors._device_backend import DeviceBackend, UnsupportedConfigurationError
from pybasin.predictors._device_backend import resolve_backend as real_resolve_backend
from pybasin.predictors.kneighbors_classifier import KNeighborsClassifier


def make_training_data() -> tuple[np.ndarray, list[str]]:
    X = np.array([[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 9.9]])
    y = ["a", "a", "b", "b"]
    return X, y


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
        clf = KNeighborsClassifier()
        assert clf.n_neighbors == 5
        assert clf.weights == "uniform"
        assert clf.algorithm == "auto"
        assert not hasattr(clf, "estimator_")

    def test_clone_roundtrip(self):
        clf = KNeighborsClassifier(n_neighbors=3)
        cloned = clone(clf)
        assert cloned.n_neighbors == 3


class TestFitPredictCpu:
    def test_string_labels_roundtrip(self):
        X, y = make_training_data()
        clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
        preds = clf.predict(X)
        assert list(preds) == y
        assert preds.dtype == object or preds.dtype.kind in "US"

    def test_matches_plain_sklearn_after_encoding(self):
        X, y = make_training_data()
        ours = KNeighborsClassifier(n_neighbors=1).fit(X, y).predict(X)

        classes, y_encoded = np.unique(y, return_inverse=True)
        reference_encoded = (
            SklearnKNeighborsClassifier(n_neighbors=1)
            .fit(X, y_encoded)  # pyright: ignore[reportUnknownMemberType]
            .predict(X)  # pyright: ignore[reportUnknownMemberType]
        )
        reference = classes[reference_encoded]

        np.testing.assert_array_equal(ours, reference)  # pyright: ignore[reportUnknownArgumentType]

    def test_predict_before_fit_raises(self):
        clf = KNeighborsClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(np.zeros((1, 2)))

    def test_fit_resolves_backend_predict_reuses_it(self):
        X, y = make_training_data()
        clf = KNeighborsClassifier(n_neighbors=1).fit(X, y)
        assert clf.backend_ == "cpu"
        clf.predict(X)
        assert clf.backend_ == "cpu"


class TestAlgorithmValidation:
    def test_cpu_backend_allows_kd_tree(self):
        X, y = make_training_data()
        KNeighborsClassifier(n_neighbors=1, algorithm="kd_tree").fit(X, y)

    def test_gpu_backend_rejects_kd_tree(self, monkeypatch: pytest.MonkeyPatch):
        X, y = make_training_data()
        monkeypatch.setattr(knn_module, "resolve_backend", fake_resolve_backend_gpu)

        with pytest.raises(UnsupportedConfigurationError, match="algorithm"):
            KNeighborsClassifier(n_neighbors=1, algorithm="kd_tree").fit(X, y)

    def test_gpu_backend_allows_brute(self, monkeypatch: pytest.MonkeyPatch):
        X, y = make_training_data()
        monkeypatch.setattr(knn_module, "resolve_backend", fake_resolve_backend_gpu)

        clf = KNeighborsClassifier(n_neighbors=1, algorithm="brute").fit(X, y)
        preds = clf.predict(X)
        assert list(preds) == y
