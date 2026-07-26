"""Tests for the shared CPU/GPU backend resolution module."""

import importlib.util
from dataclasses import replace

import numpy as np
import pytest
import torch

import pybasin.predictors._device_backend as device_backend_module
from pybasin.predictors._device_backend import (
    UnsupportedConfigurationError,
    assign_noise_to_clusters,
    from_native,
    require_supported,
    resolve_backend,
    standardize,
    to_native,
    to_numpy,
)


def _cuml_available() -> bool:
    return importlib.util.find_spec("cuml") is not None


class TestResolveBackend:
    def test_numpy_array_resolves_cpu(self):
        backend = resolve_backend(np.zeros((5, 2)))
        assert backend.name == "cpu"
        assert not backend.is_gpu
        assert backend.fallback_reason is None

    def test_cpu_tensor_resolves_cpu(self):
        backend = resolve_backend(torch.zeros(5, 2))
        assert backend.name == "cpu"
        assert not backend.is_gpu

    def test_requested_cpu_forces_cpu(self):
        backend = resolve_backend(torch.zeros(5, 2), requested="cpu")
        assert backend.name == "cpu"

    def test_requested_gpu_raises_without_cuml(self):
        if _cuml_available():
            pytest.skip("cuML is installed; see test_requested_gpu_resolves_with_cuml")
        with pytest.raises(ImportError, match="cuML"):
            resolve_backend(np.zeros((5, 2)), requested="gpu")

    def test_cuda_without_cuml_raises_by_default(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA GPU")
        if _cuml_available():
            pytest.skip("cuML is installed; see test_cuda_resolves_gpu_with_cuml_installed")
        with pytest.raises(ImportError, match="cuML"):
            resolve_backend(torch.zeros(5, 2, device="cuda"))

    def test_cuda_without_cuml_falls_back_with_fallback_cpu(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA GPU")
        if _cuml_available():
            pytest.skip("cuML is installed; fallback='cpu' only matters when cuML is absent")
        backend = resolve_backend(torch.zeros(5, 2, device="cuda"), fallback="cpu")
        assert backend.name == "cpu"
        assert backend.fallback_reason is not None

    def test_requested_gpu_resolves_with_cuml(self):
        if not _cuml_available():
            pytest.skip("requires cuML installed")
        backend = resolve_backend(np.zeros((5, 2)), requested="gpu")
        assert backend.name == "gpu"
        assert backend.is_gpu

    def test_cuda_resolves_gpu_with_cuml_installed(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA GPU")
        if not _cuml_available():
            pytest.skip("requires cuML installed")
        backend = resolve_backend(torch.zeros(5, 2, device="cuda"))
        assert backend.name == "gpu"
        assert backend.fallback_reason is None

    def test_unsupported_device_warns_and_falls_back(self, monkeypatch: pytest.MonkeyPatch):
        def fake_device_kind(X: object) -> str:
            return "mps"

        monkeypatch.setattr(device_backend_module, "_device_kind", fake_device_kind)

        with pytest.warns(UserWarning, match="No GPU backend"):
            backend = resolve_backend(np.zeros(3))
        assert backend.name == "cpu"
        assert backend.fallback_reason is not None


class TestConversions:
    def test_to_native_cpu_tensor_to_numpy(self):
        backend = resolve_backend(torch.zeros(3))
        t = torch.tensor([1.0, 2.0, 3.0])
        native = to_native(t, backend=backend)
        assert isinstance(native, np.ndarray)
        np.testing.assert_allclose(native, [1.0, 2.0, 3.0])  # pyright: ignore[reportUnknownArgumentType]

    def test_to_native_passthrough_numpy(self):
        backend = resolve_backend(np.zeros(3))
        arr = np.array([1.0, 2.0])
        assert to_native(arr, backend=backend) is arr

    def test_to_numpy_from_tensor(self):
        backend = resolve_backend(torch.zeros(3))
        result = to_numpy(torch.tensor([1, 2, 3]), backend=backend)
        assert isinstance(result, np.ndarray)

    def test_from_native_mirrors_tensor_device(self):
        backend = resolve_backend(torch.zeros(3))
        like = torch.zeros(3)
        result = from_native(np.array([1, 2, 3]), like=like, backend=backend)
        assert isinstance(result, torch.Tensor)
        assert result.device == like.device

    def test_from_native_mirrors_numpy(self):
        backend = resolve_backend(np.zeros(3))
        like = np.zeros(3)
        result = from_native(np.array([1, 2, 3]), like=like, backend=backend)
        assert isinstance(result, np.ndarray)


class TestRequireSupported:
    def test_cpu_backend_never_raises(self):
        backend = resolve_backend(np.zeros(3))
        require_supported(
            "anything", gpu_supported={"euclidean"}, param_name="metric", backend=backend
        )

    def test_gpu_backend_raises_for_unsupported_value(self):
        backend = resolve_backend(np.zeros(3))
        gpu_backend = replace(backend, name="gpu", is_gpu=True)
        with pytest.raises(UnsupportedConfigurationError, match="metric"):
            require_supported(
                "manhattan", gpu_supported={"euclidean"}, param_name="metric", backend=gpu_backend
            )

    def test_gpu_backend_allows_supported_value(self):
        backend = resolve_backend(np.zeros(3))
        gpu_backend = replace(backend, name="gpu", is_gpu=True)
        require_supported(
            "euclidean", gpu_supported={"euclidean"}, param_name="metric", backend=gpu_backend
        )


class TestStandardize:
    def test_zero_mean_unit_ish_scale(self):
        x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        result = standardize(x, np)
        np.testing.assert_allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_constant_column_does_not_divide_by_zero(self):
        x = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        result = standardize(x, np)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result[:, 0], [0.0, 0.0, 0.0])


class TestAssignNoiseToClusters:
    def test_no_noise_is_noop(self):
        backend = resolve_backend(np.zeros(3))
        features = np.array([[0.0], [1.0], [2.0]])
        labels = np.array([0, 0, 1])
        result = assign_noise_to_clusters(features, labels, backend=backend)
        np.testing.assert_array_equal(result, labels)

    def test_noise_assigned_to_nearest_cluster(self):
        backend = resolve_backend(np.zeros(3))
        features = np.array([[0.0], [0.1], [10.0], [10.1], [0.05]])
        labels = np.array([0, 0, 1, 1, -1])
        result = assign_noise_to_clusters(features, labels, n_neighbors=2, backend=backend)
        assert result[4] == 0
