"""Tests for the shared CPU/GPU array-backend core (pybasin._device_backend)."""

import importlib.util

import numpy as np
import pytest
import torch

import pybasin._device_backend as device_backend_module
from pybasin._device_backend import (
    from_native,
    resolve_array_backend,
    to_native,
    to_numpy,
)


def _cupy_available() -> bool:
    return importlib.util.find_spec("cupy") is not None


class TestResolveArrayBackend:
    def test_numpy_array_resolves_cpu(self):
        backend = resolve_array_backend(np.zeros((5, 2)))
        assert backend.name == "cpu"
        assert not backend.is_gpu
        assert backend.fallback_reason is None
        assert backend.xp is np

    def test_cpu_tensor_resolves_cpu(self):
        backend = resolve_array_backend(torch.zeros(5, 2))
        assert backend.name == "cpu"
        assert not backend.is_gpu

    def test_requested_cpu_forces_cpu(self):
        backend = resolve_array_backend(torch.zeros(5, 2), requested="cpu")
        assert backend.name == "cpu"

    def test_requested_gpu_raises_without_cupy(self):
        if _cupy_available():
            pytest.skip("cupy is installed; see test_requested_gpu_resolves_with_cupy")
        with pytest.raises(ImportError, match="cupy"):
            resolve_array_backend(np.zeros((5, 2)), requested="gpu")

    def test_requested_gpu_resolves_with_cupy(self):
        if not _cupy_available():
            pytest.skip("requires cupy installed")
        backend = resolve_array_backend(np.zeros((5, 2)), requested="gpu")
        assert backend.name == "gpu"
        assert backend.is_gpu

    def test_cuda_tensor_resolves_gpu_with_cupy_installed(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA GPU")
        if not _cupy_available():
            pytest.skip("requires cupy installed")
        backend = resolve_array_backend(torch.zeros(5, 2, device="cuda"))
        assert backend.name == "gpu"
        assert backend.fallback_reason is None

    def test_cuda_without_cupy_falls_back_with_fallback_cpu(self, monkeypatch: pytest.MonkeyPatch):
        def fake_gpu_array_backend() -> device_backend_module.ArrayBackend:
            raise ImportError("cupy not installed")

        def fake_device_kind(X: object) -> str:
            return "cuda"

        monkeypatch.setattr(device_backend_module, "_gpu_array_backend", fake_gpu_array_backend)
        monkeypatch.setattr(device_backend_module, "device_kind", fake_device_kind)

        backend = resolve_array_backend(torch.zeros(5, 2), fallback="cpu")
        assert backend.name == "cpu"
        assert backend.fallback_reason is not None

    def test_unsupported_device_warns_and_falls_back(self, monkeypatch: pytest.MonkeyPatch):
        def fake_device_kind(X: object) -> str:
            return "mps"

        monkeypatch.setattr(device_backend_module, "device_kind", fake_device_kind)

        with pytest.warns(UserWarning, match="No GPU backend"):
            backend = resolve_array_backend(np.zeros(3))
        assert backend.name == "cpu"
        assert backend.fallback_reason is not None


class TestConversions:
    def test_to_native_cpu_tensor_to_numpy(self):
        backend = resolve_array_backend(torch.zeros(3))
        t = torch.tensor([1.0, 2.0, 3.0])
        native = to_native(t, backend=backend)
        assert isinstance(native, np.ndarray)
        np.testing.assert_allclose(native, [1.0, 2.0, 3.0])  # pyright: ignore[reportUnknownArgumentType]

    def test_to_native_passthrough_numpy(self):
        backend = resolve_array_backend(np.zeros(3))
        arr = np.array([1.0, 2.0])
        assert to_native(arr, backend=backend) is arr

    def test_to_numpy_from_tensor(self):
        backend = resolve_array_backend(torch.zeros(3))
        result = to_numpy(torch.tensor([1, 2, 3]), backend=backend)
        assert isinstance(result, np.ndarray)

    def test_from_native_mirrors_tensor_device(self):
        backend = resolve_array_backend(torch.zeros(3))
        like = torch.zeros(3)
        result = from_native(np.array([1, 2, 3]), like=like, backend=backend)
        assert isinstance(result, torch.Tensor)
        assert result.device == like.device

    def test_from_native_mirrors_numpy(self):
        backend = resolve_array_backend(np.zeros(3))
        like = np.zeros(3)
        result = from_native(np.array([1, 2, 3]), like=like, backend=backend)
        assert isinstance(result, np.ndarray)

    def test_gpu_roundtrip_preserves_dtype_and_device(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA GPU")
        if not _cupy_available():
            pytest.skip("requires cupy installed")
        backend = resolve_array_backend(torch.zeros(3, device="cuda"))
        t = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float32)
        native = to_native(t, backend=backend)
        assert type(native).__module__.startswith("cupy")
        back = from_native(native, like=t, backend=backend)
        assert isinstance(back, torch.Tensor)
        assert back.device == t.device
        assert back.dtype == t.dtype
        torch.testing.assert_close(back, t)
