# pyright: basic
import numpy as np
import pytest
import torch
from sklearn.base import clone  # type: ignore[import-untyped]

from pybasin.feature_selector import DefaultFeatureSelector


def _make_features() -> np.ndarray:
    rng = np.random.default_rng(1)
    n = 300
    base = rng.standard_normal(n)
    return np.column_stack([
        np.zeros(n),  # zero variance -> dropped by variance step
        base,  # correlated pair -> one dropped by correlation step
        base + rng.standard_normal(n) * 0.01,
        rng.standard_normal(n),  # independent -> kept
        rng.standard_normal(n),  # independent -> kept
    ])


class TestInitIsAssignmentOnly:
    def test_defaults(self):
        selector = DefaultFeatureSelector()
        assert selector.variance_threshold == 0.01
        assert selector.correlation_threshold == 0.9
        assert selector.min_features == 3
        assert selector.backend == "auto"
        assert selector.fallback == "raise"

    def test_backend_forwarded_to_steps(self):
        selector = DefaultFeatureSelector(backend="cpu", fallback="cpu")
        assert selector.named_steps["variance"].backend == "cpu"
        assert selector.named_steps["variance"].fallback == "cpu"
        assert selector.named_steps["correlation"].backend == "cpu"
        assert selector.named_steps["correlation"].fallback == "cpu"

    def test_clone_roundtrip(self):
        selector = DefaultFeatureSelector(variance_threshold=0.05, backend="cpu")
        cloned = clone(selector)
        assert cloned.variance_threshold == 0.05
        assert cloned.backend == "cpu"


class TestFitTransformCpu:
    def test_two_stage_filtering(self):
        X = _make_features()
        selector = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        out = selector.fit_transform(X)

        support = selector.get_support()
        assert support.shape == (5,)
        assert not support[0], "zero-variance feature dropped"
        assert support[1] != support[2], "exactly one of the correlated pair survives"
        assert support[3] and support[4], "independent features kept"
        assert out.shape[1] == int(support.sum())

    def test_get_support_indices(self):
        X = _make_features()
        selector = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        selector.fit(X)

        mask = selector.get_support(indices=False)
        indices = selector.get_support(indices=True)
        np.testing.assert_array_equal(np.where(mask)[0], indices)


class TestDeviceNative:
    def test_cpu_tensor_in_cpu_tensor_out(self):
        X = torch.from_numpy(_make_features()).to(torch.float32)
        selector = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        out = selector.fit_transform(X)

        assert isinstance(out, torch.Tensor)
        assert out.device == X.device
        assert out.dtype == X.dtype

    def test_matches_numpy_result(self):
        X_np = _make_features().astype(np.float32)
        X_t = torch.from_numpy(X_np)

        sel_np = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        out_np = sel_np.fit_transform(X_np)

        sel_t = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        out_t = sel_t.fit_transform(X_t)

        np.testing.assert_array_equal(sel_np.get_support(), sel_t.get_support())
        np.testing.assert_allclose(out_np, out_t.numpy())


class TestGpuBackend:
    def test_cuda_tensor_stays_on_device_and_matches_cpu(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA GPU")
        pytest.importorskip("cupy")

        X_np = _make_features().astype(np.float32)
        X_gpu = torch.from_numpy(X_np).cuda()

        sel_cpu = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        out_cpu = sel_cpu.fit_transform(X_np)

        sel_gpu = DefaultFeatureSelector(correlation_threshold=0.95, min_features=2)
        out_gpu = sel_gpu.fit_transform(X_gpu)

        assert isinstance(out_gpu, torch.Tensor)
        assert out_gpu.device.type == "cuda"
        assert out_gpu.dtype == torch.float32
        np.testing.assert_array_equal(sel_cpu.get_support(), sel_gpu.get_support())
        np.testing.assert_allclose(out_cpu, out_gpu.cpu().numpy(), rtol=1e-4, atol=1e-5)
