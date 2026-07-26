# pyright: basic
import numpy as np
import pytest
import torch
from sklearn.base import clone  # type: ignore[import-untyped]
from sklearn.feature_selection import VarianceThreshold

from pybasin.feature_selector.variance_threshold_selector import VarianceThresholdSelector


def _make_features() -> np.ndarray:
    rng = np.random.default_rng(0)
    n = 200
    return np.column_stack([
        np.zeros(n),  # zero variance -> always dropped
        rng.standard_normal(n) * 0.05,  # low variance -> dropped at default threshold
        rng.standard_normal(n),  # kept
        rng.standard_normal(n) * 10,  # kept
    ])


class TestInitIsAssignmentOnly:
    def test_defaults(self):
        selector = VarianceThresholdSelector()
        assert selector.threshold == 0.01
        assert selector.backend == "auto"
        assert selector.fallback == "raise"
        assert not hasattr(selector, "support_")

    def test_clone_roundtrip(self):
        selector = VarianceThresholdSelector(threshold=0.05, backend="cpu")
        cloned = clone(selector)
        assert cloned.threshold == 0.05
        assert cloned.backend == "cpu"


class TestMatchesSklearn:
    def test_support_matches_reference(self):
        X = _make_features()
        ours = VarianceThresholdSelector(threshold=0.01)
        ours.fit(X)

        reference = VarianceThreshold(threshold=0.01).fit(X)  # type: ignore[reportUnknownMemberType]

        np.testing.assert_array_equal(ours.get_support(), reference.get_support())  # type: ignore[reportUnknownArgumentType]

    def test_transform_matches_reference(self):
        X = _make_features()
        ours = VarianceThresholdSelector(threshold=0.01).fit_transform(X)
        reference = VarianceThreshold(threshold=0.01).fit_transform(X)  # type: ignore[reportUnknownMemberType]

        np.testing.assert_allclose(ours, reference)  # type: ignore[reportUnknownArgumentType]

    def test_raises_when_no_feature_survives(self):
        X = np.zeros((50, 3))
        with pytest.raises(ValueError, match="variance threshold"):
            VarianceThresholdSelector(threshold=0.01).fit(X)

    def test_get_support_indices(self):
        X = _make_features()
        selector = VarianceThresholdSelector(threshold=0.01)
        selector.fit(X)

        indices = selector.get_support(indices=True)
        mask = selector.get_support(indices=False)
        np.testing.assert_array_equal(np.where(mask)[0], indices)


class TestDeviceNative:
    def test_has_backend_attribute(self):
        assert hasattr(VarianceThresholdSelector(), "backend")

    def test_cpu_tensor_in_cpu_tensor_out(self):
        X = torch.from_numpy(_make_features()).to(torch.float32)
        selector = VarianceThresholdSelector(threshold=0.01)
        out = selector.fit_transform(X)

        assert isinstance(out, torch.Tensor)
        assert out.device == X.device
        assert out.dtype == X.dtype
        assert selector.backend_ == "cpu"

    def test_matches_numpy_result(self):
        X_np = _make_features().astype(np.float32)
        X_t = torch.from_numpy(X_np)

        sel_np = VarianceThresholdSelector(threshold=0.01)
        out_np = sel_np.fit_transform(X_np)

        sel_t = VarianceThresholdSelector(threshold=0.01)
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

        sel_cpu = VarianceThresholdSelector(threshold=0.01)
        out_cpu = sel_cpu.fit_transform(X_np)

        sel_gpu = VarianceThresholdSelector(threshold=0.01)
        out_gpu = sel_gpu.fit_transform(X_gpu)

        assert sel_gpu.backend_ == "gpu"
        assert isinstance(out_gpu, torch.Tensor)
        assert out_gpu.device.type == "cuda"
        assert out_gpu.dtype == torch.float32
        np.testing.assert_array_equal(sel_cpu.get_support(), sel_gpu.get_support())
        np.testing.assert_allclose(out_cpu, out_gpu.cpu().numpy(), rtol=1e-4, atol=1e-5)
