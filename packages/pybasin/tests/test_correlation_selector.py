# pyright: basic
import numpy as np
import pytest
import torch
from sklearn.base import clone  # type: ignore[import-untyped]

from pybasin.feature_selector.correlation_selector import CorrelationSelector


def _make_correlated_features() -> np.ndarray:
    """Create two highly correlated features plus two independent features (4 total)."""
    rng: np.random.Generator = np.random.default_rng(0)
    n: int = 500
    base: np.ndarray = rng.standard_normal(n)
    f1: np.ndarray = base + rng.standard_normal(n) * 0.01
    f2: np.ndarray = base + rng.standard_normal(n) * 0.01
    f3: np.ndarray = rng.standard_normal(n)
    f4: np.ndarray = rng.standard_normal(n)
    return np.column_stack([f1, f2, f3, f4])


def test_removes_one_of_correlated_pair() -> None:
    """When two features are nearly identical, exactly one is removed."""
    X: np.ndarray = _make_correlated_features()
    selector = CorrelationSelector(threshold=0.95, min_features=2)
    X_out: np.ndarray = selector.fit_transform(X)  # type: ignore[reportUnknownMemberType]

    assert X_out.shape[1] == 3
    support: np.ndarray = selector.get_support(indices=False)
    assert support.sum() == 3
    assert support[2], "Independent feature f3 should be kept"
    assert support[3], "Independent feature f4 should be kept"


def test_keeps_all_when_below_threshold() -> None:
    """No features removed when all correlations are below threshold."""
    rng: np.random.Generator = np.random.default_rng(1)
    X: np.ndarray = np.column_stack([
        rng.standard_normal(200),
        rng.standard_normal(200),
        rng.standard_normal(200),
        rng.standard_normal(200),
    ])
    selector = CorrelationSelector(threshold=0.9, min_features=2)
    X_out: np.ndarray = selector.fit_transform(X)  # type: ignore[reportUnknownMemberType]

    assert X_out.shape[1] == 4


def test_drops_more_globally_redundant_feature() -> None:
    """The feature with higher mean absolute correlation is dropped.

    Setup: f0 correlates with f1 and f2 (high mean abs corr).
    f1 only correlates with f0 (lower mean abs corr).
    f2 only correlates with f0 (lower mean abs corr).
    f3 is independent.
    f0 should be dropped first because it's globally more redundant.
    """
    rng: np.random.Generator = np.random.default_rng(2)
    n: int = 1000
    base: np.ndarray = rng.standard_normal(n)
    f0: np.ndarray = base
    f1: np.ndarray = base + rng.standard_normal(n) * 0.01
    f2: np.ndarray = base + rng.standard_normal(n) * 0.01
    f3: np.ndarray = rng.standard_normal(n)
    X: np.ndarray = np.column_stack([f0, f1, f2, f3])

    selector = CorrelationSelector(threshold=0.95, min_features=2)
    selector.fit(X)

    support: np.ndarray = selector.get_support(indices=False)

    assert not support[0], "f0 (highest mean abs corr) should be dropped first"
    assert support[3], "f3 (independent) should be kept"


def test_min_features_respected() -> None:
    """Removal stops when min_features is reached."""
    rng: np.random.Generator = np.random.default_rng(3)
    base: np.ndarray = rng.standard_normal(300)
    X: np.ndarray = np.column_stack([
        base + rng.standard_normal(300) * 0.001,
        base + rng.standard_normal(300) * 0.001,
        base + rng.standard_normal(300) * 0.001,
    ])

    selector = CorrelationSelector(threshold=0.5, min_features=2)
    X_out: np.ndarray = selector.fit_transform(X)  # type: ignore[reportUnknownMemberType]

    assert X_out.shape[1] == 2


def test_min_features_with_fewer_input_features() -> None:
    """When input has fewer features than min_features, all are kept."""
    X: np.ndarray = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    selector = CorrelationSelector(threshold=0.5, min_features=3)
    X_out: np.ndarray = selector.fit_transform(X)  # type: ignore[reportUnknownMemberType]

    assert X_out.shape[1] == 2
    assert selector.get_support(indices=False).all()


def test_get_support_indices() -> None:
    """get_support(indices=True) returns integer indices."""
    X: np.ndarray = _make_correlated_features()

    selector = CorrelationSelector(threshold=0.95, min_features=2)
    selector.fit(X)

    indices: np.ndarray = selector.get_support(indices=True)
    mask: np.ndarray = selector.get_support(indices=False)

    assert len(indices) == mask.sum()
    np.testing.assert_array_equal(np.where(mask)[0], indices)


def test_transform_is_consistent_with_support() -> None:
    """transform output columns match get_support mask."""
    X: np.ndarray = _make_correlated_features()
    selector = CorrelationSelector(threshold=0.95, min_features=2)
    X_out: np.ndarray = selector.fit_transform(X)  # type: ignore[reportUnknownMemberType]

    support: np.ndarray = selector.get_support(indices=False)
    np.testing.assert_array_equal(X_out, X[:, support])


def test_order_independent_of_column_position() -> None:
    """Removal decision depends on global redundancy, not column order.

    Shuffling columns should not change which underlying feature is dropped.
    """
    rng: np.random.Generator = np.random.default_rng(5)
    n: int = 1000
    base: np.ndarray = rng.standard_normal(n)
    f_redundant: np.ndarray = base
    f_copy: np.ndarray = base + rng.standard_normal(n) * 0.005
    f_independent_a: np.ndarray = rng.standard_normal(n)
    f_independent_b: np.ndarray = rng.standard_normal(n)

    X_original: np.ndarray = np.column_stack([
        f_redundant,
        f_copy,
        f_independent_a,
        f_independent_b,
    ])
    X_shuffled: np.ndarray = np.column_stack([
        f_independent_a,
        f_copy,
        f_independent_b,
        f_redundant,
    ])

    sel_orig = CorrelationSelector(threshold=0.95, min_features=2)
    sel_orig.fit(X_original)

    sel_shuf = CorrelationSelector(threshold=0.95, min_features=2)
    sel_shuf.fit(X_shuffled)

    assert sel_orig.get_support(indices=False).sum() == sel_shuf.get_support(indices=False).sum()
    assert sel_orig.get_support(indices=False)[2], "Independent feature kept in original"
    assert sel_orig.get_support(indices=False)[3], "Independent feature kept in original"
    assert sel_shuf.get_support(indices=False)[0], "Independent feature kept in shuffled"
    assert sel_shuf.get_support(indices=False)[2], "Independent feature kept in shuffled"


class TestInitIsAssignmentOnly:
    def test_defaults(self):
        selector = CorrelationSelector()
        assert selector.threshold == 0.9
        assert selector.min_features == 3
        assert selector.backend == "auto"
        assert selector.fallback == "raise"
        assert not hasattr(selector, "support_")

    def test_clone_roundtrip(self):
        selector = CorrelationSelector(threshold=0.8, min_features=2, backend="cpu")
        cloned = clone(selector)
        assert cloned.threshold == 0.8
        assert cloned.min_features == 2
        assert cloned.backend == "cpu"


class TestDeviceNative:
    """CorrelationSelector exposes `.backend`, the marker BSE/TemplateIntegrator use to
    decide whether a selector accepts a tensor directly instead of forcing numpy."""

    def test_has_backend_attribute(self):
        assert hasattr(CorrelationSelector(), "backend")

    def test_cpu_tensor_in_cpu_tensor_out(self):
        X = torch.from_numpy(_make_correlated_features()).to(torch.float32)
        selector = CorrelationSelector(threshold=0.95, min_features=2)
        out = selector.fit_transform(X)

        assert isinstance(out, torch.Tensor)
        assert out.device == X.device
        assert out.dtype == X.dtype
        assert selector.backend_ == "cpu"

    def test_matches_numpy_result(self):
        X_np = _make_correlated_features().astype(np.float32)
        X_t = torch.from_numpy(X_np)

        sel_np = CorrelationSelector(threshold=0.95, min_features=2)
        out_np = sel_np.fit_transform(X_np)

        sel_t = CorrelationSelector(threshold=0.95, min_features=2)
        out_t = sel_t.fit_transform(X_t)

        np.testing.assert_array_equal(sel_np.get_support(), sel_t.get_support())
        np.testing.assert_allclose(out_np, out_t.numpy())

    def test_get_support_is_always_host_numpy(self):
        X = torch.from_numpy(_make_correlated_features()).to(torch.float32)
        selector = CorrelationSelector(threshold=0.95, min_features=2)
        selector.fit(X)
        assert isinstance(selector.get_support(), np.ndarray)
        assert isinstance(selector.get_support(indices=True), np.ndarray)


class TestGpuBackend:
    def test_cuda_tensor_stays_on_device_and_matches_cpu(self):
        if not torch.cuda.is_available():
            pytest.skip("requires a CUDA GPU")
        pytest.importorskip("cupy")

        X_np = _make_correlated_features().astype(np.float32)
        X_gpu = torch.from_numpy(X_np).cuda()

        sel_cpu = CorrelationSelector(threshold=0.95, min_features=2)
        out_cpu = sel_cpu.fit_transform(X_np)

        sel_gpu = CorrelationSelector(threshold=0.95, min_features=2)
        out_gpu = sel_gpu.fit_transform(X_gpu)

        assert sel_gpu.backend_ == "gpu"
        assert isinstance(out_gpu, torch.Tensor)
        assert out_gpu.device.type == "cuda"
        assert out_gpu.dtype == torch.float32
        np.testing.assert_array_equal(sel_cpu.get_support(), sel_gpu.get_support())
        np.testing.assert_allclose(out_cpu, out_gpu.cpu().numpy(), rtol=1e-4, atol=1e-5)
