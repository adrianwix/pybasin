# pyright: basic
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn


def _load_experiment_module():
    module_path = Path(__file__).with_name("backend_agnostic_features.py")
    sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("backend_agnostic_features", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment_module()
PartialAutocorrelation = experiment.PartialAutocorrelation
PartialAutocorrelationParams = experiment.PartialAutocorrelationParams
backend = experiment.backend
set_backend = experiment.set_backend


PARAMS: list[PartialAutocorrelationParams] = [
    {"lag": 0},
    {"lag": 1},
    {"lag": 2},
    {"lag": 5},
    {"lag": 8},
]


def _sample_numpy() -> np.ndarray:
    rng = np.random.default_rng(20260720)
    trend = np.linspace(-0.2, 0.4, 32, dtype=np.float64)
    periodic = np.sin(np.linspace(0, 4 * np.pi, 32, dtype=np.float64))
    noise = rng.normal(scale=0.05, size=(3, 2, 32))
    return (trend.reshape(1, 1, -1) + periodic.reshape(1, 1, -1) + noise).astype(np.float32)


def _torch_sample() -> Tensor:
    return torch.from_numpy(_sample_numpy())


def test_names_are_backend_independent() -> None:
    feature = PartialAutocorrelation(PARAMS)
    assert feature.names == [
        "partial_autocorrelation__lag_0",
        "partial_autocorrelation__lag_1",
        "partial_autocorrelation__lag_2",
        "partial_autocorrelation__lag_5",
        "partial_autocorrelation__lag_8",
    ]


def test_numpy_and_torch_are_numerically_close() -> None:
    x_np = _sample_numpy()
    set_backend("numpy")
    expected = PartialAutocorrelation(PARAMS)(x_np)
    set_backend("torch")
    actual = PartialAutocorrelation(PARAMS)(torch.from_numpy(x_np)).detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_torch_compiles_fullgraph() -> None:
    set_backend("torch")
    x = _torch_sample()
    feature = PartialAutocorrelation(PARAMS)
    compiled = torch.compile(feature, fullgraph=True, dynamic=True, backend="eager")
    torch.testing.assert_close(compiled(x), feature(x), atol=2e-5, rtol=2e-5)


class _FeatureModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = PartialAutocorrelation(PARAMS)

    def forward(self, x: Tensor) -> Tensor:
        return self.feature(x)


def test_torch_exports() -> None:
    set_backend("torch")
    x = _torch_sample()
    exported = torch.export.export(_FeatureModule(), (x,))
    actual = exported.module()(x)
    expected = PartialAutocorrelation(PARAMS)(x)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_jax_jits_and_matches_numpy() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    x_np = _sample_numpy()
    set_backend("numpy")
    feature = PartialAutocorrelation(PARAMS)
    expected = feature(x_np)

    set_backend("jax")
    jitted = jax.jit(feature)
    actual = np.asarray(jitted(jnp.asarray(x_np)))
    np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=1e-2)


def test_backend_is_global_not_input_inferred() -> None:
    set_backend("numpy")
    assert backend() == "numpy"
    np_result = PartialAutocorrelation(PARAMS)(_sample_numpy())

    set_backend("torch")
    assert backend() == "torch"
    with pytest.raises(TypeError):
        PartialAutocorrelation(PARAMS)(_sample_numpy())

    torch_result = PartialAutocorrelation(PARAMS)(_torch_sample()).detach().cpu().numpy()
    np.testing.assert_allclose(torch_result, np_result, atol=2e-5, rtol=2e-5)
