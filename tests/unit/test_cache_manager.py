import os
from typing import Any

import pytest
import torch

from pybasin.cache_manager import CacheManager


class StubODESystem:
    """Minimal ODE system stub satisfying ODESystemProtocol."""

    def __init__(self, params: dict[str, float]) -> None:
        self.params = params
        self.default_params = params

    def to(self, device: Any) -> "StubODESystem":
        return self

    def get_str(self) -> str:
        return "dy/dt = -y"

    def params_to_array(self) -> Any:
        return []


@pytest.fixture
def cache_dir(tmp_path: Any) -> str:
    return str(tmp_path / "cache")


@pytest.fixture
def manager(cache_dir: str) -> CacheManager:
    return CacheManager(cache_dir)


@pytest.fixture
def ode() -> StubODESystem:
    return StubODESystem({"decay": -1.0})


def _make_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0, 1, 10)
    y = torch.randn(10, 2, 3)
    return t, y


def test_save_and_load_roundtrip(manager: CacheManager) -> None:
    """Save tensors to cache and load them back — values must match exactly."""
    t, y = _make_tensors()

    manager.save("key1", t, y)
    result = manager.load("key1", torch.device("cpu"))

    assert result is not None
    t_loaded, y_loaded = result
    assert torch.equal(t_loaded, t)
    assert torch.equal(y_loaded, y)


def test_load_returns_none_on_cache_miss(manager: CacheManager) -> None:
    """Loading a key that was never saved returns None."""
    assert manager.load("nonexistent", torch.device("cpu")) is None


def test_build_key_deterministic(manager: CacheManager, ode: StubODESystem) -> None:
    """Same inputs always produce the same cache key."""
    y0 = torch.tensor([[1.0]])
    t_eval = torch.linspace(0, 1, 10)
    config: dict[str, Any] = {"rtol": 1e-8, "atol": 1e-6}

    key1 = manager.build_key("Solver", ode, y0, t_eval, config)
    key2 = manager.build_key("Solver", ode, y0, t_eval, config)
    assert key1 == key2


def test_build_key_varies_with_inputs(manager: CacheManager, ode: StubODESystem) -> None:
    """Changing any input (solver name, y0, config, t_eval) produces a different cache key."""
    y0 = torch.tensor([[1.0]])
    t_eval = torch.linspace(0, 1, 10)
    config: dict[str, Any] = {"rtol": 1e-8}

    base_key = manager.build_key("Solver", ode, y0, t_eval, config)

    different_solver = manager.build_key("OtherSolver", ode, y0, t_eval, config)
    different_y0 = manager.build_key("Solver", ode, torch.tensor([[2.0]]), t_eval, config)
    different_config = manager.build_key("Solver", ode, y0, t_eval, {"rtol": 1e-4})
    different_t_eval = manager.build_key("Solver", ode, y0, torch.linspace(0.5, 1, 10), config)
    # t_span in config distinguishes same t_eval with different integration starts
    different_t_span = manager.build_key(
        "Solver", ode, y0, t_eval, {**config, "t_span": (0.5, 1.0)}
    )

    assert base_key != different_solver
    assert base_key != different_y0
    assert base_key != different_config
    assert base_key != different_t_eval
    assert base_key != different_t_span


def test_cached_call_computes_on_miss_and_caches(manager: CacheManager, ode: StubODESystem) -> None:
    """First call computes the result; second call returns the cached copy without recomputing."""
    y0 = torch.tensor([[1.0]])
    t_eval = torch.linspace(0, 1, 10)
    config: dict[str, Any] = {"rtol": 1e-8}
    device = torch.device("cpu")

    t_expected = torch.linspace(0, 1, 10)
    y_expected = torch.ones(10, 1, 1)
    call_count = 0

    def compute() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal call_count
        call_count += 1
        return t_expected, y_expected

    t1, y1 = manager.cached_call("S", ode, y0, t_eval, config, device, compute)
    assert call_count == 1
    assert torch.equal(t1, t_expected)
    assert torch.equal(y1, y_expected)

    t2, _ = manager.cached_call("S", ode, y0, t_eval, config, device, compute)
    assert call_count == 1, "compute_fn should not be called on cache hit"
    assert torch.equal(t2, t_expected)


def test_corrupted_cache_file_is_deleted_and_recomputed(
    manager: CacheManager, ode: StubODESystem, cache_dir: str
) -> None:
    """A corrupted .safetensors file is silently deleted and the result is recomputed."""
    y0 = torch.tensor([[1.0]])
    t_eval = torch.linspace(0, 1, 5)
    config: dict[str, Any] = {"rtol": 1e-8}
    device = torch.device("cpu")

    key = manager.build_key("S", ode, y0, t_eval, config)
    cache_file = os.path.join(cache_dir, f"{key}.safetensors")
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as f:
        f.write("not a valid safetensors file")

    t_expected = torch.linspace(0, 1, 5)
    y_expected = torch.ones(5, 1, 1)

    t_out, _ = manager.cached_call(
        "S", ode, y0, t_eval, config, device, lambda: (t_expected, y_expected)
    )

    assert not os.path.exists(cache_file) or os.path.getsize(cache_file) > 27
    assert torch.equal(t_out, t_expected)
