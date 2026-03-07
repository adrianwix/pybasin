"""Tests for solver caching logic.

CacheManager is replaced with a MagicMock so no files are written to disk.
Cache hit/miss behavior is verified via behavioral assertions:
- hit: the returned result matches the mock's sentinel value (compute was skipped)
- miss: a ``compute_count`` counter confirms that ``compute_fn`` was invoked exactly once
"""

from collections.abc import Callable
from typing import Any, TypedDict
from unittest.mock import MagicMock

import pytest
import torch
from jax import Array

from pybasin.cache_manager import CacheManager
from pybasin.solvers import JaxSolver, ScipyParallelSolver, TorchDiffEqSolver, TorchOdeSolver
from pybasin.solvers.jax_ode_system import JaxODESystem
from pybasin.solvers.torch_ode_system import ODESystem

# ── shared ODE definitions ────────────────────────────────────────────────────


class ExponentialParams(TypedDict):
    decay: float


class ExponentialDecayODE(ODESystem[ExponentialParams]):
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.params["decay"] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['decay']} * y"


class ExponentialDecayJaxODE(JaxODESystem[ExponentialParams]):
    def ode(self, t: Array, y: Array, args: Any = None) -> Array:
        return self.params["decay"] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['decay']} * y"


@pytest.fixture
def ode() -> ExponentialDecayODE:
    return ExponentialDecayODE({"decay": -1.0})


@pytest.fixture
def jax_ode() -> ExponentialDecayJaxODE:
    return ExponentialDecayJaxODE({"decay": -1.0})


@pytest.fixture
def y0() -> torch.Tensor:
    return torch.tensor([[1.0]])


# ── helpers ───────────────────────────────────────────────────────────────────

# Sentinel tensors returned by the hit mock — real integration of dy/dt=-y from y₀=1
# produces e^{-t} ≠ 1, so all-ones output proves the computation was bypassed.
_HIT_Y = torch.ones(10, 1, 1)


def _hit_mock() -> MagicMock:
    """Mock CacheManager that simulates a hit: returns a fixed result without calling compute_fn."""
    mock = MagicMock(spec=CacheManager)
    mock.cached_call.return_value = (torch.linspace(0, 1, 10), _HIT_Y.clone())
    return mock


def _miss_counting_mock() -> tuple[MagicMock, list[int]]:
    """Mock CacheManager that simulates a miss: delegates to compute_fn and counts calls."""
    call_count: list[int] = [0]
    mock = MagicMock(spec=CacheManager)

    def call_through(*, compute_fn: Callable[[], Any], **_: Any) -> Any:
        call_count[0] += 1
        return compute_fn()

    mock.cached_call.side_effect = call_through
    return mock, call_count


def _inject_cache_manager(solver: Any, mock_cm: MagicMock) -> None:
    """Attach a mock CacheManager to a solver, bypassing constructor file-system setup."""
    solver._cache_manager = mock_cm  # pyright: ignore[reportPrivateUsage]


# ── PyTorch solvers (TorchDiffEq, TorchOde, Scipy) ───────────────────────────

# Each entry: (SolverClass, extra constructor kwargs needed beyond the shared base set)
TORCH_SOLVER_PARAMS = [
    pytest.param(TorchDiffEqSolver, {}, id="torchdiffeq"),
    pytest.param(TorchOdeSolver, {}, id="torchode"),
    pytest.param(ScipyParallelSolver, {"n_jobs": 1}, id="scipy"),
]


@pytest.mark.parametrize("SolverClass, extra", TORCH_SOLVER_PARAMS)
def test_cached_call_invoked_when_cache_enabled(
    SolverClass: type,
    extra: dict[str, Any],
    ode: ExponentialDecayODE,
    y0: torch.Tensor,
) -> None:
    """integrate() delegates to cache_manager.cached_call when a cache manager is set."""
    solver = SolverClass(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None, **extra)
    mock_cm = _hit_mock()
    _inject_cache_manager(solver, mock_cm)

    solver.integrate(ode, y0)

    mock_cm.cached_call.assert_called_once()


@pytest.mark.parametrize("SolverClass, extra", TORCH_SOLVER_PARAMS)
def test_cache_hit_skips_integration(
    SolverClass: type,
    extra: dict[str, Any],
    ode: ExponentialDecayODE,
    y0: torch.Tensor,
) -> None:
    """On a cache hit, the mock's sentinel value is returned — real compute is skipped.

    Real dy/dt=-y integration from y₀=1 produces e^{-t} ≠ 1, so all-ones output
    proves the computation was bypassed.
    """
    solver = SolverClass(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None, **extra)
    _inject_cache_manager(solver, _hit_mock())

    _, y = solver.integrate(ode, y0)

    assert torch.all(y == 1.0)


@pytest.mark.parametrize("SolverClass, extra", TORCH_SOLVER_PARAMS)
def test_cache_miss_calls_integration(
    SolverClass: type,
    extra: dict[str, Any],
    ode: ExponentialDecayODE,
    y0: torch.Tensor,
) -> None:
    """On a cache miss, compute_fn is invoked exactly once."""
    solver = SolverClass(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None, **extra)
    mock_cm, call_count = _miss_counting_mock()
    _inject_cache_manager(solver, mock_cm)

    solver.integrate(ode, y0)

    assert call_count[0] == 1


@pytest.mark.parametrize("SolverClass, extra", TORCH_SOLVER_PARAMS)
def test_cached_call_receives_correct_solver_config(
    SolverClass: type,
    extra: dict[str, Any],
    ode: ExponentialDecayODE,
    y0: torch.Tensor,
) -> None:
    """cached_call is passed solver_config that matches the constructed solver's parameters."""
    solver = SolverClass(
        t_span=(0, 1), t_steps=10, rtol=1e-8, atol=1e-6, device="cpu", cache_dir=None, **extra
    )
    mock_cm = _hit_mock()
    _inject_cache_manager(solver, mock_cm)

    solver.integrate(ode, y0)

    config = mock_cm.cached_call.call_args.kwargs["solver_config"]
    assert config["rtol"] == 1e-8
    assert config["atol"] == 1e-6
    assert config["t_span"] == (0, 1)


@pytest.mark.parametrize("SolverClass, extra", TORCH_SOLVER_PARAMS)
def test_solver_config_contains_required_keys(
    SolverClass: type,
    extra: dict[str, Any],
    ode: ExponentialDecayODE,
    y0: torch.Tensor,
) -> None:
    """solver_config passed to cached_call always includes rtol, atol, and t_span."""
    solver = SolverClass(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None, **extra)
    mock_cm = _hit_mock()
    _inject_cache_manager(solver, mock_cm)

    solver.integrate(ode, y0)

    config = mock_cm.cached_call.call_args.kwargs["solver_config"]
    assert "rtol" in config
    assert "atol" in config
    assert "t_span" in config


@pytest.mark.parametrize("SolverClass, extra", TORCH_SOLVER_PARAMS)
def test_solver_config_is_stable(
    SolverClass: type,
    extra: dict[str, Any],
    ode: ExponentialDecayODE,
    y0: torch.Tensor,
) -> None:
    """Identical solver parameters always produce the same solver_config."""
    base_kwargs: dict[str, Any] = {
        "t_span": (0, 1),
        "t_steps": 10,
        "rtol": 1e-8,
        "atol": 1e-6,
        "device": "cpu",
        "cache_dir": None,
        **extra,
    }
    solver_a = SolverClass(**base_kwargs)
    solver_b = SolverClass(**base_kwargs)

    mock_a, mock_b = _hit_mock(), _hit_mock()
    _inject_cache_manager(solver_a, mock_a)
    _inject_cache_manager(solver_b, mock_b)

    solver_a.integrate(ode, y0)
    solver_b.integrate(ode, y0)

    config_a = mock_a.cached_call.call_args.kwargs["solver_config"]
    config_b = mock_b.cached_call.call_args.kwargs["solver_config"]
    assert config_a == config_b


@pytest.mark.parametrize(
    "changed_kwargs",
    [
        pytest.param({"rtol": 1e-4}, id="rtol"),
        pytest.param({"atol": 1e-4}, id="atol"),
        pytest.param({"t_span": (0, 2)}, id="t_span"),
    ],
)
@pytest.mark.parametrize("SolverClass, extra", TORCH_SOLVER_PARAMS)
def test_solver_config_varies_with_params(
    SolverClass: type,
    extra: dict[str, Any],
    changed_kwargs: dict[str, Any],
    ode: ExponentialDecayODE,
    y0: torch.Tensor,
) -> None:
    """Changing rtol, atol, or t_span produces a different solver_config (different cache key)."""
    base_kwargs: dict[str, Any] = {
        "t_span": (0, 1),
        "t_steps": 10,
        "rtol": 1e-8,
        "atol": 1e-6,
        "device": "cpu",
        "cache_dir": None,
        **extra,
    }
    solver_base = SolverClass(**base_kwargs)
    solver_changed = SolverClass(**{**base_kwargs, **changed_kwargs})

    mock_base, mock_changed = _hit_mock(), _hit_mock()
    _inject_cache_manager(solver_base, mock_base)
    _inject_cache_manager(solver_changed, mock_changed)

    solver_base.integrate(ode, y0)
    solver_changed.integrate(ode, y0)

    config_base = mock_base.cached_call.call_args.kwargs["solver_config"]
    config_changed = mock_changed.cached_call.call_args.kwargs["solver_config"]
    assert config_base != config_changed


# ── JaxSolver ─────────────────────────────────────────────────────────────────


def test_jax_cached_call_invoked_when_cache_enabled(
    jax_ode: ExponentialDecayJaxODE, y0: torch.Tensor
) -> None:
    """JaxSolver.integrate() delegates to cache_manager.cached_call when a cache manager is set."""
    solver = JaxSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)
    mock_cm = _hit_mock()
    _inject_cache_manager(solver, mock_cm)

    solver.integrate(jax_ode, y0)

    mock_cm.cached_call.assert_called_once()


def test_jax_cache_hit_skips_integration(jax_ode: ExponentialDecayJaxODE, y0: torch.Tensor) -> None:
    """On a cache hit, the mock's sentinel value is returned — _integrate_jax is not called."""
    solver = JaxSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)
    _inject_cache_manager(solver, _hit_mock())

    _, y = solver.integrate(jax_ode, y0)

    assert torch.all(y == 1.0)


def test_jax_cache_miss_calls_integration(
    jax_ode: ExponentialDecayJaxODE, y0: torch.Tensor
) -> None:
    """On a cache miss, compute_fn is invoked exactly once."""
    solver = JaxSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)
    mock_cm, call_count = _miss_counting_mock()
    _inject_cache_manager(solver, mock_cm)

    solver.integrate(jax_ode, y0)

    assert call_count[0] == 1


def test_jax_cached_call_receives_correct_solver_config(
    jax_ode: ExponentialDecayJaxODE, y0: torch.Tensor
) -> None:
    """JaxSolver passes solver_config that matches the constructed solver's parameters."""
    solver = JaxSolver(
        t_span=(0, 1), t_steps=10, rtol=1e-8, atol=1e-6, device="cpu", cache_dir=None
    )
    mock_cm = _hit_mock()
    _inject_cache_manager(solver, mock_cm)

    solver.integrate(jax_ode, y0)

    config = mock_cm.cached_call.call_args.kwargs["solver_config"]
    assert config["rtol"] == 1e-8
    assert config["atol"] == 1e-6
    assert config["t_span"] == (0, 1)


def test_jax_solver_config_contains_required_keys(
    jax_ode: ExponentialDecayJaxODE, y0: torch.Tensor
) -> None:
    """JaxSolver config passed to cached_call includes method, rtol, atol, max_steps, t_span."""
    solver = JaxSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)
    mock_cm = _hit_mock()
    _inject_cache_manager(solver, mock_cm)

    solver.integrate(jax_ode, y0)

    config = mock_cm.cached_call.call_args.kwargs["solver_config"]
    assert "method" in config
    assert "rtol" in config
    assert "atol" in config
    assert "max_steps" in config
    assert "t_span" in config


def test_jax_solver_config_is_stable(jax_ode: ExponentialDecayJaxODE, y0: torch.Tensor) -> None:
    """Identical JaxSolver parameters always produce the same solver_config."""
    base_kwargs: dict[str, Any] = {
        "t_span": (0, 1),
        "t_steps": 10,
        "rtol": 1e-8,
        "atol": 1e-6,
        "device": "cpu",
        "cache_dir": None,
    }
    solver_a = JaxSolver(**base_kwargs)
    solver_b = JaxSolver(**base_kwargs)

    mock_a, mock_b = _hit_mock(), _hit_mock()
    _inject_cache_manager(solver_a, mock_a)
    _inject_cache_manager(solver_b, mock_b)

    solver_a.integrate(jax_ode, y0)
    solver_b.integrate(jax_ode, y0)

    config_a = mock_a.cached_call.call_args.kwargs["solver_config"]
    config_b = mock_b.cached_call.call_args.kwargs["solver_config"]
    assert config_a == config_b


@pytest.mark.parametrize(
    "changed_kwargs",
    [
        pytest.param({"rtol": 1e-4}, id="rtol"),
        pytest.param({"atol": 1e-4}, id="atol"),
        pytest.param({"t_span": (0, 2)}, id="t_span"),
    ],
)
def test_jax_solver_config_varies_with_params(
    changed_kwargs: dict[str, Any],
    jax_ode: ExponentialDecayJaxODE,
    y0: torch.Tensor,
) -> None:
    """Changing rtol, atol, or t_span produces a different config in JaxSolver."""
    base_kwargs: dict[str, Any] = {
        "t_span": (0, 1),
        "t_steps": 10,
        "rtol": 1e-8,
        "atol": 1e-6,
        "device": "cpu",
        "cache_dir": None,
    }
    solver_base = JaxSolver(**base_kwargs)
    solver_changed = JaxSolver(**{**base_kwargs, **changed_kwargs})

    mock_base, mock_changed = _hit_mock(), _hit_mock()
    _inject_cache_manager(solver_base, mock_base)
    _inject_cache_manager(solver_changed, mock_changed)

    solver_base.integrate(jax_ode, y0)
    solver_changed.integrate(jax_ode, y0)

    config_base = mock_base.cached_call.call_args.kwargs["solver_config"]
    config_changed = mock_changed.cached_call.call_args.kwargs["solver_config"]
    assert config_base != config_changed
