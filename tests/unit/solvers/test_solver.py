from typing import TypedDict

import pytest
import torch

from pybasin.solvers import ScipyParallelSolver, TorchDiffEqSolver, TorchOdeSolver
from pybasin.solvers.torch_ode_system import ODESystem


class ExponentialParams(TypedDict):
    decay: float


class ExponentialDecayODE(ODESystem[ExponentialParams]):
    def ode(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.params["decay"] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['decay']} * y"


@pytest.fixture
def simple_ode() -> ExponentialDecayODE:
    params: ExponentialParams = {"decay": -1.0}
    return ExponentialDecayODE(params)


def test_torchdiffeq_solver_integration(simple_ode: ExponentialDecayODE) -> None:
    solver = TorchDiffEqSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_ode, y0)

    # 10 time evaluation points
    assert t.shape == (10,)
    # Single trajectory: 10 steps × 1 batch × 1 state
    assert y.shape == (10, 1, 1)
    # Initial condition preserved (y₀=1.0)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    # Exponential decay: final value < initial value
    assert y[-1].item() < y[0].item()


def test_torchode_solver_integration(simple_ode: ExponentialDecayODE) -> None:
    solver = TorchOdeSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_ode, y0)

    # 10 time points
    assert t.shape == (10,)
    # Consistent shape: 10 steps × 1 batch × 1 state
    assert y.shape == (10, 1, 1)
    # Initial condition correct
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]


def test_solver_batched_integration(simple_ode: ExponentialDecayODE) -> None:
    solver = TorchOdeSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0], [2.0]])
    t, y = solver.integrate(simple_ode, y0)

    # 10 time points
    assert t.shape == (10,)
    # 10 steps × 2 batches × 1 state
    assert y.shape == (10, 2, 1)
    # Ratio between trajectories maintained (y₀=[1.0] and [2.0], so ratio=2.0 throughout)
    assert torch.allclose(y[:, 1, :] / y[:, 0, :], torch.tensor([[2.0]]), atol=1e-5)


def test_solver_y0_shape_validation(simple_ode: ExponentialDecayODE) -> None:
    solver = TorchOdeSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)

    # 1D tensor should be rejected
    y0_1d = torch.tensor([1.0])
    with pytest.raises(ValueError, match="y0 must be 2D with shape"):
        solver.integrate(simple_ode, y0_1d)

    # 3D tensor should be rejected
    y0_3d = torch.tensor([[[1.0]]])
    with pytest.raises(ValueError, match="y0 must be 2D with shape"):
        solver.integrate(simple_ode, y0_3d)

    # 2D tensor should work
    y0_2d = torch.tensor([[1.0]])
    _, y_result = solver.integrate(simple_ode, y0_2d)
    assert y_result.shape == (10, 1, 1)


def test_sklearn_solver_integration(simple_ode: ExponentialDecayODE) -> None:
    solver = ScipyParallelSolver(t_span=(0, 1), t_steps=11, device="cpu", n_jobs=1, cache_dir=None)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_ode, y0)

    # 11 time points
    assert t.shape == (11,)
    # Single trajectory: 11 steps × 1 batch × 1 state
    assert y.shape == (11, 1, 1)
    # Initial condition preserved (y₀=1.0)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    # Exponential decay: final value < initial value
    assert y[-1].item() < y[0].item()


def test_sklearn_solver_batched(simple_ode: ExponentialDecayODE) -> None:
    solver = ScipyParallelSolver(t_span=(0, 1), t_steps=11, device="cpu", n_jobs=2, cache_dir=None)

    y0 = torch.tensor([[1.0], [2.0]])
    t, y = solver.integrate(simple_ode, y0)
    # 11 time points
    assert t.shape == (11,)
    # 11 steps × 2 batches × 1 state
    assert y.shape == (11, 2, 1)
    # Ratio between trajectories maintained (y₀=[1.0] and [2.0], so ratio=2.0 throughout)
    assert torch.allclose(y[:, 1, :] / y[:, 0, :], torch.tensor([[2.0]]), atol=1e-5)  # type: ignore[misc]


def test_sklearn_solver_single_trajectory_with_parallel_enabled(
    simple_ode: ExponentialDecayODE,
) -> None:
    """Test single trajectory works correctly even when n_jobs > 1."""
    solver = ScipyParallelSolver(t_span=(0, 1), t_steps=11, device="cpu", n_jobs=2, cache_dir=None)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_ode, y0)

    # 11 time points
    assert t.shape == (11,)
    # Single trajectory: 11 steps × 1 batch × 1 state
    assert y.shape == (11, 1, 1)
    # Initial condition preserved
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    # Exponential decay: final value should be e^(-1) ≈ 0.368
    assert y[-1].item() == pytest.approx(0.368, abs=0.01)  # type: ignore[misc]


@pytest.mark.parametrize(
    "solver",
    [
        TorchDiffEqSolver(t_span=(0, 2), t_steps=10, t_eval=(1, 2), device="cpu", cache_dir=None),
        TorchOdeSolver(t_span=(0, 2), t_steps=10, t_eval=(1, 2), device="cpu", cache_dir=None),
        ScipyParallelSolver(
            t_span=(0, 2), t_steps=10, t_eval=(1, 2), device="cpu", n_jobs=1, cache_dir=None
        ),
    ],
)
def test_t_eval_saves_only_subrange(
    solver: TorchDiffEqSolver | TorchOdeSolver | ScipyParallelSolver,
    simple_ode: ExponentialDecayODE,
) -> None:
    """t_eval=(1,2) with t_span=(0,2): output covers [1,2], but integration started at 0."""
    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)
    assert t[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert t[-1].item() == pytest.approx(2.0, abs=1e-5)  # type: ignore[misc]
    # y[0] at t=1 should be e^(-1) ≈ 0.368, not 1.0 — proves integration started at t=0
    assert y[0].item() == pytest.approx(0.368, abs=0.01)  # type: ignore[misc]


@pytest.mark.parametrize(
    "SolverClass",
    [TorchDiffEqSolver, TorchOdeSolver, ScipyParallelSolver],
)
def test_t_eval_out_of_span_raises(SolverClass: type) -> None:
    with pytest.raises(ValueError, match="t_eval"):
        SolverClass(t_span=(0, 1), t_steps=10, t_eval=(0.5, 1.5), device="cpu", cache_dir=None)


@pytest.mark.parametrize(
    "SolverClass",
    [TorchDiffEqSolver, TorchOdeSolver, ScipyParallelSolver],
)
def test_t_eval_propagates_to_clone(SolverClass: type) -> None:
    solver = SolverClass(t_span=(0, 2), t_steps=10, t_eval=(1, 2), device="cpu", cache_dir=None)
    cloned = solver.clone(device="cpu", cache_dir=None)
    assert cloned.t_eval == (1, 2)
