from typing import Any, TypedDict

import jax.numpy as jnp
import pytest
import torch
from diffrax import ConstantStepSize, Dopri5, ODETerm, PIDController, SaveAt, Tsit5
from jax import Array

from pybasin.solvers import JaxSolver
from pybasin.solvers.jax_ode_system import JaxODESystem


class ExponentialParams(TypedDict):
    decay: float


class ExponentialDecayJaxODE(JaxODESystem[ExponentialParams]):
    def ode(self, t: Array, y: Array) -> Array:
        return self.params["decay"] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['decay']} * y"


@pytest.fixture
def simple_jax_ode() -> ExponentialDecayJaxODE:
    params: ExponentialParams = {"decay": -1.0}
    return ExponentialDecayJaxODE(params)


def test_jax_solver_integration(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(time_span=(0, 1), n_steps=10, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert y[-1].item() < y[0].item()


def test_jax_solver_batched(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(time_span=(0, 1), n_steps=10, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0], [2.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 2, 1)
    assert torch.allclose(y[:, 1, :] / y[:, 0, :], torch.tensor([[2.0]]), atol=1e-5)  # type: ignore[misc]


def test_jax_solver_y0_shape_validation(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(time_span=(0, 1), n_steps=10, device="cpu", cache_dir=None)

    y0_1d = torch.tensor([1.0])
    with pytest.raises(ValueError, match="y0 must be 2D with shape"):
        solver.integrate(simple_jax_ode, y0_1d)

    y0_3d = torch.tensor([[[1.0]]])
    with pytest.raises(ValueError, match="y0 must be 2D with shape"):
        solver.integrate(simple_jax_ode, y0_3d)

    y0_2d = torch.tensor([[1.0]])
    _, y_result = solver.integrate(simple_jax_ode, y0_2d)
    assert y_result.shape == (10, 1, 1)


def test_jax_solver_custom_solver(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    custom_solver = Tsit5()
    solver = JaxSolver(
        time_span=(0, 1), n_steps=10, device="cpu", method=custom_solver, cache_dir=None
    )

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert y[-1].item() < y[0].item()


def test_jax_solver_clone(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(time_span=(0, 1), n_steps=10, device="cpu", cache_dir=None)
    new_solver = solver.clone(device="cpu")

    assert new_solver is not solver

    y0 = torch.tensor([[1.0]])
    t, y = new_solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)


def test_jax_solver_default_n_steps(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(time_span=(0, 1), device="cpu", cache_dir=None)

    steps = 1000
    assert solver.n_steps == steps

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (steps,)
    assert y.shape == (steps, 1, 1)


def test_jax_solver_2d_system() -> None:
    class LorenzLikeODE(JaxODESystem[dict[str, float]]):
        def ode(self, t: Array, y: Array) -> Array:
            x, v = y[..., 0], y[..., 1]
            dx = v
            dv = -x
            return jnp.stack([dx, dv], axis=-1)

        def get_str(self) -> str:
            return "harmonic_oscillator"

    ode = LorenzLikeODE({})
    solver = JaxSolver(time_span=(0, 2 * 3.14159), n_steps=100, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0, 0.0]])
    t, y = solver.integrate(ode, y0)

    assert t.shape == (100,)
    assert y.shape == (100, 1, 2)
    assert y[0, 0, 0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert y[0, 0, 1].item() == pytest.approx(0.0, abs=1e-5)  # type: ignore[misc]


# --- solver_args overload tests ---


class EmptyParams(TypedDict):
    pass


class DecaySystemNoODE(JaxODESystem[EmptyParams]):
    """System that does NOT override ode() — uses solver_args terms instead."""

    def get_str(self) -> str:
        return "decay_no_ode"


def test_solver_args_basic_integration(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    """solver_args with explicit ODETerm from ode_system.ode()."""

    def ode_wrapper(t: Any, y: Any, args: Any) -> Any:
        return simple_jax_ode.ode(t, y)

    t_eval = jnp.linspace(0, 1, 10)  # type: ignore[reportUnknownMemberType]
    solver = JaxSolver(
        solver_args={
            "terms": ODETerm(ode_wrapper),
            "solver": Dopri5(),
            "t0": 0,
            "t1": 1,
            "dt0": None,
            "saveat": SaveAt(ts=t_eval),
            "stepsize_controller": PIDController(rtol=1e-8, atol=1e-6),
        },
        cache_dir=None,
    )

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert y[-1].item() < y[0].item()


def test_solver_args_with_custom_terms() -> None:
    """solver_args with user-provided terms — ode() is NOT called."""

    def decay_fn(t: Array, y: Array, args: Any) -> Array:  # type: ignore[type-arg]
        return -1.0 * y

    t_eval = jnp.linspace(0, 1, 10)  # type: ignore[reportUnknownMemberType]
    solver = JaxSolver(
        solver_args={
            "terms": ODETerm(decay_fn),  # type: ignore[reportArgumentType]
            "solver": Dopri5(),
            "t0": 0,
            "t1": 1,
            "dt0": None,
            "saveat": SaveAt(ts=t_eval),
            "stepsize_controller": PIDController(rtol=1e-8, atol=1e-6),
        },
        cache_dir=None,
    )

    system = DecaySystemNoODE({})
    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(system, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert y[-1].item() < y[0].item()


def test_solver_args_constant_step_size(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    """solver_args with ConstantStepSize instead of PIDController."""

    def ode_wrapper(t: Any, y: Any, args: Any) -> Any:
        return simple_jax_ode.ode(t, y)

    t_eval = jnp.linspace(0, 1, 50)  # type: ignore[reportUnknownMemberType]
    solver = JaxSolver(
        solver_args={
            "terms": ODETerm(ode_wrapper),
            "solver": Dopri5(),
            "t0": 0,
            "t1": 1,
            "dt0": 0.01,
            "saveat": SaveAt(ts=t_eval),
            "stepsize_controller": ConstantStepSize(),
        },
        cache_dir=None,
    )

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (50,)
    assert y.shape == (50, 1, 1)
    assert y[-1].item() < y[0].item()


def test_solver_args_basic_attributes() -> None:
    """solver_args mode stores solver_args without introspecting them."""
    t_eval = jnp.linspace(0, 5, 200)  # type: ignore[reportUnknownMemberType]
    solver = JaxSolver(
        solver_args={
            "solver": Tsit5(),
            "t0": 0,
            "t1": 5,
            "dt0": None,
            "saveat": SaveAt(ts=t_eval),
            "stepsize_controller": PIDController(rtol=1e-5, atol=1e-5),
        },
        cache_dir=None,
    )

    assert solver.solver_args is not None


def test_solver_args_clone_propagates() -> None:
    """clone() preserves solver_args."""
    t_eval = jnp.linspace(0, 1, 10)  # type: ignore[reportUnknownMemberType]
    solver = JaxSolver(
        solver_args={
            "solver": Dopri5(),
            "t0": 0,
            "t1": 1,
            "dt0": None,
            "saveat": SaveAt(ts=t_eval),
            "stepsize_controller": PIDController(rtol=1e-8, atol=1e-6),
        },
        cache_dir=None,
    )

    new_solver = solver.clone(device="cpu")
    assert new_solver is not solver
    assert new_solver.solver_args is not None
    assert new_solver.solver_args == solver.solver_args
    assert new_solver._cache_dir == solver._cache_dir  # type: ignore[reportPrivateUsage]


def test_solver_args_batched(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    """solver_args path works with batched initial conditions."""

    def ode_wrapper(t: Any, y: Any, args: Any) -> Any:
        return simple_jax_ode.ode(t, y)

    t_eval = jnp.linspace(0, 1, 10)  # type: ignore[reportUnknownMemberType]
    solver = JaxSolver(
        solver_args={
            "terms": ODETerm(ode_wrapper),
            "solver": Dopri5(),
            "t0": 0,
            "t1": 1,
            "dt0": None,
            "saveat": SaveAt(ts=t_eval),
            "stepsize_controller": PIDController(rtol=1e-8, atol=1e-6),
        },
        cache_dir=None,
    )

    y0 = torch.tensor([[1.0], [2.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 2, 1)
    assert torch.allclose(y[:, 1, :] / y[:, 0, :], torch.tensor([[2.0]]), atol=1e-5)  # type: ignore[misc]


def test_system_without_ode_raises_on_direct_call() -> None:
    """JaxODESystem without ode() override raises NotImplementedError."""
    system = DecaySystemNoODE({})
    with pytest.raises(NotImplementedError, match="ode\\(\\) is not implemented"):
        system.ode(jnp.array(0.0), jnp.array([1.0]))  # type: ignore[reportUnknownMemberType]


def test_generic_and_solver_args_produce_same_results(
    simple_jax_ode: ExponentialDecayJaxODE,
) -> None:
    """Generic API and solver_args produce identical results with equivalent config."""
    time_span: tuple[float, float] = (0, 1)
    n_steps = 50
    rtol = 1e-8
    atol = 1e-6

    generic_solver = JaxSolver(
        time_span=time_span,
        n_steps=n_steps,
        device="cpu",
        method=Dopri5(),
        rtol=rtol,
        atol=atol,
        cache_dir=None,
    )

    t_eval = jnp.linspace(time_span[0], time_span[1], n_steps)  # type: ignore[reportUnknownMemberType]

    def ode_wrapper(t: Any, y: Any, args: Any) -> Any:
        return simple_jax_ode.ode(t, y)

    solver_args_solver = JaxSolver(
        solver_args={
            "terms": ODETerm(ode_wrapper),
            "solver": Dopri5(),
            "t0": time_span[0],
            "t1": time_span[1],
            "dt0": None,
            "saveat": SaveAt(ts=t_eval),
            "stepsize_controller": PIDController(rtol=rtol, atol=atol),
        },
        cache_dir=None,
    )

    y0 = torch.tensor([[1.0], [2.0], [0.5]])

    t_generic, y_generic = generic_solver.integrate(simple_jax_ode, y0)
    t_args, y_args = solver_args_solver.integrate(simple_jax_ode, y0)

    assert torch.allclose(t_generic, t_args, atol=1e-6)  # type: ignore[misc]
    assert torch.allclose(y_generic, y_args, atol=1e-6)  # type: ignore[misc]
