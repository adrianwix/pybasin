import math
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
    def ode(self, t: Array, y: Array, p: Array) -> Array:  # type: ignore[override]
        return p[..., 0:1] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['decay']} * y"


@pytest.fixture
def simple_jax_ode() -> ExponentialDecayJaxODE:
    params: ExponentialParams = {"decay": -1.0}
    return ExponentialDecayJaxODE(params)


def test_jax_solver_integration(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert y[-1].item() < y[0].item()


def test_jax_solver_batched(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0], [2.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 2, 1)
    assert torch.allclose(y[:, 1, :] / y[:, 0, :], torch.tensor([[2.0]]), atol=1e-5)  # type: ignore[misc]


def test_jax_solver_y0_shape_validation(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)

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
        t_span=(0, 1), t_steps=10, device="cpu", method=custom_solver, cache_dir=None
    )

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)
    assert y[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert y[-1].item() < y[0].item()


def test_jax_solver_clone(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)
    new_solver = solver.clone(device="cpu")

    assert new_solver is not solver

    y0 = torch.tensor([[1.0]])
    t, y = new_solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)


def test_jax_solver_default_t_steps(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    solver = JaxSolver(t_span=(0, 1), device="cpu", cache_dir=None)

    steps = 1000
    assert solver.t_steps == steps

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (steps,)
    assert y.shape == (steps, 1, 1)


def test_jax_solver_t_eval_saves_only_subrange(
    simple_jax_ode: ExponentialDecayJaxODE,
) -> None:
    """t_eval=(1,2) with t_span=(0,2): output covers [1,2], but integration started at 0."""
    solver = JaxSolver(t_span=(0, 2), t_steps=10, t_eval=(1, 2), device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(simple_jax_ode, y0)

    assert t.shape == (10,)
    assert y.shape == (10, 1, 1)
    assert t[0].item() == pytest.approx(1.0, abs=1e-5)  # type: ignore[misc]
    assert t[-1].item() == pytest.approx(2.0, abs=1e-5)  # type: ignore[misc]
    # y[0] at t=1 should be e^(-1) ≈ 0.368, not 1.0 — proves integration started at t=0
    assert y[0].item() == pytest.approx(0.368, abs=0.01)  # type: ignore[misc]


def test_jax_solver_t_eval_out_of_span_raises() -> None:
    with pytest.raises(ValueError, match="t_eval"):
        JaxSolver(t_span=(0, 1), t_steps=10, t_eval=(0.5, 1.5), device="cpu", cache_dir=None)


def test_jax_solver_t_eval_propagates_to_clone(
    simple_jax_ode: ExponentialDecayJaxODE,
) -> None:
    solver = JaxSolver(t_span=(0, 2), t_steps=10, t_eval=(1, 2), device="cpu", cache_dir=None)
    cloned = solver.clone(device="cpu", cache_dir=None)
    assert cloned.t_eval == (1, 2)


def test_jax_solver_2d_system() -> None:
    class LorenzLikeODE(JaxODESystem[dict[str, float]]):
        def ode(self, t: Array, y: Array, p: Array) -> Array:  # type: ignore[override]
            x, v = y[0], y[1]
            dx = v
            dv = -x
            return jnp.array([dx, dv])  # pyright: ignore[reportUnknownMemberType]

        def get_str(self) -> str:
            return "harmonic_oscillator"

    ode = LorenzLikeODE({})
    solver = JaxSolver(t_span=(0, 2 * 3.14159), t_steps=100, device="cpu", cache_dir=None)

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
    default_p = simple_jax_ode.params_to_array()

    def ode_wrapper(t: Any, y: Any, args: Any) -> Any:
        return simple_jax_ode.ode(t, y, default_p)

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
    default_p = simple_jax_ode.params_to_array()

    def ode_wrapper(t: Any, y: Any, args: Any) -> Any:
        return simple_jax_ode.ode(t, y, default_p)

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
    default_p = simple_jax_ode.params_to_array()

    def ode_wrapper(t: Any, y: Any, args: Any) -> Any:
        return simple_jax_ode.ode(t, y, default_p)

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
        system.ode(jnp.array(0.0), jnp.array([1.0]), jnp.array([]))  # type: ignore[reportUnknownMemberType]


def test_generic_and_solver_args_produce_same_results(
    simple_jax_ode: ExponentialDecayJaxODE,
) -> None:
    """Generic API and solver_args produce identical results with equivalent config."""
    time_span: tuple[float, float] = (0, 1)
    n_steps = 50
    rtol = 1e-8
    atol = 1e-6

    generic_solver = JaxSolver(
        t_span=time_span,
        t_steps=n_steps,
        device="cpu",
        method=Dopri5(),
        rtol=rtol,
        atol=atol,
        cache_dir=None,
    )

    t_eval = jnp.linspace(time_span[0], time_span[1], n_steps)  # type: ignore[reportUnknownMemberType]
    default_p = simple_jax_ode.params_to_array()

    def ode_wrapper(t: Any, y: Any, args: Any) -> Any:
        return simple_jax_ode.ode(t, y, default_p)

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


# --- Parameter batching tests ---
#
# The solver's params argument accepts a 2D tensor of shape (batch, n_params).
# The caller pre-flattens B initial conditions × P parameter combinations into a
# flat (B*P, ...) batch before calling integrate().
#
# Flattening convention — IC-major order (each IC runs with all P combos):
#   params_flat = param_combos.repeat(B, 1)              # (B*P, n_params)
#   y0_flat     = torch.repeat_interleave(y0, P, dim=0)  # (B*P, n_dims)
#
# Trajectory index  ic * P + p  carries (IC[ic], params[p]).
#
# The three patterns below differ only in how param_combos is built:
#   Sweep — vary one parameter:     param_combos shape (P, 1)
#   Zip   — pair-wise of two arrays: param_combos shape (P, 2)  (zip the arrays)
#   Grid  — cartesian product:       param_combos shape (n_a*n_b, 2)  (meshgrid)


class TwoDecayParams(TypedDict):
    alpha: float
    beta: float


class TwoDecayJaxODE(JaxODESystem[TwoDecayParams]):
    """d[y0]/dt = alpha * y[0],  d[y1]/dt = beta * y[1].

    Exact solution: y[i](t) = y0[i] * exp(p[i] * t).
    """

    def ode(self, t: Array, y: Array, p: Array) -> Array:  # type: ignore[override]
        return p * y

    def get_str(self) -> str:
        return "two_decay"


@pytest.fixture
def two_decay_ode() -> TwoDecayJaxODE:
    return TwoDecayJaxODE({"alpha": -1.0, "beta": -1.0})


def test_jax_solver_params_sweep(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    """Sweep: vary one parameter over P values; each of the B ICs runs with all P values.

    B=2 ICs, P=3 decay values -> B*P=6 trajectories.
    Trajectory ic*P+p at t=1: y0[ic] * exp(decay[p]).
    """
    solver = JaxSolver(t_span=(0, 1), t_steps=11, device="cpu", cache_dir=None)

    B, P = 2, 3
    y0 = torch.tensor([[1.0], [2.0]])  # (B, 1)
    param_combos = torch.tensor([[-0.5], [-1.0], [-2.0]])  # (P, 1)

    _, y = solver.integrate(simple_jax_ode, y0, params=param_combos)

    assert y.shape == (11, B * P, 1)
    y0_vals = [1.0, 2.0]
    decays = [-0.5, -1.0, -2.0]
    for ic in range(B):
        for p in range(P):
            expected = y0_vals[ic] * math.exp(decays[p])
            assert y[-1, ic * P + p, 0].item() == pytest.approx(expected, abs=0.01)  # type: ignore[misc]


def test_jax_solver_params_zip(two_decay_ode: TwoDecayJaxODE) -> None:
    """Zip: pair-wise combination of two parameter arrays; each IC runs with all P pairs.

    alphas=[a1,a2,a3], betas=[b1,b2,b3] -> P=3 pairs (a1,b1),(a2,b2),(a3,b3).
    B=2 ICs, each runs with all 3 pairs -> B*P=6 trajectories.
    Trajectory ic*P+p at t=1: y[dim] = y0[ic][dim] * exp(param[p][dim]).
    """
    solver = JaxSolver(t_span=(0, 1), t_steps=11, device="cpu", cache_dir=None)

    B, P = 2, 3
    alphas = torch.tensor([-0.5, -1.0, -2.0])
    betas = torch.tensor([-1.0, -2.0, -0.5])
    param_combos = torch.stack([alphas, betas], dim=1)  # (P, 2)
    y0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (B, 2)

    _, y = solver.integrate(two_decay_ode, y0, params=param_combos)

    assert y.shape == (11, B * P, 2)
    y0_vals = [[1.0, 2.0], [3.0, 4.0]]
    pairs: list[tuple[float, float]] = list(zip(alphas.tolist(), betas.tolist(), strict=True))  # type: ignore[arg-type]
    for ic in range(B):
        for p, (a, b) in enumerate(pairs):
            idx = ic * P + p
            assert y[-1, idx, 0].item() == pytest.approx(y0_vals[ic][0] * math.exp(a), abs=0.01)  # type: ignore[misc]
            assert y[-1, idx, 1].item() == pytest.approx(y0_vals[ic][1] * math.exp(b), abs=0.01)  # type: ignore[misc]


def test_jax_solver_params_grid(two_decay_ode: TwoDecayJaxODE) -> None:
    """Grid: cartesian product of two parameter arrays; each IC runs with all combinations.

    alphas=[a1,a2], betas=[b1,b2] -> P=4 combos: (a1,b1),(a1,b2),(a2,b1),(a2,b2).
    B=2 ICs, each runs with all 4 combos -> B*P=8 trajectories.
    Trajectory ic*P+p at t=1: y[dim] = y0[ic][dim] * exp(param[p][dim]).
    """
    solver = JaxSolver(t_span=(0, 1), t_steps=11, device="cpu", cache_dir=None)

    B = 2
    alphas = torch.tensor([-0.5, -1.0])
    betas = torch.tensor([-1.0, -2.0])
    ag, bg = torch.meshgrid(alphas, betas, indexing="ij")  # each (2, 2)
    param_combos = torch.stack([ag.flatten(), bg.flatten()], dim=1)  # (P=4, 2)
    P = param_combos.shape[0]
    y0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (B, 2)

    _, y = solver.integrate(two_decay_ode, y0, params=param_combos)

    assert y.shape == (11, B * P, 2)
    y0_vals = [[1.0, 2.0], [3.0, 4.0]]
    for ic in range(B):
        for p in range(P):
            a = param_combos[p, 0].item()
            b = param_combos[p, 1].item()
            idx = ic * P + p
            assert y[-1, idx, 0].item() == pytest.approx(y0_vals[ic][0] * math.exp(a), abs=0.01)  # type: ignore[misc]
            assert y[-1, idx, 1].item() == pytest.approx(y0_vals[ic][1] * math.exp(b), abs=0.01)  # type: ignore[misc]


def test_jax_solver_params_none_uses_defaults(simple_jax_ode: ExponentialDecayJaxODE) -> None:
    """When params=None, uses ode_system.params_to_array() (decay=-1)."""
    solver = JaxSolver(t_span=(0, 1), t_steps=11, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0]])
    _, y = solver.integrate(simple_jax_ode, y0, params=None)

    assert y[-1, 0, 0].item() == pytest.approx(0.368, abs=0.01)  # type: ignore[misc]
