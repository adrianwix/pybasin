import math
from typing import TypedDict

import jax.numpy as jnp
import pytest
import torch

from pybasin.solvers import JaxSolver, TorchOdeSolver
from pybasin.solvers.jax_ode_system import JaxODESystem
from pybasin.solvers.torch_ode_system import ODESystem


class DecayParams(TypedDict):
    decay: float


class ExponentialDecayODE(ODESystem[DecayParams]):
    def ode(self, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return p[..., 0:1] * y


class ExponentialDecayJaxODE(JaxODESystem[DecayParams]):
    def ode(self, t, y, p):  # type: ignore[no-untyped-def,override]
        return p[..., 0:1] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['decay']} * y"


class HarmonicParams(TypedDict):
    pass


class HarmonicOscillatorODE(ODESystem[HarmonicParams]):
    """dx/dt = v, dv/dt = -x. Analytic solution for y0=[1, 0]: x(t)=cos(t), v(t)=-sin(t)."""

    def ode(self, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x, v = y[..., 0], y[..., 1]
        return torch.stack([v, -x], dim=-1)


class HarmonicOscillatorJaxODE(JaxODESystem[HarmonicParams]):
    def ode(self, t, y, p):  # type: ignore[no-untyped-def,override]
        x, v = y[..., 0], y[..., 1]
        return jnp.stack([v, -x], axis=-1)  # type: ignore[no-any-return]

    def get_str(self) -> str:
        return "harmonic_oscillator"


@pytest.fixture
def decay_ode() -> ExponentialDecayODE:
    return ExponentialDecayODE({"decay": -1.0})


@pytest.fixture
def harmonic_ode() -> HarmonicOscillatorODE:
    return HarmonicOscillatorODE({})


def test_dopri5_default_ignores_missing_step_size(decay_ode: ExponentialDecayODE) -> None:
    """Adaptive methods don't need step_size: IntegralController's own error-based control
    subdivides regardless of how sparse the requested output times are."""
    solver = TorchOdeSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0]])
    _, y = solver.integrate(decay_ode, y0)

    assert y[..., -1].item() == pytest.approx(math.exp(-1.0), abs=1e-4)


def test_heun_default_already_adapts(decay_ode: ExponentialDecayODE) -> None:
    """Unlike euler, heun has an embedded lower-order error estimate, so it already adapts
    its own step size under the default IntegralController -- no step_size needed even
    across a large anchor gap."""
    solver = TorchOdeSolver(
        t_span=(0, 10),
        t_steps=5,
        t_eval=(9, 10),
        method="heun",
        device="cpu",
        cache_dir=None,
    )

    y0 = torch.tensor([[1.0]])
    _, y = solver.integrate(decay_ode, y0)

    exact = math.exp(-9.0)
    assert y[..., 0].item() == pytest.approx(exact, rel=1e-2)


def test_euler_without_step_size_is_wrong(decay_ode: ExponentialDecayODE) -> None:
    """Documents the gap this fix addresses: euler has no embedded error estimate, so
    IntegralController accepts every step without ever changing the step size -- the
    integration silently runs at a single step size auto-selected from the dynamics at
    t_span[0], reused unchanged across the whole (possibly much larger) anchor gap.

    Uses deliberately loose rtol/atol for two reasons. First, performance: under the class
    default (rtol=1e-8, atol=1e-6), IntegralController's initial-step heuristic picks
    dt0~1e-8 for this ODE, and since euler's dt never grows from there, integrating even
    t_span=(0, 5) would need ~1e8 steps -- not just inaccurate but a genuine multi-minute
    hang. Second, to actually demonstrate wrongness at all: a *tighter* loose tolerance
    (e.g. rtol=atol=0.1) still picks a small enough dt0 that plain exponential decay --
    whose relative rate of change never varies -- stays accurate anyway. rtol=atol=100
    pushes dt0 up to exactly the ODE's own timescale, at which forward Euler's update
    (y_new = y * (1 - dt)) collapses to exactly 0 and stays there.
    """
    solver = TorchOdeSolver(
        t_span=(0, 5),
        t_steps=3,
        t_eval=(4, 5),
        method="euler",
        rtol=100.0,
        atol=100.0,
        device="cpu",
        cache_dir=None,
    )

    y0 = torch.tensor([[1.0]])
    _, y = solver.integrate(decay_ode, y0)

    exact = math.exp(-4.0)
    assert abs(y[..., 0].item() - exact) > 0.5 * abs(exact)


def test_euler_with_step_size_matches_analytic(decay_ode: ExponentialDecayODE) -> None:
    solver = TorchOdeSolver(
        t_span=(0, 10),
        t_steps=5,
        t_eval=(9, 10),
        method="euler",
        step_size=0.001,
        device="cpu",
        cache_dir=None,
    )

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(decay_ode, y0)

    assert t[0].item() == pytest.approx(9.0, abs=1e-5)
    assert t[-1].item() == pytest.approx(10.0, abs=1e-5)
    for i in range(t.shape[0]):
        exact = math.exp(-t[i].item())
        assert y[0, 0, i].item() == pytest.approx(exact, rel=1e-2)


def test_fixed_step_matches_diffrax_trajectory(harmonic_ode: HarmonicOscillatorODE) -> None:
    """Trajectory-level accuracy against JaxSolver/diffrax (Dopri5, rtol=1e-8, atol=1e-6),
    across the same t_span-anchor + narrow t_eval save-window pattern used by the case
    studies."""
    t_span = (0.0, 20.0)
    t_eval = (18.0, 20.0)
    t_steps = 50
    y0 = torch.tensor([[1.0, 0.0]])

    euler_solver = TorchOdeSolver(
        t_span=t_span,
        t_steps=t_steps,
        t_eval=t_eval,
        method="euler",
        step_size=0.001,
        device="cpu",
        cache_dir=None,
    )
    _, y_euler = euler_solver.integrate(harmonic_ode, y0)

    jax_ode = HarmonicOscillatorJaxODE({})
    jax_solver = JaxSolver(
        t_span=t_span,
        t_steps=t_steps,
        t_eval=t_eval,
        device="cpu",
        rtol=1e-8,
        atol=1e-6,
        cache_dir=None,
    )
    _, y_jax = jax_solver.integrate(jax_ode, y0)

    max_abs_err = (y_euler - y_jax).abs().max().item()
    assert max_abs_err < 5e-2


def test_clone_propagates_step_size() -> None:
    solver = TorchOdeSolver(
        t_span=(0, 1), t_steps=10, method="euler", step_size=0.01, device="cpu", cache_dir=None
    )
    cloned = solver.clone(device="cpu")
    assert cloned.step_size == 0.01


def test_unknown_method_error_lists_only_real_methods() -> None:
    """`midpoint` used to be listed as available despite torchode having no such solver and
    the branch below never handling it -- the error message must only list methods that are
    actually implemented."""
    solver = TorchOdeSolver(method="not_a_method", device="cpu", cache_dir=None)
    y0 = torch.tensor([[1.0]])
    with pytest.raises(
        ValueError, match=r"^Unknown method: not_a_method\. Available: dopri5, tsit5, euler, heun$"
    ):
        solver.integrate(ExponentialDecayODE({"decay": -1.0}), y0)
