import math
from typing import TypedDict

import jax.numpy as jnp
import pytest
import torch

from pybasin.solvers import JaxSolver, TorchDiffEqSolver
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
    """Adaptive methods don't need step_size: their own control subdivides regardless
    of how sparse the requested output times are."""
    solver = TorchDiffEqSolver(t_span=(0, 1), t_steps=10, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0]])
    _, y = solver.integrate(decay_ode, y0)

    assert y[..., -1].item() == pytest.approx(math.exp(-1.0), abs=1e-4)


def test_fixed_step_without_step_size_is_wrong(decay_ode: ExponentialDecayODE) -> None:
    """Documents the gap this fix addresses: method="rk4" with no step_size takes one
    step per requested output interval, which is wildly inaccurate whenever the anchor
    gap (t_span[0] -> save window) is large relative to the dynamics' timescale."""
    solver = TorchDiffEqSolver(
        t_span=(0, 10),
        t_steps=5,
        t_eval=(9, 10),
        method="rk4",
        device="cpu",
        cache_dir=None,
    )

    y0 = torch.tensor([[1.0]])
    _, y = solver.integrate(decay_ode, y0)

    exact = math.exp(-9.0)
    assert abs(y[..., 0].item() - exact) > 0.1 * abs(exact) + 1e-3


def test_fixed_step_with_step_size_matches_analytic(decay_ode: ExponentialDecayODE) -> None:
    solver = TorchDiffEqSolver(
        t_span=(0, 10),
        t_steps=5,
        t_eval=(9, 10),
        method="rk4",
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
        assert y[0, 0, i].item() == pytest.approx(exact, rel=1e-3)


def test_fixed_step_matches_diffrax_trajectory(harmonic_ode: HarmonicOscillatorODE) -> None:
    """Trajectory-level accuracy against JaxSolver/diffrax (Dopri5, rtol=1e-8, atol=1e-6),
    across the same t_span-anchor + narrow t_eval save-window pattern used by the case
    studies (e.g. Duffing's t_span=(0, 1000), t_eval=(900, 1000))."""
    t_span = (0.0, 20.0)
    t_eval = (18.0, 20.0)
    t_steps = 50
    y0 = torch.tensor([[1.0, 0.0]])

    rk4_solver = TorchDiffEqSolver(
        t_span=t_span,
        t_steps=t_steps,
        t_eval=t_eval,
        method="rk4",
        step_size=0.0005,
        device="cpu",
        cache_dir=None,
    )
    _, y_rk4 = rk4_solver.integrate(harmonic_ode, y0)

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

    max_abs_err = (y_rk4 - y_jax).abs().max().item()
    max_rel_err = ((y_rk4 - y_jax).abs() / y_jax.abs().clamp_min(1e-8)).max().item()
    assert max_abs_err < 1e-3
    assert max_rel_err < 1e-2


def test_clone_propagates_step_size() -> None:
    solver = TorchDiffEqSolver(
        t_span=(0, 1), t_steps=10, method="rk4", step_size=0.01, device="cpu", cache_dir=None
    )
    cloned = solver.clone(device="cpu")
    assert cloned.step_size == 0.01
