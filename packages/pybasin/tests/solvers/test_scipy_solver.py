import math
from typing import TypedDict

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from pybasin.solvers import JaxSolver, ScipyParallelSolver
from pybasin.solvers.jax_ode_system import JaxODESystem
from pybasin.solvers.numpy_ode_system import NumpyODESystem


class DecayParams(TypedDict):
    decay: float


class ExponentialDecayNumpyODE(NumpyODESystem[DecayParams]):
    def ode(self, t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return p[0] * y


class ExponentialDecayJaxODE(JaxODESystem[DecayParams]):
    def ode(self, t, y, p):  # type: ignore[no-untyped-def,override]
        return p[..., 0:1] * y

    def get_str(self) -> str:
        return f"dy/dt = {self.params['decay']} * y"


class HarmonicParams(TypedDict):
    pass


class HarmonicOscillatorNumpyODE(NumpyODESystem[HarmonicParams]):
    """dx/dt = v, dv/dt = -x. Analytic solution for y0=[1, 0]: x(t)=cos(t), v(t)=-sin(t)."""

    def ode(self, t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        x, v = y[0], y[1]
        return np.array([v, -x])


class HarmonicOscillatorJaxODE(JaxODESystem[HarmonicParams]):
    def ode(self, t, y, p):  # type: ignore[no-untyped-def,override]
        x, v = y[..., 0], y[..., 1]
        return jnp.stack([v, -x], axis=-1)  # type: ignore[no-any-return]

    def get_str(self) -> str:
        return "harmonic_oscillator"


@pytest.fixture
def decay_ode() -> ExponentialDecayNumpyODE:
    return ExponentialDecayNumpyODE({"decay": -1.0})


@pytest.fixture
def harmonic_ode() -> HarmonicOscillatorNumpyODE:
    return HarmonicOscillatorNumpyODE({})


def test_default_rk45_matches_analytic(decay_ode: ExponentialDecayNumpyODE) -> None:
    """No fixed-step gap to worry about here: every method scipy's solve_ivp offers
    (RK45, RK23, DOP853, Radau, BDF, LSODA) is itself adaptive, so there's no equivalent of
    torchdiffeq's/torchode's "fixed-step method silently takes one giant/frozen step" bug --
    `max_step` only bounds the adaptive controller's step size, it isn't the step size."""
    solver = ScipyParallelSolver(t_span=(0, 1), t_steps=10, n_jobs=1, device="cpu", cache_dir=None)

    y0 = torch.tensor([[1.0]])
    _, y = solver.integrate(decay_ode, y0)

    assert y[..., -1].item() == pytest.approx(math.exp(-1.0), abs=1e-4)


def test_large_anchor_gap_still_accurate(decay_ode: ExponentialDecayNumpyODE) -> None:
    """The same t_span-anchor + narrow t_eval save-window pattern that exposed
    torchdiffeq's/torchode's fixed-step gaps: integrate from t_span[0]=0 through a save
    window far from it. solve_ivp integrates continuously from t_span[0] to t_eval[-1]
    regardless of how sparse t_eval is, so this is accurate without any extra configuration."""
    solver = ScipyParallelSolver(
        t_span=(0, 10), t_steps=5, t_eval=(9, 10), n_jobs=1, device="cpu", cache_dir=None
    )

    y0 = torch.tensor([[1.0]])
    t, y = solver.integrate(decay_ode, y0)

    for i in range(t.shape[0]):
        exact = math.exp(-t[i].item())
        assert y[0, 0, i].item() == pytest.approx(exact, rel=1e-4)


def test_matches_diffrax_trajectory(harmonic_ode: HarmonicOscillatorNumpyODE) -> None:
    """Trajectory-level accuracy against JaxSolver/diffrax (Dopri5, rtol=1e-8, atol=1e-6),
    across the same t_span-anchor + narrow t_eval save-window pattern used by the case
    studies."""
    t_span = (0.0, 20.0)
    t_eval = (18.0, 20.0)
    t_steps = 50
    y0 = torch.tensor([[1.0, 0.0]])

    scipy_solver = ScipyParallelSolver(
        t_span=t_span,
        t_steps=t_steps,
        t_eval=t_eval,
        n_jobs=1,
        rtol=1e-8,
        atol=1e-6,
        device="cpu",
        cache_dir=None,
    )
    _, y_scipy = scipy_solver.integrate(harmonic_ode, y0)

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

    max_abs_err = (y_scipy - y_jax).abs().max().item()
    assert max_abs_err < 1e-3
