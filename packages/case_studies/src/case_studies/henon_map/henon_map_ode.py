"""`HenonMapODE`: wraps the Hénon map rule as a `DiscreteMapSystem`, so it can be driven
by `BasinStabilityEstimator` via `DiscreteMapSolver`, exactly like every continuous-time
case study is driven by an `ODESystem` + a continuous-time `Solver`.

    x' = 1 - p1*x^2 + y
    y' = p2*x

This is the map of Hénon (1976), "A two-dimensional mapping with a strange attractor",
Commun. Math. Phys. 50, 69-77, https://doi.org/10.1007/BF01608556, with ``a = p1``,
``b = p2`` (``a = 1.4``, ``b = 0.3`` in the original paper).

A discrete map has no continuous-time right-hand side, so it does not fit
pybasin's `ODESystem` directly -- `HenonMapODE.step()` below is called by
`DiscreteMapSolver` once per iteration instead of `ODESystem.ode()`.

Source: `Attractors.jl` @ `df7c420b4c1d69e1d662bb679a46f8ef8b53bcaa`
`test/mapping/histogram_grouping.jl:6` -- `henon_rule` below is the same one-line rule.
Line numbers and content are only guaranteed to match at that commit -- re-verify against
current Attractors.jl source before trusting them at a later one.
"""

from typing import TypedDict

import torch

from pybasin.solvers.discrete_map_system import DiscreteMapSystem

HENON_DEFAULT_P: tuple[float, float] = (1.4, 0.3)


def henon_rule(
    state: torch.Tensor,
    p: tuple[float, float] | tuple[torch.Tensor, torch.Tensor] = HENON_DEFAULT_P,
) -> torch.Tensor:
    """One Hénon map step: ``x' = 1 - p1*x^2 + y``, ``y' = p2*x``.

    :param state: Current state, shape ``(..., 2)``.
    :param p: Map parameters ``(p1, p2)``.
    :return: Next state, shape ``(..., 2)``.
    """
    p1, p2 = p
    x, y = state[..., 0], state[..., 1]
    return torch.stack([1.0 - p1 * x**2 + y, p2 * x], dim=-1)


class HenonMapParams(TypedDict):
    """Parameters for the Hénon map."""

    p1: float
    p2: float


class HenonMapODE(DiscreteMapSystem[HenonMapParams]):
    """`henon_rule`, as a `DiscreteMapSystem` for use with `DiscreteMapSolver`."""

    def params_to_array(self) -> torch.Tensor:
        """Build ``[p1, p2]`` directly in float64.

        ``ODESystem.params_to_array()``'s default (``torch.tensor(list(...))``, torch's
        default float32 dtype) already rounds p1/p2 before any later dtype cast could
        run -- upcasting a float32 value to float64 afterward reproduces that rounded
        value exactly, it does not recover the original precision. That rounding is
        usually negligible, but the Hénon map is chaotic: a ~1e-8 relative error in p1/p2
        amplifies exponentially over enough iterations into a visibly different
        trajectory, not just a rounding-level one -- so this needs true float64
        precision from construction, not a cast applied too late.

        :return: Tensor ``[p1, p2]``, dtype float64.
        """
        return torch.tensor([self.params["p1"], self.params["p2"]], dtype=torch.float64)

    def step(self, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        :param y: Current state, shape ``(..., 2)``.
        :param p: Parameter tensor of shape ``(..., 2)`` ordered as ``[p1, p2]``.
        :return: Next state, shape ``(..., 2)``.
        """
        return henon_rule(y, (p[..., 0], p[..., 1]))
