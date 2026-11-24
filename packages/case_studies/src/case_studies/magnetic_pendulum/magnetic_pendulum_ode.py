from typing import TypedDict

import torch

from pybasin.solvers.torch_ode_system import ODESystem


class MagneticPendulumParams(TypedDict):
    """Parameters for the magnetic pendulum ODE system."""

    alpha: float  # damping coefficient
    omega: float  # restoring-force (spring) coefficient
    d: float  # magnet height above the pendulum's swing plane
    gamma: float  # magnet strength, shared by every magnet


class MagneticPendulumODE(ODESystem[MagneticPendulumParams]):
    """Magnetic pendulum: a damped bob above `n_magnets` point magnets placed evenly on
    the unit circle, projected onto position space (velocity fixed at ``[0, 0]``).

    State ``y`` has shape ``(..., 4)``: ``[x, y, vx, vy]``.

    A variant of the magnetic-pendulum model of Motter, Gruiz, Károlyi & Tél (2013),
    "Doubly Transient Chaos: Generic Form of Chaos in Autonomous Dissipative Systems",
    Phys. Rev. Lett. 111, 194101, https://doi.org/10.1103/PhysRevLett.111.194101,
    eqs. (2)-(3): same force law, with magnets on the unit circle (Motter et al. place
    them on a triangle of unit edge, circumradius ``1/sqrt(3)``) and a magnet strength
    ``gamma``.

    Source: ``Attractors.jl`` @ ``df7c420b4c1d69e1d662bb679a46f8ef8b53bcaa``
    ``test/mapping/attractor_mapping.jl:196-247`` (``@testset "Magnetic pendulum:
    projected system"``) -- same equations of motion (``magnetic_pendulum_rule``, lines
    204-216) as ``case_studies/magnetic_pendulum/magnetic_pendulum.jl`` in this same
    directory, ported to `torch` for pybasin-side use. Line numbers and content are only
    guaranteed to match at that commit -- re-verify against current Attractors.jl source
    before trusting them at a later one.
    """

    magnets: torch.Tensor
    param_values: torch.Tensor

    def __init__(self, params: MagneticPendulumParams, *, n_magnets: int = 3):
        super().__init__(params)
        self.n_magnets = n_magnets
        angles = torch.arange(n_magnets, dtype=torch.float64) * (2 * torch.pi / n_magnets)
        magnets = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        self.register_buffer("magnets", magnets)

        param_values = torch.tensor(
            [params["alpha"], params["omega"], params["d"], params["gamma"]],
            dtype=torch.float64,
        )
        self.register_buffer("param_values", param_values)

    def params_to_array(self) -> torch.Tensor:
        """``ODESystem.params_to_array()``'s default (``torch.tensor(list(...))``) builds a
        plain, CPU-only, float32 tensor that a solver never moves to its own device --
        unlike registered buffers, which ``Solver.integrate()``'s ``ode_system.to(device)``
        call does move. Overriding to return the registered ``param_values`` buffer instead
        keeps parameters on the same device as everything else `ode()` reads, and in full
        float64 precision.

        :return: Tensor ``[alpha, omega, d, gamma]``, same device as this module's buffers.
        """
        return self.param_values

    def ode(self, t: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Right-hand side of the magnetic pendulum ODE.

        :param t: Current time, scalar tensor.
        :param y: State tensor of shape ``(..., 4)``: ``[x, y, vx, vy]``.
        :param p: Parameter tensor of shape ``(..., 4)`` ordered as ``[alpha, omega, d, gamma]``.
        :return: Time derivatives of shape ``(..., 4)``.
        """
        # Some solvers (e.g. torchode's Runge-Kutta step) carry internal buffers matching
        # y's own dtype and reject mixed-precision arithmetic outright; p and magnets are
        # both built (in params_to_array()/__init__) independently of any particular y, so
        # both need matching here -- letting either stay at its own fixed dtype silently
        # upcasts every downstream term that touches it, not just that one value.
        p = p.to(y.dtype)
        magnets = self.magnets.to(y.dtype)
        alpha, omega, d, gamma = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
        x, y_pos, vx, vy = y[..., 0], y[..., 1], y[..., 2], y[..., 3]

        dx_dt = vx
        dy_dt = vy

        # Broadcast every batch element against every magnet at once: (..., 1) vs
        # (n_magnets,) -> (..., n_magnets), then sum each magnet's pull over the last axis.
        delta_x = x.unsqueeze(-1) - magnets[:, 0]
        delta_y = y_pos.unsqueeze(-1) - magnets[:, 1]
        distance = torch.sqrt(delta_x**2 + delta_y**2 + d.unsqueeze(-1) ** 2)
        magnet_pull_x = (gamma.unsqueeze(-1) * delta_x / distance**3).sum(dim=-1)
        magnet_pull_y = (gamma.unsqueeze(-1) * delta_y / distance**3).sum(dim=-1)

        dvx_dt = -(omega**2) * x - alpha * vx - magnet_pull_x
        dvy_dt = -(omega**2) * y_pos - alpha * vy - magnet_pull_y

        return torch.stack([dx_dt, dy_dt, dvx_dt, dvy_dt], dim=-1)
