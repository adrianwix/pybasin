import logging
from typing import Any

import torch

from pybasin.constants import DEFAULT_CACHE_DIR, UNSET
from pybasin.solvers.base import Solver
from pybasin.solvers.discrete_map_system import DiscreteMapSystem
from pybasin.solvers.torch_ode_system import ODESystem

logger = logging.getLogger(__name__)


class DiscreteMapSolver(Solver):
    """Drives a :class:`~pybasin.solvers.discrete_map_system.DiscreteMapSystem` by treating
    each unit of time as one map iteration: ``save_ts = [0, 1, 2, ..., t_steps - 1]``,
    computed by calling ``step()`` repeatedly from the initial condition.

    ``rtol``/``atol`` are accepted (for :class:`~pybasin.solvers.base.Solver` constructor
    compatibility) but unused -- a discrete map has no adaptive step-size control.
    """

    def __init__(
        self,
        t_steps: int = 1000,
        device: str | None = None,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
    ):
        """
        :param t_steps: Number of map states to record, including the initial one.
            Iterates ``step()`` exactly ``t_steps - 1`` times.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param cache_dir: Directory for caching integration results. ``None`` disables caching.
        """
        super().__init__(
            t_span=(0, t_steps - 1),
            t_steps=t_steps,
            device=device,
            rtol=0.0,
            atol=0.0,
            cache_dir=cache_dir,
            t_eval=None,
        )

    def clone(
        self,
        *,
        device: str | None = None,
        t_steps_factor: int = 1,
        cache_dir: str | None | object = UNSET,
    ) -> "DiscreteMapSolver":
        """Create a copy of this solver, optionally overriding device, resolution, or caching."""
        effective_cache_dir = self._cache_dir if cache_dir is UNSET else cache_dir
        return DiscreteMapSolver(
            t_steps=self.t_steps * t_steps_factor,
            device=device or self._device_str,
            cache_dir=effective_cache_dir,  # type: ignore[arg-type]
        )

    def _integrate(
        self,
        ode_system: ODESystem[Any],
        y0: torch.Tensor,
        save_ts: torch.Tensor,
        params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Iterate ``ode_system.step()`` once per unit of ``save_ts``.

        :param ode_system: A :class:`DiscreteMapSystem` instance.
        :param y0: Initial conditions with shape (batch, n_dims).
        :param save_ts: Save-region time points; must be exactly ``[0, 1, ..., t_steps - 1]``
            (:class:`DiscreteMapSolver` does not support ``t_eval`` sub-windowing).
        :param params: Pre-expanded parameters of shape ``(B*P, n_params)``.
        :return: ``(save_ts, y_values)`` where ``y_values`` has shape ``(batch, n_dims, t_steps)``.
        :raises TypeError: If ``ode_system`` is not a :class:`DiscreteMapSystem`.
        :raises ValueError: If ``save_ts`` is not unit-spaced starting at 0.
        """
        if not isinstance(ode_system, DiscreteMapSystem):
            raise TypeError(
                f"DiscreteMapSolver requires a DiscreteMapSystem, got {type(ode_system).__name__}."
            )
        n_steps = save_ts.shape[0]
        expected_ts = torch.arange(n_steps, dtype=save_ts.dtype, device=save_ts.device)
        if not torch.equal(save_ts, expected_ts):
            raise ValueError(
                f"DiscreteMapSolver requires save_ts == [0, 1, ..., t_steps-1], got {save_ts}."
            )

        p = params if params is not None else ode_system.params_to_array().to(y0.device)

        trajectory = torch.empty((n_steps, *y0.shape), dtype=y0.dtype, device=y0.device)
        trajectory[0] = y0
        state = y0
        for step_idx in range(1, n_steps):
            state = ode_system.step(state, p)
            trajectory[step_idx] = state

        y_result = trajectory.permute(1, 2, 0).contiguous()
        return save_ts, y_result
