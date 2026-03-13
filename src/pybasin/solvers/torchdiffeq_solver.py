import logging
from typing import Any

import torch
from torchdiffeq import odeint  # type: ignore[import-untyped]

from pybasin.constants import DEFAULT_CACHE_DIR, UNSET
from pybasin.solvers.base import Solver
from pybasin.solvers.torch_ode_system import ODESystem

logger = logging.getLogger(__name__)


class TorchDiffEqSolver(Solver):
    """
    Differentiable ODE solver with full GPU support and O(1)-memory backpropagation.

    Uses the adjoint method for memory-efficient gradient computation through ODE solutions.
    Supports adaptive-step (dopri5, dopri8, bosh3) and fixed-step (euler, rk4) methods.

    See also: [torchdiffeq GitHub](https://github.com/rtqichen/torchdiffeq)

    Citation:

    ```bibtex
    @misc{torchdiffeq,
        author={Chen, Ricky T. Q.},
        title={torchdiffeq},
        year={2018},
        url={https://github.com/rtqichen/torchdiffeq},
    }
    ```
    """

    def __init__(
        self,
        t_span: tuple[float, float] = (0, 1000),
        t_steps: int = 1000,
        device: str | None = None,
        method: str = "dopri5",
        rtol: float = 1e-8,
        atol: float = 1e-6,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
        t_eval: tuple[float, float] | None = None,
    ):
        """
        Initialize TorchDiffEqSolver.

        :param t_span: Tuple (t_start, t_end) defining the integration interval.
        :param t_steps: Number of evaluation points in the save region.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param method: Integration method from tordiffeq.odeint.
        :param rtol: Relative tolerance (used by adaptive-step methods only).
        :param atol: Absolute tolerance (used by adaptive-step methods only).
        :param cache_dir: Directory for caching integration results. ``None`` disables caching.
        :param t_eval: Optional save region ``(save_start, save_end)``. Only time points
            in this range are stored. Must be contained within ``t_span``. If ``None``,
            defaults to ``t_span``.
        """
        super().__init__(
            t_span,
            t_steps=t_steps,
            device=device,
            rtol=rtol,
            atol=atol,
            cache_dir=cache_dir,
            t_eval=t_eval,
        )
        self.method = method

    def _get_cache_config(self) -> dict[str, Any]:
        """Include method in cache key (rtol/atol handled by base class)."""
        config = super()._get_cache_config()
        config["method"] = self.method
        return config

    def clone(
        self,
        *,
        device: str | None = None,
        t_steps_factor: int = 1,
        cache_dir: str | None | object = UNSET,
    ) -> "TorchDiffEqSolver":
        """Create a copy of this solver, optionally overriding device, resolution, or caching."""
        effective_cache_dir = self._cache_dir if cache_dir is UNSET else cache_dir
        return TorchDiffEqSolver(
            t_span=self.t_span,
            t_steps=self.t_steps * t_steps_factor,
            device=device or self._device_str,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            cache_dir=effective_cache_dir,  # type: ignore[arg-type]
            t_eval=self.t_eval,
        )

    def _integrate(
        self,
        ode_system: ODESystem[Any],
        y0: torch.Tensor,
        save_ts: torch.Tensor,
        params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate using torchdiffeq's odeint.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions with shape (batch, n_dims).
        :param save_ts: Save-region time points (1D tensor).
        :param params: Pre-expanded parameters of shape ``(B*P, n_params)``.
            Stored on the ODE module so ``forward(t, y)`` passes them to
            ``ode(t, y, p)`` instead of calling ``params_to_array()``.
        :return: (save_ts, y_values) where y_values has shape (t_steps, batch, n_dims).
        """
        if params is not None:
            ode_system._batched_params = params  # type: ignore[attr-defined]

        try:
            t_span_start = float(self.t_span[0])
            save_start = float(save_ts[0])

            ts = save_ts
            if save_start > t_span_start:
                anchor = torch.tensor([t_span_start], dtype=save_ts.dtype, device=save_ts.device)
                ts = torch.cat([anchor, save_ts])

            with torch.no_grad():
                y_all: torch.Tensor = odeint(  # type: ignore[assignment]
                    ode_system, y0, ts, method=self.method, rtol=self.rtol, atol=self.atol
                )
            return save_ts, y_all[-len(save_ts) :]
        finally:
            if params is not None:
                del ode_system._batched_params  # type: ignore[attr-defined]
