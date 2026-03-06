# pyright: reportUntypedBaseClass=false
import logging
from typing import Any

import torch
import torchode as to  # type: ignore[import-untyped]

from pybasin.constants import DEFAULT_CACHE_DIR, UNSET
from pybasin.solvers.base import Solver
from pybasin.solvers.torch_ode_system import ODESystem

logger = logging.getLogger(__name__)


class TorchOdeSolver(Solver):
    """
    Parallel ODE solver with independent step sizes per batch element.

    Compatible with PyTorch's JIT compiler for performance optimization. Unlike other
    solvers, torchode can take different step sizes for each sample in a batch, avoiding
    performance traps for problems of varying stiffness.

    See also: [torchode documentation](https://torchode.readthedocs.io/en/latest/)

    Citation:

    ```bibtex
    @inproceedings{lienen2022torchode,
        title = {torchode: A Parallel {ODE} Solver for PyTorch},
        author = {Marten Lienen and Stephan G{"u}nnemann},
        booktitle = {The Symbiosis of Deep Learning and Differential Equations II, NeurIPS},
        year = {2022},
        url = {https://openreview.net/forum?id=uiKVKTiUYB0}
    }
    ```
    """

    def __init__(
        self,
        time_span: tuple[float, float] = (0, 1000),
        n_steps: int = 1000,
        device: str | None = None,
        method: str = "dopri5",
        rtol: float = 1e-8,
        atol: float = 1e-6,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
    ):
        """
        Initialize TorchOdeSolver.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param n_steps: Number of evaluation points.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param method: Integration method ('dopri5', 'tsit5', 'euler', 'heun').
        :param rtol: Relative tolerance (used by adaptive-step methods only).
        :param atol: Absolute tolerance (used by adaptive-step methods only).
        :param cache_dir: Directory for caching integration results. ``None`` disables caching.
        """
        super().__init__(
            time_span, n_steps=n_steps, device=device, rtol=rtol, atol=atol, cache_dir=cache_dir
        )
        self.method = method.lower()

    def _get_cache_config(self) -> dict[str, Any]:
        """Include method in cache key (rtol/atol handled by base class)."""
        config = super()._get_cache_config()
        config["method"] = self.method
        return config

    def clone(
        self,
        *,
        device: str | None = None,
        n_steps_factor: int = 1,
        cache_dir: str | None | object = UNSET,
    ) -> "TorchOdeSolver":
        """Create a copy of this solver, optionally overriding device, resolution, or caching."""
        effective_cache_dir = self._cache_dir if cache_dir is UNSET else cache_dir
        return TorchOdeSolver(
            time_span=self.time_span,
            n_steps=self.n_steps * n_steps_factor,
            device=device or self._device_str,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            cache_dir=effective_cache_dir,  # type: ignore[arg-type]
        )

    def _integrate(
        self, ode_system: ODESystem[Any], y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate using torchode.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions with shape (batch, n_dims).
        :param t_eval: Time points at which the solution is evaluated (1D tensor).
        :return: (t_eval, y_values) where y_values has shape (n_steps, batch, n_dims).
        """
        batch_size = y0.shape[0]

        # For torchode, we need t_start and t_end as (batch,) tensors
        # Repeat for each sample in the batch
        t_start = torch.full(
            (batch_size,), t_eval[0].item(), device=t_eval.device, dtype=t_eval.dtype
        )
        t_end = torch.full(
            (batch_size,), t_eval[-1].item(), device=t_eval.device, dtype=t_eval.dtype
        )

        # t_eval needs to be (batch, n_steps) - repeat for each sample
        t_eval_batched = t_eval.unsqueeze(0).expand(batch_size, -1) if t_eval.ndim == 1 else t_eval

        # Create ODE function wrapper for torchode
        # torchode calls f(t, y) where t is scalar and y is (batch, n_dims)
        def torchode_func(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # y shape: (batch, n_dims)
            # Call ODE system which handles batched input correctly
            return ode_system(t, y)

        # Create torchode components
        term = to.ODETerm(torchode_func)  # pyright: ignore[reportArgumentType]

        # Select step method
        if self.method == "dopri5":
            step_method = to.Dopri5(term=term)
        elif self.method == "tsit5":
            step_method = to.Tsit5(term=term)
        elif self.method == "euler":
            step_method = to.Euler(term=term)
        elif self.method == "heun":
            step_method = to.Heun(term=term)
        else:
            raise ValueError(
                f"Unknown method: {self.method}. Available: dopri5, tsit5, euler, midpoint, heun"
            )

        step_size_controller = to.IntegralController(atol=self.atol, rtol=self.rtol, term=term)
        solver = to.AutoDiffAdjoint(step_method, step_size_controller)  # pyright: ignore[reportArgumentType]

        # Create initial value problem with matching batch sizes
        problem = to.InitialValueProblem(
            y0=y0,  # pyright: ignore[reportArgumentType]
            t_start=t_start,  # pyright: ignore[reportArgumentType]
            t_end=t_end,  # pyright: ignore[reportArgumentType]
            t_eval=t_eval_batched,  # pyright: ignore[reportArgumentType]
        )

        # Solve
        with torch.inference_mode():
            solution = solver.solve(problem)

            # Extract solution and transpose to match expected format
            # torchode returns (batch, n_steps, n_dims)
            # We need (n_steps, batch, n_dims) to match TorchDiffEqSolver
            y_result = solution.ys.transpose(0, 1)

        return t_eval, y_result
