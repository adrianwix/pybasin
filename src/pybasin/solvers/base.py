import logging
from abc import ABC, abstractmethod
from typing import Any, cast

import torch

from pybasin.cache_manager import CacheManager
from pybasin.constants import DEFAULT_CACHE_DIR, UNSET
from pybasin.protocols import ODESystemProtocol, SolverProtocol
from pybasin.solvers.torch_ode_system import ODESystem
from pybasin.utils import DisplayNameMixin, resolve_cache_dir

logger = logging.getLogger(__name__)


class Solver(SolverProtocol, DisplayNameMixin, ABC):
    """Abstract base class for ODE solvers with persistent caching.

    The cache is stored both in-memory and on disk.
    The cache key is built using:
      - The solver class name,
      - The solver-specific configuration (rtol, atol, method, etc.),
      - The ODE system's string representation via ode_system.get_str(),
      - The serialized initial conditions (y0),
      - The serialized evaluation time points (t_eval).
    """

    def __init__(
        self,
        t_span: tuple[float, float] = (0, 1000),
        t_steps: int = 1000,
        device: str | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
        t_eval: tuple[float, float] | None = None,
    ):
        """
        Initialize the solver with integration parameters.

        :param t_span: Tuple (t_start, t_end) defining the integration interval.
        :param t_steps: Number of evaluation points in the save region.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param rtol: Relative tolerance (used by adaptive-step methods only).
        :param atol: Absolute tolerance (used by adaptive-step methods only).
        :param cache_dir: Directory for caching integration results. Relative paths are
            resolved from the project root. ``None`` disables caching.
        :param t_eval: Optional save region ``(save_start, save_end)``. Only time points
            in this range are stored. Must be contained within ``t_span``. Integration
            runs from ``t_span[0]`` to ``t_eval[1]`` (not ``t_span[1]``), so only
            ``t_span[0]`` is used as a hard start when ``t_eval`` is provided. If ``None``,
            defaults to ``t_span`` (save all points).
        """
        if t_eval is not None and (t_eval[0] < t_span[0] or t_eval[1] > t_span[1]):
            raise ValueError(f"t_eval {t_eval} must be within t_span {t_span}.")
        self.t_span = t_span
        self.t_eval = t_eval
        self.rtol = rtol
        self.atol = atol

        self.t_steps = t_steps
        self._set_device(device)

        self._cache_manager: CacheManager | None = None
        self._cache_dir: str | None = None
        if cache_dir is not None:
            self._cache_dir = resolve_cache_dir(cache_dir)
            self._cache_manager = CacheManager(self._cache_dir)

    def _set_device(self, device: str | None) -> None:
        """
        Set the device for tensor operations with auto-detection and normalization.

        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        # Store original device string for clone()
        self._device_str = device

        # Auto-detect device if not specified and normalize cuda to cuda:0
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            # Normalize "cuda" to "cuda:0" for consistency
            dev = torch.device(device)
            # For CUDA devices, normalize to cuda:0 if no specific index given
            if dev.type == "cuda":
                # If device string was just "cuda" without index, index will be None
                # torch.device("cuda").index returns None, not 0
                idx = dev.index if dev.index is not None else 0  # pyright: ignore[reportUnnecessaryComparison]
                self.device = torch.device(f"cuda:{idx}")
            else:
                self.device = dev

    @abstractmethod
    def clone(
        self,
        *,
        device: str | None = None,
        t_steps_factor: int = 1,
        cache_dir: str | None | object = UNSET,
    ) -> "Solver":
        """
        Create a copy of this solver, optionally overriding device, resolution, or caching.

        :param device: Target device ('cpu', 'cuda'). If None, keeps the current device.
        :param t_steps_factor: Multiply the number of evaluation points by this factor.
        :param cache_dir: Override cache directory. Pass ``None`` to disable caching.
            If not provided, keeps the current setting.
        :return: New solver instance.
        """
        pass

    def _prepare_tensors(self, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare save-region time points and initial conditions with correct device and dtype."""
        save_start, save_end = self.t_eval if self.t_eval is not None else self.t_span

        save_ts = torch.linspace(
            save_start, save_end, self.t_steps, dtype=y0.dtype, device=self.device
        )

        # Warn if y0 is on wrong device or has wrong dtype
        if y0.device != self.device:
            logger.warning(
                "  Warning: y0 is on device %s but solver expects %s", y0.device, self.device
            )

        return save_ts, y0

    def integrate(
        self, ode_system: ODESystemProtocol, y0: torch.Tensor, params: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the ODE system and return the evaluation time points and solution.
        Uses caching to avoid recomputation if the same problem was solved before.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions with shape (batch, n_dims) where batch is the number
                   of initial conditions and n_dims is the number of state variables.
        :param params: Optional 2-D tensor of shape ``(P, n_params)`` with P parameter
            combinations. The solver runs every IC against every combination, producing
            ``B*P`` output trajectories in IC-major order: trajectory ``ic*P + p`` carries
            ``(y0[ic], params[p])``. When ``None``, the ODE system's default parameters
            are used for all ICs.
        :return: Tuple (t_eval, y_values) where y_values has shape ``(t_steps, B*P, n_dims)``.
        """
        if y0.ndim != 2:
            raise ValueError(
                f"y0 must be 2D with shape (batch, n_dims), got shape {y0.shape}. "
                f"For single initial condition, use y0.unsqueeze(0) or y0.reshape(1, -1)."
            )

        save_ts, y0 = self._prepare_tensors(y0)
        ode_system = ode_system.to(self.device)

        if params is not None:
            B = y0.shape[0]
            P = params.shape[0]
            y0 = torch.repeat_interleave(y0, P, dim=0)  # (B*P, n_dims)
            params = params.to(self.device).repeat(B, 1)  # (B*P, n_params)

        def compute() -> tuple[torch.Tensor, torch.Tensor]:
            ode_system_concrete = cast(ODESystem[Any], ode_system)
            logger.debug("[%s] Integrating on %s...", self.__class__.__name__, self.device)
            t_result, y_result = self._integrate(ode_system_concrete, y0, save_ts, params)
            logger.debug("[%s] Integration complete", self.__class__.__name__)
            return t_result, y_result

        if self._cache_manager is not None:
            return self._cache_manager.cached_call(
                solver_name=self.__class__.__name__,
                ode_system=ode_system,
                y0=y0,
                save_ts=save_ts,
                solver_config=self._get_cache_config(),
                device=self.device,
                compute_fn=compute,
                params=params,
            )

        logger.debug(
            "[%s] Cache disabled - integrating on %s...", self.__class__.__name__, self.device
        )
        return compute()

    def _get_cache_config(self) -> dict[str, Any]:
        """
        Get solver-specific configuration for cache key.
        Subclasses should override this to include additional relevant parameters.

        :return: Dictionary of configuration parameters that affect integration results.
        """
        return {"rtol": self.rtol, "atol": self.atol, "t_span": self.t_span}

    @abstractmethod
    def _integrate(
        self,
        ode_system: ODESystem[Any],
        y0: torch.Tensor,
        save_ts: torch.Tensor,
        params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the actual integration using the given solver.
        This method is implemented by subclasses.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions.
        :param save_ts: Save-region time points tensor (computed linspace over the save window).
        :param params: Pre-expanded parameter tensor of shape ``(B*P, n_params)`` as
            produced by :meth:`integrate`. ``None`` when default params should be used.
        :return: (save_ts, y_values)
        """
        pass
