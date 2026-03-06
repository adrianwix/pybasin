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
        time_span: tuple[float, float] = (0, 1000),
        n_steps: int = 1000,
        device: str | None = None,
        rtol: float = 1e-8,
        atol: float = 1e-6,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
    ):
        """
        Initialize the solver with integration parameters.

        :param time_span: Tuple (t_start, t_end) defining the integration interval.
        :param n_steps: Number of evaluation points.
        :param device: Device to use ('cuda', 'cpu', or None for auto-detect).
        :param rtol: Relative tolerance (used by adaptive-step methods only).
        :param atol: Absolute tolerance (used by adaptive-step methods only).
        :param cache_dir: Directory for caching integration results. Relative paths are
            resolved from the project root. ``None`` disables caching.
        """
        self.time_span = time_span
        self.rtol = rtol
        self.atol = atol

        self.n_steps = n_steps
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
        n_steps_factor: int = 1,
        cache_dir: str | None | object = UNSET,
    ) -> "Solver":
        """
        Create a copy of this solver, optionally overriding device, resolution, or caching.

        :param device: Target device ('cpu', 'cuda'). If None, keeps the current device.
        :param n_steps_factor: Multiply the number of evaluation points by this factor.
        :param cache_dir: Override cache directory. Pass ``None`` to disable caching.
            If not provided, keeps the current setting.
        :return: New solver instance.
        """
        pass

    def _prepare_tensors(self, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare time evaluation points and initial conditions with correct device and dtype."""
        t_start, t_end = self.time_span

        t_eval = torch.linspace(t_start, t_end, self.n_steps, dtype=y0.dtype, device=self.device)

        # Warn if y0 is on wrong device or has wrong dtype
        if y0.device != self.device:
            logger.warning(
                "  Warning: y0 is on device %s but solver expects %s", y0.device, self.device
            )

        return t_eval, y0

    def integrate(
        self, ode_system: ODESystemProtocol, y0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Solve the ODE system and return the evaluation time points and solution.
        Uses caching to avoid recomputation if the same problem was solved before.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions with shape (batch, n_dims) where batch is the number
                   of initial conditions and n_dims is the number of state variables.
        :return: Tuple (t_eval, y_values) where y_values has shape (n_steps, batch, n_dims).
        """
        if y0.ndim != 2:
            raise ValueError(
                f"y0 must be 2D with shape (batch, n_dims), got shape {y0.shape}. "
                f"For single initial condition, use y0.unsqueeze(0) or y0.reshape(1, -1)."
            )

        t_eval, y0 = self._prepare_tensors(y0)
        ode_system = ode_system.to(self.device)

        def compute() -> tuple[torch.Tensor, torch.Tensor]:
            ode_system_concrete = cast(ODESystem[Any], ode_system)
            logger.debug("[%s] Integrating on %s...", self.__class__.__name__, self.device)
            t_result, y_result = self._integrate(ode_system_concrete, y0, t_eval)
            logger.debug("[%s] Integration complete", self.__class__.__name__)
            return t_result, y_result

        if self._cache_manager is not None:
            return self._cache_manager.cached_call(
                solver_name=self.__class__.__name__,
                ode_system=ode_system,
                y0=y0,
                t_eval=t_eval,
                solver_config=self._get_cache_config(),
                device=self.device,
                compute_fn=compute,
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
        return {"rtol": self.rtol, "atol": self.atol}

    @abstractmethod
    def _integrate(
        self, ode_system: ODESystem[Any], y0: torch.Tensor, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform the actual integration using the given solver.
        This method is implemented by subclasses.

        :param ode_system: An instance of ODESystem.
        :param y0: Initial conditions.
        :param t_eval: Time points at which the solution is evaluated.
        :return: (t_eval, y_values)
        """
        pass
