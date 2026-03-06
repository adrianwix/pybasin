import hashlib
import logging
import os
import pickle
import shutil
from collections.abc import Callable
from typing import Any

import torch
from safetensors.torch import load_file, save_file  # pyright: ignore[reportUnknownVariableType]

from pybasin.protocols import ODESystemProtocol

logger = logging.getLogger(__name__)

__all__ = ["CacheManager"]

Result = tuple[torch.Tensor, torch.Tensor]


class CacheManager:
    """Manages persistent caching of integration results."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def build_key(
        self,
        solver_name: str,
        ode_system: ODESystemProtocol,
        y0: torch.Tensor,
        save_ts: torch.Tensor,
        solver_config: dict[str, Any] | None = None,
    ) -> str:
        """Build a unique cache key based on solver type, configuration, ODE system, and initial conditions.

        :param solver_name: Name of the solver class.
        :param ode_system: The ODE system being solved.
        :param y0: Initial conditions.
        :param save_ts: Save-region time points tensor (the computed linspace).
        :param solver_config: Dictionary of solver-specific parameters (rtol, atol, method, etc.).
        """
        params_tuple = (
            tuple(sorted(ode_system.params.items())) if hasattr(ode_system, "params") else ()
        )

        key_data = (
            solver_name,
            ode_system.get_str(),
            params_tuple,
            y0.detach().cpu().numpy().tobytes(),
            save_ts.detach().cpu().numpy().tobytes(),
            tuple(sorted(solver_config.items())) if solver_config else (),
        )
        key_bytes = pickle.dumps(key_data)
        return hashlib.md5(key_bytes).hexdigest()

    def cached_call(
        self,
        solver_name: str,
        ode_system: ODESystemProtocol,
        y0: torch.Tensor,
        save_ts: torch.Tensor,
        solver_config: dict[str, Any],
        device: torch.device,
        compute_fn: Callable[[], Result],
    ) -> Result:
        """Check cache, compute on miss, save, and return the result.

        :param solver_name: Name of the solver class (for the cache key).
        :param ode_system: The ODE system being solved.
        :param y0: Initial conditions (CPU tensor for key building).
        :param save_ts: Save-region time points tensor (CPU, for key building).
        :param solver_config: Solver-specific parameters that affect results.
        :param device: Device to load cached tensors onto.
        :param compute_fn: Callable that performs the actual integration.
        :return: Tuple of (save_ts, y_values).
        """
        cache_key = self.build_key(solver_name, ode_system, y0, save_ts, solver_config)
        cached_result = self.load(cache_key, device)

        if cached_result is not None:
            logger.debug("[%s] Loaded result from cache", solver_name)
            return cached_result

        logger.debug("[%s] Cache miss - computing...", solver_name)
        t_result, y_result = compute_fn()
        self.save(cache_key, t_result, y_result)
        return t_result, y_result

    def load(
        self, cache_key: str, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Load cached result from disk if it exists."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.safetensors")

        if not os.path.exists(cache_file):
            return None

        try:
            data = load_file(cache_file, device=str(device))
            return data["t"], data["y"]
        except Exception:
            logger.warning("Cache file corrupted. Deleting and recomputing.")
            os.remove(cache_file)
            return None

    def save(self, cache_key: str, t: torch.Tensor, y: torch.Tensor) -> None:
        """Save integration result to disk cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.safetensors")

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        usage = shutil.disk_usage(os.path.dirname(cache_file))
        free_gb = usage.free / (1024**3)
        if free_gb < 1:
            logger.warning("Only %.2fGB free space available.", free_gb)

        try:
            save_file({"t": t.cpu().contiguous(), "y": y.cpu().contiguous()}, cache_file)
        except OSError as e:
            logger.error("Error saving to cache: %s", e)
            raise
