"""Shared trajectory cache for AIO components."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pybasin.basin_stability_estimator import BasinStabilityEstimator


class TrajectoryCache:
    """
    Centralized in-memory cache for template trajectories.

    All trajectory plot components share the same cache to avoid redundant
    expensive JAX integrations. The cache is keyed by BSE instance ID.
    """

    _cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def get_or_integrate(cls, bse: "BasinStabilityEstimator") -> tuple[np.ndarray, np.ndarray]:
        """
        Get cached trajectories or integrate and cache them.

        Uses the BSE's template_integrator for proper ODE params and solver selection.

        :param bse: Basin stability estimator with computed results.
        :return: Tuple of (time_array, trajectories_array) where time_array has
            shape (n_timesteps,) and trajectories_array has shape
            (n_templates, n_states, n_timesteps).
        """
        bse_id = id(bse)

        if bse_id in cls._cache:
            return cls._cache[bse_id]

        if bse.template_integrator is None:
            empty_time = np.array([0.0, 1.0])
            empty_traj = np.zeros((1, 1, 2))
            cls._cache[bse_id] = (empty_time, empty_traj)
            return empty_time, empty_traj

        if bse.template_integrator.solution is not None:
            time_array = bse.template_integrator.solution.time.cpu().numpy()
            traj_array = bse.template_integrator.solution.y.cpu().numpy()
        else:
            bse.template_integrator.integrate(
                solver=bse.solver,
                ode_system=bse.ode_system,
            )

            if bse.template_integrator.solution is None:
                raise RuntimeError("Template integration failed to create solution")

            time_array = bse.template_integrator.solution.time.cpu().numpy()
            traj_array = bse.template_integrator.solution.y.cpu().numpy()

        cls._cache[bse_id] = (time_array, traj_array)
        return time_array, traj_array

    @classmethod
    def clear(cls, bse: "BasinStabilityEstimator | None" = None) -> None:
        """
        Clear cache for specific BSE instance or all instances.

        :param bse: If provided, clear only this BSE's cache. Otherwise clear all.
        """
        if bse is not None:
            cls._cache.pop(id(bse), None)
        else:
            cls._cache.clear()
