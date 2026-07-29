# pyright: basic
"""Centralized ID registry for Dash components.

This module provides a single source of truth for all component IDs used
across the interactive plotter pages, ensuring consistent naming and
avoiding ID collisions.
"""

from pybasin.plotters.types import ViewType


class IDs:
    """Registry of component ID prefixes and utilities.

    Each page has a unique prefix. Use the ``id()`` method to generate
    fully qualified component IDs.

    Example:

    ```python
    IDs.id(IDs.STATE_SPACE, "plot")  # returns 'state_space-plot'
    IDs.id(IDs.FEATURE_SPACE, "x-select")  # returns 'feature_space-x-select'
    ```
    """

    # Page prefixes - aligned with ViewType from types.py
    BASIN_STABILITY: ViewType = "basin_stability"
    STATE_SPACE: ViewType = "state_space"
    FEATURE_SPACE: ViewType = "feature_space"
    TEMPLATES_PHASE_SPACE: ViewType = "templates_phase_space"
    TEMPLATES_TIME_SERIES: ViewType = "templates_time_series"
    PARAM_OVERVIEW: ViewType = "param_overview"
    PARAM_ORBIT_DIAGRAM: ViewType = "param_orbit_diagram"

    @staticmethod
    def id(prefix: str, suffix: str) -> str:
        """Generate a fully qualified component ID.

        :param prefix: The page or component prefix (e.g., IDs.STATE_SPACE).
        :param suffix: The component-specific suffix (e.g., "plot", "x-select").
        :return: Combined ID string like "state_space-plot".
        """
        return f"{prefix}-{suffix}"
