# pyright: basic
"""Type definitions for interactive plotter options.

This module provides TypedDict-based options for configuring the InteractivePlotter.
All options use ``total=False`` so every field is optional - unspecified values
use sensible defaults at runtime.

Example:

```python
from pybasin.plotters import InteractivePlotter

plotter = InteractivePlotter(
    bse,
    state_labels={0: "θ", 1: "ω"},
    options={
        "initial_view": "templates_phase_space",
        "templates_time_series": {"state_variable": 1},
        "templates_phase_space": {"exclude_templates": ["unbounded"]},
    },
)
```
"""

from copy import deepcopy
from typing import Any, Literal, TypedDict


class StateSpaceOptions(TypedDict, total=False):
    """Options for the State Space page."""

    x_axis: int
    """Index of the state variable to plot on the X axis. Default: 0."""
    y_axis: int
    """Index of the state variable to plot on the Y axis. Default: 1."""
    time_range: tuple[float, float]
    """Fraction of trajectory to display as (start, end) where 0.0 is the beginning and 1.0 is the end. Default: (0.0, 1.0)."""


class FeatureSpaceOptions(TypedDict, total=False):
    """Options for the Feature Space page."""

    x_axis: int
    """Index of the feature to plot on the X axis. Default: 0."""
    y_axis: int | None
    """Index of the feature to plot on the Y axis. None for 1D strip plot. Default: 1."""
    use_filtered: bool
    """Whether to use filtered features. Default: True."""
    include_labels: list[str]
    """Show only these attractor labels. Mutually exclusive with exclude_labels."""
    exclude_labels: list[str]
    """Hide these attractor labels. Mutually exclusive with include_labels."""


class TemplatesPhaseSpaceOptions(TypedDict, total=False):
    """Options for the Templates Phase Space page (2D or 3D)."""

    x_axis: int
    """Index of the state variable to plot on the X axis. Default: 0."""
    y_axis: int
    """Index of the state variable to plot on the Y axis. Default: 1."""
    z_axis: int | None
    """Index of the state variable for 3D plot Z axis. None for 2D plot. Default: None (2D)."""
    include_templates: list[str]
    """Show only these template labels. Mutually exclusive with exclude_templates."""
    exclude_templates: list[str]
    """Hide these template labels. Mutually exclusive with include_templates."""


class TemplatesTimeSeriesOptions(TypedDict, total=False):
    """Options for the Templates Time Series page."""

    state_variable: int
    """Index of the state variable to plot over time. Default: 0."""
    time_range: tuple[float, float]
    """Fraction of trajectory to display as (start, end) where 0.0 is the beginning and 1.0 is the end. Default: (0.0, 1.0)."""
    include_templates: list[str]
    """Show only these template labels. Mutually exclusive with exclude_templates."""
    exclude_templates: list[str]
    """Hide these template labels. Mutually exclusive with include_templates."""
    y_limits: tuple[float, float] | dict[str, tuple[float, float]]
    """Y-axis limits. Tuple applies to all, dict maps label to (y_min, y_max)."""


class ParamOverviewOptions(TypedDict, total=False):
    """Options for the Parameter Overview page."""

    x_scale: Literal["linear", "log"]
    """Scale for the X axis. Default: "linear"."""
    selected_labels: list[str]
    """Attractor labels to display. Default: all labels."""


class ParamOrbitDiagramOptions(TypedDict, total=False):
    """Options for the Parameter Orbit Diagram page."""

    state_dimensions: list[int]
    """Indices of state dimensions to show. Default: all dimensions."""


ViewType = Literal[
    "basin_stability",
    "state_space",
    "feature_space",
    "templates_phase_space",
    "templates_time_series",
    "param_overview",
    "param_orbit_diagram",
]


class InteractivePlotterOptions(TypedDict, total=False):
    """Configuration options for InteractivePlotter defaults.

    All options are optional. Unspecified values use sensible defaults.
    Invalid values (e.g., out-of-bounds indices) trigger a warning and
    fall back to safe defaults.

    Example:

    ```python
    options: InteractivePlotterOptions = {
        "initial_view": "templates_phase_space",
        "templates_time_series": {"state_variable": 1, "time_range": (0.0, 1.0)},
        "templates_phase_space": {"x_axis": 0, "y_axis": 2, "exclude_templates": ["unbounded"]},
    }
    plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "y"}, options=options)
    ```
    """

    initial_view: ViewType
    """The page to display when the plotter opens. Default: "basin_stability"."""
    state_space: StateSpaceOptions
    """Options for the State Space visualization."""
    feature_space: FeatureSpaceOptions
    """Options for the Feature Space visualization."""
    templates_phase_space: TemplatesPhaseSpaceOptions
    """Options for the Templates Phase Space visualization."""
    templates_time_series: TemplatesTimeSeriesOptions
    """Options for the Templates Time Series visualization."""
    param_overview: ParamOverviewOptions
    """Options for the Parameter Overview visualization."""
    param_orbit_diagram: ParamOrbitDiagramOptions
    """Options for the Parameter Orbit Diagram visualization."""


# Default values for all options
_DEFAULT_STATE_SPACE: StateSpaceOptions = {
    "x_axis": 0,
    "y_axis": 1,
    "time_range": (0.0, 1.0),
}

_DEFAULT_FEATURE_SPACE: FeatureSpaceOptions = {
    "x_axis": 0,
    "y_axis": 1,
    "use_filtered": True,
}

_DEFAULT_TEMPLATES_PHASE_SPACE: TemplatesPhaseSpaceOptions = {
    "x_axis": 0,
    "y_axis": 1,
    "z_axis": None,
}

_DEFAULT_TEMPLATES_TIME_SERIES: TemplatesTimeSeriesOptions = {
    "state_variable": 0,
    "time_range": (0.0, 1.0),
}

_DEFAULT_PARAM_OVERVIEW: ParamOverviewOptions = {
    "x_scale": "linear",
}

_DEFAULT_PARAM_ORBIT_DIAGRAM: ParamOrbitDiagramOptions = {}

_DEFAULT_OPTIONS: InteractivePlotterOptions = {
    "initial_view": "basin_stability",
    "state_space": _DEFAULT_STATE_SPACE,
    "feature_space": _DEFAULT_FEATURE_SPACE,
    "templates_phase_space": _DEFAULT_TEMPLATES_PHASE_SPACE,
    "templates_time_series": _DEFAULT_TEMPLATES_TIME_SERIES,
    "param_overview": _DEFAULT_PARAM_OVERVIEW,
    "param_orbit_diagram": _DEFAULT_PARAM_ORBIT_DIAGRAM,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override dict into base dict.

    :param base: Base dictionary with default values.
    :param override: Dictionary with user-provided overrides.
    :return: Merged dictionary.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def merge_options(user_options: InteractivePlotterOptions | None) -> InteractivePlotterOptions:
    """Merge user-provided options with defaults.

    :param user_options: Partial options dict from user, or None for all defaults.
    :return: Complete options dict with all defaults filled in.
    """
    if user_options is None:
        return deepcopy(_DEFAULT_OPTIONS)
    return _deep_merge(_DEFAULT_OPTIONS, user_options)  # type: ignore[return-value]


def filter_by_include_exclude(
    all_items: list[str],
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    """Filter items based on include/exclude lists.

    :param all_items: All available items.
    :param include: If provided, show only these items (mutually exclusive with exclude).
    :param exclude: If provided, hide these items (mutually exclusive with include).
    :return: Filtered list of items.
    :raises ValueError: If both include and exclude are provided.
    """
    if include is not None and exclude is not None:
        raise ValueError(
            "Cannot specify both include and exclude lists. "
            "Use include to show only specific items, or exclude to hide specific items."
        )
    if include is not None:
        return [item for item in all_items if item in include]
    if exclude is not None:
        return [item for item in all_items if item not in exclude]
    return all_items


def infer_z_axis(x_axis: int, y_axis: int, n_states: int) -> int:
    """Infer the Z-axis variable index from unused state variables.

    :param x_axis: X-axis state variable index.
    :param y_axis: Y-axis state variable index.
    :param n_states: Total number of state variables.
    :return: Z-axis variable index (first unused state, or 0 if none available).
    """
    if n_states >= 3:
        used = {x_axis, y_axis}
        remaining = [i for i in range(n_states) if i not in used]
        return remaining[0] if remaining else 0
    return 0


# Mapping from old ViewType to URL paths (for navbar routing)
VIEW_TO_PATH: dict[ViewType, str] = {
    "basin_stability": "/basin-stability",
    "state_space": "/state-space",
    "feature_space": "/feature-space",
    "templates_phase_space": "/phase",
    "templates_time_series": "/time-series",
    "param_overview": "/param-overview",
    "param_orbit_diagram": "/param-orbit-diagram",
}
