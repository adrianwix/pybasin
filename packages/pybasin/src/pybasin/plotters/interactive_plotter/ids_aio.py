"""
Pattern-matching ID utilities for Dash AIO components.

This module provides utilities for creating pattern-matching callback IDs
that enable instance-scoped component isolation in Dash applications.

Pattern-matching callbacks use dictionary-based IDs with standardized keys:
- 'component': The component type (e.g., 'StateSpace', 'TrajectoryModal')
- 'aio_id': The instance identifier (unique per component instance)
- 'subcomponent': The specific element within the component (e.g., 'plot', 'button')

These dict IDs enable Dash's pattern-matching callback selectors:
- MATCH: Callback targets components with the same aio_id
- ALL: Callback targets all instances of a component type
- ALLSMALLER: Callback targets parent-child component hierarchies
"""

from typing import Any

from dash.dependencies import _Wildcard  # pyright: ignore[reportPrivateUsage]


def aio_id(component: str, instance_id: str | _Wildcard, subcomponent: str) -> dict[str, Any]:
    """
    Generate a pattern-matching callback ID for AIO components.

    ```python
    aio_id("StateSpace", "uuid-123", "plot")
    # returns {'component': 'StateSpace', 'aio_id': 'uuid-123', 'subcomponent': 'plot'}


    # Usage in Dash callback with MATCH pattern
    @callback(
        Output(aio_id("StateSpace", MATCH, "plot"), "figure"),
        Input(aio_id("StateSpace", MATCH, "scale-dropdown"), "value"),
    )
    def update_plot(scale):
        # This callback only triggers for matching aio_id instances
        return create_figure(scale)
    ```

    :param component: Component type identifier (e.g., 'StateSpace', 'TrajectoryModal').
    :param instance_id: Unique instance identifier (typically UUID or parameter index).
    :param subcomponent: Specific element within the component (e.g., 'plot', 'button', 'store').
    :return: Dictionary ID for pattern-matching callbacks with keys:
        component, aio_id, and subcomponent.
    """
    return {
        "component": component,
        "aio_id": instance_id,
        "subcomponent": subcomponent,
    }
