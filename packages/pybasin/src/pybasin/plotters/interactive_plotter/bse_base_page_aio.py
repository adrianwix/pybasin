"""Base class for AIO page components."""

from abc import ABC, abstractmethod

from dash import html

from pybasin.basin_stability_estimator import BasinStabilityEstimator


class BseBasePageAIO(ABC):
    """
    Abstract base class for AIO (All-in-One) page components that work with BasinStabilityEstimator.

    AIO components encapsulate layout generation and callbacks within a single
    class using pattern-matching IDs for instance-scoped isolation. Each instance
    has a unique aio_id that prevents callback collisions across multiple instances.

    Subclasses must implement:
    - render(): Generate the component layout with pattern-matching IDs
    - Module-level callbacks using MATCH pattern for instance isolation

    Pattern-matching callbacks use dict IDs via aio_id():
        @callback(
            Output(aio_id('Component', MATCH, 'element'), 'property'),
            Input(aio_id('Component', MATCH, 'control'), 'value')
        )

    This ensures callbacks only trigger for matching aio_id instances.
    """

    _instances: dict[str, "BseBasePageAIO"] = {}

    @classmethod
    def get_instance(cls, instance_id: str) -> "BseBasePageAIO | None":
        """Get instance by ID."""
        return cls._instances.get(instance_id)

    def __init__(
        self,
        bse: BasinStabilityEstimator,
        aio_id: str,
        state_labels: dict[int, str] | None = None,
    ):
        """
        Initialize AIO page component.

        :param bse: Basin stability estimator instance.
        :param aio_id: Unique identifier for this component instance.
        :param state_labels: Optional mapping of state indices to display labels.
        """
        self.bse = bse
        self.aio_id = aio_id
        self.state_labels = state_labels or {}

    def get_state_label(self, idx: int) -> str:
        """Get label for a state variable."""
        return self.state_labels.get(idx, f"State {idx}")

    def get_n_states(self) -> int:
        """Get number of state variables."""
        if self.bse.y0 is None:
            return 0
        return self.bse.y0.shape[1]

    def get_state_options(self) -> list[dict[str, str]]:
        """Get dropdown options for state variable selection."""
        return [
            {"value": str(i), "label": self.get_state_label(i)} for i in range(self.get_n_states())
        ]

    @abstractmethod
    def render(self) -> html.Div:
        """
        Render the complete component layout.

        Returns a div containing the full page structure with pattern-matching
        IDs using this instance's aio_id. Callbacks registered at module level
        will use MATCH pattern to target this specific instance.

        :return: Dash html.Div containing the complete page layout.
        """
        pass
