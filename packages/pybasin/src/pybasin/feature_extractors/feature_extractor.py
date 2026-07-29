"""Abstract base class for extracting features from ODE solution trajectories."""

from abc import ABC, abstractmethod

import torch

from pybasin.constants import DEFAULT_STEADY_FRACTION
from pybasin.feature_extractors.utils import to_snake_case
from pybasin.solution import Solution


class FeatureExtractor(ABC):
    """Abstract base class for extracting features from ODE solutions.

    Feature extractors transform ODE solution trajectories into feature vectors
    that can be used for basin of attraction classification. This class provides
    utilities for filtering solutions by time (to remove transients).

    ```python
    class AmplitudeExtractor(FeatureExtractor):
        def extract_features(self, solution: Solution) -> torch.Tensor:
            y_filtered = self.filter_time(solution)
            return torch.max(torch.abs(y_filtered), dim=-1)[0]
    ```

    :param time_steady: Time threshold for filtering transients. Only data after this
        time will be used for feature extraction. If ``None`` (default), uses 85%
        of the integration time span (derived from ``solution.time`` at extraction
        time). Set to ``0.0`` to explicitly use the entire time series.
    """

    def __init__(self, time_steady: float | None = None):
        self.time_steady = time_steady
        self._feature_names: list[str] | None = None
        self._num_features: int | None = None

    @abstractmethod
    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract features from an ODE solution.

        This method must be implemented by subclasses to define how features
        are computed from solution trajectories.

        :param solution: ODE solution containing time series data for one or more
            trajectories. The solution.y tensor has shape (B, S, N) where
            B is the batch size (number of initial conditions), S is the number
            of state variables, and N is the number of time steps.
        :return: Feature tensor of shape (B, F) where B is the batch size and F is
            the number of features extracted per trajectory.
        """
        pass

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature names.

        If not explicitly set by a subclass, automatically generates names using
        the pattern: <class_name_snake_case>_<index>. The class name is converted
        to snake_case and the suffix 'FeatureExtractor' is removed (if present).

        :return: List of feature names. Length must match the number of features (F)
            in the output tensor from extract_features().
        """
        if self._feature_names is not None:
            return self._feature_names

        if self._num_features is None:
            return []

        class_name = self.__class__.__name__
        snake_case_name = to_snake_case(class_name)

        if snake_case_name.endswith("_feature_extractor") and len(snake_case_name) > len(
            "_feature_extractor"
        ):
            snake_case_name = snake_case_name[: -len("_feature_extractor")]
        elif snake_case_name == "feature_extractor":
            snake_case_name = "feature"

        if len(snake_case_name) == 0:
            snake_case_name = "feature"

        return [f"{snake_case_name}_{i + 1}" for i in range(self._num_features)]

    def _resolve_time_steady(self, solution: Solution) -> float:
        """Resolve the effective time_steady value.

        If ``time_steady`` was explicitly set, returns that value. Otherwise
        computes a default as 85% of the integration time span from the
        solution's time array.

        :param solution: ODE solution with time tensor of shape (N,).
        :return: The resolved time_steady threshold.
        """
        if self.time_steady is not None:
            return self.time_steady
        t = solution.time
        t0 = float(t[0])
        t1 = float(t[-1])
        return t0 + DEFAULT_STEADY_FRACTION * (t1 - t0)

    def filter_time(self, solution: Solution) -> torch.Tensor:
        """Filter out transient behavior by removing early time steps.

        Removes time steps before ``time_steady`` to exclude transient dynamics
        from feature extraction. This ensures features are computed only from
        steady-state or long-term behavior.

        If ``time_steady`` is ``None``, defaults to 85% of the integration
        time span (i.e. keeps the last 15% of time points).

        ```python
        extractor = FeatureExtractor(time_steady=9.0)  # if time_span=(0, 10)
        filtered = extractor.filter_time(solution)
        # Only time points t >= 9.0 are included
        ```

        :param solution: ODE solution with time tensor of shape (N,) and y tensor
            of shape (B, S, N) where B is batch size, S is number of state
            variables, and N is time steps.
        :return: Filtered tensor of shape (B, S, N') where N' is the number of time
            steps after time_steady. If time_steady is 0 or less than all time
            points, returns the original solution.y unchanged.
        """
        effective_steady = self._resolve_time_steady(solution)
        time_arr = solution.time
        idx_steady = torch.where(time_arr >= effective_steady)[0]
        start_idx = idx_steady[0] if len(idx_steady) > 0 else 0
        y_filtered = solution.y[..., start_idx:]
        return y_filtered
