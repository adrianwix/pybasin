"""Default feature extractor with a fixed statistical feature set."""

import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution

_FEATURE_NAMES: tuple[str, ...] = (
    "median",
    "mean",
    "standard_deviation",
    "variance",
    "root_mean_square",
    "maximum",
    "absolute_maximum",
    "minimum",
    "delta",
    "log_delta",
)


def _median(x: torch.Tensor) -> torch.Tensor:
    """Median along the last (time) dim, matching ``numpy.median`` for even-length input.

    ``torch.median`` returns the lower of the two middle values for an
    even-length input; this averages them instead, like ``numpy.median``.
    """
    n = x.shape[-1]
    sorted_x = x.sort(dim=-1).values
    if n % 2 == 1:
        return sorted_x[..., n // 2]
    return (sorted_x[..., n // 2 - 1] + sorted_x[..., n // 2]) / 2


def _delta(x: torch.Tensor) -> torch.Tensor:
    """Absolute difference between the maximum and the mean along the time dim."""
    return (x.max(dim=-1).values - x.mean(dim=-1)).abs()


class DefaultFeatureExtractor(FeatureExtractor):
    """Extracts a fixed set of statistical features from every state variable.

    Always computes the same 10 features per state variable: ``median``, ``mean``,
    ``standard_deviation``, ``variance``, ``root_mean_square``, ``maximum``,
    ``absolute_maximum``, ``minimum``, ``delta`` (``|max - mean|``), and
    ``log_delta`` (``log(delta + 1e-10)``). Unlike :class:`TsfreshFeatureExtractor`
    or :class:`NoldsFeatureExtractor`, the feature set is not configurable, which
    keeps this the simplest extractor to reach for when no particular feature
    selection is required.

    ```python
    extractor = DefaultFeatureExtractor(time_steady=9.0)
    features = extractor.extract_features(solution)
    ```

    :param time_steady: Time threshold for filtering transients. If ``None``
        (default), uses 85% of the integration time span.
    """

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract the fixed statistical feature set from an ODE solution.

        :param solution: ODE solution with y tensor of shape (B, S, N) where B is
            batch size, S is number of state variables, and N is time steps.
        :return: Feature tensor of shape (B, S * 10), with features grouped by
            state variable in the order given by :attr:`feature_names`.
        """
        y_filtered = self.filter_time(solution)
        _n_batch, n_states, _n_steps = y_filtered.shape

        delta = _delta(y_filtered)
        stats = {
            "median": _median(y_filtered),
            "mean": y_filtered.mean(dim=-1),
            "standard_deviation": y_filtered.std(dim=-1, correction=0),
            "variance": y_filtered.var(dim=-1, correction=0),
            "root_mean_square": torch.sqrt((y_filtered**2).mean(dim=-1)),
            "maximum": y_filtered.max(dim=-1).values,
            "absolute_maximum": y_filtered.abs().max(dim=-1).values,
            "minimum": y_filtered.min(dim=-1).values,
            "delta": delta,
            "log_delta": torch.log(delta + 1e-10),
        }

        per_state_blocks = [
            torch.stack([stats[name][:, state_idx] for name in _FEATURE_NAMES], dim=1)
            for state_idx in range(n_states)
        ]
        features = torch.cat(per_state_blocks, dim=1)

        self._feature_names = [
            f"state_{state_idx}__{name}" for state_idx in range(n_states) for name in _FEATURE_NAMES
        ]
        self._num_features = features.shape[1]

        return features
