"""Default feature extractor with a fixed statistical feature set."""

import threading

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
    :param normalize: Whether to z-score normalize features. Default ``True``. Mean/std
        are computed on the first call and reused (frozen) for every subsequent call,
        matching ``TorchFeatureExtractorAdapter``'s normalization behavior. Set ``False``
        to get the raw statistics.

    .. warning::
        ``BasinStabilityStudy`` reuses one ``feature_extractor`` instance across every
        point of a parameter sweep. With ``normalize=True`` the mean/std frozen from the
        *first* point's features are silently applied to every later point too, even if
        that point's parameter value shifts the feature distribution onto a different
        scale. There is currently no per-point re-fit or re-scaling hook for this case --
        anyone adding continuation (reusing one point's fit as a warm start for the next)
        should treat this frozen-scale reuse as a likely source of silently wrong
        normalization, not an oversight to just delete ``reset_scaler()`` calls around.
    """

    def __init__(self, time_steady: float | None = None, normalize: bool = True):
        super().__init__(time_steady)
        self.normalize = normalize
        self._is_fitted = False
        self._fit_lock = threading.Lock()
        self._feature_mean: torch.Tensor | None = None
        self._feature_std: torch.Tensor | None = None

    def reset_scaler(self) -> None:
        """Reset the normalization parameters, so the next call re-fits mean/std."""
        with self._fit_lock:
            self._feature_mean = None
            self._feature_std = None
            self._is_fitted = False

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

        if self.normalize:
            with self._fit_lock:
                if not self._is_fitted:
                    self._feature_mean = features.mean(dim=0, keepdim=True)
                    self._feature_std = features.std(dim=0, keepdim=True)
                    self._feature_std = torch.where(
                        self._feature_std == 0,
                        torch.ones_like(self._feature_std),
                        self._feature_std,
                    )
                    self._is_fitted = True

                assert self._feature_mean is not None
                assert self._feature_std is not None
                features = (features - self._feature_mean.to(features.device)) / (
                    self._feature_std.to(features.device)
                )

        return features
