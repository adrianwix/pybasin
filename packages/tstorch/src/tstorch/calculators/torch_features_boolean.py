from typing import TypedDict

import torch
from torch import Tensor

from tstorch.feature_factory import Feature, ParameterlessFeature

# =============================================================================
# BOOLEAN FEATURES (5 features)
# =============================================================================


class HasDuplicate(ParameterlessFeature):
    """Whether any value in the time series occurs more than once.

    Two elements are considered duplicates when they compare exactly equal.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1), with 1.0 if any value repeats and 0.0 otherwise.
        """
        batch_size, n_states, T = x.shape
        x_flat = x.reshape(-1, T)
        sorted_x = x_flat.sort(dim=-1).values
        has_adj_dup = (sorted_x[:, 1:] == sorted_x[:, :-1]).any(dim=1)
        return has_adj_dup.to(x.dtype).reshape(batch_size, n_states, 1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``has_duplicate``.
        """
        return ["has_duplicate"]


class HasDuplicateMax(ParameterlessFeature):
    """Whether the maximum value in the time series occurs more than once.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1), with 1.0 if the maximum value repeats and 0.0 otherwise.
        """
        max_val = x.max(dim=2, keepdim=True).values
        return ((x == max_val).sum(dim=2, keepdim=True) > 1).to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``has_duplicate_max``.
        """
        return ["has_duplicate_max"]


class HasDuplicateMin(ParameterlessFeature):
    """Whether the minimum value in the time series occurs more than once.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1), with 1.0 if the minimum value repeats and 0.0 otherwise.
        """
        min_val = x.min(dim=2, keepdim=True).values
        return ((x == min_val).sum(dim=2, keepdim=True) > 1).to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``has_duplicate_min``.
        """
        return ["has_duplicate_min"]


class HasVarianceLargerThanStandardDeviation(ParameterlessFeature):
    """Whether the variance of the time series is larger than its standard deviation.

    Evaluated as ``std(x) > 1`` rather than comparing ``std(x)**2`` against
    ``std(x)`` directly. The two are algebraically equivalent for ``std(x) >= 0``.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1), with 1.0 if variance exceeds standard deviation and
            0.0 otherwise.
        """
        std = x.std(dim=2, correction=0, keepdim=True)
        return (std > 1).to(x.dtype)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``has_variance_larger_than_standard_deviation``.
        """
        return ["has_variance_larger_than_standard_deviation"]


class LargeStandardDeviationParams(TypedDict):
    """Parameters for one `LargeStandardDeviation` output column.

    Attributes:
        r: Threshold multiplier applied to the range ``max(x) - min(x)``.
    """

    r: float


class LargeStandardDeviation(Feature[LargeStandardDeviationParams]):
    """Whether the standard deviation exceeds a fraction of the series range.

    The feature returns 1.0 when ``std(x) > r * (max(x) - min(x))`` and 0.0 otherwise. For
    several requested ``r`` values, the standard deviation and range are computed once and
    broadcast against all thresholds.

    Args:
        params: One dict per requested output, each with an ``r`` key.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[LargeStandardDeviationParams]) -> None:
        self.params: list[LargeStandardDeviationParams] = params
        self._rs: Tensor = torch.tensor([p["r"] for p in params], dtype=torch.float64)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(params)).
        """
        return _has_large_standard_deviation(x, self._rs)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in the same order as ``params``. Format: ``large_standard_deviation__r_<r>``.
        """
        return [f"large_standard_deviation__r_{p['r']}" for p in self.params]


def _has_large_standard_deviation(x: Tensor, rs: Tensor) -> Tensor:
    std = x.std(dim=2, correction=0, keepdim=True)  # (B, S, 1)
    range_val = x.max(dim=2, keepdim=True).values - x.min(dim=2, keepdim=True).values
    rs_tensor = rs.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
    return (std > rs_tensor * range_val).to(x.dtype)
