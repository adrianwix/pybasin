import torch
from torch import Tensor

from tstorch.feature_factory import ParameterlessFeature


def _sorted_adjacent_duplicates(x: Tensor, *, nan_as_equal: bool = False) -> tuple[Tensor, Tensor]:
    """Sort each series and flag adjacent duplicate pairs."""
    batch_size, n_states, time_steps = x.shape
    sorted_x = x.reshape(batch_size * n_states, time_steps).sort(dim=-1).values
    is_duplicate = sorted_x[:, 1:] == sorted_x[:, :-1]
    if nan_as_equal:
        # NumPy unique groups NaNs, while pandas value_counts drops them.
        both_nan = torch.isnan(sorted_x[:, 1:]) & torch.isnan(sorted_x[:, :-1])
        is_duplicate = is_duplicate | both_nan
    return sorted_x, is_duplicate


def _duplicate_neighbor_masks(is_duplicate: Tensor) -> tuple[Tensor, Tensor]:
    """Expand adjacent-pair flags to flags for each sorted position."""
    duplicate_before = torch.nn.functional.pad(is_duplicate, (1, 0), value=False)
    duplicate_after = torch.nn.functional.pad(is_duplicate, (0, 1), value=False)
    return duplicate_before, duplicate_after


class PercentageOfReoccurringDatapointsToAllDatapoints(ParameterlessFeature):
    """Fraction of datapoints whose value occurs more than once.

    Every occurrence of a repeated value contributes to the numerator. NaNs do not count as
    repeated, matching the default NaN-dropping behavior of pandas ``Series.value_counts`` used
    by tsfresh, but they remain part of the denominator. An empty series produces NaN.

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
            Tensor of shape (B, S, 1).
        """
        batch_size, n_states, time_steps = x.shape
        if time_steps == 0:
            return torch.full(
                (batch_size, n_states, 1),
                float("nan"),
                dtype=x.dtype,
                device=x.device,
            )
        if time_steps == 1:
            return torch.zeros(batch_size, n_states, 1, dtype=x.dtype, device=x.device)

        _, is_duplicate = _sorted_adjacent_duplicates(x)
        duplicate_before, duplicate_after = _duplicate_neighbor_masks(is_duplicate)
        is_reoccurring = duplicate_before | duplicate_after
        result = is_reoccurring.sum(dim=1).to(x.dtype) / time_steps
        return result.reshape(batch_size, n_states, 1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing
            ``percentage_of_reoccurring_datapoints_to_all_datapoints``.
        """
        return ["percentage_of_reoccurring_datapoints_to_all_datapoints"]


class PercentageOfReoccurringValuesToAllValues(ParameterlessFeature):
    """Fraction of distinct values that occur more than once.

    Each repeated value contributes once to the numerator, regardless of its multiplicity.
    NaNs form one distinct value and count as repeated when at least two are present, matching
    NumPy ``unique`` as used by tsfresh. An empty series produces NaN.

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
            Tensor of shape (B, S, 1).
        """
        batch_size, n_states, time_steps = x.shape
        if time_steps == 0:
            return torch.full(
                (batch_size, n_states, 1),
                float("nan"),
                dtype=x.dtype,
                device=x.device,
            )
        _, is_duplicate = _sorted_adjacent_duplicates(x, nan_as_equal=True)
        duplicate_before, duplicate_after = _duplicate_neighbor_masks(is_duplicate)
        is_first_in_repeated_group = ~duplicate_before & duplicate_after
        reoccurring_value_count = is_first_in_repeated_group.sum(dim=1).to(x.dtype)
        unique_value_count = (~is_duplicate).sum(dim=1).to(x.dtype) + 1
        result = reoccurring_value_count / unique_value_count
        return result.reshape(batch_size, n_states, 1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing
            ``percentage_of_reoccurring_values_to_all_values``.
        """
        return ["percentage_of_reoccurring_values_to_all_values"]


class SumOfReoccurringDataPoints(ParameterlessFeature):
    """Sum every datapoint whose value occurs more than once.

    For example, the result for ``[2, 2, 2, 2, 1]`` is 8 because all four occurrences of 2
    contribute. The result is NaN when the input contains any NaN, matching tsfresh's NumPy
    ``unique`` multiplication semantics. An empty series produces zero.

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
            Tensor of shape (B, S, 1).
        """
        batch_size, n_states, time_steps = x.shape
        if time_steps == 0:
            return torch.zeros(batch_size, n_states, 1, dtype=x.dtype, device=x.device)
        sorted_x, is_duplicate = _sorted_adjacent_duplicates(x)
        duplicate_before, duplicate_after = _duplicate_neighbor_masks(is_duplicate)
        is_reoccurring = duplicate_before | duplicate_after
        result = torch.where(is_reoccurring, sorted_x, 0.0).sum(dim=1)
        has_nan = torch.isnan(x).reshape(batch_size * n_states, -1).any(dim=1)
        result = torch.where(has_nan, torch.full_like(result, float("nan")), result)
        return result.reshape(batch_size, n_states, 1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``sum_of_reoccurring_data_points``.
        """
        return ["sum_of_reoccurring_data_points"]


class SumOfReoccurringValues(ParameterlessFeature):
    """Sum each distinct value that occurs more than once.

    For example, the result for ``[2, 2, 2, 2, 1]`` is 2 because the repeated value contributes
    once. The result is NaN when the input contains any NaN, matching tsfresh's NumPy ``unique``
    multiplication semantics. An empty series produces zero.

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
            Tensor of shape (B, S, 1).
        """
        batch_size, n_states, time_steps = x.shape
        if time_steps == 0:
            return torch.zeros(batch_size, n_states, 1, dtype=x.dtype, device=x.device)
        sorted_x, is_duplicate = _sorted_adjacent_duplicates(x)
        duplicate_before, duplicate_after = _duplicate_neighbor_masks(is_duplicate)
        is_first_in_repeated_group = ~duplicate_before & duplicate_after
        result = torch.where(is_first_in_repeated_group, sorted_x, 0.0).sum(dim=1)
        has_nan = torch.isnan(x).reshape(batch_size * n_states, -1).any(dim=1)
        result = torch.where(has_nan, torch.full_like(result, float("nan")), result)
        return result.reshape(batch_size, n_states, 1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``sum_of_reoccurring_values``.
        """
        return ["sum_of_reoccurring_values"]


class RatioValueNumberToTimeSeriesLength(ParameterlessFeature):
    """Ratio of the number of distinct values to the time-series length.

    The result is one when every datapoint is unique and below one when values repeat. All NaNs
    form one distinct value, matching NumPy ``unique`` as used by tsfresh. An empty series
    produces NaN.

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
            Tensor of shape (B, S, 1).
        """
        batch_size, n_states, time_steps = x.shape
        if time_steps == 0:
            return torch.full(
                (batch_size, n_states, 1),
                float("nan"),
                dtype=x.dtype,
                device=x.device,
            )
        _, is_duplicate = _sorted_adjacent_duplicates(x, nan_as_equal=True)
        unique_value_count = time_steps - is_duplicate.sum(dim=1).to(x.dtype)
        result = unique_value_count / time_steps
        return result.reshape(batch_size, n_states, 1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``ratio_value_number_to_time_series_length``.
        """
        return ["ratio_value_number_to_time_series_length"]
