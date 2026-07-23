"""Change and difference-based feature calculators for BST time series."""

from typing import Literal, TypedDict

import torch
from torch import Tensor

from tstorch.feature_factory import Feature, ParameterlessFeature


def _absolute_differences(x: Tensor) -> Tensor:
    return (x[..., 1:] - x[..., :-1]).abs()


def _absolute_change_total(x: Tensor) -> Tensor:
    return _absolute_differences(x).sum(dim=2, keepdim=True)


class AbsoluteSumOfChanges(ParameterlessFeature):
    r"""Sum of absolute consecutive differences across time.

    \[
    \sum_{t=1}^{T-1} |x_t - x_{t-1}|
    \]

    For a series with fewer than two samples, the sum is empty and returns 0.

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
        return _absolute_change_total(x)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``absolute_sum_of_changes``.
        """
        return ["absolute_sum_of_changes"]


class MeanAbsChange(ParameterlessFeature):
    r"""Mean absolute consecutive difference across time.

    \[
    \frac{1}{T-1} \sum_{t=1}^{T-1} |x_t - x_{t-1}|
    \]

    For a series with fewer than two samples, the result is NaN.

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
        T = x.shape[2]
        total = _absolute_change_total(x).squeeze(-1)
        result = total / (T - 1) if T > 1 else torch.full_like(total, float("nan"))
        return result.unsqueeze(-1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``mean_abs_change``.
        """
        return ["mean_abs_change"]


class MeanChange(ParameterlessFeature):
    r"""Average rate of change from the first to the last value.

    \[
    \frac{x_{T-1} - x_0}{T - 1}
    \]

    This is algebraically equivalent to the mean of first differences. For a series
    with fewer than two samples, the result is NaN.

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
        T = x.shape[2]
        if T > 1:
            result = (x[..., -1] - x[..., 0]) / (T - 1)
        else:
            result = torch.full(x.shape[:2], float("nan"), dtype=x.dtype, device=x.device)
        return result.unsqueeze(-1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``mean_change``.
        """
        return ["mean_change"]


class MeanSecondDerivativeCentral(ParameterlessFeature):
    r"""Mean central second derivative across time.

    \[
    \frac{x_{T-1} - x_{T-2} - x_1 + x_0}{2(T - 2)}
    \]

    For a series with fewer than three samples, the result is NaN.

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
        T = x.shape[2]
        if T > 2:
            result = (x[..., -1] - x[..., -2] - x[..., 1] + x[..., 0]) / (2 * (T - 2))
        else:
            result = torch.full(x.shape[:2], float("nan"), dtype=x.dtype, device=x.device)
        return result.unsqueeze(-1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``mean_second_derivative_central``.
        """
        return ["mean_second_derivative_central"]


class ChangeQuantilesParams(TypedDict):
    """Parameters for one `ChangeQuantiles` output column.

    Attributes:
        ql: Lower quantile in ``[0, 1]``.
        qh: Upper quantile in ``[0, 1]``.
        isabs: Whether to aggregate absolute first differences.
        f_agg: Aggregation for selected differences, either ``"mean"`` or ``"var"``.
    """

    ql: float
    qh: float
    isabs: bool
    f_agg: Literal["mean", "var"]


def _quantiles_via_sort(x: Tensor, qs: Tensor) -> Tensor:
    batch_size, n_states, T = x.shape
    Q = qs.shape[0]
    sorted_x = x.sort(dim=2).values

    q_tensor = qs.to(dtype=x.dtype, device=x.device)
    index = q_tensor * (T - 1)
    lower = index.floor().long()
    upper = index.ceil().long()
    frac = (index - lower.to(x.dtype)).view(1, 1, Q)

    lower_idx = lower.clamp(0, T - 1).view(1, 1, Q).expand(batch_size, n_states, Q)
    upper_idx = upper.clamp(0, T - 1).view(1, 1, Q).expand(batch_size, n_states, Q)
    lower_vals = sorted_x.gather(dim=2, index=lower_idx)
    upper_vals = sorted_x.gather(dim=2, index=upper_idx)
    return lower_vals + frac * (upper_vals - lower_vals)


class ChangeQuantiles(Feature[ChangeQuantilesParams]):
    r"""Statistic of consecutive changes within a quantile corridor.

    Let \(q_l\) and \(q_h\) be the requested quantiles of the series. A consecutive
    difference \(x_{t+1} - x_t\), or its absolute value when ``isabs`` is true, is
    selected only when both endpoints lie in \([q_l, q_h]\). The selected differences
    are aggregated by ``f_agg``.

    When ``ql >= qh`` or no consecutive pair falls inside the corridor, the feature
    returns 0. Several requested parameter combinations share one sorted series and
    broadcast the corridor masks across the parameter axis.

    Args:
        params: One dict per requested output, each with ``ql``, ``qh``, ``isabs``,
            and ``f_agg`` keys.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[ChangeQuantilesParams]) -> None:
        self.params: list[ChangeQuantilesParams] = params
        unique_q_values = sorted({p["ql"] for p in params} | {p["qh"] for p in params})
        q_to_idx = {q: i for i, q in enumerate(unique_q_values)}
        self._unique_q_values = torch.tensor(unique_q_values, dtype=torch.float64)
        self._ql_indices = torch.tensor([q_to_idx[p["ql"]] for p in params], dtype=torch.long)
        self._qh_indices = torch.tensor([q_to_idx[p["qh"]] for p in params], dtype=torch.long)
        self._isabs = torch.tensor([p["isabs"] for p in params], dtype=torch.bool)
        self._is_var = torch.tensor([p["f_agg"] == "var" for p in params], dtype=torch.bool)
        self._degenerate = torch.tensor([p["ql"] >= p["qh"] for p in params], dtype=torch.bool)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(params)).
        """
        batch_size, n_states = x.shape[:2]
        if not self.params:
            return torch.zeros(batch_size, n_states, 0, dtype=x.dtype, device=x.device)

        q_values = self._unique_q_values.to(dtype=x.dtype, device=x.device)
        ql_indices = self._ql_indices.to(device=x.device)
        qh_indices = self._qh_indices.to(device=x.device)
        all_quantiles = _quantiles_via_sort(x, q_values)
        q_lows = all_quantiles.index_select(dim=2, index=ql_indices).unsqueeze(2)
        q_highs = all_quantiles.index_select(dim=2, index=qh_indices).unsqueeze(2)

        x_by_param = x.unsqueeze(-1)
        mask = (x_by_param >= q_lows) & (x_by_param <= q_highs)

        diff = x[..., 1:] - x[..., :-1]
        diff_by_param = diff.unsqueeze(-1)
        isabs = self._isabs.to(device=x.device).view(1, 1, 1, -1)
        values = torch.where(isabs, diff_by_param.abs(), diff_by_param)

        valid_changes = mask[..., :-1, :] & mask[..., 1:, :]
        valid_float = valid_changes.to(dtype=x.dtype)
        count = valid_float.sum(dim=2)
        safe_count = count + 1e-10

        mean_result = (values * valid_float).sum(dim=2) / safe_count
        squared = (values - mean_result.unsqueeze(2)) ** 2 * valid_float
        var_result = squared.sum(dim=2) / safe_count

        is_var = self._is_var.to(device=x.device).view(1, 1, -1)
        result = torch.where(is_var, var_result, mean_result)
        result = torch.where(count > 0, result, torch.zeros_like(result))

        degenerate = self._degenerate.to(device=x.device).view(1, 1, -1)
        return torch.where(degenerate, torch.zeros_like(result), result)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format:
            ``change_quantiles__ql_<ql>__qh_<qh>__isabs_<isabs>__f_agg_<f_agg>``.
        """
        return [
            (
                f"change_quantiles__ql_{p['ql']}__qh_{p['qh']}"
                f"__isabs_{p['isabs']}__f_agg_{p['f_agg']}"
            )
            for p in self.params
        ]
