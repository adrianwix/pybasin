"""Torch-native augmented Dickey-Fuller test."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from tstorch.statsmodels.regression.linear_model import (
    ols_coefficient_tvalue,
    ols_parameters,
)
from tstorch.statsmodels.tsa.adfvalues import mackinnonp

Autolag = Literal["AIC", "BIC", "t-stat", None]


def _adf_design(x: Tensor, lag: int) -> tuple[Tensor, Tensor]:
    """Build the constant, lagged-level, and lagged-difference ADF regression."""
    differences = x.diff(dim=-1)
    if lag == 0:
        y = differences
        level = x[..., :-1]
        lagged_differences = differences.unsqueeze(-1)[..., :0]
    else:
        windows = differences.unfold(-1, lag + 1, 1)
        y = windows[..., -1]
        lagged_differences = windows[..., :-1].flip(-1)
        level = x[..., lag:-1]
    ones = torch.ones(*level.shape, 1, dtype=x.dtype, device=x.device)
    return y, torch.cat((level.unsqueeze(-1), lagged_differences, ones), dim=-1)


def adfuller(x: Tensor, autolag: Autolag = "AIC") -> Tensor:
    """Run a batched constant-regression augmented Dickey-Fuller test.

    Args:
        x: Tensor whose final axis contains observations.
        autolag: Lag-selection method.

    Returns:
        Tensor with final-axis columns ``teststat``, ``pvalue``, and ``usedlag``.
    """
    time_steps = x.shape[-1]
    max_lag = min(
        time_steps // 2 - 2,
        math.ceil(12.0 * (time_steps / 100.0) ** 0.25),
    )
    differences = x.diff(dim=-1)
    common_windows = differences.unfold(-1, max_lag + 1, 1)
    common_y = common_windows[..., -1]
    common_level = x[..., max_lag:-1]
    common_lags = common_windows[..., :-1].flip(-1)
    common_ones = torch.ones(
        *common_level.shape,
        1,
        dtype=x.dtype,
        device=x.device,
    )
    full_design = torch.cat((common_ones, common_level.unsqueeze(-1), common_lags), dim=-1)

    criterion_values: list[Tensor] = []
    last_tvalues: list[Tensor] = []
    for lag in range(max_lag + 1):
        design = full_design[..., : lag + 2]
        parameters = ols_parameters(common_y, design)
        residuals = common_y - (design @ parameters.unsqueeze(-1)).squeeze(-1)
        residual_sum_squares = residuals.square().sum(dim=-1)
        parameter_count = lag + 2
        observation_count = common_y.shape[-1]
        if autolag == "BIC":
            penalty = math.log(float(observation_count)) * parameter_count
        else:
            penalty = 2.0 * parameter_count
        criterion_values.append(
            float(observation_count) * torch.log(residual_sum_squares / float(observation_count))
            + penalty
        )
        last_tvalues.append(ols_coefficient_tvalue(common_y, design, -1).abs())

    if autolag == "AIC" or autolag == "BIC":
        selected_lag = torch.stack(criterion_values, dim=-1).argmin(dim=-1)
    elif autolag == "t-stat":
        significant = torch.stack(last_tvalues, dim=-1) >= 1.6448536269514722
        lag_indices = torch.arange(max_lag + 1, device=x.device)
        selected_lag = torch.where(significant, lag_indices, 0).amax(dim=-1)
    else:
        selected_lag = torch.full(
            x.shape[:-1],
            max_lag,
            dtype=torch.long,
            device=x.device,
        )

    candidate_test_statistics: list[Tensor] = []
    for lag in range(max_lag + 1):
        y, design = _adf_design(x, lag)
        candidate_test_statistics.append(ols_coefficient_tvalue(y, design, 0))
    candidates = torch.stack(candidate_test_statistics, dim=-1)
    test_statistic = candidates.gather(-1, selected_lag.unsqueeze(-1)).squeeze(-1)
    pvalue = mackinnonp(test_statistic)
    constant = x.amax(dim=-1) == x.amin(dim=-1)
    test_statistic = torch.where(
        constant, torch.full_like(test_statistic, float("nan")), test_statistic
    )
    pvalue = torch.where(constant, torch.full_like(pvalue, float("nan")), pvalue)
    used_lag = selected_lag.to(x.dtype)
    used_lag = torch.where(constant, torch.full_like(used_lag, float("nan")), used_lag)
    return torch.stack((test_statistic, pvalue, used_lag), dim=-1)
