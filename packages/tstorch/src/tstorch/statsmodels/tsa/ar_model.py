"""Torch-native autoregressive model fitting."""

from __future__ import annotations

import torch
from torch import Tensor

from tstorch.statsmodels.regression.linear_model import ols_parameters


def auto_reg(x: Tensor, lags: int) -> Tensor:
    """Fit a batched autoregressive model with a constant trend.

    Args:
        x: Tensor whose final axis contains observations.
        lags: Maximum autoregressive lag.

    Returns:
        Tensor containing the intercept followed by lag coefficients.
    """
    windows = x.unfold(-1, lags + 1, 1)
    y = windows[..., -1]
    lagged = windows[..., :-1].flip(-1)
    ones = torch.ones(*lagged.shape[:-1], 1, dtype=x.dtype, device=x.device)
    design = torch.cat((ones, lagged), dim=-1)
    return ols_parameters(y, design)
