"""MacKinnon p-values for augmented Dickey-Fuller statistics."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def mackinnonp(test_statistic: Tensor) -> Tensor:
    """Calculate MacKinnon's constant-regression ADF p-value approximation.

    Args:
        test_statistic: Augmented Dickey-Fuller test statistic.

    Returns:
        Approximate p-value with the same shape as ``test_statistic``.
    """
    small = 2.1659 + 1.4412 * test_statistic + 0.038269 * test_statistic.square()
    large = (
        1.7339
        + 0.93202 * test_statistic
        - 0.12745 * test_statistic.square()
        - 0.010368 * test_statistic.pow(3)
    )
    polynomial = torch.where(test_statistic <= -1.61, small, large)
    normal_cdf = 0.5 * (1.0 + torch.erf(polynomial / math.sqrt(2.0)))
    normal_cdf = torch.where(test_statistic > 2.74, torch.ones_like(normal_cdf), normal_cdf)
    return torch.where(test_statistic < -18.83, torch.zeros_like(normal_cdf), normal_cdf)
