"""Torch-native batched linear regression."""

from __future__ import annotations

import torch
from torch import Tensor


def _regularized_incomplete_beta(x: Tensor, a: float, b: float) -> Tensor:
    """Evaluate the regularized incomplete beta function by continued fraction."""
    tiny = torch.finfo(x.dtype).tiny
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = torch.ones_like(x)
    d = 1.0 - qab * x / qap
    d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
    d = d.reciprocal()
    fraction = d
    for iteration in range(1, 33):
        m = float(iteration)
        m2 = 2.0 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
        c = 1.0 + aa / c
        c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
        d = d.reciprocal()
        fraction = fraction * d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = torch.where(d.abs() < tiny, torch.full_like(d, tiny), d)
        c = 1.0 + aa / c
        c = torch.where(c.abs() < tiny, torch.full_like(c, tiny), c)
        d = d.reciprocal()
        fraction = fraction * d * c

    log_front = (
        torch.lgamma(torch.tensor(qab, dtype=x.dtype, device=x.device))
        - torch.lgamma(torch.tensor(a, dtype=x.dtype, device=x.device))
        - torch.lgamma(torch.tensor(b, dtype=x.dtype, device=x.device))
        + a * torch.log(x)
        + b * torch.log1p(-x)
    )
    return torch.exp(log_front) * fraction / a


def _student_t_two_sided_pvalue(t_stat: Tensor, degrees_of_freedom: int) -> Tensor:
    """Return the two-sided Student-t p-value."""
    df = float(degrees_of_freedom)
    x = df / (df + t_stat.square())
    x = x.clamp(torch.finfo(x.dtype).eps, 1.0 - torch.finfo(x.dtype).eps)
    a = df / 2.0
    b = 0.5
    direct = _regularized_incomplete_beta(x, a, b)
    reflected = 1.0 - _regularized_incomplete_beta(1.0 - x, b, a)
    threshold = (a + 1.0) / (a + b + 2.0)
    return torch.where(x < threshold, direct, reflected).clamp(0.0, 1.0)


def linregress(y: Tensor) -> Tensor:
    """Calculate batched linear regressions against a zero-based sample index.

    Args:
        y: Tensor whose final axis contains observations.

    Returns:
        Tensor with final-axis columns ``slope``, ``intercept``, ``rvalue``, ``pvalue``,
        and ``stderr``.
    """
    time_steps = y.shape[-1]
    x = torch.arange(time_steps, dtype=y.dtype, device=y.device)
    x = x.view(*([1] * (y.ndim - 1)), time_steps)
    x_mean = x.mean()
    y_mean = y.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    ss_xx = x_centered.square().mean()
    ss_xy = (x_centered * y_centered).mean(dim=-1, keepdim=True)
    ss_yy = y_centered.square().mean(dim=-1, keepdim=True)

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean
    rvalue = (ss_xy / torch.sqrt(ss_xx * ss_yy)).clamp(-1.0, 1.0)
    rvalue = torch.where(
        (ss_xx == 0.0) | (ss_yy == 0.0),
        torch.full_like(rvalue, float("nan")),
        rvalue,
    )

    degrees_of_freedom = time_steps - 2
    fitted = slope * x + intercept
    residual_sum_squares = (y - fitted).square().sum(dim=-1, keepdim=True)
    stderr = torch.sqrt(
        residual_sum_squares / float(degrees_of_freedom) / x_centered.square().sum()
    )
    pvalue = _student_t_two_sided_pvalue(slope / stderr, degrees_of_freedom)
    return torch.cat((slope, intercept, rvalue, pvalue, stderr), dim=-1)
