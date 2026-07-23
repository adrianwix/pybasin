"""Torch-native batched ordinary least squares."""

from __future__ import annotations

import torch
from torch import Tensor


def _cuda_least_squares(design: Tensor, response: Tensor) -> Tensor:
    parameter_count = design.shape[-1]
    zero = torch.zeros_like(design[..., 0, 0])
    q_columns: list[Tensor] = []
    r_columns: list[Tensor] = []
    for column_index in range(parameter_count):
        vector = design[..., :, column_index]
        projections: list[Tensor] = []
        for orthonormal_column in q_columns:
            projection = (orthonormal_column * vector).sum(dim=-1)
            projections.append(projection)
            vector = vector - projection.unsqueeze(-1) * orthonormal_column
        for projection_index, orthonormal_column in enumerate(q_columns):
            correction = (orthonormal_column * vector).sum(dim=-1)
            projections[projection_index] = projections[projection_index] + correction
            vector = vector - correction.unsqueeze(-1) * orthonormal_column
        diagonal = vector.square().sum(dim=-1).sqrt()
        safe_diagonal = torch.where(
            diagonal > torch.finfo(design.dtype).eps,
            diagonal,
            torch.ones_like(diagonal),
        )
        q_columns.append(vector / safe_diagonal.unsqueeze(-1))
        r_columns.append(
            torch.stack(
                [
                    *projections,
                    safe_diagonal,
                    *[zero for _ in range(parameter_count - column_index - 1)],
                ],
                dim=-1,
            )
        )

    q = torch.stack(q_columns, dim=-1)
    r = torch.stack(r_columns, dim=-1)
    projected_response = q.mT @ response
    reversed_parameters: list[Tensor] = []
    for row_index in range(parameter_count - 1, -1, -1):
        rhs = projected_response[..., row_index, :]
        for offset, solved_parameter in enumerate(reversed_parameters, start=1):
            rhs = rhs - r[..., row_index, -offset].unsqueeze(-1) * solved_parameter
        reversed_parameters.append(rhs / r[..., row_index, row_index].unsqueeze(-1))
    return torch.stack(list(reversed(reversed_parameters)), dim=-2)


def ols_parameters(y: Tensor, design: Tensor) -> Tensor:
    """Fit batched ordinary least-squares parameters.

    Args:
        y: Response tensor with observations on the final axis.
        design: Design tensor with shape ``(..., observations, parameters)``.

    Returns:
        Parameters in design-column order.
    """
    if design.is_cuda:
        return _cuda_least_squares(design, y.unsqueeze(-1)).squeeze(-1)
    parameters, _, _ = ols_fit(y, design)
    return parameters


def ols_coefficient_tvalue(y: Tensor, design: Tensor, index: int) -> Tensor:
    """Calculate the t-value for one batched regression coefficient.

    Args:
        y: Response tensor with observations on the final axis.
        design: Design tensor with shape ``(..., observations, parameters)``.
        index: Design-column index of the coefficient.

    Returns:
        T-value for the selected coefficient.
    """
    if not design.is_cuda:
        _, tvalues, _ = ols_fit(y, design)
        return tvalues[..., index]

    column_index = index % design.shape[-1]
    selected = design[..., column_index]
    before = design[..., :column_index]
    after = design[..., column_index + 1 :]
    nuisance = torch.cat((before, after), dim=-1)
    responses = torch.stack((y, selected), dim=-1)
    nuisance_parameters = _cuda_least_squares(nuisance, responses)
    residualized = responses - nuisance @ nuisance_parameters
    residualized_y = residualized[..., 0]
    residualized_selected = residualized[..., 1]
    selected_sum_squares = residualized_selected.square().sum(dim=-1)
    coefficient = (residualized_selected * residualized_y).sum(dim=-1) / selected_sum_squares
    residuals = residualized_y - coefficient.unsqueeze(-1) * residualized_selected
    degrees_of_freedom = design.shape[-2] - design.shape[-1]
    residual_variance = residuals.square().sum(dim=-1) / float(degrees_of_freedom)
    standard_error = torch.sqrt(residual_variance / selected_sum_squares)
    return coefficient / standard_error


def ols_fit(y: Tensor, design: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Fit batched ordinary least squares.

    Args:
        y: Response tensor with observations on the final axis.
        design: Design tensor with shape ``(..., observations, parameters)``.

    Returns:
        Parameters, t-values, and residual sums of squares.
    """
    q, r = torch.qr(design, some=True)
    projected_response = q.mT @ y.unsqueeze(-1)
    parameters = torch.triangular_solve(
        projected_response,
        r,
        upper=True,
    ).solution
    normalized_covariance = torch.cholesky_inverse(r, upper=True)
    residuals = y.unsqueeze(-1) - design @ parameters
    residual_sum_squares = residuals.square().sum(dim=-2).squeeze(-1)
    observation_count = design.shape[-2]
    parameter_count = design.shape[-1]
    scale = residual_sum_squares / float(observation_count - parameter_count)
    standard_errors = torch.sqrt(
        normalized_covariance.diagonal(dim1=-2, dim2=-1) * scale.unsqueeze(-1)
    )
    tvalues = parameters.squeeze(-1) / standard_errors
    return parameters.squeeze(-1), tvalues, residual_sum_squares
