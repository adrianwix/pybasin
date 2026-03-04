# pyright: basic
"""Shared utilities for plotter components."""

import torch

from pybasin.plotters.colors import COLORS, get_color

__all__ = ["COLORS", "get_color", "tensor_to_float_list"]


def tensor_to_float_list(tensor: torch.Tensor) -> list[float]:
    """Convert a torch tensor to a list of floats.

    :param tensor: 1D torch tensor to convert.
    :return: List of float values from the tensor.
    """
    return [float(x) for x in tensor.cpu().tolist()]
