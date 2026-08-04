# pyright: basic
"""Dask block <-> ``torch.Tensor`` conversion for the feature-extraction kernel path.

The only place that converts a Dask block into a ``torch.Tensor`` and back. A calculator
never imports Dask or NumPy directly - every worker task goes through this adapter before
calling the shared raw ``(B, S, T) -> (B, F)`` kernel path from
``tstorch.torch_feature_extractor``, and back through it to produce a NumPy-backed
result block.
"""

from typing import Any, Literal

import torch
from torch import Tensor

from tstorch._array_backend import is_cupy_array

_DeviceType = Literal["cpu", "cuda"]


def _block_to_tensor(block: Any, device_type: _DeviceType) -> Tensor:
    """Convert one worker's ``(B_chunk, S, T)`` array block into a contiguous tensor.

    :param block: A NumPy ``ndarray``, or (when ``device_type == "cuda"``) a NumPy or
        CuPy array. CuPy input is converted via DLPack, which requires it to already sit
        on the CUDA device visible to this worker.
    :param device_type: ``"cpu"`` or ``"cuda"`` - the execution policy for this task.
    :return: A ``torch.Tensor`` of shape ``(B_chunk, S, T)``, contiguous along the time
        axis, placed on ``device_type``.
    :raises RuntimeError: If ``device_type == "cuda"`` but no CUDA device is visible to
        this worker. Never falls back to CPU.
    """
    if device_type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "DaskFeatureExtractor(device='cuda') scheduled a task on a worker with no "
            "usable CUDA device. This is a hard failure, not a silent CPU fallback."
        )

    if is_cupy_array(block):
        tensor = torch.from_dlpack(block)
    else:
        tensor = torch.from_numpy(block)
        if device_type == "cuda":
            tensor = tensor.to("cuda")

    if tensor.stride(-1) != 1:
        tensor = tensor.contiguous()
    return tensor


def _tensor_to_block(features: Tensor) -> Any:
    """Convert a computed ``(B_chunk, F)`` feature tensor into a NumPy result block.

    Every distributed output block is NumPy-backed, even when extraction executed on
    CUDA, so downstream storage and preprocessing see one stable representation. Moving
    a CUDA tensor to host performs the synchronization needed before the task reports
    completion.

    :param features: Tensor of shape ``(B_chunk, F)``, any device.
    :return: ``numpy.ndarray`` of the same shape and dtype.
    """
    return features.detach().cpu().numpy()
