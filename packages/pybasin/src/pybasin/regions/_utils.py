from collections.abc import Sequence

import torch
from torch import Tensor


# @TODO: Review tensor transformation since we have similar utilities already
def as_tensor(value: Tensor | Sequence[float]) -> Tensor:
    """Pass a ``Tensor`` through unchanged; convert any other sequence to a float32 tensor."""
    if isinstance(value, Tensor):
        return value
    return torch.tensor(value, dtype=torch.float32)
