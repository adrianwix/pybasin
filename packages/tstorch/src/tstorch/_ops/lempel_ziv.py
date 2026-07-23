"""PyTorch operator for batched Lempel-Ziv phrase counting."""

import importlib
import importlib.util
from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor


class _RegisterFake(Protocol):
    def __call__(
        self,
        op: str,
        function: Callable[[Tensor], Tensor],
        /,
    ) -> object: ...


class _Library(Protocol):
    def define(self, schema: str, /) -> object: ...

    def impl(
        self,
        op_name: str,
        function: Callable[[Tensor], Tensor],
        dispatch_key: str,
        /,
    ) -> None: ...


class _LibraryFactory(Protocol):
    def __call__(self, namespace: str, kind: str, /) -> _Library: ...


class _RegisterCudaKernel(Protocol):
    def __call__(self, library: _Library, /) -> None: ...


_register_fake: _RegisterFake = vars(torch.library)["register_fake"]
_library_factory: _LibraryFactory = vars(torch.library)["Library"]

_LIBRARY = _library_factory("tstorch", "FRAGMENT")
_LIBRARY.define("lempel_ziv_phrase_count(Tensor sequences) -> Tensor")


def _lempel_ziv_phrase_count_cpu(sequences: Tensor) -> Tensor:
    """Count Lempel-Ziv phrases in each row of a CPU symbol tensor."""
    rows, sequence_length = sequences.shape
    output = torch.zeros(rows, dtype=torch.int64, device=sequences.device)
    if sequence_length == 0:
        return output

    sequence_rows = sequences.contiguous().numpy()
    output_rows = output.numpy()
    symbol_width = sequences.element_size()
    for row in range(rows):
        sequence = sequence_rows[row].tobytes()
        phrases: set[bytes] = set()
        position = 0
        candidate_length = 1
        while position + candidate_length <= sequence_length:
            byte_start = position * symbol_width
            byte_stop = (position + candidate_length) * symbol_width
            phrase = sequence[byte_start:byte_stop]
            if phrase in phrases:
                candidate_length += 1
            else:
                phrases.add(phrase)
                position += candidate_length
                candidate_length = 1
        output_rows[row] = len(phrases)
    return output


_LIBRARY.impl(
    "lempel_ziv_phrase_count",
    _lempel_ziv_phrase_count_cpu,
    "CPU",
)


def _lempel_ziv_phrase_count_fake(sequences: Tensor) -> Tensor:
    return sequences.new_empty((sequences.shape[0],), dtype=torch.int64)


_register_fake(
    "tstorch::lempel_ziv_phrase_count",
    _lempel_ziv_phrase_count_fake,
)

lempel_ziv_phrase_count: Callable[[Tensor], Tensor] = (
    torch.ops.tstorch.lempel_ziv_phrase_count.default
)

if importlib.util.find_spec("triton") is not None:
    _cuda_module = importlib.import_module("tstorch._ops.lempel_ziv_cuda")
    _register_cuda_kernel: _RegisterCudaKernel = vars(_cuda_module)["register_cuda_kernel"]
    _register_cuda_kernel(_LIBRARY)
