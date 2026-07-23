"""CUDA dispatch for batched Lempel-Ziv phrase counting."""

from collections.abc import Callable
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor


class _LibraryImpl(Protocol):
    def impl(
        self,
        op_name: str,
        function: Callable[[Tensor], Tensor],
        dispatch_key: str,
        /,
    ) -> None: ...


@triton.jit
def _lempel_ziv_phrase_count_kernel(
    sequences_ptr: tl.tensor,
    phrase_starts_ptr: tl.tensor,
    phrase_lengths_ptr: tl.tensor,
    output_ptr: tl.tensor,
    sequence_length: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    row_offset = row * sequence_length
    position = 0
    candidate_length = 1
    phrase_count = 0

    while position + candidate_length <= sequence_length:
        found = False
        phrase_index = 0

        while phrase_index < phrase_count:
            previous_length = tl.load(phrase_lengths_ptr + row_offset + phrase_index)
            if previous_length == candidate_length:
                previous_start = tl.load(phrase_starts_ptr + row_offset + phrase_index)
                equal = True
                offset = 0
                while offset < candidate_length:
                    previous_symbol = tl.load(sequences_ptr + row_offset + previous_start + offset)
                    candidate_symbol = tl.load(sequences_ptr + row_offset + position + offset)
                    equal = equal and previous_symbol == candidate_symbol
                    offset += 1

                found = found or equal

            phrase_index += 1

        if found:
            candidate_length += 1
        else:
            tl.store(phrase_starts_ptr + row_offset + phrase_count, position)
            tl.store(phrase_lengths_ptr + row_offset + phrase_count, candidate_length)
            phrase_count += 1
            position += candidate_length
            candidate_length = 1

    tl.store(output_ptr + row, phrase_count)


def _lempel_ziv_phrase_count_cuda(sequences: Tensor) -> Tensor:
    rows, sequence_length = sequences.shape
    output = torch.empty(rows, dtype=torch.int64, device=sequences.device)
    if rows == 0 or sequence_length == 0:
        return output.zero_()

    contiguous_sequences = sequences.contiguous()
    phrase_starts = torch.empty(rows, sequence_length, dtype=torch.int32, device=sequences.device)
    phrase_lengths = torch.empty_like(phrase_starts)
    _lempel_ziv_phrase_count_kernel[(rows,)](
        contiguous_sequences,
        phrase_starts,
        phrase_lengths,
        output,
        sequence_length=sequence_length,
        num_warps=1,
    )
    return output


def register_cuda_kernel(library: _LibraryImpl) -> None:
    """Register the CUDA implementation with the owning operator library."""
    library.impl(
        "lempel_ziv_phrase_count",
        _lempel_ziv_phrase_count_cuda,
        "CUDA",
    )
