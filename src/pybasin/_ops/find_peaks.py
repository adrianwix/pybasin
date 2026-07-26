"""PyTorch operator for topographic-prominence peak finding.

The CPU dispatch-key implementation is real ``scipy.signal.find_peaks`` --
not a reimplementation. The CUDA dispatch-key implementation (registered
below when Triton is available) computes the same local-maxima and
prominence definitions with a Triton kernel.
"""

import importlib.util
from collections.abc import Callable
from typing import Protocol

import scipy.signal  # type: ignore[import-untyped]
import torch
from torch import Tensor


class _RegisterFake(Protocol):
    def __call__(
        self,
        op: str,
        function: Callable[[Tensor, float], tuple[Tensor, Tensor]],
        /,
    ) -> object: ...


class _Library(Protocol):
    def define(self, schema: str, /) -> object: ...

    def impl(
        self,
        op_name: str,
        function: Callable[[Tensor, float], tuple[Tensor, Tensor]],
        dispatch_key: str,
        /,
    ) -> None: ...


class _LibraryFactory(Protocol):
    def __call__(self, namespace: str, kind: str, /) -> _Library: ...


_register_fake: _RegisterFake = vars(torch.library)["register_fake"]
_library_factory: _LibraryFactory = vars(torch.library)["Library"]

_LIBRARY = _library_factory("pybasin", "FRAGMENT")
_LIBRARY.define("find_peaks(Tensor s, float height) -> (Tensor peaks, Tensor prominences)")


def _find_peaks_cpu(s: Tensor, height: float) -> tuple[Tensor, Tensor]:
    """Real scipy.signal.find_peaks, wrapped as the CPU dispatch-key implementation."""
    peaks, props = scipy.signal.find_peaks(s.numpy(), height=height, prominence=0)
    peaks_tensor = torch.from_numpy(peaks).to(torch.int64)  # type: ignore[arg-type]
    prominences_tensor = torch.from_numpy(props["prominences"]).to(s.dtype)  # type: ignore[arg-type]
    return peaks_tensor, prominences_tensor


_LIBRARY.impl("find_peaks", _find_peaks_cpu, "CPU")


def _find_peaks_fake(s: Tensor, height: float) -> tuple[Tensor, Tensor]:
    n = s.shape[0]
    return s.new_empty((n,), dtype=torch.int64), s.new_empty((n,))


_register_fake("pybasin::find_peaks", _find_peaks_fake)

find_peaks: Callable[[Tensor, float], tuple[Tensor, Tensor]] = torch.ops.pybasin.find_peaks.default

if importlib.util.find_spec("triton") is not None:
    import triton
    import triton.language as tl

    @triton.jit
    def _find_peaks_kernel(
        s_ptr: tl.tensor,
        height: float,
        is_peak_ptr: tl.tensor,
        prominence_ptr: tl.tensor,
        n: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        s = tl.load(s_ptr + offs, mask=mask, other=float("-inf"))

        left_val = tl.load(s_ptr + offs - 1, mask=mask & (offs > 0), other=float("inf"))
        right_val = tl.load(s_ptr + offs + 1, mask=mask & (offs < n - 1), other=float("inf"))
        is_local_max = (s > left_val) & (s > right_val) & (s >= height)

        # prominence[i]: broadcast s against itself (BLOCK x BLOCK), same definition as the
        # CPU/scipy path -- nearest higher neighbor on each side, min value up to that point.
        s_row = s[:, None]
        s_col = s[None, :]
        idx_row = offs[:, None]
        idx_col = offs[None, :]
        greater = s_col > s_row
        left_bound = tl.max(tl.where(greater & (idx_col < idx_row), idx_col, -1), axis=1)
        right_bound = tl.min(tl.where(greater & (idx_col > idx_row), idx_col, n), axis=1)
        left_min = tl.min(
            tl.where((idx_col > left_bound[:, None]) & (idx_col <= idx_row), s_col, float("inf")),
            axis=1,
        )
        right_min = tl.min(
            tl.where((idx_col < right_bound[:, None]) & (idx_col >= idx_row), s_col, float("inf")),
            axis=1,
        )
        prominence = s - tl.maximum(left_min, right_min)

        tl.store(is_peak_ptr + offs, is_local_max.to(tl.int8), mask=mask)
        tl.store(prominence_ptr + offs, prominence, mask=mask)

    def _find_peaks_cuda(s: Tensor, height: float) -> tuple[Tensor, Tensor]:
        n = s.shape[0]
        if n == 0:
            empty = s.new_empty((0,))
            return empty.to(torch.int64), empty

        contiguous_s = s.contiguous()
        is_peak = torch.empty(n, dtype=torch.int8, device=s.device)
        prominence = torch.empty(n, dtype=s.dtype, device=s.device)
        block = triton.next_power_of_2(n)
        _find_peaks_kernel[(1,)](contiguous_s, height, is_peak, prominence, n=n, BLOCK=block)

        peaks = torch.nonzero(is_peak.bool(), as_tuple=True)[0]
        prominences = prominence[peaks]
        return peaks.to(torch.int64), prominences

    _LIBRARY.impl("find_peaks", _find_peaks_cuda, "CUDA")
