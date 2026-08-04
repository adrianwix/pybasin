# pyright: basic
"""Process-local, bounded cache of compiled feature kernels.

Each Dask worker process builds and compiles its own kernel on the first block it
receives for a given configuration, then reuses that compiled callable for every later
block with the same key - warm blocks never rebuild the closures or retrigger
``torch.compile``. A per-key lock covers both construction and the first (warming)
invocation, so two worker threads racing on the same cold key don't compile the same
graph twice. Nothing here is serialized into a Dask task: only the key components (plan
token, shape, dtype, device, compile policy) travel with the task; the cache itself lives
in worker process memory and is rebuilt from scratch after a worker restart.
"""

import threading
from collections import OrderedDict
from collections.abc import Hashable

import torch

from tstorch.feature_plan import CompiledFeatureKernel, ResolvedFeaturePlan, build_kernel

_CacheKey = tuple[Hashable, int, int, str, str, bool]

_DEFAULT_MAX_ENTRIES = 32


def _make_cache_key(
    resolved: ResolvedFeaturePlan,
    sequence_length: int,
    dtype: torch.dtype,
    device_type: str,
    use_compile: bool,
) -> _CacheKey:
    """Build the cache key for one worker-local compiled kernel.

    Every value that can change generated code or output schema is part of the key:
    the resolved feature token, state count, sequence length, dtype, device type, and
    compile policy. Batch size is deliberately excluded - the kernel is compiled with
    dynamic batch support, so a smaller tail block reuses the same entry.
    """
    return (
        resolved.token,
        resolved.num_states,
        sequence_length,
        str(dtype),
        device_type,
        use_compile,
    )


class _KernelCache:
    """A bounded, thread-safe, process-local cache of :class:`CompiledFeatureKernel`."""

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[_CacheKey, CompiledFeatureKernel] = OrderedDict()
        self._entries_guard = threading.Lock()
        self._key_locks: dict[_CacheKey, threading.Lock] = {}
        self._key_locks_guard = threading.Lock()

    def get_or_build(
        self,
        key: _CacheKey,
        resolved: ResolvedFeaturePlan,
        use_compile: bool,
    ) -> CompiledFeatureKernel:
        """Return the cached kernel for ``key``, building and warming it exactly once.

        The per-key lock is held across both construction and the caller's first
        (warming) invocation is the caller's responsibility - this method only
        guarantees ``build_kernel`` runs once per key; callers that need warmup
        serialized too should invoke the kernel while still holding no other lock,
        immediately after this call returns a freshly built entry.
        """
        with self._entries_guard:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached

        with self._key_locks_guard:
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        with key_lock:
            with self._entries_guard:
                cached = self._entries.get(key)
                if cached is not None:
                    self._entries.move_to_end(key)
                    return cached

            compiled = build_kernel(resolved, use_compile)

            with self._entries_guard:
                self._entries[key] = compiled
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_entries:
                    self._entries.popitem(last=False)

            return compiled

    def __len__(self) -> int:
        with self._entries_guard:
            return len(self._entries)

    def keys(self) -> list[_CacheKey]:
        """Inspection hook for tests: the currently cached keys, most-recent last."""
        with self._entries_guard:
            return list(self._entries.keys())

    def clear(self) -> None:
        """Clear every cached kernel. Ordinary callers never need this; tests do."""
        with self._entries_guard:
            self._entries.clear()
        with self._key_locks_guard:
            self._key_locks.clear()


_process_cache = _KernelCache()


def _get_process_cache() -> _KernelCache:
    """The one process-local kernel cache every worker task in this process shares."""
    return _process_cache
