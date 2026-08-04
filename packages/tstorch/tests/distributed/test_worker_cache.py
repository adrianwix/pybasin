"""Worker-local kernel cache behavior: build-once, warm-reuse, bounded eviction."""

import torch
from tstorch.dask._cache import _KernelCache, _make_cache_key
from tstorch.feature_plan import normalize_features_config, resolve_feature_plan


def _resolved(num_states: int = 1):
    state_config = {0: normalize_features_config({"mean": None})}
    return resolve_feature_plan(state_config, num_states)


class TestKernelCache:
    def test_same_key_builds_once(self) -> None:
        cache = _KernelCache()
        resolved = _resolved()
        key = _make_cache_key(
            resolved, sequence_length=32, dtype=torch.float32, device_type="cpu", use_compile=False
        )

        first = cache.get_or_build(key, resolved, False)
        second = cache.get_or_build(key, resolved, False)

        assert first is second
        assert len(cache) == 1

    def test_different_sequence_length_is_a_different_key(self) -> None:
        cache = _KernelCache()
        resolved = _resolved()
        key_a = _make_cache_key(
            resolved, sequence_length=32, dtype=torch.float32, device_type="cpu", use_compile=False
        )
        key_b = _make_cache_key(
            resolved, sequence_length=64, dtype=torch.float32, device_type="cpu", use_compile=False
        )

        cache.get_or_build(key_a, resolved, False)
        cache.get_or_build(key_b, resolved, False)

        assert len(cache) == 2

    def test_clear_removes_all_entries(self) -> None:
        cache = _KernelCache()
        resolved = _resolved()
        key = _make_cache_key(
            resolved, sequence_length=32, dtype=torch.float32, device_type="cpu", use_compile=False
        )
        cache.get_or_build(key, resolved, False)
        assert len(cache) == 1

        cache.clear()
        assert len(cache) == 0

    def test_bounded_eviction(self) -> None:
        cache = _KernelCache(max_entries=2)
        resolved = _resolved()
        keys = [
            _make_cache_key(
                resolved,
                sequence_length=n,
                dtype=torch.float32,
                device_type="cpu",
                use_compile=False,
            )
            for n in (8, 16, 24)
        ]
        for key in keys:
            cache.get_or_build(key, resolved, False)

        cached_keys = cache.keys()
        assert len(cache) == 2
        assert keys[0] not in cached_keys
        assert keys[-1] in cached_keys

    def test_tail_batch_reuses_entry_not_a_new_key(self) -> None:
        # Batch size is not part of the cache key: dynamic-batch compilation means a
        # smaller tail chunk reuses the same warm entry instead of creating another one.
        cache = _KernelCache()
        resolved = _resolved()
        key = _make_cache_key(
            resolved, sequence_length=32, dtype=torch.float32, device_type="cpu", use_compile=False
        )

        cache.get_or_build(key, resolved, False)
        cache.get_or_build(key, resolved, False)

        assert len(cache) == 1
