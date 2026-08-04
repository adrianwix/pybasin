# pyright: basic, reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false
"""End-to-end worker kernel cache behavior over real Dask worker processes.

Uses process-based workers (not threads) so each worker's `_KernelCache` is a genuinely
separate, isolated process-local cache, then instruments `tstorch.dask._cache.build_kernel`
on every worker to count how many times it actually runs - the thing `_KernelCache` exists
to keep at one per (worker, cache key), never once per block or per `.compute()` call.
"""

import dask.array as da
import numpy as np
import pytest
import tstorch.dask._cache as cache_mod
from dask.distributed import Client, LocalCluster
from tstorch.dask import DaskFeatureExtractor


@pytest.fixture
def client():
    with (
        LocalCluster(
            n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None
        ) as cluster,
        Client(cluster) as client,
    ):
        yield client


def _make_array(batch_size: int, n_states: int, n_steps: int, chunks: int, seed: int) -> da.Array:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((batch_size, n_states, n_steps)).astype(np.float32)
    return da.from_array(data, chunks=(chunks, n_states, n_steps))


def _install_build_counter() -> None:
    """Wrap `build_kernel` with a call counter, once per worker process.

    Runs inside a worker process via `client.run()`. `cache_mod` is a module-level
    import, but the worker executing this reimports it independently - each worker
    process gets its own `tstorch.dask._cache` module object and its own counter.
    """
    if getattr(cache_mod, "_test_build_count", None) is not None:
        return
    cache_mod._test_build_count = 0
    original_build_kernel = cache_mod.build_kernel

    def counting_build_kernel(*args, **kwargs):
        cache_mod._test_build_count = cache_mod._test_build_count + 1
        return original_build_kernel(*args, **kwargs)

    cache_mod.build_kernel = counting_build_kernel


def _read_build_counts(client: Client) -> dict[str, int]:
    def _read() -> int:
        return getattr(cache_mod, "_test_build_count", 0)

    return client.run(_read)


class TestWorkerKernelCacheEndToEnd:
    def test_reused_across_blocks_within_one_call(self, client: Client) -> None:
        client.run(_install_build_counter)

        array = _make_array(batch_size=40, n_states=1, n_steps=32, chunks=5, seed=0)
        extractor = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=True)
        client.compute(extractor.extract_features(array)).result()

        counts = _read_build_counts(client)
        # 8 blocks spread over 2 workers: every worker compiles its kernel at most once,
        # however many blocks it ends up handling.
        assert all(count <= 1 for count in counts.values())
        assert sum(counts.values()) >= 1

    def test_reused_across_separate_compute_calls(self, client: Client) -> None:
        client.run(_install_build_counter)

        array_a = _make_array(batch_size=20, n_states=1, n_steps=32, chunks=5, seed=1)
        extractor_a = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=True)
        client.compute(extractor_a.extract_features(array_a)).result()
        counts_after_first = _read_build_counts(client)
        assert sum(counts_after_first.values()) >= 1

        # A second, independent extractor over fresh data, but the same feature/shape/
        # dtype/device/compile configuration: the resolved plan token and cache key are
        # identical, so no worker should compile again.
        array_b = _make_array(batch_size=20, n_states=1, n_steps=32, chunks=5, seed=2)
        extractor_b = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=True)
        client.compute(extractor_b.extract_features(array_b)).result()
        counts_after_second = _read_build_counts(client)

        assert counts_after_second == counts_after_first

    def test_different_sequence_length_forces_a_new_build(self, client: Client) -> None:
        client.run(_install_build_counter)

        array_a = _make_array(batch_size=20, n_states=1, n_steps=32, chunks=5, seed=1)
        extractor_a = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=True)
        client.compute(extractor_a.extract_features(array_a)).result()
        counts_after_first = sum(_read_build_counts(client).values())

        array_b = _make_array(batch_size=20, n_states=1, n_steps=64, chunks=5, seed=2)
        extractor_b = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=True)
        client.compute(extractor_b.extract_features(array_b)).result()
        counts_after_second = sum(_read_build_counts(client).values())

        assert counts_after_second > counts_after_first
