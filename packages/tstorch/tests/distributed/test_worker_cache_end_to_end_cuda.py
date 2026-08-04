"""End-to-end worker kernel cache behavior on a one-worker Dask-CUDA cluster.

Mirrors `test_worker_cache_end_to_end.py`'s CPU checks on the CUDA path: the plan scopes
CUDA support to exactly one worker and one GPU for the first release (see
`test_dask_cuda.py`), so there is only one worker process whose `_KernelCache` to inspect,
but the same question applies - does `torch.compile` actually run once and get reused, or
does every block/call recompile.
"""

import dask.array as da
import numpy as np
import pytest
import torch
import tstorch.dask._cache as cache_mod
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from tstorch.dask import DaskFeatureExtractor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def _cuda_cluster_client() -> Client:
    return Client(LocalCUDACluster(n_workers=1, threads_per_worker=1))


@pytest.fixture
def client():
    with _cuda_cluster_client() as client:
        yield client


def _make_array(batch_size: int, n_states: int, n_steps: int, chunks: int, seed: int):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((batch_size, n_states, n_steps)).astype(np.float32)
    return da.from_array(data, chunks=(chunks, n_states, n_steps))


def _install_build_counter() -> None:
    """Wrap `build_kernel` with a call counter, once per worker process.

    Runs inside the worker process via `client.run()`. `cache_mod` is a module-level
    import, but the worker executing this reimports it independently - each worker
    process gets its own `tstorch.dask._cache` module object and its own counter.
    """
    if getattr(cache_mod, "_test_build_count", None) is not None:
        return
    cache_mod._test_build_count = 0
    original_build_kernel = cache_mod.build_kernel

    def counting_build_kernel(*args, **kwargs):
        cache_mod._test_build_count += 1
        return original_build_kernel(*args, **kwargs)

    cache_mod.build_kernel = counting_build_kernel


def _read_build_counts(client: Client) -> dict[str, int]:
    def _read() -> int:
        return getattr(cache_mod, "_test_build_count", 0)

    return client.run(_read)


class TestWorkerKernelCacheEndToEndCuda:
    def test_reused_across_blocks_within_one_call(self, client: Client) -> None:
        client.run(_install_build_counter)

        array = _make_array(batch_size=40, n_states=1, n_steps=32, chunks=5, seed=0)
        extractor = DaskFeatureExtractor(features="minimal", device="cuda", use_compile=True)
        client.compute(extractor.extract_features(array)).result()

        counts = _read_build_counts(client)
        # 8 blocks, one worker: the kernel is compiled at most once regardless.
        assert sum(counts.values()) == 1

    def test_reused_across_separate_compute_calls(self, client: Client) -> None:
        client.run(_install_build_counter)

        array_a = _make_array(batch_size=20, n_states=1, n_steps=32, chunks=5, seed=1)
        extractor_a = DaskFeatureExtractor(features="minimal", device="cuda", use_compile=True)
        client.compute(extractor_a.extract_features(array_a)).result()
        counts_after_first = sum(_read_build_counts(client).values())
        assert counts_after_first == 1

        # A second, independent extractor over fresh data, but the same feature/shape/
        # dtype/device/compile configuration: the resolved plan token and cache key are
        # identical, so the worker should not compile again.
        array_b = _make_array(batch_size=20, n_states=1, n_steps=32, chunks=5, seed=2)
        extractor_b = DaskFeatureExtractor(features="minimal", device="cuda", use_compile=True)
        client.compute(extractor_b.extract_features(array_b)).result()
        counts_after_second = sum(_read_build_counts(client).values())

        assert counts_after_second == counts_after_first

    def test_different_sequence_length_forces_a_new_build(self, client: Client) -> None:
        client.run(_install_build_counter)

        array_a = _make_array(batch_size=20, n_states=1, n_steps=32, chunks=5, seed=1)
        extractor_a = DaskFeatureExtractor(features="minimal", device="cuda", use_compile=True)
        client.compute(extractor_a.extract_features(array_a)).result()
        counts_after_first = sum(_read_build_counts(client).values())

        array_b = _make_array(batch_size=20, n_states=1, n_steps=64, chunks=5, seed=2)
        extractor_b = DaskFeatureExtractor(features="minimal", device="cuda", use_compile=True)
        client.compute(extractor_b.extract_features(array_b)).result()
        counts_after_second = sum(_read_build_counts(client).values())

        assert counts_after_second > counts_after_first
