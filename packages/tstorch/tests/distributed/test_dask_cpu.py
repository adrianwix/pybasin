# pyright: basic, reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false
"""CPU distributed correctness tests for `DaskFeatureExtractor`.

Uses process-based Dask workers (not threads only) so serialization and per-process
caches are actually exercised, per the plan's test-matrix requirement.
"""

import dask.array as da
import numpy as np
import pytest
import torch
from dask.distributed import Client, LocalCluster
from tstorch.dask import DaskFeatureExtractor
from tstorch.torch_feature_extractor import TorchFeatureExtractor


@pytest.fixture(scope="module")
def client():
    with (
        LocalCluster(
            n_workers=2, threads_per_worker=1, processes=True, dashboard_address=None
        ) as cluster,
        Client(cluster) as client,
    ):
        yield client


def _make_array(batch_size: int, n_states: int, n_steps: int, chunks: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((batch_size, n_states, n_steps)).astype(np.float32)
    return data, da.from_array(data, chunks=(chunks, n_states, n_steps))


class TestDaskCpuCorrectness:
    def test_matches_local_concatenated_extraction(self, client: Client) -> None:
        data, array = _make_array(batch_size=20, n_states=2, n_steps=64, chunks=7)

        dask_extractor = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=False)
        lazy = dask_extractor.extract_features(array)
        distributed_result = client.compute(lazy).result()

        local_extractor = TorchFeatureExtractor(features="minimal", use_compile=False)
        local_result = local_extractor.extract_features(torch.from_numpy(data)).numpy()

        np.testing.assert_allclose(distributed_result, local_result, rtol=1e-4, atol=1e-5)
        assert dask_extractor.feature_names == local_extractor.feature_names

    def test_output_is_numpy_backed_dask_array(self, client: Client) -> None:
        _, array = _make_array(batch_size=10, n_states=1, n_steps=32, chunks=5)
        extractor = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=False)
        lazy = extractor.extract_features(array)

        assert isinstance(lazy, da.Array)
        result = client.compute(lazy).result()
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, len(extractor.feature_names))

    def test_smaller_tail_chunk_preserved_in_output(self, client: Client) -> None:
        data, array = _make_array(batch_size=13, n_states=1, n_steps=16, chunks=5)
        extractor = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=False)
        lazy = extractor.extract_features(array)

        assert lazy.chunks[0] == array.chunks[0]
        result = client.compute(lazy).result()
        assert result.shape[0] == 13

    def test_partitioning_does_not_change_values(self, client: Client) -> None:
        data, array_one_block = _make_array(batch_size=16, n_states=1, n_steps=32, chunks=16)
        _, array_many_blocks = _make_array(batch_size=16, n_states=1, n_steps=32, chunks=3)

        extractor_a = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=False)
        extractor_b = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=False)

        result_a = client.compute(extractor_a.extract_features(array_one_block)).result()
        result_b = client.compute(extractor_b.extract_features(array_many_blocks)).result()

        np.testing.assert_allclose(result_a, result_b, rtol=1e-4, atol=1e-5)

    def test_rejects_split_state_axis(self, client: Client) -> None:
        data = np.random.default_rng(0).standard_normal((10, 2, 16)).astype(np.float32)
        array = da.from_array(data, chunks=(5, 1, 16))
        extractor = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=False)
        with pytest.raises(ValueError):
            extractor.extract_features(array)

    def test_rejects_split_time_axis(self, client: Client) -> None:
        data = np.random.default_rng(0).standard_normal((10, 1, 16)).astype(np.float32)
        array = da.from_array(data, chunks=(5, 1, 8))
        extractor = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=False)
        with pytest.raises(ValueError):
            extractor.extract_features(array)

    def test_reusing_extractor_with_different_state_count_fails(self, client: Client) -> None:
        _, array_2_states = _make_array(batch_size=8, n_states=2, n_steps=16, chunks=4)
        _, array_3_states = _make_array(batch_size=8, n_states=3, n_steps=16, chunks=4)

        extractor = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=False)
        extractor.extract_features(array_2_states)
        names_before = extractor.feature_names

        with pytest.raises(ValueError):
            extractor.extract_features(array_3_states)

        assert extractor.feature_names == names_before

    def test_feature_names_raises_before_first_call(self) -> None:
        extractor = DaskFeatureExtractor(features="minimal", device="cpu", use_compile=False)
        with pytest.raises(RuntimeError):
            _ = extractor.feature_names
