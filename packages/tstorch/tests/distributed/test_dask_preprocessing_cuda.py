# pyright: basic, reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false
"""Distributed preprocessing over CuPy-backed Dask Arrays on a one-worker CUDA cluster.

Mirrors `test_dask_preprocessing.py`'s NumPy-backed checks, but every array here is
CuPy-backed (RAPIDS-style), so the assertions also confirm the result stays GPU-resident
end to end rather than round-tripping through NumPy on the host.
"""

import cupy
import dask.array as da
import numpy as np
import pytest
import torch
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from tstorch.preprocessing import ExtremeImputer, StandardScaler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def _cuda_cluster_client() -> Client:
    return Client(LocalCUDACluster(n_workers=1, threads_per_worker=1))


def _cupy_array(data: np.ndarray, chunks: int) -> da.Array:
    return da.from_array(cupy.asarray(data), chunks=(chunks, data.shape[1]))


class TestExtremeImputerDaskCuda:
    def test_matches_eager_result_and_stays_cupy_backed(self) -> None:
        data = np.array(
            [[1.0, np.nan], [np.inf, 2.0], [-np.inf, 3.0], [4.0, 5.0]], dtype=np.float32
        )
        eager_result = ExtremeImputer().transform(torch.from_numpy(data)).numpy()

        with _cuda_cluster_client() as client:
            array = _cupy_array(data, chunks=1)
            lazy = ExtremeImputer().transform(array)
            assert isinstance(lazy._meta, cupy.ndarray)

            result = client.compute(lazy).result()

        assert isinstance(result, cupy.ndarray)
        np.testing.assert_allclose(cupy.asnumpy(result), eager_result)


class TestStandardScalerDaskCuda:
    def test_fit_state_stays_on_cuda(self) -> None:
        data = np.random.default_rng(0).standard_normal((50, 2)).astype(np.float32)

        with _cuda_cluster_client():
            array = _cupy_array(data, chunks=10)
            state = StandardScaler().fit(array).state

        assert state.mean.device.type == "cuda"
        assert state.scale.device.type == "cuda"
        assert state.n_samples_seen == 50

    def test_transform_matches_eager_result_and_stays_cupy_backed(self) -> None:
        data = np.random.default_rng(1).standard_normal((60, 3)).astype(np.float32)

        eager_scaler = StandardScaler().fit(torch.from_numpy(data))
        eager_transformed = eager_scaler.transform(torch.from_numpy(data)).numpy()

        with _cuda_cluster_client() as client:
            array = _cupy_array(data, chunks=13)
            scaler = StandardScaler().fit(array)
            lazy = scaler.transform(array)
            assert isinstance(lazy._meta, cupy.ndarray)

            result = client.compute(lazy).result()

        assert isinstance(result, cupy.ndarray)
        np.testing.assert_allclose(cupy.asnumpy(result), eager_transformed, rtol=1e-4, atol=1e-5)
