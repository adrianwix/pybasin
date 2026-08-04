# pyright: basic, reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false
"""One-worker Dask-CUDA integration test.

Skipped entirely on hosts without a usable CUDA device: this is an integration/
correctness check for the array-conversion and worker-cache path, not a scaling claim,
and the plan explicitly scopes CUDA support to exactly one worker and one GPU for the
first release.
"""

import dask.array as da
import numpy as np
import pytest
import torch
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from tstorch.dask import DaskFeatureExtractor
from tstorch.torch_feature_extractor import TorchFeatureExtractor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


def _cuda_cluster_client() -> Client:
    return Client(LocalCUDACluster(n_workers=1, threads_per_worker=1))


class TestDaskCudaCorrectness:
    def test_matches_local_cuda_extraction(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((16, 2, 32)).astype(np.float32)
        array = da.from_array(data, chunks=(8, 2, 32))

        with _cuda_cluster_client() as client:
            extractor = DaskFeatureExtractor(features="minimal", device="cuda", use_compile=False)
            lazy = extractor.extract_features(array)
            distributed_result = client.compute(lazy).result()

        local_extractor = TorchFeatureExtractor(features="minimal", use_compile=False)
        local_result = local_extractor.extract_features(torch.from_numpy(data).cuda()).cpu().numpy()

        np.testing.assert_allclose(distributed_result, local_result, rtol=1e-4, atol=1e-5)

    def test_output_is_numpy_backed_even_from_cuda(self) -> None:
        data = np.random.default_rng(1).standard_normal((8, 1, 16)).astype(np.float32)
        array = da.from_array(data, chunks=(4, 1, 16))

        with _cuda_cluster_client() as client:
            extractor = DaskFeatureExtractor(features="minimal", device="cuda", use_compile=False)
            result = client.compute(extractor.extract_features(array)).result()

        assert isinstance(result, np.ndarray)
