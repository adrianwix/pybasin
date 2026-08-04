# tstorch

tstorch provides PyTorch-native time-series feature calculators for batched CPU
and GPU workloads. Its calculators are tested for numerical agreement,
`torch.compile(fullgraph=True)`, dynamic-batch `torch.export`, and performance
against tsfresh where a reference implementation exists.

The package is being developed toward array and dataframe agnostic front ends,
distributed execution with Dask and RAPIDS, and features from recent
publications.

See the [tstorch documentation](https://adrianwix.github.io/pybasin/tstorch/)
for API details and the current evaluation record.

## Distributed extraction

`TorchFeatureExtractor` always runs locally and is the preferred path whenever one
machine can process the workload in memory-bounded batches. `tstorch.dask` adds
out-of-core and multi-machine orchestration on top of the same calculators, feature
selection, and raw `(B, F)` output - it never changes what a feature computes.

Dask, Dask-ML, `distributed`, and Dask-CUDA are all core dependencies, so `uv sync`
installs everything needed for both CPU and one-worker GPU distributed extraction by
default - no extra flags.

Local Dask CPU cluster:

```python
from dask.distributed import Client, LocalCluster

from tstorch.dask import DaskFeatureExtractor

with LocalCluster(n_workers=4, threads_per_worker=1) as cluster:
    with Client(cluster) as client:
        extractor = DaskFeatureExtractor(features="minimal", device="cpu")
        lazy_features = extractor.extract_features(trajectories)  # dask.array.Array (B, S, T)
        features = client.compute(lazy_features).result()  # NumPy (B, F)
```

One-worker Dask-CUDA (integration/correctness only - direct local CUDA execution
remains the recommended single-GPU path, not this wrapper):

```python
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

with LocalCUDACluster(n_workers=1, threads_per_worker=1) as cluster:
    with Client(cluster) as client:
        extractor = DaskFeatureExtractor(features="minimal", device="cuda")
        lazy_features = extractor.extract_features(trajectories)
        features = client.compute(lazy_features).result()  # still NumPy-backed
```

Only the batch axis (`B`) may have several chunks - `S` and `T` must each already be
one complete chunk:

```python
trajectories = trajectories.rechunk({0: 4096})
```

Raw features are unnormalized and un-imputed. Compose them explicitly with
`tstorch.preprocessing`, which accepts both eager `torch.Tensor` and (with `dask`
installed) NumPy-backed `dask.array.Array`:

```python
from tstorch.preprocessing import ExtremeImputer, StandardScaler

imputed = ExtremeImputer().transform(lazy_features)
scaler = StandardScaler().fit(imputed)
scaled = scaler.transform(imputed)  # same collection family as the input
mean, scale = scaler.state.mean, scaler.state.scale  # always CPU torch.Tensor
```
