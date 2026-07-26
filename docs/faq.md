# FAQ

Answers to questions that come up repeatedly when running pybasin, especially around GPU usage.

## Why does my pipeline crash with `CUDA_ERROR_OUT_OF_MEMORY` when using `JaxSolver` with a GPU predictor?

[`JaxSolver`](user-guide/solvers.md#jaxsolver) uses JAX/Diffrax for integration, and by default JAX
preallocates roughly 75% of GPU memory the first time it touches the device -- regardless of how
much the integration itself actually needs. If your pipeline also runs a GPU predictor in the same
process (`HDBSCANClusterer`, `DBSCANClusterer`, `KNeighborsClassifier` with `backend="gpu"`, or any
other cuML/cupy-backed component), that preallocation can leave too little memory for cuML/cupy to
allocate their own pools, causing a `CUDA_ERROR_OUT_OF_MEMORY` crash partway through a run -- often
several minutes in, after JAX has already integrated many trajectories successfully.

Fix it by setting these environment variables **before** JAX is imported, directly or
transitively (e.g. before importing `pybasin` at all in your entry-point script):

```python
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import pybasin  # import pybasin (and anything that imports JaxSolver) after the env vars are set
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false` stops JAX from grabbing a large block of memory upfront and
lets it grow allocations on demand instead. `XLA_PYTHON_CLIENT_ALLOCATOR=platform` switches JAX to
the native `cudaMalloc`-based allocator, which shares the device more cooperatively with other CUDA
libraries in the same process than JAX's default BFC allocator does.
