# Regions & Samplers

Generating initial conditions is split into two concerns:

- A **region** (`pybasin.regions`) defines geometric membership only: does this point fall inside? It carries no opinion on how points get generated, or what device/dtype they end up on.
- A **sampler** (`pybasin.samplers`), by contrast, generates the complete initial conditions inside a region. Every sampler returns a PyTorch tensor and supports GPU acceleration.

```python
import numpy as np
from pybasin.regions import BoxRegion
from pybasin.samplers import UniformRandomSampler

region = BoxRegion([-np.pi, -2.0], [np.pi, 2.0])
sampler = UniformRandomSampler(region)
samples = sampler.sample(n=10_000)
```

## Regions

| Class                | Shape                                            |
| --------------------- | ------------------------------------------------- |
| `BoxRegion`            | Axis-aligned box (`min`, `max` per dimension)      |
| `BallRegion`           | Filled interior of a Euclidean ball                |
| `SphereSurfaceRegion`  | Surface of a sphere (a thin shell, not the interior) |

`min`/`max`/`center` each accept a `torch.Tensor` or a plain Python sequence of floats. A plain sequence is converted to a `float32` CPU tensor automatically:

```python
from pybasin.regions import BoxRegion, BallRegion, SphereSurfaceRegion

box = BoxRegion([-10.0, -20.0, 0.0], [10.0, 20.0, 0.0])
ball = BallRegion([0.0, 0.0], radius=1.0)
shell = SphereSurfaceRegion([0.0, 0.0, 0.0], radius=1.0)
```

### Fixed dimensions

To fix a state variable to a constant value (e.g. initial velocity = 0), set the same value for both `min` and `max` in that dimension of a `BoxRegion`. Every sampler that supports `BoxRegion` handles this correctly:

```python
# 3D state space (x, y, z) with z fixed at 0
region = BoxRegion([-10.0, -20.0, 0.0], [10.0, 20.0, 0.0])  # z is fixed at 0
```

`BallRegion` and `SphereSurfaceRegion` have no equivalent "fixed dimension" concept, since every axis shares the same radius.

## Samplers

| Class                      | Regions                    | Returns exact `n`? | `is_deterministic` |
| --------------------------- | --------------------------- | ------------------- | --------------- |
| `UniformRandomSampler`      | Box, Ball, SphereSurface    | ✓                    | ✗ |
| `EvenlySpacedSampler`       | Box, Ball, SphereSurface\*  | Approximate (exact for SphereSurface) | ✓ |
| `CartesianProductSampler`   | Box, Ball                   | Determined by `coordinates`, not `n` | ✓ |
| `TruncatedGaussianSampler`  | Box, Ball                   | ✓                    | ✗ |
| `LatinHypercubeSampler`     | Box                          | ✓                    | ✗ |
| `SobolSampler`              | Box                          | ✓                    | ✗ |
| `CsvSampler`                | n/a -- region is fitted from the CSV | ✓ (or all rows) | ✓ |

\* `EvenlySpacedSampler` on `SphereSurfaceRegion` requires `state_dim == 3` (Fibonacci-sphere spacing is only implemented in 3D).

`is_deterministic=False` means the sampler draws randomly on every `sample()` call unless seeded; see [Reproducibility](#reproducibility) for how each of the four random samplers is seeded.

Passing an unsupported region to a sampler raises `UnsupportedSamplerRegionError` immediately at construction. An unsupported *configuration* of an otherwise-supported region type (a 2D `SphereSurfaceRegion` passed to `EvenlySpacedSampler`, for example) raises `UnsupportedSamplerConfigurationError` instead.

### `UniformRandomSampler`

Uniform sampling, dispatched by region type: independent `U(a, b)` per dimension for a box, uniform *by volume* for a ball, uniform *by surface area* for a sphere surface.

```python
from pybasin.samplers import UniformRandomSampler

sampler = UniformRandomSampler(region)
samples = sampler.sample(n=10_000)
```

### `EvenlySpacedSampler`

Deterministic, geometry-aware coverage. `sample(n)` works like every other sampler, but for `BoxRegion`/`BallRegion` the count is only *approximately* `n`. It gets distributed across active dimensions as `ceil(n ** (1 / n_active))` per axis, matching the old `GridSampler`'s behavior, so the actual count usually lands slightly above `n`. `BallRegion` complicates this further: it filters the enclosing-box grid by `contains` afterward, so its count can also fall below that approximation. Check `n_points(n)` first if you need the exact resulting count before generating anything.

```python
from pybasin.samplers import EvenlySpacedSampler

sampler = EvenlySpacedSampler(region)
sampler.n_points(10_000)  # exact count sample(10_000) would return
samples = sampler.sample(10_000)
```

For `SphereSurfaceRegion` (3D only), `n` is the exact Fibonacci-sphere point count.

### `CartesianProductSampler`

Builds the Cartesian product of explicit per-dimension coordinate tensors. Unlike every other sampler, `sample()` takes no `n`: the point count is entirely determined by `coordinates`, and there's no meaningful "first n" of a Cartesian grid to slice.

```python
import torch
from pybasin.samplers import CartesianProductSampler

sampler = CartesianProductSampler(
    region, coordinates=[torch.arange(0.0, 10.01, 0.01), torch.arange(100.0, 1001.0, 1.0)]
)
sampler.n_points()  # total points before region filtering
samples = sampler.sample()
```

Points are filtered by `region.contains` afterward, since user-supplied coordinates aren't guaranteed to respect the region's own bounds.

### `TruncatedGaussianSampler`

Gaussian candidates are kept only when `region.contains` accepts them; none are ever clipped. Clipping would pile artificial mass on the boundary, so rejection sampling is used instead, preserving the true truncated-Gaussian shape.

```python
from pybasin.samplers import TruncatedGaussianSampler

sampler = TruncatedGaussianSampler(region, mean=torch.zeros(2), std=0.3 * torch.ones(2))
samples = sampler.sample(n=10_000)
```

For a `BoxRegion` with fixed dimensions, `mean` must equal the fixed value and `std` must be zero on those dimensions. Raises `RuntimeError` if `max_candidate_batches` is exhausted before collecting `n` accepted points.

### `LatinHypercubeSampler` / `SobolSampler`

Latin hypercube and Sobol quasi-random sampling, `BoxRegion` only. Only active (non-fixed) dimensions participate; fixed dimensions are inserted afterward.

```python
from pybasin.samplers import LatinHypercubeSampler, SobolSampler

lhs_samples = LatinHypercubeSampler(region, seed=42).sample(n=4096)
sobol_samples = SobolSampler(region, scramble=True, seed=42).sample(n=4096)
```

For `SobolSampler`, batch sizes that are powers of two tend to be the most balanced.

### `CsvSampler`

Loads pre-defined samples from a CSV file, which is essential for reproducing results from MATLAB or other reference implementations. Its region does not come from the caller. Instead, it is a `BoxRegion` fitted to each coordinate column's own min/max.

```python
from pybasin.samplers import CsvSampler

sampler = CsvSampler(
    csv_path="data/initial_conditions.csv",
    coordinate_columns=["x1", "x2"],
    label_column="attractor",  # optional: ground truth labels
    device="cuda",  # optional: auto-detects GPU if None
)

sampler.region  # BoxRegion fitted from the CSV's own data
samples = sampler.sample()  # every row, in file order
samples = sampler.sample(n=1000)  # the first 1000 rows
labels = sampler.labels  # numpy array or None
```

| Exception            | Condition                                      |
| --------------------- | ----------------------------------------------- |
| `FileNotFoundError`   | CSV file does not exist at the specified path    |
| `ValueError`          | Coordinate columns not found in CSV              |
| `ValueError`          | Label column not found in CSV (when specified)   |
| `ValueError`          | Requested `n` samples exceeds available data     |

## Device and dtype

A region carries only whatever device/dtype its own tensors happen to have; building one from a plain Python list always produces a `float32` CPU tensor. Every sampler, except `CsvSampler` (which keeps its own `device` parameter unchanged for backward compatibility), accepts an optional `device`/`dtype` override. Applied once at construction, this override takes precedence over the region's own placement:

```python
sampler = UniformRandomSampler(region, device="cuda", dtype=torch.float64)
sampler.device  # torch.device("cuda:0") -- resolved, never None after construction
sampler.region.device  # same -- the sampler's region is moved to match
```

The region instance you passed in is left unmodified (regions are immutable); the sampler resolves its own, possibly-relocated copy.

## Reproducibility

`EvenlySpacedSampler`, `CartesianProductSampler`, and `CsvSampler` are always deterministic (`is_deterministic=True`) and need no seed at all. The other four samplers -- `UniformRandomSampler`, `TruncatedGaussianSampler`, `LatinHypercubeSampler`, `SobolSampler` -- draw randomly, and all four share the same seeding shape: a `seed: int | None = None` constructor argument, and a `seed: int | None = None` keyword on `sample()` that overrides it for one call. Passing the same seed always returns the same points, however many times you call it:

```python
from pybasin.regions import BoxRegion
from pybasin.samplers import UniformRandomSampler

sampler = UniformRandomSampler(BoxRegion([-1.0, -1.0], [1.0, 1.0]), seed=42)
samples = sampler.sample(n=10_000)  # always the same points, every call

other_samples = sampler.sample(n=10_000, seed=7)  # overrides the constructor seed, once
```

Internally, each call with a resolved seed builds a fresh `torch.Generator` and discards it afterward -- there's no `torch.Generator` in the public API, and no shared, advancing RNG state to accidentally reuse across calls. `seed=None` (the default, at either the constructor or the call) leaves the sampler drawing from PyTorch's global random state instead, so results vary from call to call unless you seed that global state yourself.

There is no `set_seed()` *method* on any sampler -- that name refers to a single global function, `pybasin.set_seed()`, which seeds PyTorch (CPU and CUDA), NumPy, and Python's `random` module in one call, covering every stochastic step in the pipeline: sampling (when a sampler's own `seed` is `None`), feature extraction, and HDBSCAN clustering. Use the per-sampler `seed` for reproducible initial conditions specifically; use `pybasin.set_seed()` when you need the rest of the pipeline reproducible too.

### Reproducibility across a parameter study

Both `BasinStabilityEstimator` and `BasinStabilityStudy` accept a `seed: int | None = None` constructor argument, passed straight through to `sampler.sample(n, seed=...)` wherever they sample. `BasinStabilityStudy` passes its `seed` to every `BasinStabilityEstimator` it creates:

```python
study = BasinStabilityStudy(..., seed=42)
```

- If every varied assignment targets `ode_system.params[...]`, `BasinStabilityStudy` already draws `y0` once and reuses it for every parameter combination, regardless of `seed` -- `seed` here just makes that one draw reproducible across repeated runs of the whole study.
- Otherwise, `BasinStabilityStudy` creates one `BasinStabilityEstimator` per combination, each sampling independently. Passing the same `seed` to all of them makes every combination draw the *same* initial conditions, so parameter values are compared on equal footing, and the whole sweep is reproducible across repeated runs.

`seed` here only reaches the sampler -- it does not seed feature extraction or the predictor. Combine it with `pybasin.set_seed()` if you need those reproducible too.

## Implementing a custom sampler

Any object with `region`, `is_deterministic`, `device`, and `dtype` attributes, plus a `sample(n, *, seed=None)` method, structurally satisfies the `Sampler` protocol. No base class or inheritance is required:

```python
from dataclasses import dataclass, field

import torch
from pybasin.regions import BoxRegion, Region


@dataclass
class MyCustomSampler:
    region: Region
    device: torch.device
    dtype: torch.dtype
    is_deterministic: bool = field(default=False, init=False)

    def sample(self, n: int, *, seed: int | None = None) -> torch.Tensor:
        # Must return a tensor of shape (n, region.state_dim), on self.device/self.dtype,
        # satisfying region.contains for every row. If randomness is involved, build a
        # fresh torch.Generator from `seed` when it's not None, matching every built-in
        # random sampler.
        ...
```

Custom regions work the same way: implement `state_dim`, `contains`, `device`, and `dtype` to satisfy the `Region` protocol.
