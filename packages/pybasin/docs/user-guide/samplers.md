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

lhs_samples = LatinHypercubeSampler(region).sample(n=4096)
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

`EvenlySpacedSampler`, `CartesianProductSampler`, and `CsvSampler` are always deterministic (`is_deterministic=True`) and need no seed at all. The other four samplers draw randomly, and there is no `set_seed()` *method* on any sampler -- the name refers to a single global function, `pybasin.set_seed()`, unrelated to any per-sampler API.

`UniformRandomSampler`, `TruncatedGaussianSampler`, and `LatinHypercubeSampler` draw from PyTorch's global random state, so calling `sampler.sample()` twice gives different results by default. Fix this by calling `set_seed()` once at the top of your script, before any sampling or estimation:

```python
from pybasin import set_seed
from pybasin.regions import BoxRegion
from pybasin.samplers import UniformRandomSampler

set_seed(42)

sampler = UniformRandomSampler(BoxRegion([-1.0, -1.0], [1.0, 1.0]))
samples = sampler.sample(n=10_000)  # always the same, given the same call order
```

`set_seed()` seeds PyTorch (CPU and CUDA), NumPy, and Python's `random` module in one call, covering every stochastic step in the pipeline: sampling, feature extraction, and HDBSCAN clustering. Because it reseeds a single global stream, each subsequent `sample()` call still advances that stream -- two calls in the same script never return the same points, even right after `set_seed()`; only the *sequence* of draws across a full script run is reproducible.

Every sampler's `sample(n, *, generator=None)` also accepts an explicit `torch.Generator`, letting you control one call's draw locally instead of reseeding global state:

```python
generator = torch.Generator().manual_seed(42)
samples = sampler.sample(n=10_000, generator=generator)
```

`UniformRandomSampler`, `TruncatedGaussianSampler`, and `LatinHypercubeSampler` honor it. `SobolSampler` does not: it wraps `torch.quasirandom.SobolEngine`, whose scrambling only accepts an integer seed at construction time, not a `torch.Generator` and not the global RNG state -- so `generator` is accepted for structural compatibility with the `Sampler` protocol but has no effect. Control `SobolSampler` with its own `seed=` constructor argument instead, and construct a new instance to change it:

```python
sobol_samples = SobolSampler(region, scramble=True, seed=42).sample(n=4096)
```

### Reproducibility across a parameter study

`BasinStabilityStudy` samples initial conditions independently for each parameter combination unless every varied assignment targets `ode_system.params[...]`. In that all-ODE-params case, `BasinStabilityEstimator` draws `y0` once and reuses it for every combination, so all parameter values are compared against the same initial conditions. If any other component varies instead (sampler arguments, solver settings, feature extractor, ...), each combination gets its own independent `sampler.sample()` call, so initial conditions differ across the study by default -- calling `set_seed()` beforehand does not change this, since each call still advances the same global stream. There is currently no built-in option to force shared initial conditions across a mixed-parameter study.

## Implementing a custom sampler

Any object with `region`, `is_deterministic`, `device`, and `dtype` attributes, plus a `sample(n, *, generator=None)` method, structurally satisfies the `Sampler` protocol. No base class or inheritance is required:

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

    def sample(self, n: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        # Must return a tensor of shape (n, region.state_dim), on self.device/self.dtype,
        # satisfying region.contains for every row.
        ...
```

Custom regions work the same way: implement `state_dim`, `contains`, `device`, and `dtype` to satisfy the `Region` protocol.
