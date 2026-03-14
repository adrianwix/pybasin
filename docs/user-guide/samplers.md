# Samplers

Samplers generate initial conditions for basin stability estimation. All samplers return PyTorch tensors (`float32`) and support GPU acceleration.

!!! info "Tensor Precision"
All samplers use `float32` precision for GPU efficiency (5-10x faster than `float64`). Samples are returned as `torch.Tensor` on the configured device.

## Available Samplers

| Class                  | Description                 | Returns Exact N? | Deterministic?    |
| ---------------------- | --------------------------- | ---------------- | ----------------- |
| `UniformRandomSampler` | Uniform random in hypercube | ✓                | With `set_seed()` |
| `GridSampler`          | Evenly spaced regular grid  | ✗ (scales up)    | ✓                 |
| `GaussianSampler`      | Gaussian around midpoint    | ✓                | With `set_seed()` |
| `CsvSampler`           | Load from CSV file          | ✓                | ✓                 |

## Common Parameters

All samplers share these constructor parameters:

| Parameter    | Type            | Description                                  |
| ------------ | --------------- | -------------------------------------------- |
| `min_limits` | `list[float]`   | Minimum value for each state dimension       |
| `max_limits` | `list[float]`   | Maximum value for each state dimension       |
| `device`     | `str` or `None` | `"cuda"`, `"cpu"`, or `None` for auto-detect |

## Fixed Dimensions

To fix a state variable to a constant value (e.g., initial velocity = 0), set the same value for both `min` and `max` in that dimension. All samplers handle this correctly:

```python
# 3D state space (x, y, z) with z fixed at 0
sampler = UniformRandomSampler(
    min_limits=[-10.0, -20.0, 0.0],
    max_limits=[10.0, 20.0, 0.0],  # z is fixed at 0
)
```

!!! note "CsvSampler Exception"
`CsvSampler` does not accept `min_limits`/`max_limits`—bounds are computed from the CSV data.

---

## UniformRandomSampler

Generates random samples uniformly distributed within the bounding hypercube.

```python
from pybasin.sampler import UniformRandomSampler
import numpy as np

sampler = UniformRandomSampler(
    min_limits=[-np.pi, -2.0],
    max_limits=[np.pi, 2.0],
    device="cuda",  # optional, auto-detects GPU
)

# Generate 10,000 samples (returns exactly 10,000)
samples = sampler.sample(n=10000)
```

For reproducible results, call `set_seed()` before sampling. See [Reproducibility](#reproducibility) below.

---

## GridSampler

Generates evenly spaced samples in a regular grid pattern. Ideal for 2D visualizations and deterministic sampling.

```python
from pybasin.sampler import GridSampler
import numpy as np

sampler = GridSampler(
    min_limits=[-np.pi, -2.0],
    max_limits=[np.pi, 2.0],
)

samples = sampler.sample(n=10000)  # Returns 10,000 samples (100×100 grid)
```

### Sample Count Scaling

The grid sampler **rounds up** the requested sample count to form a complete grid. For a _d_-dimensional space, it computes:

$$n_{\text{per dim}} = \lceil n^{1/d} \rceil$$

The actual number of samples returned is $n_{\text{per dim}}^d$.

**2D Examples:**

| Requested N | Points per Dimension | Actual Samples    |
| ----------- | -------------------- | ----------------- |
| 50          | ⌈50^0.5⌉ = 8         | 8² = **64**       |
| 100         | ⌈100^0.5⌉ = 10       | 10² = **100**     |
| 1,000       | ⌈1000^0.5⌉ = 32      | 32² = **1,024**   |
| 20,000      | ⌈20000^0.5⌉ = 142    | 142² = **20,164** |

### Fixed Dimensions and Sample Count

When using fixed dimensions (see [Fixed Dimensions](#fixed-dimensions)), only the varying dimensions contribute to the grid calculation.

Given $d$ varying dimensions and requested $n$ samples, the points per varying dimension is $\lceil n^{1/d} \rceil$. Fixed dimensions always contribute exactly 1 point, so the total number of samples is:

$$\text{total} = \underbrace{\lceil n^{1/d} \rceil \times \lceil n^{1/d} \rceil \times \cdots}_{d \text{ varying dims}} \times \underbrace{1 \times 1 \times \cdots}_{\text{fixed dims}} = (\lceil n^{1/d} \rceil)^d$$

**Example:** 3D space with 1 fixed dimension and $n = 20000$:

- $d = 2$ varying dimensions → $20000^{1/2} = 141.42...$, so $\lceil 141.42 \rceil = 142$ points per dimension
- Total: $142 \times 142 \times 1 = 20164$ samples

---

## GaussianSampler

Generates samples from a Gaussian distribution centered at the midpoint of each dimension. Samples are clamped to stay within bounds.

```python
from pybasin.sampler import GaussianSampler

sampler = GaussianSampler(
    min_limits=[-np.pi, -2.0],
    max_limits=[np.pi, 2.0],
    std_factor=0.2,  # σ = 20% of the range (default)
)

samples = sampler.sample(n=10000)
```

The distribution parameters are computed as:

$$\mu_i = \frac{\text{min}_i + \text{max}_i}{2}, \quad \sigma_i = \text{std_factor} \times (\text{max}_i - \text{min}_i)$$

---

## CsvSampler

Loads pre-defined samples from a CSV file. Essential for reproducing results from MATLAB or other reference implementations.

### Constructor Parameters

| Parameter            | Type            | Default  | Description                                  |
| -------------------- | --------------- | -------- | -------------------------------------------- |
| `csv_path`           | `str` or `Path` | Required | Path to the CSV file containing samples      |
| `coordinate_columns` | `list[str]`     | Required | Column names to use as state coordinates     |
| `label_column`       | `str` or `None` | `None`   | Column name for ground truth labels          |
| `device`             | `str` or `None` | `None`   | `"cuda"`, `"cpu"`, or `None` for auto-detect |

```python
from pybasin.sampler import CsvSampler

sampler = CsvSampler(
    csv_path="data/initial_conditions.csv",
    coordinate_columns=["x1", "x2"],      # Column names for state variables
    label_column="attractor",             # Optional: ground truth labels
    device="cuda",                        # Optional: auto-detects GPU if None
)

# Get all samples from the file
samples = sampler.sample()

# Or get the first n samples
samples = sampler.sample(n=1000)

# Access ground truth labels (if provided)
labels = sampler.labels  # numpy array or None
```

!!! note "Bounds Auto-Detection"
Unlike other samplers, `CsvSampler` does not require `min_limits` and `max_limits`. These are automatically computed from the data in the CSV file.

### Exceptions

| Exception           | Condition                                      |
| ------------------- | ---------------------------------------------- |
| `FileNotFoundError` | CSV file does not exist at the specified path  |
| `ValueError`        | Coordinate columns not found in CSV            |
| `ValueError`        | Label column not found in CSV (when specified) |
| `ValueError`        | Requested `n` samples exceeds available data   |

### Properties

| Property    | Type                   | Description                         |
| ----------- | ---------------------- | ----------------------------------- |
| `labels`    | `np.ndarray` or `None` | Ground truth labels from CSV        |
| `n_samples` | `int`                  | Total number of samples in the file |

---

## Reproducibility

Because `UniformRandomSampler` and `GaussianSampler` draw from the global PyTorch random state, calling `sampler.sample()` twice gives different results by default. Fix this by calling `set_seed()` once at the top of your script before any sampling or estimation:

```python
from pybasin import set_seed
from pybasin.sampler import UniformRandomSampler

set_seed(42)

sampler = UniformRandomSampler(min_limits=[-1.0, -1.0], max_limits=[1.0, 1.0])
samples = sampler.sample(n=10000)  # always the same
```

`set_seed()` seeds PyTorch (CPU and CUDA), NumPy, and Python's `random` module in one call. This covers every stochastic step in the pipeline -- sampling, feature extraction, and HDBSCAN clustering.

`GridSampler` and `CsvSampler` are always deterministic and do not require a seed.

---

## Creating Custom Samplers

Inherit from `Sampler` and implement the `sample` method:

```python
from pybasin.sampler import Sampler
import torch

class LatinHypercubeSampler(Sampler):
    """Latin Hypercube sampling for better space coverage."""

    display_name: str = "Latin Hypercube Sampler"

    def __init__(
        self,
        min_limits: list[float],
        max_limits: list[float],
        device: str | None = None,
    ):
        super().__init__(min_limits, max_limits, device)

    def sample(self, n: int) -> torch.Tensor:
        # Your implementation here
        # Must return tensor of shape (n, self.state_dim)
        ...
```

### Base Class Attributes

After calling `super().__init__()`, these attributes are available:

| Attribute    | Type           | Description                                    |
| ------------ | -------------- | ---------------------------------------------- |
| `device`     | `torch.device` | Target device (`cuda:0` or `cpu`)              |
| `min_limits` | `torch.Tensor` | Minimum bounds as `float32` tensor on `device` |
| `max_limits` | `torch.Tensor` | Maximum bounds as `float32` tensor on `device` |
| `state_dim`  | `int`          | Number of state dimensions (length of limits)  |

### Requirements

1. **Call `super().__init__()`** with `min_limits`, `max_limits`, and `device`
2. **Return a `torch.Tensor`** of shape `(n, self.state_dim)`
3. **Use `self.device`** when creating tensors to ensure GPU compatibility
4. **Use `float32` dtype** for consistency with the base class
