# Solution

Every basin stability run produces a `Solution` object that accumulates results as the pipeline progresses -- trajectories from integration, feature vectors from extraction, and basin labels from classification. You can inspect it directly for analysis or hand it to a plotter for visualization.

During `estimate_bs()`, the `BasinStabilityEstimator` creates the `Solution` and mutates it at each pipeline step. Access the finished object via `bse.solution`.

## What's Stored

| Property                  | Type                       | Description                                                                 | Available After           |
| ------------------------- | -------------------------- | --------------------------------------------------------------------------- | ------------------------- |
| `initial_condition`       | `torch.Tensor`             | Initial conditions (shape: `(N, n_states)`)                                 | Integration               |
| `time`                    | `torch.Tensor`             | Time evaluation points (shape: `(n_steps,)`)                                | Integration               |
| `y`                       | `torch.Tensor`             | Trajectories (shape: `(n_steps, N, n_states)`)                              | Integration               |
| `extracted_features`      | `torch.Tensor` or `None`   | Raw features pre-filtering (shape: `(N, F)`)                                | Feature extraction        |
| `extracted_feature_names` | `list[str]` or `None`      | Names of raw features                                                       | Feature extraction        |
| `features`                | `torch.Tensor` or `None`   | Filtered features (shape: `(N, F')`)                                        | Feature selection or None |
| `feature_names`           | `list[str]` or `None`      | Names of filtered features                                                  | Feature selection or None |
| `labels`                  | `np.ndarray` or `None`     | Basin assignments for each IC (shape: `(N,)`)                               | Classification            |
| `model_params`            | `dict[str, Any]` or `None` | ODE parameters used during integration                                      | Integration               |
| `orbit_data`              | `OrbitData` or `None`      | Peak amplitudes per attractor for orbit diagrams (see `compute_orbit_data`) | Integration               |

## Accessing the Solution

Retrieve the solution object from the estimator after running `estimate_bs()`:

```python
from pybasin.basin_stability_estimator import BasinStabilityEstimator

bse = BasinStabilityEstimator(ode_system=ode, sampler=sampler)
results = bse.estimate_bs()

# Access the complete solution
solution = bse.solution

# Trajectories for all initial conditions
trajectories = solution.y  # (n_steps, N, n_states)

# Initial conditions
ics = solution.initial_condition  # (N, n_states)

# Cluster labels
labels = solution.labels  # (N,)
```

## Features and Feature Names

Extracted features and filtered features are stored separately. After feature extraction, `extracted_features` holds the full raw matrix. When a feature selector is active, `features` contains the reduced subset; otherwise the BSE copies the unfiltered values into `features` as well. Both are always populated after `estimate_bs()` completes.

```python
# Raw features before filtering
raw_features = solution.extracted_features  # (N, F)
raw_names = solution.extracted_feature_names  # list[str] with F names

# Filtered features after selection (or unfiltered copy if no selector)
filtered_features = solution.features  # (N, F') where F' <= F
filtered_names = solution.feature_names  # list[str] with F' names
```

## Summary Output

Call `get_summary()` to produce a JSON-serializable dictionary with key information. This is useful for logging or saving results to disk.

```python
summary = solution.get_summary()
print(summary)
```

Output structure:

```python
{
    "initial_condition": [[x1, y1], [x2, y2], ...],  # N initial conditions
    "num_time_steps": 1000,
    "trajectory_shape": [1000, 10000, 2],
    "features": [[f1, f2, ...], ...] or None,
    "labels": [0, 1, 0, 2, ...] or None,
    "model_params": {"alpha": 0.1, "K": 1.0, ...} or None,
}
```

## Manual Construction

While `BasinStabilityEstimator` normally creates the solution object for you, you can also construct one manually for standalone feature extraction or custom workflows:

```python
import torch
from pybasin.solution import Solution

# Create synthetic trajectory data
time = torch.linspace(0, 1000, 1000)  # 1000 time points
y0 = torch.randn(500, 2)  # 500 initial conditions, 2-state system
y = torch.randn(1000, 500, 2)  # trajectories

solution = Solution(
    initial_condition=y0,
    time=time,
    y=y,
    model_params={"param1": 1.0, "param2": 2.5},
)

# Now you can pass this to a feature extractor
from pybasin.feature_extractors import TorchFeatureExtractor

extractor = TorchFeatureExtractor(features="minimal")
features = extractor.extract_features(solution)
```

### Shape Requirements

The constructor enforces strict shape constraints and raises `AssertionError` if shapes mismatch:

- `initial_condition`: 2D tensor of shape `(N, n_states)`
- `time`: 1D tensor of shape `(n_steps,)`
- `y`: 3D tensor of shape `(n_steps, N, n_states)`

The number of time steps in `y.shape[0]` must match `time.shape[0]`. The batch dimension `y.shape[1]` must match `initial_condition.shape[0]`. The state dimension `y.shape[2]` must match `initial_condition.shape[1]`.

## Modifying the Solution

Three setter methods allow pipeline components to update the solution as results become available:

### `set_labels(labels)`

Assigns cluster or classification labels. Labels must be a NumPy array with length matching the number of initial conditions.

```python
import numpy as np

labels = np.array([0, 1, 0, 2, 1, ...])  # length N
solution.set_labels(labels)
```

### `set_extracted_features(features, names)`

Stores raw features extracted from trajectories before any filtering step.

```python
features = torch.randn(500, 50)  # 500 samples, 50 features
names = [f"feature_{i}" for i in range(50)]

solution.set_extracted_features(features, names)
```

### `set_features(features, names=None)`

Stores filtered features after feature selection. Names are optional -- if omitted, `feature_names` remains unchanged.

```python
filtered_features = torch.randn(500, 20)  # 500 samples, 20 selected features
filtered_names = [f"selected_feature_{i}" for i in range(20)]

solution.set_features(filtered_features, filtered_names)
```

!!! tip "Pipeline Usage"
In normal usage, you never call these setters yourself. Feature extractors, feature selectors, and predictors call them automatically as they process the solution object. These methods are documented here for completeness and for custom pipeline development.

## Related Documentation

- [API Reference](../api/solution.md) -- Full method signatures and docstrings.
- [Feature Extractors](feature-extractors.md) -- How features are extracted and stored in the solution.
- [Feature Selectors](feature-selectors.md) -- How features are filtered and `feature_names` is updated.
- [Basin Stability Estimator Overview](basin-stability-estimator.md) -- How the solution object flows through the estimation pipeline.
