# Creating Custom Feature Extractors

Feature extractors transform ODE solution trajectories into feature vectors used for basin of attraction classification. This guide shows how to create your own.

## Basic Implementation

To create a custom feature extractor, subclass `FeatureExtractor` and implement the `extract_features` method:

```python
import torch
from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


class AmplitudeFeatureExtractor(FeatureExtractor):
    def extract_features(self, solution: Solution) -> torch.Tensor:
        # Filter out transient behavior
        y_filtered: torch.Tensor = self.filter_time(solution)

        # Compute features - here we extract max amplitude per state
        # y_filtered shape: (n_samples, n_states, n_times)
        max_amplitude: torch.Tensor = torch.max(torch.abs(y_filtered), dim=-1).values

        # Set _num_features for automatic feature naming
        self._num_features = max_amplitude.shape[1]

        # Return shape: (n_samples, n_features)
        return max_amplitude
```

### Key Points

1. **Use `filter_time`**: Call `self.filter_time(solution)` to remove transient dynamics based on `time_steady`
2. **Return a tensor**: The return type must be `torch.Tensor` with shape `(n_samples, n_features)`
3. **Set `_num_features`**: Assign `self._num_features` to enable automatic feature naming
4. **Do NOT modify the Solution object**: The `extract_features` method should be pure - read from the solution, compute features, and return them. Never assign to `solution.features`, `solution.extracted_features`, or any other solution attributes.

## Using the Extractor

```python
extractor = AmplitudeFeatureExtractor(time_steady=100.0)
features = extractor.extract_features(solution)

# Feature names are automatically generated
print(extractor.feature_names)  # ['amplitude_1', 'amplitude_2', ...]
```

## Custom Feature Names

By default, feature names are generated automatically from the class name:

- `AmplitudeFeatureExtractor` → `amplitude_1`, `amplitude_2`, ...
- `SynchronizationFeatureExtractor` → `synchronization_1`, `synchronization_2`, ...

### Overriding Feature Names

To use custom, meaningful names, set `_feature_names` in `__init__`:

```python
class SynchronizationFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        n_nodes: int,
        time_steady: float = 1000.0,
    ):
        super().__init__(time_steady=time_steady)
        self.n_nodes = n_nodes
        # Define custom feature names
        self._feature_names = [
            "max_deviation_x",
            "max_deviation_y",
            "max_deviation_z",
            "max_deviation_all",
        ]

    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered: torch.Tensor = self.filter_time(solution)
        # ... compute features ...
        return features  # shape: (n_samples, 4)
```

When `_feature_names` is set, it takes precedence over automatic name generation.

## Complete Example

```python
import torch
from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


class MeanAndStdFeatureExtractor(FeatureExtractor):
    """Extract mean and standard deviation of each state variable."""

    def __init__(self, n_states: int, time_steady: float = 0.0):
        super().__init__(time_steady=time_steady)
        self.n_states = n_states
        # Custom names: mean_0, std_0, mean_1, std_1, ...
        self._feature_names = []
        for i in range(n_states):
            self._feature_names.extend([f"mean_{i}", f"std_{i}"])

    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered: torch.Tensor = self.filter_time(solution)
        # y_filtered: (n_samples, n_states, n_times)

        mean_vals: torch.Tensor = y_filtered.mean(dim=-1)  # (n_samples, n_states)
        std_vals: torch.Tensor = y_filtered.std(dim=-1)    # (n_samples, n_states)

        # Interleave: [mean_0, std_0, mean_1, std_1, ...]
        features: torch.Tensor = torch.stack(
            [mean_vals, std_vals], dim=2
        ).reshape(mean_vals.shape[0], -1)

        return features
```
