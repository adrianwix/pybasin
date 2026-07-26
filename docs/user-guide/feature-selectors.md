# Feature Selectors

Feature selectors remove redundant or uninformative features from the extracted feature matrix before it reaches the predictor. After feature extraction, the matrix can contain hundreds of columns -- many with near-zero variance or high mutual correlation. Filtering these out reduces noise, speeds up clustering, and often improves classification accuracy.

By default, `BasinStabilityEstimator` applies a `DefaultFeatureSelector` that chains variance thresholding with pairwise correlation filtering. Any sklearn-compatible transformer can be used instead, and passing `None` disables selection entirely.

## Available Selectors

| Class                    | Strategy                                | Best for                                                  |
| ------------------------ | --------------------------------------- | --------------------------------------------------------- |
| `DefaultFeatureSelector` | Variance threshold + correlation filter | **Default.** General-purpose two-stage pipeline.          |
| `CorrelationSelector`    | Mean absolute correlation ranking       | Targeted removal of redundant correlated features.        |
| Any sklearn transformer  | Custom (e.g., `SelectKBest`, k-best)    | Domain-specific selection when defaults are insufficient. |

## Default Behavior in BasinStabilityEstimator

When no `feature_selector` argument is provided, the estimator creates a `DefaultFeatureSelector()` with these defaults:

- **Variance threshold**: 0.01 -- features with variance below this are dropped
- **Correlation threshold**: 0.95 -- among features with |correlation| above this, the more globally redundant one is dropped
- **Minimum features**: 2 -- at least two features are always retained

The selector's `feature_selector` parameter uses a sentinel pattern to distinguish "not specified" (use default) from `None` (disable). Passing `None` explicitly turns off all feature filtering.

```python
from pybasin import BasinStabilityEstimator

# Default: uses DefaultFeatureSelector()
bse = BasinStabilityEstimator(ode_system=..., sampler=..., n=5000)

# Disabled: no feature filtering at all
bse = BasinStabilityEstimator(ode_system=..., sampler=..., n=5000, feature_selector=None)
```

!!! tip "When to disable"
For pipelines with `features="minimal"` (10 features per state), feature selection may remove potentially useful columns. Consider disabling it or lowering the thresholds when working with small feature sets.

---

## DefaultFeatureSelector

A two-step sklearn `Pipeline` that first removes low-variance features, then filters out highly correlated ones. Because it subclasses `Pipeline`, it exposes the full sklearn transformer API: `fit()`, `transform()`, `fit_transform()`, `get_params()`, and `set_params()`.

```python
from pybasin.feature_selector import DefaultFeatureSelector

# Default thresholds
selector = DefaultFeatureSelector()

# Custom thresholds
selector = DefaultFeatureSelector(
    variance_threshold=0.05,
    correlation_threshold=0.90,
    min_features=3,
)
```

The pipeline steps run in order:

1. `VarianceThreshold(threshold=variance_threshold)` -- drops constant or near-constant columns
2. `CorrelationSelector(threshold=correlation_threshold, min_features=min_features)` -- among the surviving features, drops those with high pairwise |correlation|

### Constructor Parameters

| Parameter               | Type    | Default | Description                                                                 |
| ----------------------- | ------- | ------- | --------------------------------------------------------------------------- |
| `variance_threshold`    | `float` | `0.01`  | Minimum variance required to keep a feature.                                |
| `correlation_threshold` | `float` | `0.95`  | Maximum absolute pairwise correlation allowed between features.             |
| `min_features`          | `int`   | `2`     | Floor on the number of features retained by the correlation filtering step. |

### Retrieving the Selection Mask

Call `get_support()` after fitting to see which original features survived both stages:

```python
import numpy as np

selector = DefaultFeatureSelector()
X_filtered = selector.fit_transform(X)

mask = selector.get_support()  # boolean array, length = original feature count
indices = selector.get_support(indices=True)  # integer indices of kept features
```

The method composes the variance mask and the correlation mask internally, returning a single combined result against the original feature indices.

---

## CorrelationSelector

An sklearn transformer that removes features whose pairwise absolute correlation exceeds a threshold. Rather than dropping columns in index order, it applies the ranking algorithm from Kuhn & Johnson (2013), _Applied Predictive Modeling_, Chapter 3 -- the same logic behind R's `caret::findCorrelation()`.

At each step, the selector finds the most correlated pair among the remaining features and computes each member's mean absolute correlation against all other remaining features. The feature with the higher mean -- the more globally redundant one -- is dropped. Because the correlation statistics are recomputed after every removal, later decisions reflect the reduced feature set.

```python
from pybasin.feature_selector import CorrelationSelector

selector = CorrelationSelector(threshold=0.9, min_features=3)
X_filtered = selector.fit_transform(X)
```

### Constructor Parameters

| Parameter      | Type    | Default | Description                                                                          |
| -------------- | ------- | ------- | ------------------------------------------------------------------------------------ |
| `threshold`    | `float` | `0.9`   | Correlation ceiling. Pairs with \|corr\| above this trigger removal.                 |
| `min_features` | `int`   | `3`     | Minimum features to retain. Prevents aggressive filtering from removing all columns. |

!!! note "Column-order independence"
Because removal is driven by mean absolute correlation rather than column position, reordering the input features does not change which features survive. Two datasets containing the same features in different column order produce the same retained set.

### Retrieving the Selection Mask

```python
mask = selector.get_support()  # boolean mask
indices = selector.get_support(indices=True)  # integer indices
```

---

## Using Custom sklearn Transformers

Any sklearn-compatible transformer that implements `fit_transform()` can serve as a feature selector. If the transformer also provides `get_support()`, filtered feature names will appear in `Solution.feature_names`.

```python
from sklearn.feature_selection import VarianceThreshold

bse = BasinStabilityEstimator(
    ode_system=...,
    sampler=...,
    feature_selector=VarianceThreshold(threshold=0.1),
)
```

The transformer must be unsupervised -- its `fit()` receives only the feature matrix `X`, with no labels. Supervised selectors like `SelectKBest(f_classif)` will fail because the pipeline never passes `y`.

!!! warning "Feature name tracking"
Only selectors with a `get_support()` method enable automatic feature name filtering. Without it, `Solution.feature_names` falls back to the full unfiltered list.

---

## Standalone Usage

Feature selectors work independently outside `BasinStabilityEstimator`. This is useful for inspecting which features survive filtering before running a full pipeline.

```python
import numpy as np
from pybasin.feature_selector import DefaultFeatureSelector

# Simulated feature matrix: 100 samples, 50 features
rng = np.random.default_rng(42)
X = rng.standard_normal((100, 50))

selector = DefaultFeatureSelector(variance_threshold=0.01, correlation_threshold=0.90)
X_filtered = selector.fit_transform(X)

print(f"Before: {X.shape[1]} features")
print(f"After:  {X_filtered.shape[1]} features")
print(f"Kept indices: {selector.get_support(indices=True)}")
```

---

For full class signatures and attribute documentation, see the [Feature Selection API reference](../api/feature-selector.md).
