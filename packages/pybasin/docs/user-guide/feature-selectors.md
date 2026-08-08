# Feature Selectors

Feature selectors remove uninformative or redundant columns before the feature
matrix reaches the predictor.

## Available Selectors

| Class | Strategy | Typical use |
| --- | --- | --- |
| `DefaultFeatureSelector` | Variance threshold followed by correlation filtering | General-purpose default |
| `CorrelationSelector` | Mean absolute correlation ranking | Removing redundant columns |
| `VarianceThresholdSelector` | Population variance | Removing constant or nearly constant columns |
| `SklearnFeatureSelectorAdapter` | Wrapped scikit-learn selector | Reusing a scikit-learn selection algorithm |

## Common Interface

Each selector accepts a two-dimensional `torch.Tensor` with shape
`(n_samples, n_features)`. The filtered tensor has the same dtype and device as
the input passed to `transform()`.

After fitting, `get_support()` returns a NumPy boolean mask indexed against the
original feature columns. Pass `indices=True` to receive their integer indices.
Pybasin uses the mask to update `Solution.feature_names`.

```python
filtered = selector.fit_transform(features)
mask = selector.get_support()
indices = selector.get_support(indices=True)
```

Calling `transform()` or `get_support()` before fitting raises `RuntimeError`.
A feature-count mismatch between `fit()` and `transform()` raises `ValueError`.
Selectors also support `get_params()`, `set_params()`, and
`sklearn.base.clone()`.

## DefaultFeatureSelector

`DefaultFeatureSelector` removes low-variance columns first. It then applies
correlation filtering to the remaining feature matrix. The combined support
mask uses the column indices of the matrix passed to `fit()`.

```python
from pybasin.feature_selector import DefaultFeatureSelector

selector = DefaultFeatureSelector(
    variance_threshold=0.05,
    correlation_threshold=0.90,
    min_features=3,
)
filtered = selector.fit_transform(features)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `variance_threshold` | `float` | `0.01` | Minimum population variance required to keep a feature |
| `correlation_threshold` | `float` | `0.9` | Absolute pairwise correlation that triggers removal |
| `min_features` | `int` | `3` | Minimum columns retained by the correlation stage |

When `feature_selector` is omitted, `BasinStabilityEstimator` constructs this
selector with the defaults shown above. Passing `None` disables feature
selection.

```python
from pybasin import BasinStabilityEstimator

default_bse = BasinStabilityEstimator(
    ode_system=...,
    sampler=...,
    n=5_000,
)

unfiltered_bse = BasinStabilityEstimator(
    ode_system=...,
    sampler=...,
    n=5_000,
    feature_selector=None,
)
```

The fitted stages are available as `variance_selector_` and
`correlation_selector_`.

## CorrelationSelector

`CorrelationSelector` follows the ranking procedure from Kuhn and Johnson
(2013), *Applied Predictive Modeling*, Chapter 3. For each pair above the
threshold, it removes the feature with the higher mean absolute correlation to
the current feature set. Selection stops when all remaining pairs satisfy the
threshold or the matrix reaches `min_features` columns.

```python
from pybasin.feature_selector import CorrelationSelector

selector = CorrelationSelector(threshold=0.9, min_features=3)
filtered = selector.fit_transform(features)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `threshold` | `float` | `0.9` | Absolute correlation that triggers removal |
| `min_features` | `int` | `3` | Minimum number of retained columns |

## VarianceThresholdSelector

`VarianceThresholdSelector` calculates population variance for every column.
A feature is selected when its variance is greater than `threshold`.

```python
from pybasin.feature_selector import VarianceThresholdSelector

selector = VarianceThresholdSelector(threshold=0.01)
filtered = selector.fit_transform(features)
```

The fitted `variances_` tensor contains one value per input feature.

## Using a scikit-learn Selector

`SklearnFeatureSelectorAdapter` presents a selector from scikit-learn through
pybasin's feature-selector interface. The wrapped object must implement `fit()`
and `get_support()`.

```python
from sklearn.feature_selection import VarianceThreshold

from pybasin import BasinStabilityEstimator
from pybasin.feature_selector import SklearnFeatureSelectorAdapter

selector = SklearnFeatureSelectorAdapter(
    VarianceThreshold(threshold=0.1),
)

bse = BasinStabilityEstimator(
    ode_system=...,
    sampler=...,
    feature_selector=selector,
)
```

During `fit()`, the adapter passes a CPU NumPy array to the wrapped selector and
stores its support mask. `transform()` applies that mask to the supplied Torch
tensor.

`BasinStabilityEstimator` calls `fit(X, y=None)`. A wrapped selector must support
that call signature; supervised selectors that require target labels need a
separate labeled workflow.

For complete signatures, see the
[Feature Selection API reference](../api/feature-selector.md). The
[Feature Extractors guide](feature-extractors.md) lists the columns produced by
the built-in extractors.
