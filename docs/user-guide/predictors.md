# Predictors

Predictors classify trajectories into attractor classes based on extracted features. The [`BasinStabilityEstimator`](../api/basin-stability-estimator.md) supports any sklearn-compatible estimator, falling into two main categories: unsupervised **clusterers** that discover attractor classes automatically, and supervised **classifiers** that learn from labeled template trajectories.

## Predictor Types

When you pass a predictor to `BasinStabilityEstimator`, the library detects its type using sklearn's `is_classifier()` and `is_clusterer()` functions:

- **Clusterers** (`is_clusterer(predictor) == True`): Implement `fit_predict(X)` and discover classes from the data without prior labels. Any sklearn clusterer works.
- **Classifiers** (`is_classifier(predictor) == True`): Implement `fit(X, y)` + `predict(X)` and require labeled training data via [`TemplateIntegrator`](#supervised-classification-with-templateintegrator). Any sklearn classifier works.

If no predictor is specified, the default is `HDBSCANClusterer(auto_tune=True, assign_noise=True)`.

## Available Predictors

| Class                        | Type         | Description                                                                                |
| ---------------------------- | ------------ | ------------------------------------------------------------------------------------------ |
| `HDBSCANClusterer`           | Unsupervised | **Default**. Density-based, auto-tunes parameters                                          |
| `DBSCANClusterer`            | Unsupervised | Classic DBSCAN with epsilon auto-tuning                                                    |
| `DynamicalSystemClusterer`   | Unsupervised | Physics-based two-stage hierarchical. See [guide](../guides/dynamics-based-clustering.md). |
| `UnboundednessMetaEstimator` | Meta         | Wraps any estimator to handle unbounded cases                                              |
| Any sklearn clusterer        | Unsupervised | `KMeans`, `GaussianMixture`, `AgglomerativeClustering`, etc.                               |
| Any sklearn classifier       | Supervised   | `KNeighborsClassifier`, `SVC`, `RandomForestClassifier`, etc.                              |

The built-in predictors (`HDBSCANClusterer`, `DBSCANClusterer`) include **auto-tuning** to improve the probability of finding meaningful attractor clusters when analyzing unknown systems. Rather than requiring manual parameter selection, they search for optimal clustering parameters using silhouette analysis.

The **noise reassignment** option (`assign_noise=True`) addresses a key property of basin stability: every trajectory from an initial condition either converges to an attractor or diverges to infinity. There are no "noise" points in the traditional clustering sense. When HDBSCAN or DBSCAN labels low-density samples as noise, these samples still belong to some basin -- reassigning them to the nearest cluster ensures every initial condition receives an attractor label.

For complete API documentation, see the [Predictors API Reference](../api/predictors.md).

---

## HDBSCANClusterer (Default)

HDBSCAN excels at finding clusters of varying densities without requiring the number of clusters upfront. Our wrapper adds optional auto-tuning and noise reassignment.

```python
from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer

predictor = HDBSCANClusterer(
    auto_tune=True,      # Auto-select min_cluster_size
    assign_noise=True,   # Reassign noise points to nearest cluster
)
```

### Constructor Parameters

| Parameter           | Type                | Default                           | Description                                   |
| ------------------- | ------------------- | --------------------------------- | --------------------------------------------- |
| `hdbscan`           | `HDBSCAN` or `None` | `HDBSCAN(min_cluster_size=50)`    | Configured sklearn HDBSCAN instance           |
| `auto_tune`         | `bool`              | `False`                           | Auto-select `min_cluster_size` via silhouette |
| `assign_noise`      | `bool`              | `False`                           | Reassign noise points (-1) to nearest cluster |
| `nearest_neighbors` | `NearestNeighbors`  | `NearestNeighbors(n_neighbors=5)` | KNN for noise reassignment                    |

### Auto-Tuning Algorithm

When `auto_tune=True`, the clusterer searches for the optimal `min_cluster_size` using silhouette score analysis. Given $n$ samples, it evaluates several candidate sizes:

$$
S_{\text{candidates}} = \left\{ \max(10, 0.005n), \max(25, 0.01n), \max(50, 0.02n), \max(100, 0.03n), \max(150, 0.05n) \right\}
$$

For each candidate $s \in S_{\text{candidates}}$:

1. Set `min_cluster_size = s` and `min_samples = min(10, s // 5)`
2. Run HDBSCAN clustering to obtain labels $L$
3. Compute the silhouette score $\sigma(s)$ over non-noise samples:

$$
\sigma(s) = \frac{1}{|L_{valid}|} \sum_{i: L_i \neq -1} \frac{b_i - a_i}{\max(a_i, b_i)}
$$

where $a_i$ is the mean intra-cluster distance and $b_i$ is the mean nearest-cluster distance for sample $i$.

The algorithm selects the `min_cluster_size` with the highest $\sigma$:

$$
s^* = \arg\max_{s \in S_{\text{candidates}}} \sigma(s)
$$

!!! note "When to disable auto-tuning"
Auto-tuning adds computational overhead. If you know the approximate cluster sizes for your system, passing a pre-configured `HDBSCAN` instance is faster.

### Noise Reassignment

HDBSCAN labels low-density points as noise (label -1). When `assign_noise=True`, these points are reassigned to their nearest cluster using k-nearest neighbors voting. This ensures every trajectory receives a basin label:

```python
# All noise points assigned to clusters
predictor = HDBSCANClusterer(auto_tune=True, assign_noise=True)
```

---

## DBSCANClusterer

Standard DBSCAN with optional epsilon auto-tuning based on the MATLAB bSTAB algorithm.

```python
from pybasin.predictors.dbscan_clusterer import DBSCANClusterer

predictor = DBSCANClusterer(
    auto_tune=True,       # Automatic epsilon search
    assign_noise=True,    # Reassign noise to nearest cluster
)
```

### Constructor Parameters

| Parameter          | Type               | Default                           | Description                                   |
| ------------------ | ------------------ | --------------------------------- | --------------------------------------------- |
| `dbscan`           | `DBSCAN` or `None` | `DBSCAN(eps=0.5, min_samples=10)` | Pre-configured DBSCAN instance                |
| `auto_tune`        | `bool`             | `False`                           | Auto-find epsilon via silhouette peaks        |
| `n_eps_grid`       | `int`              | `200`                             | Number of epsilon candidates                  |
| `tune_sample_size` | `int`              | `2000`                            | Max samples for tuning (subsampled if larger) |
| `min_peak_height`  | `float`            | `0.9`                             | Minimum silhouette peak height                |
| `assign_noise`     | `bool`             | `False`                           | Reassign noise points to nearest cluster      |

### Epsilon Auto-Tuning Algorithm

DBSCAN requires an `eps` (epsilon) parameter -- the maximum distance between two samples for them to be considered neighbors. Choosing the right value is critical: too small and most points become noise with fragmented clusters; too large and distinct clusters merge together. The auto-tuning algorithm automates this selection so users do not need to manually tune epsilon for each feature space.

The epsilon search replicates the MATLAB bSTAB `classify_solution.m` unsupervised branch. Given a feature matrix $X \in \mathbb{R}^{n \times d}$ where $n$ is the number of samples and $d$ is the number of features:

**Step 1: Build epsilon grid**

Let $X_j$ denote the $j$-th column (feature) of $X$. Compute feature ranges $R_j = \max(X_j) - \min(X_j)$ for each dimension $j \in \{1, \ldots, d\}$. The minimum range $R_{\min} = \min_j R_j$ determines the search grid:

$$
\varepsilon_{\text{grid}} = \left\{ \frac{R_{\min}}{N}, \frac{2 R_{\min}}{N}, \ldots, R_{\min} \right\}
$$

where $N$ is the number of grid points (`n_eps_grid`, default 200).

Using $R_{\min}$ as the upper bound is a heuristic from the original bSTAB implementation. The reasoning: if epsilon exceeds the smallest feature dimension's range, all points become neighbors in that dimension, losing discriminative power. This approach also provides scale invariance -- the search adapts to the actual data range rather than using arbitrary fixed values.

**Step 2: Evaluate silhouette scores**

For each candidate $\varepsilon$ in the grid, run DBSCAN to obtain cluster labels $L_i$ for each sample $i$. DBSCAN assigns $L_i = -1$ to noise points (samples not belonging to any cluster). Compute the **minimum per-sample silhouette score** over non-noise samples:

$$
\sigma_{\min}(\varepsilon) = \min_{i: L_i \neq -1} s_i
$$

where $s_i$ is the silhouette coefficient of sample $i$. The silhouette coefficient measures how similar a sample is to its own cluster compared to other clusters, ranging from -1 (wrong cluster) to +1 (well-matched). Using the minimum rather than the mean provides a worst-case quality measure.

**Step 3: Peak detection**

Rather than simply taking the maximum silhouette score, the algorithm uses scipy's `find_peaks` to detect local maxima in $\sigma_{\min}$ and selects the most **prominent** peak above `min_peak_height`. Let $P$ denote the set of grid indices where local maxima with $\sigma_{\min} \geq$ `min_peak_height` were detected. The optimal epsilon $\varepsilon^*$ is:

$$
\varepsilon^* = \varepsilon_{\text{grid}}[k^*] \quad \text{where} \quad k^* = \arg\max_{k \in P} \text{prominence}(k)
$$

**Why prominence instead of maximum?** The silhouette curve often has multiple local maxima. A peak's prominence measures how much you would need to descend before climbing to a higher peak -- it captures how "significant" a peak is, not just how tall. Using `argmax` alone might select a sharp spike caused by noise or edge effects. Prominence identifies stable, well-separated clustering regimes.

If no peak exceeds the threshold (i.e., $P = \emptyset$), the algorithm falls back to the global maximum: $\varepsilon^* = \varepsilon_{\text{grid}}[\arg\max_k \sigma_{\min}(\varepsilon_{\text{grid}}[k])]$.

!!! tip "Performance consideration"
When the dataset exceeds `tune_sample_size`, a random subsample is used for the search. This keeps tuning fast while still finding a good epsilon.

---

## Supervised Classification with TemplateIntegrator

When the attractor types are known beforehand (e.g., from bifurcation analysis), supervised classification produces more reliable results than unsupervised clustering. The workflow requires two components:

1. A **classifier** -- any sklearn estimator where `is_classifier(predictor)` returns `True` (e.g., `KNeighborsClassifier`, `SVC`, `RandomForestClassifier`)
2. A **TemplateIntegrator** -- holds template initial conditions and their labels

### How It Works

The `TemplateIntegrator` integrates template initial conditions (one per known attractor) and extracts features from the resulting trajectories. These features, paired with their labels, form the training set for the classifier.

### Example: Duffing Oscillator with KNN Classifier

```python
from sklearn.neighbors import KNeighborsClassifier

from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.template_integrator import TemplateIntegrator
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver

# Template initial conditions -- one per known attractor
template_y0 = [
    [-0.21, 0.02],   # Attractor 1
    [1.05, 0.77],    # Attractor 2
    [-0.67, 0.02],   # Attractor 3
]

# Labels corresponding to each template
labels = ["y1", "y2", "y3"]

# ODE parameters for template integration (may differ from main study)
ode_params = {"delta": 0.08, "k3": 1, "A": 0.2}

template_integrator = TemplateIntegrator(
    template_y0=template_y0,
    labels=labels,
    ode_params=ode_params,
)

# Any sklearn classifier works
classifier = KNeighborsClassifier(n_neighbors=1)

bse = BasinStabilityEstimator(
    n=10_000,
    ode_system=ode_system,
    sampler=sampler,
    predictor=classifier,
    template_integrator=template_integrator,  # Required for classifiers
)

bs_vals = bse.estimate_bs()
```

### TemplateIntegrator Parameters

| Parameter     | Type                | Description                                         |
| ------------- | ------------------- | --------------------------------------------------- |
| `template_y0` | `list[list[float]]` | Initial conditions for template trajectories        |
| `labels`      | `list[str]`         | Ground truth labels (one per template IC)           |
| `ode_params`  | `Mapping[str, Any]` | ODE parameters for template integration             |
| `solver`      | `SolverProtocol`    | Optional dedicated solver (defaults to CPU variant) |

For complete API documentation, see the [TemplateIntegrator API](../api/template-integrator.md).

### When to Use Supervised Classification

Use supervised classification when:

- You know the attractor types from bifurcation analysis or domain knowledge
- The system has been studied before and attractors are well-characterized
- Unsupervised clustering fails to separate known attractor types
- You need consistent labeling across parameter studies

Unsupervised clustering is preferable when:

- Attractor types are unknown or exploratory analysis is needed
- The number of attractors varies with parameters
- Manual labeling of templates is impractical

---

## Creating Custom Estimators

Any sklearn-compatible estimator works with `BasinStabilityEstimator`. Write a custom clusterer by subclassing `BaseEstimator` and `ClusterMixin`, then implementing `fit_predict()`. The key requirement: return an array of labels (strings or integers) with one label per sample.

For the complete sklearn developer guide, see [Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html).

### Quick Reference

**Clusterer (unsupervised):**

```python
from sklearn.base import BaseEstimator, ClusterMixin

class MyClusterer(BaseEstimator, ClusterMixin):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit_predict(self, X, y=None):
        # X is a numpy array of shape (n_samples, n_features)
        # Return labels array of shape (n_samples,)
        labels = ...
        return labels
```

**Classifier (supervised):**

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class MyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        # Store training data
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        # Return predicted labels
        labels = ...
        return labels
```

### Example: SynchronizationClassifier

The Rossler network case study uses a threshold-based classifier that labels trajectories as "synchronized" or "desynchronized" based on the maximum deviation between network nodes:

```python
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class SynchronizationClassifier(BaseEstimator, ClusterMixin):
    """
    Classifier that labels trajectories as 'synchronized' or 'desynchronized'.

    Works with features from SynchronizationFeatureExtractor, which computes
    the max deviation across all node pairs. Synchronization is achieved when:
        max_deviation_all < epsilon

    :param epsilon: Synchronization threshold.
    :param feature_index: Index of the feature to use for thresholding.
        Options: 0=max_deviation_x, 1=max_deviation_y, 2=max_deviation_z, 3=max_deviation_all
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        feature_index: int = 3,
    ):
        self.epsilon = epsilon
        self.feature_index = feature_index

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        """
        Classify each trajectory as synchronized or desynchronized.

        :param X: Feature matrix of shape (n_samples, 4).
        :param y: Ignored. Present for API compatibility.
        :return: Labels array with 'synchronized' or 'desynchronized'.
        """
        X_arr = np.asarray(X)
        max_deviation = X_arr[:, self.feature_index]

        labels = np.where(
            max_deviation < self.epsilon,
            "synchronized",
            "desynchronized",
        )

        return labels
```

This classifier inherits from `ClusterMixin` because it implements `fit_predict()` directly -- it does not need separate training. Despite returning semantic labels, sklearn treats it as a clusterer.

### String Labels vs Integer Labels

pybasin handles both string and integer labels. HDBSCAN returns integers (0, 1, 2, ..., -1 for noise), while the `SynchronizationClassifier` returns strings. The `BasinStabilityEstimator` converts all labels to strings internally when computing basin stability fractions.

---

## Using Sklearn Estimators Directly

Beyond the built-in predictors, any sklearn estimator works. A few examples:

```python
# Gaussian Mixture Model clustering
from sklearn.mixture import GaussianMixture
predictor = GaussianMixture(n_components=3)

# K-Means clustering
from sklearn.cluster import KMeans
predictor = KMeans(n_clusters=2)

# Support Vector Classification (supervised)
from sklearn.svm import SVC
predictor = SVC(kernel='rbf')

# Decision Tree (supervised)
from sklearn.tree import DecisionTreeClassifier
predictor = DecisionTreeClassifier(max_depth=5)
```

!!! warning "Supervised classifiers require TemplateIntegrator"
When using a classifier (`is_classifier(predictor)` returns `True`), you must provide a `template_integrator` to the `BasinStabilityEstimator`. Without it, the estimator raises a `ValueError`.

---

## Feature Name Awareness

Some predictors need feature names to select specific columns (for example, `DynamicalSystemClusterer` -- see the [Dynamics-Based Clustering](../guides/dynamics-based-clustering.md) guide). If your custom predictor needs this capability, implement the `FeatureNameAware` protocol:

```python
from pybasin.protocols import FeatureNameAware

class MyFeatureAwareClusterer(BaseEstimator, ClusterMixin, FeatureNameAware):
    def set_feature_names(self, names: list[str]) -> None:
        self.feature_names_ = names

    def fit_predict(self, X, y=None):
        # Access self.feature_names_ to find specific features by name
        ...
```

The `BasinStabilityEstimator` automatically calls `set_feature_names()` on predictors that implement it.

---

## Summary

| Use Case                        | Recommended Predictor                                                        |
| ------------------------------- | ---------------------------------------------------------------------------- |
| Unknown attractors, exploratory | `HDBSCANClusterer(auto_tune=True)`                                           |
| Known attractors, labeled data  | `KNeighborsClassifier` + `TemplateIntegrator`                                |
| Physics-aware classification    | `DynamicalSystemClusterer` ([guide](../guides/dynamics-based-clustering.md)) |
| Simple threshold-based logic    | Custom `ClusterMixin` class                                                  |
| Need MATLAB bSTAB compatibility | `DBSCANClusterer(auto_tune=True)`                                            |
