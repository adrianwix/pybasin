# Basin Stability Estimator

`BasinStabilityEstimator` is the core class for computing basin stability values. It orchestrates the full pipeline: sampling initial conditions, integrating the ODE, extracting features from trajectories, classifying them into attractor basins, and computing the stability fractions. Only two arguments are required -- an ODE system and a sampler -- while every other component has sensible defaults.

## Minimal Example

At its simplest, basin stability estimation requires an ODE definition and a sampler. Everything else is auto-configured:

```python
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.sampler import UniformRandomSampler
from case_studies.duffing_oscillator.duffing_jax_ode import DuffingJaxODE

ode = DuffingJaxODE(params={"delta": 0.08, "k3": 1.0, "A": 0.2, "omega": 1.0})
sampler = UniformRandomSampler(min_limits=[-3.0, -3.0], max_limits=[3.0, 3.0])

bse = BasinStabilityEstimator(ode_system=ode, sampler=sampler)
result = bse.estimate_bs()

print(result["basin_stability"])  # e.g. {'0': 0.42, '1': 0.58}
```

With these defaults, the estimator generates 10,000 initial conditions, integrates them using `JaxSolver` (since the ODE inherits from `JaxODESystem`), extracts 10 statistical features per state variable, filters redundant features, clusters the results with HDBSCAN, and returns basin stability fractions.

## Pipeline Steps

The `estimate_bs()` method runs seven sequential steps. Each step maps to a configurable component:

| Step | What Happens                             | Component             | Default                                       |
| ---- | ---------------------------------------- | --------------------- | --------------------------------------------- |
| 1    | Sample N initial conditions              | `sampler`             | Required                                      |
| 2    | Integrate the ODE for each IC            | `solver`              | `JaxSolver` or `TorchDiffEqSolver` (auto)     |
| 3    | Create [Solution](solution.md) object    | --                    | --                                            |
| 3b   | Detect unbounded trajectories (optional) | `detect_unbounded`    | `True` (active only with `JaxSolver` + event) |
| 4    | Extract features from steady-state       | `feature_extractor`   | `TorchFeatureExtractor` (minimal, 10/state)   |
| 5    | Filter redundant features                | `feature_selector`    | `DefaultFeatureSelector`                      |
| 5b   | Fit classifier on templates (supervised) | `template_integrator` | `None` (unsupervised by default)              |
| 6    | Cluster or classify features             | `predictor`           | `HDBSCANClusterer(auto_tune=True)`            |
| 7    | Compute basin stability fractions        | --                    | --                                            |

```
Sample ICs --> Integrate ODEs --> Detect Unbounded --> Extract Features
--> Filter Features --> Cluster/Classify --> Compute BS Values
```

## Constructor Parameters

| Parameter             | Type                           | Default                  | Description                                                         |
| --------------------- | ------------------------------ | ------------------------ | ------------------------------------------------------------------- |
| `ode_system`          | `ODESystemProtocol`            | Required                 | The dynamical system to analyze. Use `JaxODESystem` or `ODESystem`. |
| `sampler`             | `Sampler`                      | Required                 | Generates initial conditions from the region of interest.           |
| `n`                   | `int`                          | `10_000`                 | Number of initial conditions to sample.                             |
| `solver`              | `SolverProtocol` or `None`     | Auto-detect              | ODE integrator. Auto-selects based on ODE type if `None`.           |
| `feature_extractor`   | `FeatureExtractor` or `None`   | `TorchFeatureExtractor`  | Computes feature vectors from trajectories.                         |
| `predictor`           | `BaseEstimator` or `None`      | `HDBSCANClusterer`       | Sklearn-compatible classifier or clusterer.                         |
| `template_integrator` | `TemplateIntegrator` or `None` | `None`                   | Required for supervised classifiers. Holds template ICs and labels. |
| `feature_selector`    | `BaseEstimator` or `None`      | `DefaultFeatureSelector` | Filters redundant features. `None` disables filtering.              |
| `detect_unbounded`    | `bool`                         | `True`                   | Separate diverging trajectories before feature extraction.          |
| `output_dir`          | `str`, `Path`, or `None`       | `None`                   | Directory path for saving results to JSON or Excel.                 |

For full method signatures and docstrings, see the [API reference](../api/basin-stability-estimator.md).

## Automatic Solver Selection

When no `solver` is passed, the estimator picks one based on the ODE class:

- `JaxODESystem` --> `JaxSolver(time_span=(0, 1000), n_steps=1000)`
- `ODESystem` --> `TorchDiffEqSolver(time_span=(0, 1000), n_steps=1000)`

Both auto-selected solvers inherit the `device` from the sampler. For most workloads, `JaxSolver` delivers the best GPU performance. Override the solver when you need custom time spans, tolerances, or caching:

```python
from pybasin.solvers import JaxSolver

solver = JaxSolver(
    time_span=(0, 500),
    n_steps=5000,
    device="cuda",
    rtol=1e-8,
    atol=1e-6,
    cache_dir=".pybasin_cache/duffing",
)

bse = BasinStabilityEstimator(
    ode_system=ode, sampler=sampler, solver=solver
)
```

See the [Solvers guide](solvers.md) for a detailed comparison of available solvers.

---

## Customizing Feature Extraction

The default feature extractor computes 10 statistical features per state variable (mean, variance, min, max, etc.) from the steady-state portion of each trajectory. Override it for richer or more targeted feature sets:

```python
from pybasin.feature_extractors import TorchFeatureExtractor

extractor = TorchFeatureExtractor(
    features="comprehensive",  # ~800 features per state
    time_steady=800.0,         # discard transient before t=800
    device="cuda",
)

bse = BasinStabilityEstimator(
    ode_system=ode,
    sampler=sampler,
    feature_extractor=extractor,
)
```

Per-state configuration is also supported -- useful when different state variables carry different physical meaning:

```python
extractor = TorchFeatureExtractor(
    features="minimal",
    features_per_state={
        0: {"maximum": None, "minimum": None},  # position: just extrema
        1: None,                                  # velocity: skip entirely
    },
)
```

See the [Feature Extractors guide](feature-extractors.md) for the full feature catalog and configuration options.

---

## Feature Selection

After extraction, a feature selector removes uninformative columns (near-zero variance) and redundant ones (high pairwise correlation). The default `DefaultFeatureSelector` handles this automatically:

```python
# Default: variance + correlation filtering
bse = BasinStabilityEstimator(ode_system=ode, sampler=sampler)

# Disable filtering entirely
bse = BasinStabilityEstimator(
    ode_system=ode, sampler=sampler, feature_selector=None
)

# Custom sklearn selector
from sklearn.feature_selection import VarianceThreshold

bse = BasinStabilityEstimator(
    ode_system=ode,
    sampler=sampler,
    feature_selector=VarianceThreshold(threshold=0.1),
)
```

!!! tip "Minimal features + filtering"
When using `features="minimal"` (10 features per state), the default selector may drop useful columns. Consider disabling it or lowering thresholds for small feature sets.

See the [Feature Selectors guide](feature-selectors.md) for threshold configuration and custom selectors.

---

## Unboundedness Detection

Some dynamical systems produce trajectories that diverge to infinity. When `detect_unbounded=True` (the default), the estimator separates these trajectories before feature extraction to prevent extreme values from contaminating the clustering. Unbounded ICs receive the label `"unbounded"` in the final results.

This detection only activates when the solver is a `JaxSolver` configured with an `event_fn` for early termination. Without an event function, trajectories are not stopped early and no Inf values appear.

```python
from pybasin.solvers import JaxSolver

solver = JaxSolver(
    time_span=(0, 1000),
    n_steps=5000,
    event_fn=lambda t, y: jnp.max(jnp.abs(y)) - 1e6,  # stop at |y| > 1e6
)

bse = BasinStabilityEstimator(
    ode_system=ode,
    sampler=sampler,
    solver=solver,
    detect_unbounded=True,  # default
)
result = bse.estimate_bs()
# e.g. {'0': 0.45, '1': 0.40, 'unbounded': 0.15}
```

See the [Handling Unbounded Trajectories](../guides/unbounded-trajectories.md) guide for a deeper look at event functions and unboundedness strategies.

---

## Unsupervised Clustering (Default)

By default, the estimator uses `HDBSCANClusterer` to discover attractor basins without any prior knowledge. This is the simplest workflow -- no templates, no labels:

```python
bse = BasinStabilityEstimator(
    ode_system=ode,
    sampler=sampler,
    n=10_000,
)
result = bse.estimate_bs()```

HDBSCAN auto-tunes its `min_cluster_size` parameter and reassigns noise points so that every trajectory receives a basin label. To swap in a different clusterer, pass any sklearn-compatible estimator:

```python
from sklearn.cluster import KMeans

bse = BasinStabilityEstimator(
    ode_system=ode,
    sampler=sampler,
    predictor=KMeans(n_clusters=3),
)
```

See the [Predictors guide](predictors.md) for all available clusterers and their tuning options.

---

## Supervised Classification

When the attractor structure of the system is known, supervised classification produces more reliable basin labels. This requires a `TemplateIntegrator` that provides labeled initial conditions -- one per known attractor -- along with a sklearn classifier.

```python
import torch
from sklearn.neighbors import KNeighborsClassifier
from pybasin.template_integrator import TemplateIntegrator

# Template ICs: one per known attractor
template_y0 = torch.tensor([
    [1.2, 0.0],   # IC converging to attractor "fp"
    [2.5, 0.0],   # IC converging to attractor "lc"
])
template_labels = ["fp", "lc"]

template_integrator = TemplateIntegrator(
    template_y0=template_y0,
    labels=template_labels,
)

bse = BasinStabilityEstimator(
    ode_system=ode,
    sampler=sampler,
    predictor=KNeighborsClassifier(n_neighbors=1),
    template_integrator=template_integrator,
)
result = bse.estimate_bs()
# e.g. {'fp': 0.35, 'lc': 0.65}
```

The estimator integrates template trajectories alongside the main batch, extracts features from both, fits the classifier on the template features, and then predicts basin labels for all N sampled ICs. By default, template and main integrations run in parallel.

!!! warning "Classifier requires templates"
Passing a classifier without a `template_integrator` raises `ValueError`. Regressors are rejected outright with `TypeError`.

See the [Predictors guide](predictors.md) for more on supervised vs. unsupervised workflows.

---

## Output Attributes

After `estimate_bs()` completes, three attributes hold the results:

| Attribute      | Type                      | Description                                                        |
| -------------- | ------------------------- | ------------------------------------------------------------------ |
| `bse.result`   | `StudyResult \| None`     | Full estimation result with basin stability, errors, labels, and orbit data |
| `bse.y0`       | `torch.Tensor`            | Sampled initial conditions, shape `(N, n_states)`                  |
| `bse.solution` | [`Solution`](solution.md) | Full results: trajectories, features, labels, metadata             |

The `Solution` object carries everything downstream components need -- trajectories for plotting, features for analysis, labels for visualization. See the [Solution guide](solution.md) for details on its properties.

## Error Estimation

Basin stability values are Monte Carlo estimates, so they carry statistical uncertainty. Call `get_errors()` to compute the absolute and relative standard errors based on Bernoulli experiment statistics:

```python
result = bse.estimate_bs()

for label, err in result["errors"].items():
    print(f"Basin {label}: S_B = {result['basin_stability'][label]:.3f} +/- {err['e_abs']:.4f}")
```

The absolute error for each basin is:

$$e_{\text{abs}} = \sqrt{\frac{S_B(A) \cdot (1 - S_B(A))}{N}}$$

Increasing `n` reduces the error proportionally to $1/\sqrt{N}$.

---

## Saving Results

Two export methods are available after running `estimate_bs()`. Both require `output_dir` to be set in the constructor.

### JSON Export

`save()` writes basin stability values, ODE system definition, sampler settings, and feature selection metadata:

```python
bse = BasinStabilityEstimator(
    ode_system=ode,
    sampler=sampler,
    output_dir="results/pendulum",
)
bse.estimate_bs()
bse.save()  # writes results/pendulum/basin_stability_results_<timestamp>.json
```

### Excel Export

`save_to_excel()` writes initial conditions, labels, and bifurcation amplitudes in tabular form:

```python
bse.save_to_excel()  # writes results/pendulum/basin_stability_results_<timestamp>.xlsx
```

---

## Visualization

Pass the estimator to a plotter for visual inspection of the results. `MatplotlibPlotter` produces static figures suitable for publications, while `InteractivePlotter` launches a Dash web app for exploration.

```python
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter

plotter = MatplotlibPlotter(bse)
plotter.plot_bse_results()              # 4-panel diagnostic
plotter.plot_basin_stability_bars()     # bar chart of BS values
plotter.plot_state_space()              # labeled phase portrait
plotter.plot_feature_space()            # feature space clusters
plotter.save()  # save all to bse.output_dir
plotter.show()  # or use plt.show() directly
```

```python
from pybasin.plotters.interactive_plotter import InteractivePlotter

plotter = InteractivePlotter(bse, state_labels={0: "x", 1: "v"})
plotter.run(port=8050)
```

See the [Plotters guide](plotters.md) for the full set of visualization methods.

---

## Full Configured Example

Below is a complete example showing every configurable component:

```python
import torch
from sklearn.neighbors import KNeighborsClassifier
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.sampler import UniformRandomSampler
from pybasin.solvers import JaxSolver
from pybasin.feature_extractors import TorchFeatureExtractor
from pybasin.feature_selector import DefaultFeatureSelector
from pybasin.template_integrator import TemplateIntegrator
from pybasin.plotters.matplotlib_plotter import MatplotlibPlotter

# 1. ODE system (defined elsewhere)
from case_studies.pendulum.pendulum_jax_ode import PendulumJaxODE

ode = PendulumJaxODE(params={"alpha": 0.1, "T": 0.5, "K": 1.0})

# 2. Sampler
sampler = UniformRandomSampler(
    min_limits=[-3.14, -2.0],
    max_limits=[3.14, 2.0],
    device="cuda",
)

# 3. Solver
solver = JaxSolver(
    time_span=(0, 1000),
    n_steps=5000,
    device="cuda",
    cache_dir=".pybasin_cache/pendulum",
)

# 4. Feature extractor
extractor = TorchFeatureExtractor(
    features="minimal",
    time_steady=850.0,
    device="cuda",
)

# 5. Feature selector
selector = DefaultFeatureSelector(
    variance_threshold=0.01,
    correlation_threshold=0.95,
)

# 6. Templates for supervised classification
template_y0 = torch.tensor([[0.5, 0.0], [2.5, 0.0]])
template_integrator = TemplateIntegrator(
    template_y0=template_y0,
    labels=["fixed_point", "limit_cycle"],
)

# 7. Classifier
predictor = KNeighborsClassifier(n_neighbors=1)

# 8. Assemble and run
bse = BasinStabilityEstimator(
    ode_system=ode,
    sampler=sampler,
    n=20_000,
    solver=solver,
    feature_extractor=extractor,
    predictor=predictor,
    template_integrator=template_integrator,
    feature_selector=selector,
    output_dir="results/pendulum",
)

result = bse.estimate_bs()

# 9. Inspect results
print(result["basin_stability"])
print(result["errors"])

# 10. Visualize
plotter = MatplotlibPlotter(bse)
plotter.plot_bse_results()
plotter.save()  # or plotter.show() for interactive display

# 11. Save
bse.save()
bse.save_to_excel()
```

## Related Documentation

- [Solution](solution.md) -- what `bse.solution` contains and how to inspect it
- [Samplers](samplers.md) -- uniform, grid, Gaussian, and CSV sampling strategies
- [Solvers](solvers.md) -- solver comparison, caching, and GPU acceleration
- [Feature Extractors](feature-extractors.md) -- feature catalogs and per-state configuration
- [Feature Selectors](feature-selectors.md) -- variance and correlation filtering
- [Predictors](predictors.md) -- HDBSCAN, DBSCAN, supervised classifiers, and custom predictors
- [Plotters](plotters.md) -- static and interactive visualization options
- [Parameter Studies](basin-stability-study.md) -- sweeping ODE parameters with `BasinStabilityStudy`
- [Handling Unbounded Trajectories](../guides/unbounded-trajectories.md) -- event functions and unboundedness detection
- [Case Studies](../case-studies/overview.md) -- worked examples with Pendulum, Duffing, Lorenz, and more
