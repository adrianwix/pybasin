# Dynamics-Based Clustering

## Overview

Standard clustering predictors like [`HDBSCANClusterer`](../api/predictors.md) group trajectories purely by statistical similarity in feature space. This works well when attractors produce visibly different feature distributions, but it has no awareness of what the features _mean_ physically. A fixed point, a limit cycle, and a chaotic attractor might overlap in some statistical features yet differ fundamentally in their dynamics.

[`DynamicalSystemClusterer`](../api/predictors.md) takes a different approach: a two-stage hierarchical classification that first identifies the _type_ of attractor (fixed point, limit cycle, or chaos) and then sub-classifies within each type. Because Stage 1 uses physics-based heuristics -- variance thresholds, periodicity measures, drift detection -- it can separate attractors that generic clustering merges.

This guide covers the full setup: the required feature configuration, the clusterer itself, how the two stages work, and tuning advice.

## When to Use This

Reach for dynamics-based clustering when:

- The system has multiple attractor types (e.g., fixed points coexisting with limit cycles)
- Generic clusterers merge attractors that differ in dynamical character but overlap in simple statistics
- You want physically interpretable labels like `FP_0`, `LC_1`, `chaos_0` rather than opaque integer IDs

Standard [`HDBSCANClusterer`](../api/predictors.md) or [`DBSCANClusterer`](../api/predictors.md) remain better choices for exploratory analysis where attractor types are unknown, or for systems where all attractors are of the same type (e.g., multiple fixed points that only differ in location).

## Required Feature Configuration

`DynamicalSystemClusterer` reads specific features by name. It expects feature names following the `state_X__feature_name` convention, and it requires these seven base features to be present for at least one state variable:

| Required feature                               | What it provides to the clusterer                            |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `variance`                                     | Fixed point detection (near-zero variance = FP)              |
| `amplitude`                                    | Limit cycle sub-classification by oscillation size           |
| `mean`                                         | Spatial location for FP and chaos sub-classification         |
| `linear_trend__attr_slope`                     | Drift detection -- identifies rotating solutions             |
| `autocorrelation_periodicity__output_strength` | Periodicity strength [0, 1] for LC detection                 |
| `autocorrelation_periodicity__output_period`   | Dominant period for LC sub-grouping                          |
| `spectral_frequency_ratio`                     | Period-n detection (period-1 vs period-2 limit cycles, etc.) |

The `DYNAMICAL_SYSTEM_FC_PARAMETERS` preset in `pybasin.ts_torch.settings` produces exactly these features. It yields 7 feature columns per state variable (since `autocorrelation_periodicity` generates two outputs: strength and period). The preset is defined as:

```python
DYNAMICAL_SYSTEM_FC_PARAMETERS: FCParameters = {
    "variance": None,
    "amplitude": None,
    "mean": None,
    "linear_trend": [{"attr": "slope"}],
    "autocorrelation_periodicity": [
        {"output": "strength"},
        {"output": "period"},
    ],
    "spectral_frequency_ratio": None,
}
```

For a 2D system, this produces 14 feature columns -- `state_0__variance`, `state_0__amplitude`, ..., `state_1__spectral_frequency_ratio`. A 3D system like Lorenz would yield 21. See the [`TorchFeatureExtractor`](../api/feature-extractors.md) API for details on feature naming and extraction.

## Full Setup

Both the feature extractor and the predictor must be configured together. The extractor must produce the features the clusterer expects:

```python
from pybasin.basin_stability_estimator import BasinStabilityEstimator
from pybasin.ts_torch.torch_feature_extractor import TorchFeatureExtractor
from pybasin.predictors import DynamicalSystemClusterer
from pybasin.ts_torch.settings import DYNAMICAL_SYSTEM_FC_PARAMETERS

# Feature extractor with the required feature set
extractor = TorchFeatureExtractor(
    features=DYNAMICAL_SYSTEM_FC_PARAMETERS,
    time_steady=800.0,
    normalize=False,  # recommended -- keeps variance thresholds interpretable
)

# The dynamics-based clusterer with default parameters
predictor = DynamicalSystemClusterer()

# Wire both into the estimator
bse = BasinStabilityEstimator(
    n=10_000,
    ode_system=ode_system,
    sampler=sampler,
    solver=solver,
    feature_extractor=extractor,
    predictor=predictor,
)

bs_vals = bse.estimate_bs()
# Labels will be like: {"FP_0": 0.35, "LC_0": 0.45, "LC_1": 0.15, "chaos_0": 0.05}
```

The [`BasinStabilityEstimator`](../api/basin-stability-estimator.md) automatically calls `set_feature_names()` on the clusterer before prediction, so there is no manual wiring step for feature names.

!!! warning "Mismatched features"
If the feature extractor does not produce the required features, `DynamicalSystemClusterer` raises a `ValueError` at classification time listing which features are missing. The `DYNAMICAL_SYSTEM_FC_PARAMETERS` preset is the simplest way to guarantee compatibility.

!!! tip "Normalization"
Setting `normalize=False` on the feature extractor is recommended when using `DynamicalSystemClusterer`. The default thresholds (`fp_variance_threshold=1e-6`, etc.) are calibrated for raw, unnormalized feature values. With normalization enabled, variance gets z-score scaled and the thresholds would need manual adjustment.

## Algorithm

The classification proceeds in three stages: drift detection (Stage 0), attractor type assignment (Stage 1), and sub-classification within each type (Stage 2). The final output is a label array where each trajectory receives a string label like `FP_0`, `LC_2`, or `chaos_1`.

### Stage 0: Drift Detection

Before any classification, the clusterer scans for _drifting dimensions_ -- state variables where a significant fraction of trajectories show monotonic trends. A rotating pendulum is the canonical example: the angle variable increases without bound even though the underlying attractor (continuous rotation) is well-defined.

The detection works as follows. For each state dimension $i$, the clusterer reads the `linear_trend__attr_slope` feature and computes the fraction of trajectories whose absolute slope exceeds `drift_threshold` (i.e., $|\text{slope}_i| > \texttt{drift_threshold}$). If more than 30% (`drift_fraction=0.3`) of trajectories in that dimension have a high slope, dimension $i$ is flagged as drifting. All subsequent variance and mean calculations for FP and chaos classification exclude drifting dimensions, since their variance is inflated by the drift rather than by genuine attractor dynamics.

If _all_ dimensions are drifting, the clusterer falls back to using all of them to avoid discarding all information.

### Stage 1: Attractor Type Classification

Every trajectory is assigned exactly one of three types: FP, LC, or chaos. The `tiers` parameter controls which types are active and in what priority order. The default is `["FP", "LC", "chaos"]`, meaning FP is checked first, then LC, and anything unmatched falls into chaos.

Three aggregate features are computed per trajectory before the tier checks:

- **Variance:** the mean of per-dimension `variance` values across non-drifting dimensions only.
- **Periodicity strength:** the `autocorrelation_periodicity__output_strength` value from the first non-drifting dimension. Ranges from 0 (no periodic pattern) to 1 (perfect periodicity).
- **Max absolute slope:** the maximum of `|linear_trend__attr_slope|` across all dimensions, including drifting ones.

The tiers are checked in order. Each trajectory starts labeled as "chaos" and the first matching tier overwrites that label. Trajectories already claimed by an earlier tier are skipped by later ones.

**Fixed Point (FP):**

$$\text{variance} < \texttt{fp_variance_threshold}$$

A trajectory whose steady-state variance falls below `fp_variance_threshold` (default `1e-6`) has settled to a constant value. For unnormalized features from a well-converged integration, `1e-6` catches fixed points reliably.

**Limit Cycle (LC):** A trajectory is classified as LC if either condition holds:

1. _Periodic oscillation:_ periodicity strength $>$ `lc_periodicity_threshold` AND variance $<$ `chaos_variance_threshold`
2. _Rotating solution:_ max |slope| $>$ `drift_threshold` AND variance $>$ `fp_variance_threshold`

The first condition captures standard limit cycles -- regular oscillations with strong periodicity and bounded amplitude. The `chaos_variance_threshold` (default `5.0`) serves as the upper bound here: a trajectory with high periodicity _but_ extremely large variance is more likely chaotic than a clean limit cycle, so it gets excluded. The second condition catches rotating solutions (e.g., a pendulum spinning over the top) where the angle drifts monotonically but the trajectory is still periodic in a phase-space sense. The lower bound `variance > fp_variance_threshold` prevents fixed points from being misclassified as rotating LCs.

**Chaos:** Everything not matched by FP or LC remains labeled "chaos." Typically these trajectories have high variance combined with low periodicity strength. Note that "chaos" here is a catch-all for non-FP, non-LC behavior -- it will also capture unbounded trajectories unless those are handled separately (see [`UnboundednessMetaEstimator`](../api/predictors.md)).

Because tiers are checked in order, the ordering matters. With `tiers=["LC", "FP"]`, a trajectory with both near-zero variance and high periodicity would be labeled LC, not FP.

### Stage 2: Sub-classification

After Stage 1, each attractor type may contain multiple distinct attractors. This stage separates them. The labeling is sequential across all types -- if FP sub-classification produces labels 0 and 1, the first LC sub-cluster starts at label 2. This gives globally unique labels like `FP_0`, `FP_1`, `LC_2`, `LC_3`.

#### Fixed Point Sub-classification

Fixed points are clustered by their steady-state location, using the `mean` feature across non-drifting dimensions.

Before running any clustering, the algorithm checks whether the data range (max - min) of mean values in every dimension falls below 0.01. If so, all trajectories go into a single cluster -- this avoids spurious splits when all fixed points converge to the same location.

Otherwise, HDBSCAN runs on StandardScaler-normalized mean values with `min_cluster_size = max(50, n // 10)`, where `n` is the number of FP trajectories. Noise points are assigned to their nearest cluster via the `assign_noise=True` behavior in pybasin's HDBSCAN wrapper.

A custom sub-classifier can replace this step via `fp_sub_classifier`. It receives a feature matrix with one row per fixed-point trajectory and one column per non-drifting dimension, containing the mean values.

#### Limit Cycle Sub-classification

Limit cycles use a hierarchical two-level approach:

**Level 1 -- Period grouping:** The `spectral_frequency_ratio` is rounded to the nearest integer (clamped to [1, 10]) to determine the period number. Period-1, period-2, etc. trajectories form separate groups. This step uses a single analysis dimension -- the first non-drifting dimension.

**Level 2 -- Within-period clustering:** Inside each period group, the algorithm checks whether amplitude and mean values vary enough to warrant splitting:

- _Amplitude variation_ is measured by the coefficient of variation (CV = std/mean). If CV $> 0.1$, amplitude varies meaningfully.
- _Mean variation_ is measured by the range (max - min). If range $> 0.05$, spatial location varies.

When neither varies, the whole period group becomes one cluster. When one or both vary, the variable features are passed to a gap-based 1D clusterer (for a single varying feature) or to HDBSCAN (for two varying features). The 1D gap detector sorts values and finds gaps $>5\times$ the median spacing, creating up to 5 clusters.

A custom sub-classifier can replace the entire two-level process via `lc_sub_classifier`. It receives `[freq_ratio, amplitude, mean]` columns from the first analysis dimension.

#### Chaos Sub-classification

Chaotic attractors are clustered by their spatial mean across non-drifting dimensions, using `HDBSCANClusterer` with `auto_tune=True`. Two butterfly wings of the Lorenz attractor, for instance, have different mean $x$-values and will separate into `chaos_0` and `chaos_1`.

Non-finite values (e.g., from diverged trajectories) are zeroed before scaling. If fewer than 2 finite samples exist, everything goes into one cluster.

A custom sub-classifier via `chaos_sub_classifier` receives a matrix of mean values with one row per chaotic trajectory and one column per non-drifting dimension.

## Constructor Parameters

| Parameter                  | Type                | Default                 | Description                                                                                                                                   |
| -------------------------- | ------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `drift_threshold`          | `float`             | `0.1`                   | Minimum \|slope\| to flag a dimension as drifting. Units: state_units / time_units. Also the threshold for detecting rotating LC solutions.   |
| `drift_fraction`           | `float`             | `0.3`                   | Minimum fraction of trajectories with high slope for a dimension to be flagged as drifting. 0.3 means 30% of trajectories must show drift.    |
| `tiers`                    | `list[str] \| None` | `["FP", "LC", "chaos"]` | Attractor types to detect, in priority order. First match wins. Options: `"FP"`, `"LC"`, `"chaos"`. Set to e.g. `["FP", "LC"]` to skip chaos. |
| `fp_variance_threshold`    | `float`             | `1e-6`                  | Maximum variance for FP classification. For unnormalized features, set based on expected steady-state fluctuations.                           |
| `fp_sub_classifier`        | sklearn estimator   | `None`                  | Custom sub-classifier for FP. Receives mean values per non-drifting dimension. Must implement `fit_predict()`. Default: HDBSCAN.              |
| `lc_periodicity_threshold` | `float`             | `0.5`                   | Minimum periodicity strength [0--1]. 0.3--0.5 is weak/noisy, 0.5--0.8 is clear periodic, 0.8--1.0 is strong.                                  |
| `lc_sub_classifier`        | sklearn estimator   | `None`                  | Custom sub-classifier for LC. Receives `[freq_ratio, amplitude, mean]` columns. Must implement `fit_predict()`. Default: hierarchical.        |
| `chaos_variance_threshold` | `float`             | `5.0`                   | Upper variance bound for the periodic-LC condition. Trajectories above this with low periodicity become chaos.                                |
| `chaos_sub_classifier`     | sklearn estimator   | `None`                  | Custom sub-classifier for chaos. Receives mean values per dimension. Must implement `fit_predict()`. Default: HDBSCAN with auto_tune.         |

For the full API reference, including method signatures and attribute details, see the [`DynamicalSystemClusterer` API documentation](../api/predictors.md).

## Tuning Tips

**Start with defaults,** then adjust based on the label distribution. If too many limit cycles are classified as chaos, lower `chaos_variance_threshold` or raise `lc_periodicity_threshold`. If fixed points are missed, increase `fp_variance_threshold`.

**Normalization matters.** The variance threshold operates on the actual feature values. When using [`TorchFeatureExtractor`](../api/feature-extractors.md) with `normalize=True` (the default), steady-state variance is z-score normalized, so the raw threshold `1e-6` will not match. Either disable normalization for this workflow or increase the threshold to account for scaling.

**Inspect the tiers.** If your system has no chaotic behavior, set `tiers=["FP", "LC"]` to avoid misclassifying noisy limit cycles as chaos. Conversely, `tiers=["chaos"]` skips structured classification entirely and just clusters everything by spatial location.

**Custom sub-classifiers.** Each attractor type's Stage 2 step accepts a custom sklearn-compatible estimator. Pass any clusterer that implements `fit_predict()`. For example, use `KMeans(n_clusters=2)` as `fp_sub_classifier` if you know exactly how many fixed points exist.

**Handling unbounded trajectories.** `DynamicalSystemClusterer` by itself does not distinguish divergent trajectories from chaotic ones -- both end up in chaos. To separate them, wrap it with [`UnboundednessMetaEstimator`](../api/predictors.md), which detects unbounded solutions via a solver event function before the clusterer runs. The Lorenz case study uses this pattern.

## Validation Results

Each case study was run with initial conditions taken from MATLAB bSTAB ground truth CSVs. Predicted labels were mapped to ground truth labels via majority vote per cluster, and the Matthews Correlation Coefficient (MCC) was computed against 5,000--20,000 labeled samples per system. All runs used default clusterer parameters with unnormalized features.

| System   | N      | Attractors found       | MCC    |
| -------- | ------ | ---------------------- | ------ |
| Duffing  | 10,000 | 5 limit cycles         | 0.9977 |
| Friction | 5,000  | 1 fixed point, 1 LC    | 1.0000 |
| Pendulum | 10,000 | 1 fixed point, 1 LC    | 1.0000 |
| Lorenz   | 20,000 | 2 chaotic, 1 unbounded | 0.9985 |

Friction and pendulum achieve perfect classification -- the two attractor types (FP and LC) are cleanly separated by the variance and periodicity thresholds. Duffing is harder because it has five coexisting limit cycles of different periods, yet the hierarchical period-based sub-classification still reaches MCC 0.9977. For Lorenz, the two butterfly attractors overlap in feature space enough to produce a small number of mis-assignments between `chaos_0` and `chaos_1`, but overall accuracy remains high at 0.9985.
