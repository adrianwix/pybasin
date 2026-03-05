# Feature Extractors

Feature extractors transform ODE solution trajectories into numerical feature vectors. These vectors capture time-series characteristics -- statistical moments, spectral properties, complexity measures -- that distinguish different attractor types. The downstream predictor (clusterer or classifier) then operates on these features to assign basin labels.

All extractors inherit from `FeatureExtractor`, which defines two things every subclass must provide: an `extract_features(solution)` method returning a tensor of shape `(B, F)`, and a `feature_names` property listing the F feature names.

## Available Extractors

| Class                     | Features   | GPU | Speed   | Best for                                                      |
| ------------------------- | ---------- | --- | ------- | ------------------------------------------------------------- |
| `TorchFeatureExtractor`   | 10 -- 800+ | Yes | Fast    | **Default.** General-purpose, GPU-accelerated.                |
| `JaxFeatureExtractor`     | 10 -- 700+ | Yes | Fastest | JAX-only pipelines.                                           |
| `TsfreshFeatureExtractor` | 20 -- 800+ | No  | Slow    | **Recommended for production.** Battle-tested reference.      |
| `NoldsFeatureExtractor`   | 2 -- 8     | No  | Slow    | Nonlinear dynamics features only (Lyapunov, correlation dim). |

!!! warning "Experimental: Torch and JAX extractors"
`TorchFeatureExtractor` and `JaxFeatureExtractor` are experimental reimplementations that recreate the tsfresh feature extraction API in a GPU-optimized way. They have not been deeply validated against tsfresh for correctness in all cases. Results are close but not identical to tsfresh. For maximum confidence, prefer `TsfreshFeatureExtractor`.

## Feature Configuration (FCParameters)

All extractors except `NoldsFeatureExtractor` use a tsfresh-style dictionary to specify which features to compute. The type alias is:

```python
FCParameters = Mapping[str, list[dict[str, Any]] | None]
```

Each key is a feature calculator name, and the value is either `None` (use default parameters) or a list of parameter dictionaries. One feature output is produced per parameter combination.

```python
# No parameters -- just compute the feature once
{"mean": None, "variance": None}

# Parameterized -- compute autocorrelation at three different lags
{"autocorrelation": [{"lag": 1}, {"lag": 5}, {"lag": 10}]}

# Mixed
{
    "mean": None,
    "quantile": [{"q": 0.1}, {"q": 0.5}, {"q": 0.9}],
}
```

### Preset Configurations

`TorchFeatureExtractor` ships with string shortcuts for common presets:

| Preset            | Features per state | Description                                                                                                                                               |
| ----------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"comprehensive"` | ~800               | Full tsfresh-equivalent sweep. Includes FFT coefficients, entropy measures, trend statistics, and more.                                                   |
| `"minimal"`       | 10                 | Basic statistics: `median`, `mean`, `standard_deviation`, `variance`, `root_mean_square`, `maximum`, `absolute_maximum`, `minimum`, `delta`, `log_delta`. |

`JaxFeatureExtractor` defaults to `JAX_MINIMAL_FC_PARAMETERS`, which matches the same 10 features as Torch minimal.

!!! warning "JAX comprehensive JIT compile time"
Using `JAX_COMPREHENSIVE_FC_PARAMETERS` triggers a JIT compilation step that can take ~40 minutes. Stick to `JAX_MINIMAL_FC_PARAMETERS` or a custom subset unless compilation time is acceptable.

### Feature Calculators

Below is a summary of all available Torch feature calculators, grouped by category. Each calculator can be referenced by name in an `FCParameters` dictionary. JAX and tsfresh support overlapping but not identical sets. For full function signatures and parameter details, see the [Torch Feature Calculators API reference](../api/torch-feature-calculators.md).

| Category                    | Calculators                                                                                                                                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Statistical**             | `sum_values`, `median`, `mean`, `length`, `standard_deviation`, `variance`, `root_mean_square`, `maximum`, `absolute_maximum`, `minimum`, `abs_energy`, `kurtosis`, `skewness`, `quantile`, `variation_coefficient`   |
| **Change / Difference**     | `absolute_sum_of_changes`, `mean_abs_change`, `mean_change`, `mean_second_derivative_central`                                                                                                                         |
| **Counting**                | `count_above`, `count_above_mean`, `count_below`, `count_below_mean`, `count_in_range`, `count_value`                                                                                                                 |
| **Boolean**                 | `has_duplicate`, `has_duplicate_max`, `has_duplicate_min`, `has_variance_larger_than_standard_deviation`, `large_standard_deviation`                                                                                  |
| **Location**                | `first_location_of_maximum`, `first_location_of_minimum`, `last_location_of_maximum`, `last_location_of_minimum`, `index_mass_quantile`                                                                               |
| **Pattern / Streak**        | `longest_strike_above_mean`, `longest_strike_below_mean`, `number_crossing_m`, `number_peaks`, `number_cwt_peaks`                                                                                                     |
| **Autocorrelation**         | `autocorrelation`, `partial_autocorrelation`, `agg_autocorrelation`, `autocorrelation_periodicity`                                                                                                                    |
| **Entropy / Complexity**    | `permutation_entropy`, `binned_entropy`, `fourier_entropy`, `lempel_ziv_complexity`, `cid_ce`, `approximate_entropy`, `sample_entropy`                                                                                |
| **Frequency Domain**        | `fft_coefficient`, `fft_aggregated`, `spkt_welch_density`, `cwt_coefficients`, `spectral_frequency_ratio`                                                                                                             |
| **Trend / Regression**      | `linear_trend`, `linear_trend_timewise`, `agg_linear_trend`, `ar_coefficient`, `augmented_dickey_fuller`                                                                                                              |
| **Reoccurrence**            | `percentage_of_reoccurring_datapoints_to_all_datapoints`, `percentage_of_reoccurring_values_to_all_values`, `sum_of_reoccurring_data_points`, `sum_of_reoccurring_values`, `ratio_value_number_to_time_series_length` |
| **Advanced**                | `benford_correlation`, `c3`, `energy_ratio_by_chunks`, `time_reversal_asymmetry_statistic`                                                                                                                            |
| **Custom (Torch/JAX only)** | `delta` (max - min), `log_delta` (log of delta), `amplitude` (half of peak-to-peak)                                                                                                                                   |
| **Dynamical Systems**       | `lyapunov_r`, `lyapunov_e`, `correlation_dimension`, `friedrich_coefficients`, `max_langevin_fixed_point`                                                                                                             |

## Default Features in BasinStabilityEstimator

When no `feature_extractor` is passed to `BasinStabilityEstimator`, it creates a `TorchFeatureExtractor` with `DEFAULT_TORCH_FC_PARAMETERS`. This preset matches `TORCH_MINIMAL_FC_PARAMETERS` -- 10 features per state variable:

| Feature              | What it measures                                       |
| -------------------- | ------------------------------------------------------ |
| `median`             | Central tendency (robust to outliers)                  |
| `mean`               | Arithmetic average                                     |
| `standard_deviation` | Spread around the mean                                 |
| `variance`           | Squared spread                                         |
| `root_mean_square`   | Energy-like measure                                    |
| `maximum`            | Peak value                                             |
| `absolute_maximum`   | Peak absolute value                                    |
| `minimum`            | Trough value                                           |
| `delta`              | Range (`max - min`)                                    |
| `log_delta`          | `log(delta)` -- compresses large amplitude differences |

## Transient Filtering (time_steady)

Before computing features, extractors discard the initial transient portion of each trajectory. This prevents startup dynamics from contaminating the steady-state features.

The `time_steady` parameter controls when the transient ends:

| Value              | Behavior                                                                                                                                                                |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `None` (default)   | Automatically set to 85% of the integration time span. For a `time_span=(0, 1000)` integration, features are computed from `t > 850` onward -- the last 150 time units. |
| `0.0`              | Use the entire time series, no transient removal.                                                                                                                       |
| Any positive float | Use data after that time value. For example, `time_steady=900.0` keeps only `t > 900`.                                                                                  |

!!! tip "Choosing time_steady"
The 85% default is conservative. If your system settles quickly, setting `time_steady` to a lower value (e.g., 50% of the span) retains more data points for feature extraction, which can improve feature quality. On the other hand, systems with slow transients may need a higher threshold.

!!! note "NoldsFeatureExtractor exception"
`NoldsFeatureExtractor` defaults to `time_steady=0.0` (no transient removal), unlike all other extractors which default to `None` (85%). This is because nonlinear dynamics measures like Lyapunov exponents often benefit from longer time series. Override this explicitly if transient removal is needed.

## Normalization

All extractors except `NoldsFeatureExtractor` support z-score normalization (`normalize=True` by default). The scaler is **fitted on the first call** to `extract_features()` and then reused for subsequent calls.

This fit-on-first-call behavior has an important consequence for supervised workflows: the scaler trains on whichever dataset is extracted first. When using a `TemplateIntegrator` with a classifier, pybasin integrates both template and main trajectories, extracts features from the main dataset first (fitting the scaler on the larger sample), then transforms the template features using the same scaler. To reset the scaler between runs, call `reset_scaler()`.

| Extractor                 | Normalization backend                  | Scaler reset method |
| ------------------------- | -------------------------------------- | ------------------- |
| `TorchFeatureExtractor`   | Manual z-score (PyTorch)               | `reset_scaler()`    |
| `JaxFeatureExtractor`     | Manual z-score (JAX)                   | `reset_scaler()`    |
| `TsfreshFeatureExtractor` | `sklearn.preprocessing.StandardScaler` | `reset_scaler()`    |
| `NoldsFeatureExtractor`   | None                                   | N/A                 |

## Imputation

Trajectories that diverge or contain numerical issues can produce `NaN` or `Inf` feature values. Each extractor handles this with an imputation step after extraction.

| Method                              | Behavior                                                                        | When to use                                                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `"extreme"` (default for Torch/JAX) | Replace `NaN`/`Inf` with `1e10`.                                                | Systems with divergent solutions -- the extreme value separates unbounded trajectories from bounded ones in feature space. |
| `"tsfresh"`                         | Per-column: `+Inf` -> column max, `-Inf` -> column min, `NaN` -> column median. | Fully bounded systems where all trajectories converge.                                                                     |

`TsfreshFeatureExtractor` uses tsfresh's built-in `impute()` function, which follows the tsfresh-style strategy.

`NoldsFeatureExtractor` always uses column-wise imputation: `-Inf` -> column min, `+Inf` -> column max, `NaN` -> column median. Columns that are entirely `NaN` are filled with `0`.

## Feature Naming Convention

Feature names follow the pattern `state_<index>__<feature_name>`, where `<index>` is the zero-based state variable index. Parameterized features append parameter values:

```
state_0__mean
state_0__quantile__q_0.1
state_1__autocorrelation__lag_5
```

These names appear in `Solution.feature_names` after extraction and are used by feature selectors and plotting utilities.

!!! note "Names available only after extraction"
The `feature_names` property raises `RuntimeError` if accessed before calling `extract_features()`, because the number of states (and therefore the number of features) is only known at extraction time.

---

## TorchFeatureExtractor

The default extractor. Reimplements tsfresh feature calculators in pure PyTorch for GPU acceleration.

```python
from pybasin.feature_extractors import TorchFeatureExtractor

# Comprehensive features on GPU
extractor = TorchFeatureExtractor(
    time_steady=800.0,
    features="comprehensive",
    device="gpu",
)

# Minimal features on CPU (fast, good baseline)
extractor = TorchFeatureExtractor(
    time_steady=None,       # default: 85% of time span
    features="minimal",     # 10 features per state
    normalize=True,         # z-score normalization
    impute_method="extreme",
)

# Custom feature set
extractor = TorchFeatureExtractor(
    features={
        "mean": None,
        "variance": None,
        "autocorrelation": [{"lag": 1}, {"lag": 5}],
        "fft_aggregated": [{"aggtype": s} for s in ["centroid", "variance"]],
    },
)
```

### Per-State Configuration

Different state variables often carry different physical meaning (e.g., position vs. velocity). You can assign distinct feature sets to each state:

```python
extractor = TorchFeatureExtractor(
    features="minimal",            # default for states not listed below
    features_per_state={
        0: {"maximum": None, "minimum": None},  # override for state 0
        1: None,                                  # skip state 1 entirely
    },
)
```

When `features_per_state` is provided, it overrides the global `features` for the specified indices. Setting a state's value to `None` skips feature extraction for that state entirely.

!!! info "Uniform config optimization"
When all states share the same configuration object (by identity, not just by equality), the extractor processes all states in a single batched pass. Mixing per-state overrides disables this optimization and processes each state separately.

### Constructor Parameters

| Parameter            | Type                                                   | Default           | Description                                                               |
| -------------------- | ------------------------------------------------------ | ----------------- | ------------------------------------------------------------------------- |
| `time_steady`        | `float \| None`                                        | `None`            | Transient cutoff. `None` = 85% of span.                                   |
| `features`           | `"comprehensive" \| "minimal" \| FCParameters \| None` | `"comprehensive"` | Feature configuration for all states.                                     |
| `features_per_state` | `dict[int, FCParameters \| None] \| None`              | `None`            | Per-state overrides.                                                      |
| `normalize`          | `bool`                                                 | `True`            | Apply z-score normalization.                                              |
| `device`             | `"cpu" \| "gpu"`                                       | `"cpu"`           | Extraction device. Raises `RuntimeError` if `"gpu"` and CUDA unavailable. |
| `n_jobs`             | `int \| None`                                          | `None`            | CPU worker count. `None` = all cores. Ignored on GPU.                     |
| `impute_method`      | `"extreme" \| "tsfresh"`                               | `"extreme"`       | How to handle `NaN`/`Inf` in features.                                    |

---

## TsfreshFeatureExtractor

Wraps the tsfresh library directly. This is the most reliable extractor for correctness, since tsfresh is a widely-used, well-tested time-series feature library. The tradeoff is slower extraction (CPU-only, pandas-based).

```python
from pybasin.feature_extractors.tsfresh_feature_extractor import TsfreshFeatureExtractor
from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters

# Minimal extraction (fast, ~20 features per state)
extractor = TsfreshFeatureExtractor(
    time_steady=800.0,
    default_fc_parameters=MinimalFCParameters(),
    n_jobs=1,
    normalize=True,
)

# Comprehensive extraction (~800 features per state)
extractor = TsfreshFeatureExtractor(
    time_steady=800.0,
    default_fc_parameters=ComprehensiveFCParameters(),
    n_jobs=-1,  # use all cores
)

# Custom feature set
extractor = TsfreshFeatureExtractor(
    default_fc_parameters={"mean": None, "maximum": None, "minimum": None},
)
```

### Per-State Configuration

Uses `kind_to_fc_parameters`, keyed by integer state index:

```python
extractor = TsfreshFeatureExtractor(
    time_steady=950.0,
    kind_to_fc_parameters={
        0: {"mean": None, "maximum": None},       # position: basic stats
        1: ComprehensiveFCParameters(),             # velocity: full analysis
    },
    n_jobs=1,
)
```

### Constructor Parameters

| Parameter               | Type                              | Default                 | Description                           |
| ----------------------- | --------------------------------- | ----------------------- | ------------------------------------- |
| `time_steady`           | `float \| None`                   | `None`                  | Transient cutoff.                     |
| `default_fc_parameters` | `FCParameters`                    | `MinimalFCParameters()` | Default features for all states.      |
| `kind_to_fc_parameters` | `dict[int, FCParameters] \| None` | `None`                  | Per-state overrides.                  |
| `n_jobs`                | `int`                             | `1`                     | Parallel jobs. `-1` = all cores.      |
| `normalize`             | `bool`                            | `True`                  | Apply `StandardScaler` normalization. |

!!! warning "Non-determinism with n_jobs > 1"
Parallel feature extraction introduces non-determinism from floating-point arithmetic ordering. Classification results may vary between runs. Set `n_jobs=1` for reproducible results.

---

## JaxFeatureExtractor

A JAX-based reimplementation targeting JAX-native workflows. Features are JIT-compiled for speed, but compilation adds a one-time overhead.

```python
from pybasin.feature_extractors.jax import JaxFeatureExtractor
from pybasin.feature_extractors.jax.jax_feature_calculators import JAX_MINIMAL_FC_PARAMETERS

extractor = JaxFeatureExtractor(
    time_steady=800.0,
    features=JAX_MINIMAL_FC_PARAMETERS,  # 12 features per state (default)
    device="gpu",
    normalize=True,
)
```

### Constructor Parameters

| Parameter            | Type                                      | Default                     | Description                                                              |
| -------------------- | ----------------------------------------- | --------------------------- | ------------------------------------------------------------------------ |
| `time_steady`        | `float \| None`                           | `None`                      | Transient cutoff.                                                        |
| `features`           | `FCParameters \| None`                    | `JAX_MINIMAL_FC_PARAMETERS` | Feature configuration.                                                   |
| `features_per_state` | `dict[int, FCParameters \| None] \| None` | `None`                      | Per-state overrides.                                                     |
| `normalize`          | `bool`                                    | `True`                      | Apply z-score normalization.                                             |
| `use_jit`            | `bool`                                    | `True`                      | JIT-compile extraction functions.                                        |
| `device`             | `str \| None`                             | `None`                      | JAX device (`"cpu"`, `"gpu"`, `"cuda"`, `"cuda:N"`, or `None` for auto). |
| `impute_method`      | `"extreme" \| "tsfresh"`                  | `"extreme"`                 | How to handle `NaN`/`Inf`.                                               |

---

## NoldsFeatureExtractor

!!! warning "Currently unavailable"
    `NoldsFeatureExtractor` is temporarily non-functional due to a bug in the `nolds` library that causes import errors. Tests for this extractor are skipped in CI until the issue is resolved upstream.

Computes nonlinear dynamics measures using the [nolds](https://github.com/CSchoel/nolds) library. These features -- Lyapunov exponents, correlation dimension, sample entropy, Hurst exponent -- characterize the complexity and chaoticity of attractors. They are slow to compute but carry information that statistical features miss.

Requires `nolds` as an optional dependency (`uv add nolds`).

```python
from pybasin.feature_extractors import NoldsFeatureExtractor

# Default: Lyapunov exponent (Rosenstein) + correlation dimension
extractor = NoldsFeatureExtractor()

# Custom set with multiple parameter combinations
extractor = NoldsFeatureExtractor(
    time_steady=0.0,
    features={
        "lyap_r": [{"emb_dim": 10}, {"emb_dim": 15}],
        "sampen": None,
        "hurst_rs": None,
    },
    n_jobs=-1,
)
```

### Available Features

| Feature key  | nolds function     | What it measures                                |
| ------------ | ------------------ | ----------------------------------------------- |
| `lyap_r`     | `nolds.lyap_r`     | Largest Lyapunov exponent (Rosenstein's method) |
| `lyap_e`     | `nolds.lyap_e`     | Lyapunov exponent (Eckmann's method)            |
| `sampen`     | `nolds.sampen`     | Sample entropy                                  |
| `hurst_rs`   | `nolds.hurst_rs`   | Hurst exponent (R/S method)                     |
| `corr_dim`   | `nolds.corr_dim`   | Correlation dimension                           |
| `dfa`        | `nolds.dfa`        | Detrended fluctuation analysis                  |
| `mfhurst_b`  | `nolds.mfhurst_b`  | Multifractal Hurst exponent (basic)             |
| `mfhurst_dm` | `nolds.mfhurst_dm` | Multifractal Hurst exponent (DMA)               |

### Constructor Parameters

| Parameter            | Type                                           | Default                              | Description                                                                |
| -------------------- | ---------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------- |
| `time_steady`        | `float \| None`                                | `0.0`                                | Transient cutoff. Defaults to `0.0` (no removal), unlike other extractors. |
| `features`           | `NoldsFCParameters`                            | `{"lyap_r": None, "corr_dim": None}` | Feature configuration.                                                     |
| `features_per_state` | `dict[int, NoldsFCParameters \| None] \| None` | `None`                               | Per-state overrides.                                                       |
| `n_jobs`             | `int \| None`                                  | `None`                               | Parallel workers. `None` = all cores.                                      |

!!! note "No normalization"
`NoldsFeatureExtractor` does not support normalization. If normalization is needed, apply it externally before passing features to the predictor.

!!! note "Error handling"
Individual feature computations that fail (e.g., due to insufficient data length) silently produce `NaN`, which is then imputed column-wise. No exceptions are raised for individual failures.

---

## Standalone Usage

Feature extractors can be used independently of `BasinStabilityEstimator`, for example to inspect features from a single trajectory or a custom solution.

```python
import torch
from pybasin.feature_extractors import TorchFeatureExtractor
from pybasin.solution import Solution

# Create a solution manually (N=1000 timesteps, B=1 trajectory, S=2 states)
t = torch.linspace(0, 10, 1000)
y = torch.stack([torch.sin(t), torch.cos(t)], dim=-1).unsqueeze(1)  # (N, 1, S)

solution = Solution(
    initial_condition=torch.tensor([[0.0, 1.0]]),
    time=t,
    y=y,
)

extractor = TorchFeatureExtractor(features="minimal", time_steady=0.0, normalize=False)
features = extractor.extract_features(solution)

print(f"Feature shape: {features.shape}")     # (1, 20) for 2 states x 10 features
print(f"Feature names: {extractor.feature_names}")
```

---

## Creating Custom Feature Extractors

See the [Custom Feature Extractor](../guides/custom-feature-extractor.md) guide for details on subclassing `FeatureExtractor`.
