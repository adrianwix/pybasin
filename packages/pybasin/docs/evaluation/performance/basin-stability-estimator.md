# Basin Stability Estimator

## Interactive Flame Graph

View the profiling results in speedscope:

[Open in speedscope](https://www.speedscope.app/#profileURL=https%3A%2F%2Fraw.githubusercontent.com%2Fadrianwix%2Fpybasin%2Fmain%2Fbenchmarks%2Fprofiling%2Fprofile.speedscope.json&title=pybasin%20Profiling){ .md-button .md-button--primary }

## Example Run

The [pendulum case study](https://adrianwix.github.io/pybasin/case-studies/pendulum/) with 100,000 initial conditions, GPU only, using pybasin defaults -- only the ODE system and region of interest were provided, all other components (solver, feature extractor, predictor) use their default configurations:

```
BASIN STABILITY ESTIMATION COMPLETE
Total time: 16.3494s
Timing Breakdown:
  1. Sampling:               0.0325s  (  0.2%)
  2. Integration:           13.7112s  ( 83.9%)
  3. Solution/Amps:          0.0000s  (  0.0%)
  4. Features:               0.2640s  (  1.6%)
  5. Filtering:              0.0539s  (  0.3%)
  6. Classification:         2.2562s  ( 13.8%)
  7. BS Computation:         0.0286s  (  0.2%)
```

### Expensive Steps

The two most computationally expensive steps are:

1. **ODE Integration (~84%)** — Solving the differential equations for all initial conditions. Uses JAX/Diffrax by default with GPU acceleration, and dominates runtime at every sample size from 10,000 to 100,000.

2. **Classification (~14%)** — HDBSCAN clustering with auto-tuning enabled, followed by KNN-based noise reassignment. The auto-tuning search subsamples large datasets instead of re-fitting HDBSCAN on the full set for every candidate, and noise reassignment is vectorized across all points at once, keeping this step's cost roughly proportional to integration's rather than growing faster than N.

3. **Feature Extraction (~1.6%)** — Extracts time series features from trajectories. The default `DefaultFeatureExtractor` uses these statistical features: `median`, `mean`, `standard_deviation`, `variance`, `root_mean_square`, `maximum`, `absolute_maximum`, `minimum`, `delta`, `log_delta`, z-score normalized by default so no single feature's scale dominates downstream clustering distances.

!!! note "Feature Complexity"
More complex features (e.g., entropy, autocorrelation, frequency domain) can significantly increase extraction time. The default minimal set is chosen for speed while maintaining classification accuracy.

## Profiling Setup

The profile was generated using [Austin](https://github.com/P403n1x87/austin), a frame stack sampler for CPython.

The pendulum case study is run using pybasin defaults—only the ODE system and area of interest (sampler bounds) are defined. All other components (solver, feature extractor, predictor) use their default configurations.

To generate a new profile:

```bash
./scripts/generate_profiling.sh
```

This runs the pendulum case study and outputs `profile.speedscope.json` for visualization in speedscope.
