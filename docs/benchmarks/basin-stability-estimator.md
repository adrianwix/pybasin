# Basin Stability Estimator

## Interactive Flame Graph

View the profiling results in speedscope:

[Open in speedscope](https://www.speedscope.app/#profileURL=https%3A%2F%2Fraw.githubusercontent.com%2Fadrianwix%2Fpybasin%2Fmain%2Fbenchmarks%2Fprofiling%2Fprofile.speedscope.json&title=pybasin%20Profiling){ .md-button .md-button--primary }

## Example Run

The [pendulum case study](../case-studies/pendulum.md) with 10,000 initial conditions using pybasin defaults — only the ODE system and region of interest were provided, all other components (solver, feature extractor, predictor) use their default configurations:

```
BASIN STABILITY ESTIMATION COMPLETE
Total time: 17.0062s
Timing Breakdown:
  1. Sampling:               0.0266s  (  0.2%)
  2. Integration:           12.6641s  ( 74.5%)
     - Main:                12.6638s  ( 74.5%)
  3. Solution/Amps:          0.0000s  (  0.0%)
  4. Features:               0.3235s  (  1.9%)
  5. Filtering:              0.0033s  (  0.0%)
  6. Classification:         3.9861s  ( 23.4%)
  7. BS Computation:         0.0020s  (  0.0%)
```

### Expensive Steps

The three most computationally expensive steps are:

1. **ODE Integration (~75%)** — Solving the differential equations for all initial conditions. Uses JAX/Diffrax by default with GPU acceleration.

2. **Classification (~23%)** — HDBSCAN clustering with auto-tuning enabled, followed by KMeans to assign noise points to the nearest cluster.

3. **Feature Extraction (~2%)** — Extracts time series features from trajectories. The default `TorchFeatureExtractor` uses these statistical features: `median`, `mean`, `standard_deviation`, `variance`, `root_mean_square`, `maximum`, `absolute_maximum`, `minimum`, `delta`, `log_delta`.

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
