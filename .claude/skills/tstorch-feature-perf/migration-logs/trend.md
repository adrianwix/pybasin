# Trend domain migration log

The trend domain now exposes five `Feature` classes through the same public contract used by
`TorchFeatureExtractor`: `LinearTrend`, `LinearTrendTimewise`, `AggLinearTrend`,
`ArCoefficient`, and `AugmentedDickeyFuller`. The old factory and function surface was removed.

## Reference-compatible primitives

The migration needed behavior from SciPy and statsmodels without calling either library from a
Torch feature. Those implementations live under matching package hierarchies instead of in the
calculator module:

- `tstorch.scipy.stats.linregress` owns linear-regression statistics and the Student t p-value.
- `tstorch.statsmodels.regression.linear_model` owns batched ordinary least squares.
- `tstorch.statsmodels.tsa.ar_model` owns autoregressive fitting.
- `tstorch.statsmodels.tsa.stattools` and `adfvalues` own ADF lag selection and MacKinnon
  p-values.

This keeps the feature classes focused on parameter grouping, output selection, and names. It
also leaves each recreated library primitive at the path a reader would expect from the source
library.

## Precision and device behavior

Every computation preserves the input floating-point dtype. Float32 inputs remain float32 and
float64 inputs remain float64. The CUDA path never promotes float32 data to float64.

CPU regression uses the native QR implementation. Batched CUDA QR was accurate but prohibitively
slow for the ADF workload because the test fits many small regressions. Normal equations were
fast but failed on singular trajectories and amplified float32 conditioning errors. The final
CUDA path uses re-orthogonalized modified Gram-Schmidt with fixed-shape Torch operations. It
remains full-graph compilable, handles the benchmark batch efficiently, and agrees with CPU on
well-conditioned float32 and float64 inputs.

Accuracy reporting was extended so precision records include the input dtype. Trend tests now
compare every feature with tsfresh in both float32 and float64 and assert that the returned dtype
matches the requested precision.

## Feature-specific decisions

`LinearTrend` computes all requested regression statistics once and selects output columns.
`LinearTrendTimewise` reuses those statistics and applies the elapsed-hours scaling expected by
the tsfresh timestamp convention.

`AggLinearTrend` retains a short final chunk and uses sample variance, matching tsfresh. Its
aggregate parameter accepts only `mean`, `var`, `min`, and `max`; tsfresh cannot execute
`median` for this calculator.

`ArCoefficient` fits the same intercept-plus-lags model as statsmodels `AutoReg`. Requests are
grouped by model order, so coefficients from the same model share one fit.

`AugmentedDickeyFuller` implements the constant-regression ADF test rather than a single-lag
approximation. AIC, BIC, final-lag t-statistic, and fixed-lag selection are supported, and the
p-value uses the MacKinnon approximation. Requests using the same lag-selection method share the
same test.

## Verification

All five features have numerical, naming, full-graph compile, export, CPU/CUDA agreement, direct
call, extractor, and batched-argument coverage. The report-producing benchmark completed with
CUDA enabled. The required batched tstorch CPU comparison is faster than tsfresh for every trend
feature.
