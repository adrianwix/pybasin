# Statistical domain migration log

Relative to `origin/main`, this branch adds `torch_features_statistical.py` with 21 feature
classes and wires the complete statistical domain into `SUPPORTED_FEATURES`. Parameterless and
parametrized features expose the same class construction, `(B, S, P)` output, and `.names`
contracts used by direct callers and `TorchFeatureExtractor`.

## Direct reductions

`SumValues`, `Mean`, `Length`, `StandardDeviation`, `Variance`, `RootMeanSquare`, `Maximum`,
`AbsoluteMaximum`, `Minimum`, and `AbsEnergy` map their definitions to batched Torch reductions
over the time dimension. Population variance and standard deviation use `correction=0` to match
NumPy and tsfresh.

`Amplitude` returns `max(x) - min(x)`. `Delta` returns `abs(max(x) - mean(x))`, and `LogDelta`
returns `log(delta + 1e-10)`. These three features have no tsfresh equivalent and are covered by
their class API, shape, compile, and export tests.

## Median

`Median` sorts each trajectory and averages the two middle values for even lengths. This matches
NumPy median semantics; `torch.median` selects only the lower middle value. The explicit
sort-and-index kernel also supports symbolic shapes under full-graph compilation.

## Kurtosis and Skewness

`Kurtosis` and `Skewness` reproduce the pandas nanops algorithms used by tsfresh. They accumulate
central-moment sums in float64, apply pandas' `1e-14` floating-point zeroing rule to the sums, and
then apply the corresponding sample-size corrections. This is required for near-constant
trajectories where a textbook moment formula or a fixed denominator epsilon changes the result.

## Quantile

`Quantile` deduplicates requested `q` values, sorts each trajectory once, and gathers every
linearly interpolated output from that shared ordering. The algorithm matches NumPy quantiles
and avoids the vector-`q` graph break in `torch.quantile`. Since tsfresh invokes this simple
calculator separately for each `q`, sharing the sort removes repeated work.

## VariationCoefficient

`VariationCoefficient` returns `std(x) / mean(x)` and returns NaN when the mean is zero. The mean
remains signed, so trajectories with a negative mean produce the negative value returned by
tsfresh.

## MeanNAbsoluteMax

`MeanNAbsoluteMax` computes one `topk` over absolute values and uses cumulative sums to serve all
requested `number_of_maxima` values. Each output is NaN when `T <= number_of_maxima`, matching
tsfresh's strict valid-length condition independently for every parameter column.

## RatioBeyondRSigma

`RatioBeyondRSigma` computes the trajectory mean and population standard deviation once in
float64, then broadcasts every requested `r` threshold. Float64 reductions limit boundary flips
for values close to `abs(x - mean) == r * std`; the accuracy comparison uses the domain's
documented tolerance for irreducible NumPy-versus-Torch float32 boundary cases.

## SymmetryLooking

`SymmetryLooking` shares one mean, NumPy-compatible median, and range across every requested `r`,
then evaluates `abs(mean - median) < r * range`. This matches tsfresh's combiner implementation,
which also shares those reductions across its parameter list.

## Empty-params fix: brought in line with `autocorrelation`

`Quantile`, `MeanNAbsoluteMax`, `RatioBeyondRSigma`, and `SymmetryLooking` originally handled an
empty `params` list with a `K == 0` early return producing a well-shaped but empty `(B, S, 0)`
tensor. This predated the "Empty and default params" hard rule added to
`references/api-design.md` during the `autocorrelation` domain's own empty-params fix (see
`migration-logs/autocorrelation.md`), and left `statistical` as the one remaining domain still
silently accepting a call no real caller ever makes.

All four classes now match that rule: `params: list[ParamsT] | None = None`, defaulting to a
module-level constant mirroring `TORCH_COMPREHENSIVE_FC_PARAMETERS`'s entry for that feature in
`settings.py` (`_QUANTILE_DEFAULT_PARAMS`, `_MEAN_N_ABSOLUTE_MAX_DEFAULT_PARAMS`,
`_RATIO_BEYOND_R_SIGMA_DEFAULT_PARAMS`, `_SYMMETRY_LOOKING_DEFAULT_PARAMS`) when omitted, and
raising `ValueError` on an explicit empty list instead of returning the empty tensor. The
now-unreachable `K == 0` branches were removed from all four `__call__` methods as dead code.
