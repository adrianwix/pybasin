# Location domain migration

## Public API

The domain exposes five feature classes. `FirstLocationOfMaximum`,
`FirstLocationOfMinimum`, `LastLocationOfMaximum`, and `LastLocationOfMinimum`
inherit `ParameterlessFeature`; `IndexMassQuantile` inherits
`Feature[IndexMassQuantileParams]`. Each class owns its calculation, complete output
names, and Google-style algorithm documentation.

All five classes are registered directly in `TorchFeatureExtractor.SUPPORTED_FEATURES`.
The old public functions, NumPy reduction dispatch, deprecated factories, and factory
builders are absent from the final module. Direct calls and extractor calls therefore
construct and execute the same class.

## Extremum locations

The first-location features use `argmax` or `argmin` along the contiguous time axis and
divide the zero-based index by the series length. Ties resolve to the earliest matching
position.

For the last-location pair, the reduction runs on a reversed time axis. Their result is
`1 - index_from_end / T`, matching tsfresh's one-based relative position. In particular,
a unique extremum at zero-based position `i` produces `(i + 1) / T`; using `i / T` here
would be an off-by-one error. Ties resolve to the last matching position in the original
series. All four classes return NaN for an empty time axis.

## Index mass quantile

`IndexMassQuantile` accepts one or more `{"q": float}` parameter dictionaries. Its
default parameter list matches the comprehensive tstorch settings, while an explicit
empty list raises `ValueError`. The kernel computes absolute cumulative mass once,
broadcasts every requested threshold over a parameter axis, and uses `torch.searchsorted`
to locate each first crossing without a `(B, S, T, K)` boolean intermediate or a Python
loop over parameters.

All-zero and empty series return NaN. For a nonzero series, `q=1.0` resolves to the final
sample because the total comes from the same cumulative sum used for threshold
crossings. NumPy computes its total with a separate reduction; at float32 precision,
that independent sum can land on either side of the cumulative sum and make tsfresh
return `1 / T` instead. Reproducing that unstable artifact would require a NumPy
round-trip, which is incompatible with `torch.export`, so the class keeps the stable
cumulative-mass invariant. Away from `q=1.0`, the canonical comparison covers eight
quantiles; the observed boundary discrepancy is at most one time step and is tested
with an absolute tolerance that reflects that discrete shift.

## Verification and integration

The domain unit tests compare all five features with tsfresh across the cached canonical
trajectories, validate names and output shapes, cover tie and zero-mass behavior, and
check the empty parameter rejection. Every class also has
`torch.compile(fullgraph=True)` and `torch.export` coverage. A mixed composed kernel
containing location and statistical features traces as one graph with no graph breaks.

The benchmark module implements the complete eight-row inventory. Its batched-argument
rows exercise one, five, and ten `IndexMassQuantile` parameters; benchmark metadata
records that tsfresh's combiner also shares its cumulative-mass calculation. Current
accuracy, compile/export, and performance results live in
`packages/tstorch/reports/report.md`.

The API page documents each feature beside its own schema, and `mkdocs build --strict`
validates the rendered page.
