# Advanced domain migration log

The advanced calculator domain contributes four public feature classes:
`BenfordCorrelation`, `C3`, `EnergyRatioByChunks`, and
`TimeReversalAsymmetryStatistic`. Direct callers and `TorchFeatureExtractor` construct the
same classes, so parameter handling, output names, numerical behavior, and tensor shapes do
not depend on the entry point. The deprecated factory registry and its public function facades
are absent from the final module.

Each parametrized class accepts a non-empty list of its own `TypedDict`, defaults to the
corresponding comprehensive settings when the argument is omitted, and returns one `(B, S, F)`
tensor. An explicit empty list raises `ValueError`. Parameterless `BenfordCorrelation` follows
the same naming contract with its single `benford_correlation` column.

## BenfordCorrelation

The implementation builds a nine-bin leading-digit distribution from `abs(nan_to_num(x))` and
correlates it with Benford's expected frequencies. Observed frequencies are divided by the full
series length, matching tsfresh: zeros and NaNs occupy no digit bin but still dilute the
distribution. Positive and negative infinities are replaced by the dtype's largest finite
magnitude before digit extraction, again following `nan_to_num` semantics.

Two earlier differences from tsfresh were removed. The port had normalized by the count of
positive entries and required a minimum number of valid values; tsfresh does neither. The final
formula also has no epsilon in the correlation denominator, because an epsilon turned an
undefined constant-distribution correlation into zero instead of NaN. A dedicated regression
test covers that case.

## C3 and TimeReversalAsymmetryStatistic

Both lagged features preserve tsfresh's valid range `t + 2 * lag < T`. `C3` averages the triple
product `x[t] * x[t + lag] * x[t + 2 * lag]`, while
`TimeReversalAsymmetryStatistic` averages
`x[t + 2 * lag]**2 * x[t + lag] - x[t + lag] * x[t]**2`. A lag with no valid triple returns
zero. Lag zero is valid and is covered by the numerical test matrix.

The batching decision is broadcast, not a hidden loop of independent calls. Several lags use
one four-dimensional gather with a validity mask; out-of-range indices are clamped only to keep
the gather legal, then masked before reduction. The common single-lag case takes direct
contiguous slices instead, avoiding the gather setup without creating a second public
implementation. This keeps the one-parameter benchmark fast while preserving the batched API.

tsfresh exposes both calculators as `fctype="simple"` and invokes them once per lag. The tstorch
classes therefore gain a real argument-batching path that tsfresh's proxy does not provide.

## EnergyRatioByChunks

`EnergyRatioByChunks` matches `numpy.array_split`: the remainder is assigned one element at a
time to the earliest chunks. No tail is discarded. An all-zero series returns NaN because its
energy ratio is undefined; the implementation does not hide that edge case with an epsilon.

`num_segments` is structural because it fixes chunk boundaries, while `segment_focus` selects
an output column. Construction groups focuses by `num_segments`. At call time, each group uses
one squared cumulative sum and two prefix lookups per requested focus, then the results are
restored to the caller's original parameter order. This shares the expensive pass across all
focuses in one partition and still supports several structural partitions in a single feature
instance.

The chunk-size calculation uses integer `//` and `%` operations separately. Python's
`divmod()` produced the same eager result but rejected Dynamo's symbolic time length under the
extractor's `dynamic=True, fullgraph=True` compile path. A dynamic full-graph regression test
now covers that constraint in addition to the fixed-shape compile and export checks.

## Verification and integration

The domain is registered directly in `SUPPORTED_FEATURES`. Its unit suite checks all four
classes against tsfresh across the canonical trajectory set, exercises eight distinct parameter
combinations for every parametrized feature, validates names, rejects empty parameter lists,
and records both `torch.compile(fullgraph=True)` and `torch.export` safety. A mixed extractor
kernel containing an advanced and an existing statistical feature produces one graph with no
breaks.

The benchmark module covers the complete canonical inventory, including compiled direct calls
and the 1/5/10-argument sweeps. All required CPU rows beat their tsfresh comparisons. Current
precision, compile/export status, and performance measurements live in
`packages/tstorch/reports/report.md`.
