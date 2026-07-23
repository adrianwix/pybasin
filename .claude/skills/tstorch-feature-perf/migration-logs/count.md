# Count migration

The count domain now exposes six feature classes through both direct construction and
`TorchFeatureExtractor`. The old public functions, build closures, and
`COUNT_FEATURE_FACTORIES` registry are gone. Each class owns its numerical kernel and complete
output names, while the four parametrized classes use a dedicated `TypedDict` and reject an
explicit empty parameter list.

`CountAbove` and `CountBelow` return the fraction of samples meeting inclusive `>= t` and
`<= t` comparisons. They preserve tsfresh's boundary behavior, including a NaN threshold
matching no samples. Both broadcast all requested thresholds against the `(B, S, T)` input in
one comparison instead of issuing one Python call per threshold.

`CountAboveMean` and `CountBelowMean` count strict comparisons against each series mean. Samples
equal to the mean are excluded. A series containing NaN has a NaN mean, so every comparison is
false and the result is zero, consistent with tsfresh.

`CountInRange` counts samples in each half-open interval `[min_val, max_val)`. Broadcasting the
lower and upper bounds covers all intervals in one tensor expression, including empty and
reversed intervals, which naturally produce zero. Its direct API keeps tstorch's descriptive
`min_val` and `max_val` keys; accuracy and performance adapters translate those to tsfresh's
`min` and `max` names at the external boundary.

`CountValue` broadcasts equality checks for every requested target. NaN is handled explicitly:
a NaN target counts NaN samples rather than relying on ordinary equality, which would always be
false. The unconditional tensor selection keeps the same path safe for full-graph compilation
and export.

Default parameter lists mirror tsfresh's comprehensive settings: threshold zero for
`CountAbove` and `CountBelow`, the three standard intervals for `CountInRange`, and target values
zero, one, and minus one for `CountValue`. Every calculator returns `(B, S, F)` and moves its
closed-over parameter tensors to the input dtype and device at call time.

The domain test suite covers at least eight parameter choices per parametrized feature,
inclusive and exclusive boundaries, NaN behavior, empty-parameter rejection, name stability,
CPU/GPU agreement, numerical parity with tsfresh, `torch.compile(fullgraph=True)`, and
`torch.export`. The extractor registry contains all six classes, and a mixed composed kernel was
checked as one graph with zero graph breaks.

The benchmark suite implements the full direct/proxy, compiled/uncompiled, cold-run, and
batched-argument inventory. Raw tsfresh inspection confirmed that all four parametrized
calculators are `fctype="simple"` and loop once per argument in the proxy dispatcher; tstorch's
broadcast parameter axis is therefore a real batching improvement. Current correctness,
compile/export, and timing results live in `packages/tstorch/reports/report.md`.
