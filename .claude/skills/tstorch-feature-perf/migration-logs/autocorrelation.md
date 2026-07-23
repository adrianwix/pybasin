# Autocorrelation domain migration log

Relative to `origin/main`, this branch adds `torch_features_autocorrelation.py` with four
parametrized `Feature` classes and wires those classes into `SUPPORTED_FEATURES`. Each class
accepts the complete requested parameter list, returns one `(B, S, P)` tensor, and owns the
names for those output columns.

## Autocorrelation

`Autocorrelation` implements tsfresh's lag-adjusted normalization. Autocovariance at lag `k`
is divided by `(T - k) * var(x)`, rather than by a full-length denominator. Degenerate variance
produces NaN at every in-range lag, including lag zero, while lags at or beyond `T` produce zero.

The class computes all distinct requested lags together. Sparse, short lag lists use direct
sums over only those lags; larger requests use one FFT autocovariance pass. This preserves the
cost of a single sparse request while sharing the expensive work for multi-parameter calls.
The feature spec leaves `tsfresh_param` unset because tsfresh exposes `autocorrelation` as a
simple calculator whose direct call accepts one `lag` keyword.

## PartialAutocorrelation

`PartialAutocorrelation` computes the adjusted autocorrelation sequence and applies the
Durbin-Levinson recursion through the largest requested lag. The adjusted `(T - k)` denominator
matches `statsmodels.tsa.stattools.pacf(..., method="ld")`, which tsfresh uses as its numerical
reference. The recursion runs in float64 because oscillatory and short trajectories can produce
near-singular systems whose higher-order coefficients are unstable in float32.

The recurrence is sequential by definition, so all requested outputs share one recursion to the
maximum lag and are gathered afterward. Lag zero returns one for a nonempty series; requested
lags outside the computable range return zero.

## AggAutocorrelation

`AggAutocorrelation` shares one adjusted autocorrelation sequence across every requested
`maxlag` and selects `mean`, `median`, or `var` with constructor-precomputed tensor indices.
Small maximum lags use direct sums and larger maximum lags use an FFT. Tests cross-check both
branches through the public class.

A `maxlag` of zero has no autocorrelation values to aggregate and therefore returns NaN. This is
applied independently to each output column, including when zero and nonzero `maxlag` requests
are batched together. Raw counts are retained before clamping so the empty-column condition is
not converted to zero by the reduction machinery.

## AutocorrelationPeriodicity

`AutocorrelationPeriodicity` has no tsfresh equivalent. It groups requests by the structural
`min_lag` and `peak_threshold` parameters, then shares one normalized FFT autocorrelation and
peak-detection pass for the `strength` and `period` outputs in each group.

The shared `local_maxima_1d` utility uses tensor run grouping and `scatter_reduce_` to handle
plateaus without data-dependent Python control flow. This keeps the feature compatible with
`torch.compile(fullgraph=True)` and `torch.export` for batched trajectories.

## Empty-params bug and the switch to reject-and-default

An API-consistency review found that `Autocorrelation` and `PartialAutocorrelation` handled an
empty `params` list with an explicit `K == 0` early return (`(B, S, 0)`), while `AggAutocorrelation`
and `AutocorrelationPeriodicity` had no such guard and crashed instead — `AggAutocorrelation`
with an `IndexError` from `torch.tensor([], ...)` collapsing to the wrong rank, `AutocorrelationPeriodicity`
with a `ValueError` from `torch.cat([])` on an empty column list. Same `Feature.__call__` contract,
two different outcomes for the identical input depending on which of the four classes handled it.

Investigating *why* a caller would ever construct one of these with `params=[]` turned up no real
answer: no `FCParameters` preset in `settings.py` does it, tsfresh's own `ComprehensiveFCParameters`
always defines a non-empty default for all three of its equivalents here (`autocorrelation`,
`partial_autocorrelation`, `agg_autocorrelation` — `AutocorrelationPeriodicity` has no tsfresh
equivalent but mirrors the same shape via `DYNAMICAL_SYSTEM_FC_PARAMETERS`), and no test in this
codebase ever did either. Silently returning a well-shaped but empty tensor for `params=[]`
would have papered over a real caller bug (e.g. an upstream filter/dedup step that should have
produced at least one param) rather than surfacing it.

Given that, all four classes now reject an explicit empty `params` list with a `ValueError`
instead of unifying on the empty-tensor return, and `params` became optional
(`list[ParamsT] | None = None`), defaulting to a module-level constant mirroring the relevant
`settings.py`/tsfresh default when omitted (`_AUTOCORRELATION_DEFAULT_PARAMS`,
`_PARTIAL_AUTOCORRELATION_DEFAULT_PARAMS`, `_AGG_AUTOCORRELATION_DEFAULT_PARAMS`,
`_AUTOCORRELATION_PERIODICITY_DEFAULT_PARAMS`). This removed the now-unreachable `K == 0`
branches from `Autocorrelation.__call__`/`PartialAutocorrelation.__call__` as dead code, and the
canonical `Autocorrelation` reference example in `references/api-design.md` was updated to match
(see that file's new "Empty and default params" section, which states this as a hard rule for
every parametrized feature; `statistical`'s four parametrized features were brought in line with
it too, see `migration-logs/statistical.md`).
