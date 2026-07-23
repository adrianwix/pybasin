# Change domain migration log

Relative to `origin/main`, this branch adds `torch_features_change.py` with the complete set of
five change-domain feature classes and wires them into `SUPPORTED_FEATURES`.

## Parameterless change features

`AbsoluteSumOfChanges` sums absolute consecutive differences. `MeanAbsChange` divides that sum
by `T - 1`. `MeanChange` uses the equivalent endpoint expression
`(x[T - 1] - x[0]) / (T - 1)`. `MeanSecondDerivativeCentral` uses the endpoint form of the mean
central second derivative. The classes preserve tsfresh's short-series behavior: the empty
absolute-change sum is zero, while means that require unavailable differences are NaN.

All four implementations use Torch tensor reductions throughout, so their direct class calls
remain traceable by `torch.compile(fullgraph=True)` and exportable by `torch.export`.

## ChangeQuantiles

`ChangeQuantiles` accepts the full list of `ql`, `qh`, `isabs`, and `f_agg` combinations. The
constructor deduplicates quantile values and stores tensor indices for every requested output.
The call sorts each trajectory once, evaluates every distinct quantile with linear
interpolation, and broadcasts corridor masks across the parameter axis. Selected consecutive
differences are aggregated as either a mean or population variance.

The sort-and-interpolate quantile kernel matches NumPy's default linear quantile semantics and
avoids the data-dependent operation used by vector-valued `torch.quantile`, keeping the complete
multi-parameter feature compatible with full-graph compilation. Requests with `ql >= qh`, or
with no consecutive pair inside the corridor, return zero independently for each output column.
