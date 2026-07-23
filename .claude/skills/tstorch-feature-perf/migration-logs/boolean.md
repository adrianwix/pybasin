# Boolean domain migration log

Relative to `origin/main`, this branch adds `torch_features_boolean.py` with five feature
classes and wires the complete boolean domain into `SUPPORTED_FEATURES`.

## Duplicate features

`HasDuplicate`, `HasDuplicateMax`, and `HasDuplicateMin` are parameterless classes. They return
one float output column containing zero or one for every `(B, S)` trajectory. `HasDuplicate`
sorts each trajectory once and checks adjacent values. The maximum and minimum variants count
exact matches to the corresponding extremum.

The public API consists of the feature classes. Direct callers and `TorchFeatureExtractor`
construct the same class and receive the same `(B, S, 1)` output contract.

## HasVarianceLargerThanStandardDeviation

`HasVarianceLargerThanStandardDeviation` evaluates the tsfresh condition with population
standard deviation as `std(x) > 1`. For nonnegative standard deviation this is algebraically
equivalent to `var(x) > std(x)`. Its feature spec maps the tstorch name
`has_variance_larger_than_standard_deviation` to tsfresh's
`variance_larger_than_standard_deviation`, ensuring the numerical test executes the reference
calculator.

## LargeStandardDeviation

`LargeStandardDeviation` accepts all requested `r` values in one constructor call. Its kernel
computes population standard deviation and the `max(x) - min(x)` range once, broadcasts both
against the parameter tensor, and returns `std(x) > r * range(x)` for every output column.
This shares the trajectory reductions that tsfresh's simple calculator repeats for each `r`.

## Dtype-preservation fix

An API-consistency review found all five features ending their `__call__` with `.float()`,
hardcasting the output to `torch.float32` regardless of the input tensor's dtype — the only
calculator file in the codebase doing so; every other migrated domain (`statistical`, `count`,
`location`, etc.) uses `.to(x.dtype)` to preserve the caller's precision. Since
`TorchFeatureExtractor._compile()` concatenates every requested feature's output with a single
`torch.cat`, a config mixing a boolean feature with a non-float32 pipeline (float64 trajectories,
or float16/bfloat16 on GPU) would either fail at `torch.cat` or silently downcast every other
feature's output. All five `.float()` calls were replaced with `.to(x.dtype)` to match the
established convention: the caller's input dtype decides the output dtype, not the feature.
