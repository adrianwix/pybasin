# tstorch feature API design

This is a design specification, not a changelog: it describes the target shape every
tstorch feature is built to, and stays that way until the shape itself changes. Bugs found
and feature-specific history belong in `migration-logs/<domain>.md`, never here. How to
actually carry out a migration to this shape — steps, order, verification — is SKILL.md's
job, not this file's.

## API design

### The two public APIs

Every feature is reachable exactly two ways, and both are public:

1. **Direct call** — construct the feature's class with its params, then call it with data:
   `Feature(params)(x)`. What `ALL_FEATURE_FUNCTIONS` (`tests/benchmarks/direct_call_registry.py`),
   single-feature benchmarks, and any other external caller use.
2. **`TorchFeatureExtractor`** — configured via `FCParameters` (`{"feature": [...]}`), resolved
   through `SUPPORTED_FEATURES["feature"]`, the same class constructed the same way.

These are not two features that happen to compute similar things — they are two doors into the
exact same class, so calling `Feature(params)(x)` directly and configuring
`TorchFeatureExtractor` with only that feature and those same `params` must be indistinguishable:

- **Same feature names.** The output-name suffix each param produces (`.names`) is identical
  whichever door you came through — there is only one naming implementation, on the class
  itself, never two schemes that happen to agree.
- **Same arguments.** Both take the identical, fully-typed `params: list[ParamsT]` shape (see
  "The `Feature`/`ParameterlessFeature` base classes" below) — never a narrower single-param
  convenience signature on the direct-call side hiding what the class actually supports.
- **Same numerical result, same shape modulo stacking.** `Feature(params)(x)` and
  `TorchFeatureExtractor` configured with only that feature must produce bit-identical output.
  The *only* difference in what you get back is that the extractor concatenates every
  *configured* feature's `F` columns into one `(B, F_total)` tensor — that's aggregation across
  features, not a difference in how any single feature computes.

### The `Feature`/`ParameterlessFeature` base classes

One class per feature — no separate public/internal function pair, no factory of loose
callables. Constructing it *is* "building the kernel"; calling it *is* running the kernel.
Every feature class explicitly inherits one of two base classes, chosen by whether it takes
parameters at all — never left to satisfy either structurally without inheriting:

```python
from abc import ABC, abstractmethod
from collections.abc import Mapping
from torch import Tensor


class Feature[ParamsT: Mapping[str, object]](ABC):
    """Base class every parametrized feature class inherits, e.g.
    `class Quantile(Feature[QuantileParams]):`."""

    @abstractmethod
    def __init__(self, params: list[ParamsT]) -> None: ...

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """x: (B, S, T). Returns (B, S, len(params))."""

    @property
    @abstractmethod
    def names(self) -> list[str]:
        """
        Returns:
            One complete output name per ``params[i]``, in the same order.
        """


class ParameterlessFeature(ABC):
    """Base class every parameter-free feature class inherits, e.g.
    `class Median(ParameterlessFeature):`. Same shape as `Feature` above, minus the
    `params` argument there is nothing to pass."""

    @abstractmethod
    def __init__(self) -> None: ...

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """x: (B, S, T). Returns (B, S, 1)."""

    @property
    @abstractmethod
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing the feature's own name.
        """
```

Both are real `ABC` base classes (not `Protocol`s): `TorchFeatureExtractor` and the shared test
utilities dispatch on nominal `issubclass(feature_cls, ParameterlessFeature)`
(`tstorch.feature_factory.is_parameterless`) to pick the right constructor call, and a
structural/duck-typed check couldn't tell the two apart - both declare identically-named members
(`__init__`, `__call__`, `names`). `@abstractmethod` also means a feature class that forgets to
implement one of these members fails at instantiation, not silently at first call.

No `nn.Module` inheritance — measured to carry no performance benefit over a bare class for
this use (see `packages/tstorch/benchmark_feature_extract_api_design.ipynb`, "API design
comparison" section) while adding hook/training-mode/buffer-registration machinery this design
has no use for. `torch.compile` traces a bare class's `__call__` exactly as it would a closure's.

Every constructor argument is a plain Python object closed over by `self` — never mutated after
`__init__` — so Dynamo treats it as a compile-time constant, the same role a closure's default
argument played before. Neither base class supplies any shared executable behavior - each
concrete feature class implements `__init__`/`__call__`/`.names` itself; inheriting `Feature`/
`ParameterlessFeature` is purely a typing/dispatch signal, never a template method to hide the
kernel behind.

**Hard rule: the computational kernel goes in `__call__`, always.** No `_compute` method, no
template-method base class, no per-class indirection between `__call__` and the tensor
computation. A private module-level function (e.g. `_median`) is the only allowed form of
sharing, and only when the same logic is genuinely reused by more than one feature class.

Calling a registered low-level PyTorch operator directly from `__call__` is not per-class
indirection. It is the same boundary as calling an ATen operator. Use this boundary only for an
algorithm that ordinary traceable tensor operations cannot express efficiently. The operator
must be functional, provide CPU and accelerator dispatch implementations, register a FakeTensor
kernel, and pass `torch.library.opcheck` together with the feature's compile and export tests.
Feature-specific tensor preparation and result shaping stay in `__call__`; do not insert a
Python wrapper between the feature and the registered operator.

**Constraint: `__call__` must trace with zero graph breaks under both `torch.compile(fullgraph=True)`
and `torch.export.export()`.** `torch.export` requires an `nn.Module` at the top level — wrap the
instance in a thin, parameter/buffer/hook-free `nn.Module` shim for that call only; never make
`Feature` itself an `nn.Module`. SKILL.md has the test procedure that checks this per feature.

**Hard rule: no numpy, anywhere in `__call__`.** No `.numpy()`, no `torch.from_numpy()`, no numpy
round-trip of any kind — `torch.export.export()` raises `.numpy() is not supported for tensor
subclasses` on every one of them, no exceptions. This holds even when a numpy round-trip
measures faster than the pure-torch equivalent (confirmed for `torch.sort` at small batch
sizes — see `references/performance-notes.md`): export-compatibility always wins over a CPU
small-batch speed advantage. Reworking a feature to remove an existing numpy fast path is real,
required work, not something to defer alongside it just because it's currently fast. A
registered operator may use backend-native code internally; the feature's `__call__` still
contains no NumPy boundary and remains visible to PyTorch as one traceable operator call.

### Empty and default params

**Hard rule: `Feature.__init__` must reject an explicit empty `params` list, never silently
return a `(B, S, 0)` tensor for it.** No real caller — not `TorchFeatureExtractor` (every
`FCParameters` preset in `settings.py` gives each parametrized feature it enables a non-empty
param list), not tsfresh's own `ComprehensiveFCParameters` (it defines a non-empty default for
every parametrized feature; disabling a feature means omitting its key, not mapping it to `[]`)
— ever constructs a `Feature` with zero requested outputs. Treating `params=[]` as a valid,
silently-successful input only hides a caller bug (an empty list arriving from a filter/dedup
step that was supposed to produce at least one entry) behind a well-shaped but empty tensor.
Raise `ValueError` from `__init__` instead, naming the feature and pointing at the default:

```python
def __init__(self, params: list[AutocorrelationParams] | None = None) -> None:
    if params is None:
        params = list(_AUTOCORRELATION_DEFAULT_PARAMS)
    elif not params:
        raise ValueError(
            "Autocorrelation requires at least one param; omit `params` to use the "
            f"default {_AUTOCORRELATION_DEFAULT_PARAMS!r} instead of passing []."
        )
    self.params: list[AutocorrelationParams] = params
```

`params: list[ParamsT] | None = None` — optional, defaulting to a module-level constant that
mirrors tsfresh's own default for that feature (pull it from `settings.py`'s
`TORCH_COMPREHENSIVE_FC_PARAMETERS`/`DYNAMICAL_SYSTEM_FC_PARAMETERS`, or tsfresh's
`ComprehensiveFCParameters()` directly for a feature not yet present in `settings.py`) — is a
convenience for the direct-call door only; `TorchFeatureExtractor` always passes an explicit,
non-empty list from its own config either way, so this never creates divergent behavior between
the two doors (see "Hard rule: the two doors must be symmetric" below). Once `__init__`
guarantees `len(self.params) >= 1`, `__call__` needs no `K == 0` branch — delete one if a feature
predates this rule instead of leaving unreachable dead code.

### Where the documentation goes

**The class docstring is the algorithmic documentation** — formula, semantics, edge cases —
because the class, not any one method on it, is the public contract both doors share (see "The
two public APIs" above). It goes on the class itself, not on `__init__` or `__call__`: a reader
(or `help(Autocorrelation)`) should find it by looking at the one name they imported, not by
guessing which method carries it.

Use the same docstring layout in every concrete feature class:

- The class docstring starts with a one-line feature summary, follows with its formula,
  semantics, and edge cases, then uses `Args:` for constructor parameters and `References:` for
  citations. Omit `Args:` for a `ParameterlessFeature`.
- `__init__` has no docstring because the class-level `Args:` section owns its public
  parameters.
- `__call__` starts directly with `Args:` and ends with `Returns:`. Do not add a summary line
  that repeats the class description.
- The first content in `.names` is `Returns:`; see "Naming" below for what it must document. Put
  the opening triple quotes on their own line so Ruff preserves the Google-section indentation.
  Do not add “Output column names” or another summary that merely repeats the property name and
  `Returns:` section.

**Google style, not Sphinx/rST.** `Args:`, `Returns:`, `References:` sections; math in
Markdown/MathJax (`\(...\)` / `\[...\]`) — never Sphinx/rST directives (`:param:`, `:return:`,
`:math:`, `.. math::`, `.. rubric::`). Cover formula/semantics/edge cases, not a changelog of
what was tried — performance rationale belongs in `references/performance-notes.md`, a one-line
pointer from the docstring is fine, a paragraph is not. Document the params `TypedDict` with
Google-style `Attributes:`.

The kernel itself lives in `__call__` (see "The `Feature`/`ParameterlessFeature` base classes"
above). Tests must exercise the public class API (`Feature(params)(x)`), not a private
module-level helper directly. Shared helpers carry implementation-facing comments only; the
class docstring remains the public algorithmic documentation.

### Naming

Every feature class is a self-contained extraction framework in its own right - a direct caller
never has to go through `TorchFeatureExtractor` to get a usable result, so `.names` must be
usable on its own too. `.names` therefore returns one complete, self-sufficient output name per
requested param — e.g. `"autocorrelation__lag_5"`, `"median"` — never a bare per-param suffix and
never `""`. The feature's own identity is part of the name regardless of whether it takes
parameters; a parameter-free feature's `.names` is just that identity (`["median"]`), a
parametrized feature's is that identity plus its per-param suffix (`["autocorrelation__lag_5"]`).

The property's `Returns:` text must make that string structure explicit. For a parametrized
feature, give the complete format with angle-bracket placeholders, such as
``autocorrelation__lag_<lag>``; for a parameterless feature, give the exact literal, such as
``median``. Never replace either with a generic description such as “this feature's own name.”
The `Autocorrelation` and `Median` examples below show both forms in their complete class
context.

`TorchFeatureExtractor` does nothing more than prefix a state index onto whatever `.names`
already gives it — `state_{i}__{name}`, uniformly, with no branching on class kind and no
knowledge of params or suffixes. This format is a load-bearing contract with downstream consumers
(`pybasin.feature_extractors.utils.parse_feature_name`/`get_feature_indices_by_base_name`,
`DynamicalSystemClusterer.REQUIRED_FEATURES`), not a tstorch-internal convention.

Distinct params must produce distinct names — a param that silently fails to change `.names`
would make two different requested columns collide under the same name.

Deduping identical param requests across states (so a Type B feature's shared computation is
requested once, not once per state) needs no feature-specific code at all: a single
feature-agnostic function over the raw param dict (canonical repr, sorted by key) is enough for
every feature that exists today — this is not part of the `Feature` protocol.

### Hard rule: the two doors must be symmetric

Never ship a feature class that (a) computes its params via a different path depending on which
door constructed it — impossible by construction here, since there is only one `__init__`/
`__call__` — or (b) exposes a narrower calling convention than its `__call__` actually supports,
e.g. accepting only `list[ParamsT]` of length 1 in practice when the underlying computation
genuinely shares work across many.

`autocorrelation` shipped both variants of this bug under the old function-pair design (a
standalone public implementation that quietly diverged from the shared kernel at `lag=0`, then
— even after unifying — a public signature that still only took one lag at a time even though
the kernel shared a pass across many lags). See `migration-logs/autocorrelation.md` for the full
story. The one-class design in this document doesn't just discourage that bug, it makes it
harder to write in the first place: there is nowhere for a second implementation to live.

## Example

`Autocorrelation` — parametrized, one lag or many in a single call, `torch.compile`-safe end to
end.

```python
from typing import TypedDict

import torch
from torch import Tensor
from tstorch.feature_factory import Feature


class AutocorrelationParams(TypedDict):
    """Parameters for one `Autocorrelation` output column.

    Attributes:
        lag: Lag index to evaluate.
    """

    lag: int


# Matches tsfresh's `ComprehensiveFCParameters` default for `autocorrelation`.
_AUTOCORRELATION_DEFAULT_PARAMS: list[AutocorrelationParams] = [{"lag": lag} for lag in range(10)]


class Autocorrelation(Feature[AutocorrelationParams]):
    r"""Autocorrelation at every requested lag.

    \[
    r_k = \frac{\sum_{t=1}^{n-k} (x_t - \bar{x})(x_{t+k} - \bar{x})}{(n - k)\,\mathrm{Var}(x)}
    \]

    the adjusted estimator (matches tsfresh's ``autocorrelation``). Lag 0 is *not*
    unconditionally 1: for a degenerate series, the result is NaN at every lag including 0.

    Args:
        params: One dict per requested output, each with a ``lag`` key. May repeat a
            lag or include 0. Defaults to tsfresh's own default (lags 0-9) when omitted; an
            explicit empty list is rejected (see "Empty and default params" above).

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[AutocorrelationParams] | None = None) -> None:
        if params is None:
            params = list(_AUTOCORRELATION_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "Autocorrelation requires at least one param; omit `params` to use the "
                f"default {_AUTOCORRELATION_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[AutocorrelationParams] = params
        self._lags: list[int] = [p["lag"] for p in params]

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        lags = self._lags
        K = len(lags)  # always >= 1, guaranteed by __init__

        x_centered = x - x.mean(dim=2, keepdim=True)
        result = ...  # compute every requested lag from the same centered series
        return result

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in the same order as ``params``. Format: ``autocorrelation__lag_<lag>``.
        """
        return [f"autocorrelation__lag_{p['lag']}" for p in self.params]
```

`Median` — parameter-free, `ParameterlessFeature` instead of `Feature[ParamsT]`:

```python
class Median(ParameterlessFeature):
    """Median of the time series along the time dimension.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1).
        """
        return _median(x)  # shared sort+midpoint-average kernel

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``median``.
        """
        return ["median"]
```

### Direct call

```python
feature = Autocorrelation([{"lag": 1}, {"lag": 5}])   # type checker verifies each dict has "lag": int
y: Tensor = feature(x)                                  # (B, S, 2)
names: list[str] = feature.names                        # ["autocorrelation__lag_1", "autocorrelation__lag_5"]

compiled = torch.compile(feature, fullgraph=True)        # plain callable object, no special support needed
y = compiled(x)

median_feature = Median()                                # ParameterlessFeature: no params argument at all
y2: Tensor = median_feature(x)                            # (B, S, 1)
median_feature.names                                      # ["median"] - already complete, on its own
```

### `TorchFeatureExtractor` integration

```python
from tstorch.feature_factory import ParameterlessFeature, construct_feature, is_parameterless

SUPPORTED_FEATURES: dict[str, type[Feature[Any]] | type[ParameterlessFeature]] = {
    "autocorrelation": Autocorrelation,
    "median": Median,
    ...
}

# inside _compile(): construct_feature picks feature_cls() vs feature_cls(params_list) -
# is_parameterless(feature_cls) is the dispatch signal, nominal issubclass(_, ParameterlessFeature).
# Naming needs no such branch at all: .names is already complete, so the extractor only adds the
# state prefix, identically regardless of which base class produced the instance.
instances = [
    construct_feature(feature_cls, None if is_parameterless(feature_cls) else params_list)
    for feature_cls, params_list in ...
]
output_names = [f"state_{state_idx}__{name}" for name in instance.names]

def kernel_eager(y: Tensor, instances: tuple = tuple(instances)) -> Tensor:
    return torch.cat([f(y) for f in instances], dim=-1)
```

### Typing at the `TorchFeatureExtractor` config boundary

`Autocorrelation.__init__`/`__call__` are fully typed — a caller building `Autocorrelation(...)`
directly gets full type-checker coverage of `params`. The `FCParameters` config dict
`TorchFeatureExtractor` accepts (`Mapping[str, list[dict[str, object]] | None]`) cannot be typed
that precisely at the *mapping* level — Python's type system has no way to say "the value type
for key `"autocorrelation"` is `list[AutocorrelationParams]`, but for key `"quantile"` it's
`list[QuantileParams]`" within one homogeneous `Mapping` type, short of one overload per feature.
This is a real, stated limitation, not an oversight: a caller who wants full type-checking on
their config should import the specific `*Params` `TypedDict`s they're using
(`AutocorrelationParams(lag=1)`) when building the list passed under each key, even though the
surrounding dict's own declared type stays generic. Every feature's params are still checked at
the point they actually matter — `construct_feature(feature_cls, params_list)`, inside
`_compile()` — which is where a malformed param would otherwise cause a real bug.

## Decision tree: is this feature batchable?

Ignore what tsfresh's API shape suggests — `fctype=combiner` (called once with a list of
params) tells you nothing about whether the *computation* is shared. Some combiners genuinely
share one pass across every param (e.g. `agg_autocorrelation`: one ACF, read at several lags);
others just loop `func(param)` internally with no shared work, and the two look identical from
the outside. tsfresh's own behavior isn't the question we're answering anyway — we care about
making tstorch fast, not about matching tsfresh's internal structure.

**Never do what tsfresh itself sometimes does: wrap N independent per-param calls behind one
`__call__` and call it "batched."** `torch.stack([func(x, p) for p in params])` is not batching
— it's the exact same O(N) work as N separate calls, just hidden behind one method so the
`Feature` protocol looks satisfied. That's decoration, not a performance win, and it's the
specific anti-pattern this design exists to avoid.

The actual test, every time: **can the single-param computation be rewritten to accept many
params in one call, at no performance cost to the single-param case?** Check this against the
math itself (not tsfresh's `fctype`) — every parametrized feature migrated so far has landed in
one of these two, and neither requires a Python-level loop over params inside `__call__`:

- **Shared computation** — one FFT/ACF/sort/reduction computed once lets every param read a
  different slice/lag/threshold of that same intermediate. `Autocorrelation` above is the
  worked example: `Autocorrelation.__call__` computes one autocovariance pass, read off at every
  requested lag.
- **Broadcast** — no shared intermediate, but every param's independent comparison/threshold
  still broadcasts into one vectorized op against `x`: no shared FLOPs, but also no per-param
  Python loop (`torch_features_count.py`'s `count_above` family: `.view(1, 1, 1, -1)` against
  every requested `t` at once).

**If neither actually applies — no way to vectorize the parameter axis after genuinely trying —
stop and flag it explicitly instead of writing the N-independent-calls loop and moving on.**
Tell the user which two options were checked and why both failed, and get confirmation before
shipping a `__call__` that internally loops per param — this should be a rare, visible
exception, not a quiet third category. Record the reasoning in `migration-logs/<domain>.md`
either way. No feature in this codebase has needed this fallback yet, which is itself a strong
prior that a shared-computation or broadcast form exists if you look hard enough.

Note the decision and why in a comment near `__call__` — a one-line note, not a class-naming
convention as the only signal.
