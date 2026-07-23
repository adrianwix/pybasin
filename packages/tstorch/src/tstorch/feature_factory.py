# pyright: basic
"""The `Feature`/`ParameterlessFeature` base classes every feature class inherits."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, NamedTuple, cast

from torch import Tensor


def param_key(params: Mapping[str, Any]) -> Hashable:
    """Stable, value-based key for one parameter dict (not identity-based - two states
    requesting the same param values must dedup to one shared column even if they built
    separate dict instances). Used directly by `TorchFeatureExtractor._compile()`;
    override only if a param dict contains something that reprs unstably (e.g. a tensor)."""
    return repr(tuple(sorted((key, repr(value)) for key, value in params.items())))


class DEPRECATED_FeatureFactory(NamedTuple):
    """`build` receives the distinct param dicts requested for this feature anywhere in the
    extractor's config (deduped via `param_key`, first-seen order) and returns ONE closure
    computing all of them in a single shared call: `(B, S, T) -> (B, S, len(params))`. Every
    distinct param combo requested by *any* state goes through the same call, not one call
    per state; a state gets every requested combo's column computed for it, including
    combos only *other* states asked for (masked out via plain indexing afterward).

    `param_key`/`name` operate on one param dict at a time.
    """

    build: Callable[[list[dict[str, Any]]], Callable[[Tensor], Tensor]]
    param_key: Callable[[dict[str, Any]], Hashable]
    name: Callable[[dict[str, Any]], str]


def DEPRECATED_parameter_free_factory(fn: Callable[[Tensor], Tensor]) -> DEPRECATED_FeatureFactory:
    """Wrap a parameter-free `(B, S, T) -> (B, S, 1)` function as a trivial
    `DEPRECATED_FeatureFactory` (single implicit `{}` param, empty name suffix)."""

    def build(params: list[dict[str, Any]]) -> Callable[[Tensor], Tensor]:
        assert params == [{}], f"parameter-free feature got unexpected params: {params}"
        return fn

    return DEPRECATED_FeatureFactory(build=build, param_key=param_key, name=lambda params: "")


class Feature[ParamsT: Mapping[str, object]](ABC):
    """Base class every parametrized feature class inherits (e.g. `class
    Quantile(Feature[QuantileParams]):`). Direct calls and `TorchFeatureExtractor` are two
    doors into the same concrete class - `Feature(params)` constructs it, `feature(x)` runs
    it, `.names` gives the complete per-param output names - so there is exactly one
    implementation per feature, not a `feature()`/`_feature()` pair. Each class is a
    self-contained extraction framework in its own right, usable with no other part of
    tstorch involved - `TorchFeatureExtractor` only ever prefixes a state index onto
    whatever `.names` already gives it.

    `ParamsT` is a per-feature `TypedDict` (e.g. `AutocorrelationParams`), giving a caller
    (human or type checker) full visibility into a feature's arguments without reading its
    implementation. A feature that takes no parameters at all inherits `ParameterlessFeature`
    below instead - its `__init__` takes no `params` argument, so it cannot satisfy this
    base class's signature.
    """

    @abstractmethod
    def __init__(self, params: list[ParamsT]) -> None: ...

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """x: (B, S, T). Returns (B, S, len(params))."""

    @property
    @abstractmethod
    def names(self) -> list[str]:
        """One complete, self-sufficient output name per params[i], in the same order -
        e.g. "autocorrelation__lag_5", including the feature's own identity, not just the
        per-param suffix. `TorchFeatureExtractor` prefixes a state index onto this and
        nothing else."""


class ParameterlessFeature(ABC):
    """Base class every parameter-free feature class inherits (e.g. `class
    Median(ParameterlessFeature):`). Same two-doors relationship as `Feature` above, minus
    the `params` argument there is nothing to pass: `ParameterlessFeature()` constructs it,
    `feature(x)` runs it.

    A real base class rather than a `Protocol` deliberately: `is_parameterless()` below
    dispatches on nominal `issubclass()`, which needs actual inheritance to distinguish this
    from `Feature[ParamsT]` - the two declare identically-named members (`__init__`,
    `__call__`, `names`), so a structural/duck-typed check couldn't tell them apart.
    """

    @abstractmethod
    def __init__(self) -> None: ...

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """x: (B, S, T). Returns (B, S, 1)."""

    @property
    @abstractmethod
    def names(self) -> list[str]:
        """Always exactly one entry: this feature's own complete output name (e.g.
        `["median"]`) - never `[""]`."""


def is_parameterless(feature_cls: type) -> bool:
    """Whether `feature_cls` is constructed with no arguments (`ParameterlessFeature`) or
    with a `params` list (`Feature[ParamsT]`) - the dispatch signal `TorchFeatureExtractor`
    and the shared test utilities use to construct a `SUPPORTED_FEATURES` entry correctly."""
    return issubclass(feature_cls, ParameterlessFeature)


def construct_feature(
    feature_cls: type[Feature[Any]] | type[ParameterlessFeature],
    params: Sequence[Mapping[str, Any]] | None,
) -> Feature[Any] | ParameterlessFeature:
    """Construct `feature_cls` with the calling convention its kind requires: no arguments
    for a `ParameterlessFeature` (`params` must be `None`), `params` for a `Feature[ParamsT]`
    (must not be `None`). The one place that crosses the two base classes' incompatible
    constructor signatures, so callers never need their own `is_parameterless()` branch."""
    if is_parameterless(feature_cls):
        assert params is None, f"{feature_cls.__name__} takes no parameters; got {params}"
        return cast(type[ParameterlessFeature], feature_cls)()
    assert params is not None, f"{feature_cls.__name__} requires params"
    return cast(type[Feature[Any]], feature_cls)(list(params))
