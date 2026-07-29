"""Narrows sklearn's loosely-typed mixin/base-class stubs to pybasin's array contract.

``sklearn.base.TransformerMixin.fit_transform``, ``sklearn.base.ClusterMixin.fit_predict``,
and ``sklearn.pipeline.Pipeline.fit_transform``/``.transform`` are all stubbed (via the
``microsoft/python-type-stubs`` package) against sklearn's full generality: arbitrary
``Iterable``, ``DataFrame``, sparse matrices, etc. No pybasin estimator supports any of
that -- every one of them only ever accepts/returns :data:`pybasin._device_backend.Array`.

Subclassing sklearn's mixins directly would mean every concrete estimator re-declares its
own narrowed ``TArray -> TArray`` signature *and* re-suppresses the resulting override
mismatch against the wide sklearn stub. Subclassing the mixins here instead means that
mismatch is declared -- and suppressed -- exactly once per method shape, in one place
with no real logic of its own. Every concrete estimator then overrides an already-narrowed
signature, which is a genuine (unsuppressed) override, so adding a new estimator never
requires its own ``# type: ignore[override]``.
"""

from typing import Any

from sklearn.base import ClusterMixin, TransformerMixin  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]

from pybasin._device_backend import TArray


class ArrayTransformerMixin(TransformerMixin):
    """``TransformerMixin`` narrowed to ``fit_transform(TArray) -> TArray``."""

    def fit_transform(self, X: TArray, y: Any = None, **fit_params: Any) -> TArray:  # type: ignore[override]
        raise NotImplementedError


class ArrayClusterMixin(ClusterMixin):
    """``ClusterMixin`` narrowed to ``fit_predict(TArray) -> TArray``."""

    def fit_predict(self, X: TArray, y: Any = None) -> TArray:  # type: ignore[override]
        raise NotImplementedError


class ArrayPipeline(Pipeline):
    """``Pipeline`` narrowed to ``fit_transform``/``transform`` over ``TArray``.

    Delegates to the real ``Pipeline`` implementation (chaining ``steps``), so
    subclasses get working behavior for free and only need to build ``steps``
    in their own ``__init__``.
    """

    def fit_transform(self, X: TArray, y: Any = None, **fit_params: Any) -> TArray:  # type: ignore[override]
        return super().fit_transform(X, y, **fit_params)  # type: ignore[misc]

    def transform(self, X: TArray) -> TArray:  # type: ignore[override]
        return super().transform(X)  # type: ignore[misc]
