"""Narrow sklearn's cluster-mixin stubs to pybasin's array contract.

``sklearn.base.ClusterMixin.fit_predict`` is stubbed against sklearn's full
generality: arbitrary iterables, data frames, sparse matrices, and other input
types. Pybasin clusterers accept and return
:data:`pybasin._device_backend.Array`.

This mixin declares the narrower signature once, avoiding repeated override
suppressions in each clusterer.
"""

from typing import Any

from sklearn.base import ClusterMixin  # type: ignore[import-untyped]

from pybasin._device_backend import TArray


class ArrayClusterMixin(ClusterMixin):
    """``ClusterMixin`` narrowed to ``fit_predict(TArray) -> TArray``."""

    def fit_predict(self, X: TArray, y: Any = None) -> TArray:  # type: ignore[override]
        raise NotImplementedError
