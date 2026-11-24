"""Fixed-geometry rectangular-bin assignment for basin stability grouping."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal, overload

import numpy as np
import torch
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]

from pybasin._device_backend import Array, TArray
from pybasin._sklearn_compat import ArrayClusterMixin
from pybasin.utils import DisplayNameMixin


class HistogramClusterer(DisplayNameMixin, ArrayClusterMixin, BaseEstimator):
    """Assign feature rows to a fixed-geometry rectangular histogram bin.

    Implements the histogram grouping of Datseris, Rossi & Wagemakers (2023),
    "Framework for global stability analysis of dynamical systems", *Chaos* 33,
    073151, https://doi.org/10.1063/5.0159675, as realized by Attractors.jl's
    ``GroupViaHistogram`` (v1.38.4) on top of ComplexityMeasures.jl's
    ``FixedRectangularBinning``, with edges that may be non-uniform. A closely
    related classical procedure is the sorting algorithm of Hartigan (1975),
    *Clustering Algorithms*, Wiley, §3.5.

    Each feature axis is partitioned by a caller-supplied, fixed sequence of bin
    edges that never depends on the data. A row's label is the zero-based
    linear index of the bin containing it, first-axis-fastest
    (``strides_[0] = 1``, ``strides_[d] = prod(n_bins_[:d])``). A row that
    falls outside any axis's edges, or contains ``NaN``/``inf`` on any axis,
    gets label ``-1``.

    There is no fit phase: the bin geometry is a pure function of ``bin_edges``,
    so it stays identical across repeated :meth:`fit_predict` calls -- required
    for a continuation sweep, where bins must remain comparable across every
    parameter group.
    """

    def __init__(
        self,
        bin_edges: Sequence[ArrayLike],
        *,
        feature_names: Sequence[str] | None = None,
        boundary_mode: Literal["searchsorted", "arithmetic"] = "searchsorted",
    ):
        """
        :param bin_edges: One ascending, finite, duplicate-free sequence of edge
            values per feature axis (``m_d + 1`` edges give ``m_d`` bins on axis
            ``d``, left-closed-right-open ``[a, b)``). This is fixed geometry:
            the same edges are reused, unmodified, across every
            :meth:`fit_predict` call -- never derived from ``X``.
        :param feature_names: Expected feature-column order. When set,
            :meth:`set_feature_names` requires an exact ordered match, since a
            reordered or dropped column would silently change what each
            histogram axis measures.
        :param boundary_mode: ``"searchsorted"`` (default) locates each point's
            bin via a per-axis binary search over that axis's edges -- correct
            for arbitrary, non-uniform edges. ``"arithmetic"`` instead computes
            ``floor((x - lower) / width)`` from the axis's overall span and bin
            count, which is only correct when that axis's edges are equally
            spaced (validated at :meth:`fit_predict` time), and may disagree
            with ``"searchsorted"`` within floating-point roundoff of an edge.
            ``"arithmetic"`` is the lookup Attractors.jl uses by default
            (``FixedRectangularBinning(..., precise=false)``); ``"searchsorted"``
            corresponds to ``precise=true``.
        """
        self.bin_edges = bin_edges
        self.feature_names = feature_names
        self.boundary_mode = boundary_mode

    def set_feature_names(self, feature_names: list[str]) -> None:
        """Validate that incoming feature columns exactly match construction order.

        :param feature_names: Feature names for the columns about to be passed
            to :meth:`fit_predict`, in column order.
        :raises ValueError: If ``feature_names`` was configured at construction
            and does not match ``feature_names`` exactly, in the same order.
        """
        if self.feature_names is not None and list(feature_names) != list(self.feature_names):
            raise ValueError(
                f"HistogramClusterer.feature_names={list(self.feature_names)!r} does not "
                f"match the incoming column order {list(feature_names)!r}. Bin edges are "
                "fixed per axis position, so reordering or dropping a column would "
                "silently change what each histogram axis measures."
            )

    def fit_predict(self, X: TArray, y: Any = None) -> TArray:
        """Assign each row of ``X`` to a bin's zero-based linear index, or ``-1``.

        :param X: Feature array of shape ``(n_samples, n_features)``, with
            ``n_features == len(bin_edges)``.
        :param y: Ignored (present for sklearn API compatibility).
        :return: ``int64`` labels of shape ``(n_samples,)``, mirroring ``X``'s
            concrete type and device.
        :raises ValueError: If ``X`` isn't 2D, its column count doesn't match
            ``len(bin_edges)``, any axis's edges have fewer than 2 entries,
            aren't strictly ascending, contain non-finite values, aren't
            uniformly spaced under ``boundary_mode="arithmetic"``, or the total
            bin count doesn't fit a signed 64-bit label.
        """
        x_t = X if isinstance(X, torch.Tensor) else torch.from_numpy(np.asarray(X))  # type: ignore[arg-type]
        if x_t.ndim != 2:
            raise ValueError(
                "HistogramClusterer.fit_predict expects a 2D (n_samples, n_features) "
                f"array, got shape {tuple(x_t.shape)}."
            )

        edges = self._prepared_edges(x_t.dtype, x_t.device)
        n_features = len(edges)
        if x_t.shape[1] != n_features:
            raise ValueError(
                f"X has {x_t.shape[1]} feature columns but bin_edges defines {n_features} axes."
            )

        n_bins = [int(e.numel()) - 1 for e in edges]
        _validate_label_capacity(n_bins)
        strides = _strides_from_n_bins(n_bins)

        n_samples = x_t.shape[0]
        if n_samples == 0:
            labels_t = torch.empty(0, dtype=torch.int64, device=x_t.device)
        else:
            labels_t = self._assign(x_t, edges, n_bins, strides)

        self.n_bins_ = tuple(n_bins)
        self.strides_ = tuple(strides)
        self.resolved_params_ = self._canonical_parameters()

        return _mirror(labels_t, X)

    def _assign(
        self,
        x_t: torch.Tensor,
        edges: list[torch.Tensor],
        n_bins: list[int],
        strides: list[int],
    ) -> torch.Tensor:
        device = x_t.device
        n_samples, n_features = x_t.shape

        per_axis_index = torch.empty((n_samples, n_features), dtype=torch.int64, device=device)
        inside = torch.empty((n_samples, n_features), dtype=torch.bool, device=device)

        for d, e in enumerate(edges):
            col = x_t[:, d].to(e.dtype)
            lower = e[0]
            upper = e[-1]
            finite = torch.isfinite(col)

            if self.boundary_mode == "searchsorted":
                idx = torch.searchsorted(e, col.contiguous(), side="right") - 1
            else:
                width = (upper - lower) / n_bins[d]
                idx = torch.floor((col - lower) / width).to(torch.int64)

            per_axis_index[:, d] = idx
            # The index range check is needed on top of the value range check: in
            # arithmetic mode, rounding in `(col - lower) / width` can give
            # `idx == n_bins[d]` for a value just below `upper`, which would otherwise
            # spill into the next axis's stride and alias a different bin.
            inside[:, d] = finite & (col >= lower) & (col < upper) & (idx >= 0) & (idx < n_bins[d])

        strides_t = torch.tensor(strides, dtype=torch.int64, device=device)
        labels_t = (per_axis_index * strides_t).sum(dim=1)
        all_inside = inside.all(dim=1)
        return torch.where(all_inside, labels_t, torch.full_like(labels_t, -1))

    def _prepared_edges(self, x_dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
        if len(self.bin_edges) == 0:
            raise ValueError("HistogramClusterer requires at least one bin_edges axis.")

        raw_edges = [torch.as_tensor(np.asarray(edge)) for edge in self.bin_edges]

        compute_dtype = x_dtype
        for raw in raw_edges:
            compute_dtype = torch.promote_types(compute_dtype, raw.dtype)
        if not compute_dtype.is_floating_point:
            compute_dtype = torch.get_default_dtype()

        edges: list[torch.Tensor] = []
        for axis, raw in enumerate(raw_edges):
            e = raw.to(dtype=compute_dtype, device=device)
            if e.ndim != 1:
                raise ValueError(f"bin_edges[{axis}] must be 1D, got shape {tuple(e.shape)}.")
            if e.numel() < 2:
                raise ValueError(
                    f"bin_edges[{axis}] must have at least 2 entries (m+1 edges for "
                    f"m>=1 bins), got {e.numel()}."
                )
            if not bool(torch.isfinite(e).all()):
                raise ValueError(f"bin_edges[{axis}] must be finite, got {_edge_values(e)}.")
            if not bool(torch.all(e[1:] > e[:-1])):
                raise ValueError(
                    f"bin_edges[{axis}] must be strictly ascending with no duplicates, "
                    f"got {_edge_values(e)}."
                )
            if self.boundary_mode == "arithmetic":
                width = float(((e[-1] - e[0]) / (e.numel() - 1)).item())
                diffs = e[1:] - e[:-1]
                if not torch.allclose(diffs, torch.full_like(diffs, width), rtol=1e-5):
                    raise ValueError(
                        "boundary_mode='arithmetic' requires equally spaced edges; "
                        f"bin_edges[{axis}] is not uniform: {_edge_values(e)}."
                    )
            edges.append(e)
        return edges

    def _canonical_parameters(self) -> dict[str, Any]:
        return {
            "bin_edges": [np.asarray(edge).tolist() for edge in self.bin_edges],
            "feature_names": (list(self.feature_names) if self.feature_names is not None else None),
            "boundary_mode": self.boundary_mode,
        }


def _edge_values(e: torch.Tensor) -> list[float]:
    """Host list of edge values for an error message (never on a hot path)."""
    return e.detach().cpu().numpy().tolist()


def _strides_from_n_bins(n_bins: Sequence[int]) -> list[int]:
    strides = [1] * len(n_bins)
    for d in range(1, len(n_bins)):
        strides[d] = strides[d - 1] * n_bins[d - 1]
    return strides


def _validate_label_capacity(n_bins: Sequence[int]) -> None:
    total_bins = math.prod(n_bins)
    if total_bins > int(np.iinfo(np.int64).max):
        raise ValueError(
            f"Total bin count {total_bins} (n_bins={list(n_bins)}) does not fit a "
            "signed 64-bit label."
        )


@overload
def _mirror(labels: torch.Tensor, like: torch.Tensor) -> torch.Tensor: ...
@overload
def _mirror(labels: torch.Tensor, like: np.ndarray) -> np.ndarray: ...
def _mirror(labels: torch.Tensor, like: Array) -> Array:
    """Mirror ``like``'s concrete type: a tensor passes through, a numpy array converts."""
    if isinstance(like, torch.Tensor):
        return labels
    return labels.numpy()
