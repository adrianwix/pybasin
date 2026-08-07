"""Exportable graph: trajectory -> features. Stage 2 of 3 -- see integrate_graph.py's module
docstring for why each pipeline stage is its own `.pte` (a demo UI's feature-selection control
should only re-run this stage and the classify stage after it, not re-integrate)."""

import torch
import torch.nn as nn

from pybasin.feature_extractors.default_feature_extractor import _FEATURE_NAMES, _delta

N_FEATURES_PER_STATE = len(_FEATURE_NAMES)
FEATURE_NAMES = _FEATURE_NAMES


def _median_topk(x: torch.Tensor) -> torch.Tensor:
    """Median along the last (time) dim, matching ``numpy.median``'s even-length averaging
    behavior -- numerically identical to
    :func:`pybasin.feature_extractors.default_feature_extractor._median`, but computed via
    ``topk`` instead of ``sort``: ExecuTorch's portable CPU kernel library (what this graph's
    WASM build links) has no ``sort`` kernel at all, only ``topk``.
    """
    n = x.shape[-1]
    k = n // 2 + 1
    smallest_k = torch.topk(x, k, dim=-1, largest=False, sorted=True).values
    if n % 2 == 1:
        return smallest_k[..., -1]
    return (smallest_k[..., -1] + smallest_k[..., -2]) / 2


class FeatureExtractGraph(nn.Module):
    """Computes :class:`DefaultFeatureExtractor`'s fixed 10-statistic set for every state,
    unconditionally -- no per-case-study feature configuration. Depends only on `state_dim`
    (not on which case study or which features a UI control ends up selecting), so one
    exported `.pte` is reusable by every case study that shares the same `state_dim` (see the
    experiment README). "Choosing features" in the demo UI is a client-side column slice of
    this graph's full `(n, state_dim * 10)` output, not a per-selection re-export.

    :param state_dim: Number of ODE state variables.
    """

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        :param y: ``(n, state_dim, t_steps)`` trajectory tensor, already restricted to the
            steady-state observation window.
        :return: ``(n, state_dim * 10)`` feature tensor, state-major (state 0's 10 features,
            then state 1's, ...) in ``FEATURE_NAMES`` order.
        """
        delta = _delta(y)
        stats = {
            "median": _median_topk(y),
            "mean": y.mean(dim=-1),
            "standard_deviation": y.std(dim=-1, correction=0),
            "variance": y.var(dim=-1, correction=0),
            "root_mean_square": torch.sqrt((y**2).mean(dim=-1)),
            "maximum": y.max(dim=-1).values,
            "absolute_maximum": y.abs().max(dim=-1).values,
            "minimum": y.min(dim=-1).values,
            "delta": delta,
            "log_delta": torch.log(delta + 1e-10),
        }
        per_state_blocks = [
            torch.stack([stats[name][:, state_idx] for name in _FEATURE_NAMES], dim=1)
            for state_idx in range(self.state_dim)
        ]
        return torch.cat(per_state_blocks, dim=1)
