"""Hénon map mean-state featurizer, with a ``(200, 200)`` sentinel for divergence.

Source: `Attractors.jl` @ `df7c420b4c1d69e1d662bb679a46f8ef8b53bcaa`
`test/mapping/histogram_grouping.jl:12-20` -- `henon_mean_state_features` below is the
same mean-state-with-sentinel featurizer, ported to `torch` for pybasin-side use. Line
numbers and content are only guaranteed to match at that commit -- re-verify against
current Attractors.jl source before trusting them at a later one.
"""

import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


def henon_mean_state_features(trajectory: torch.Tensor) -> torch.Tensor:
    """Mean-state feature per batch row, with a ``(200, 200)`` sentinel for divergence.

    Any trajectory containing a non-finite value, or a coordinate with magnitude
    greater than 10, is treated as divergent and replaced by the sentinel feature
    instead of its (meaningless) mean -- unsupervised grouping predictors have no
    notion of "diverged to infinity" on their own.

    :param trajectory: Shape ``(batch, 2, n_steps)``.
    :return: Shape ``(batch, 2)``.
    """
    mean_state = trajectory.mean(dim=-1)
    divergent = ~torch.isfinite(trajectory).all(dim=(1, 2)) | (trajectory.abs() > 10).any(
        dim=(1, 2)
    )
    sentinel = torch.tensor([200.0, 200.0], dtype=mean_state.dtype, device=mean_state.device)
    return torch.where(divergent.unsqueeze(-1), sentinel, mean_state)


class HenonMeanStateFeatureExtractor(FeatureExtractor):
    """Per-axis mean of the Hénon trajectory, with a ``(200, 200)`` sentinel for
    divergence -- see `henon_mean_state_features`. Matches Attractors.jl's own
    featurizer, which uses the whole recorded trajectory with no transient discard;
    pass ``time_steady=0.0`` (the default here) to keep that behavior.
    """

    def __init__(self, time_steady: float = 0.0):
        super().__init__(time_steady=time_steady)

    def extract_features(self, solution: Solution) -> torch.Tensor:
        trajectory = self.filter_time(solution)  # (batch, 2, n_steps)
        return henon_mean_state_features(trajectory)
