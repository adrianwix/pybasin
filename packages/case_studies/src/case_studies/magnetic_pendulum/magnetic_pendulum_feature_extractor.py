import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


class MagneticPendulumFeatureExtractor(FeatureExtractor):
    """Final ``(x, y)`` position -- matches Attractors.jl's own
    ``featurizer(A, t) = SVector(A[end][1], A[end][2])`` (`Attractors.jl` @
    `df7c420b4c1d69e1d662bb679a46f8ef8b53bcaa` `test/mapping/attractor_mapping.jl:238-240`;
    line numbers only guaranteed to match at that commit).
    """

    def extract_features(self, solution: Solution) -> torch.Tensor:
        final_state = solution.y[..., -1]  # (batch, 4): x, y, vx, vy
        return final_state[:, :2]
