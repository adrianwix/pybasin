import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution

PendulumOHE = {
    "FP": torch.tensor([1, 0], dtype=torch.float64),
    "LC": torch.tensor([0, 1], dtype=torch.float64),
}


class PendulumFeatureExtractor(FeatureExtractor):
    @property
    def feature_names(self) -> list[str]:
        return ["FP", "LC"]

    def extract_features(self, solution: Solution):
        y_filtered = self.filter_time(solution)  # shape: (B, S, N_after)

        # shape: (B, N_after)
        y_1 = y_filtered[:, 1, :]  # angular velocity
        delta = (y_1.max(dim=-1).values - y_1.mean(dim=-1)).abs()
        mask = delta < 0.01

        # shape: (B, 2)
        out = torch.empty((y_1.shape[0], 2), dtype=torch.float64)
        out[mask] = PendulumOHE["FP"]  # [1, 0]
        out[~mask] = PendulumOHE["LC"]  # [0, 1]
        return out
