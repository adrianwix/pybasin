import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution

FricOHE = {"FP": [1, 0], "LC": [0, 1]}


class FrictionFeatureExtractor(FeatureExtractor):
    """
    Classifies solutions based on the second state:
    - If max|y[...,1]| <= 0.2 ⇒ FP (fixed point)
    - Otherwise ⇒ LC (limit cycle)
    """

    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered = self.filter_time(solution)  # shape: (N_after, B, S)

        # Return zeros if no time points exceed self.time_steady
        if y_filtered.shape[0] == 0:
            return torch.zeros((solution.y.shape[1], 2), dtype=torch.float64)

        # Evaluate max absolute value of second state
        max_abs_val = torch.max(torch.abs(y_filtered[..., 1]), dim=0).values

        fp_tensor = torch.tensor(FricOHE["FP"], dtype=torch.float64)  # [1, 0]
        lc_tensor = torch.tensor(FricOHE["LC"], dtype=torch.float64)  # [0, 1]

        # Classification
        out = torch.empty((y_filtered.shape[1], 2), dtype=torch.float64)
        out[max_abs_val <= 0.2] = fp_tensor
        out[max_abs_val > 0.2] = lc_tensor

        return out
