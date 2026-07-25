import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution

# One-hot encoding for Lorenz attractor states
# S1: positive x-mean, S2: negative x-mean
LorenzOHE = {"S1": [1, 0], "S2": [0, 1]}


class LorenzFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for the Lorenz system.
    Classifies trajectories based on the mean x-coordinate:
    - If mean(x) > 0: Solution 1 (positive regime)
    - If mean(x) < 0: Solution 2 (negative regime)
    Also handles unbounded solutions (|y| > 195)
    """

    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered = self.filter_time(solution)  # shape: (N_after, B, S)

        # Check for unbounded solutions first
        max_abs_vals = torch.max(torch.abs(y_filtered), dim=0)[0]  # shape: (B, S)
        unbounded = torch.any(max_abs_vals > 195, dim=1)  # shape: (B,)

        # Calculate mean x-coordinate (first state variable)
        x_mean = y_filtered[..., 0].mean(dim=0)  # shape: (B,)
        positive_regime = x_mean > 0

        s1_tensor = torch.tensor(LorenzOHE["S1"], dtype=torch.float64)  # [1, 0]
        s2_tensor = torch.tensor(LorenzOHE["S2"], dtype=torch.float64)  # [0, 1]

        # Initialize output tensor
        out = torch.empty((y_filtered.shape[1], 2), dtype=torch.float64)

        # Assign features based on x_mean and boundedness
        out[~unbounded & positive_regime] = s1_tensor
        out[~unbounded & ~positive_regime] = s2_tensor
        # Unbounded solutions get [0, 0]
        out[unbounded] = torch.zeros_like(s1_tensor)

        return out
