import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


class DuffingFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for the Duffing oscillator.
    Extracts two features from the first state variable:
    1. Maximum value
    2. Standard deviation
    """

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """
        Extract features from solution trajectories.

        Args:
            solution: Solution object with y shape (N, B, S)
                N: number of time steps
                B: batch size (number of initial conditions)
                S: state dimension

        Returns:
            torch.Tensor: Features with shape (B, 2)
        """
        y_filtered = self.filter_time(solution)  # shape: (N_after, B, S)
        displacement = y_filtered[..., 0]  # shape: (N_after, B)

        # Compute features across time dimension (dim=0)
        max_vals = torch.max(displacement, dim=0).values  # shape: (B,)
        std_vals = torch.std(displacement, dim=0)  # shape: (B,)

        # Stack features to get shape (B, 2)
        features = torch.stack([max_vals, std_vals], dim=1)
        return features
