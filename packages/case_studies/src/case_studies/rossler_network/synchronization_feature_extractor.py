"""Synchronization feature extractor for network systems.

This module provides a feature extractor that computes synchronization metrics
from trajectories of coupled oscillator networks.
"""

import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution


class SynchronizationFeatureExtractor(FeatureExtractor):
    """
    Feature extractor that computes synchronization metrics for network systems.

    For a network of N oscillators with 3 states each (x, y, z), computes:
    - max_deviation_x: max_i,j |x_i - x_j| over steady-state
    - max_deviation_y: max_i,j |y_i - y_j| over steady-state
    - max_deviation_z: max_i,j |z_i - z_j| over steady-state
    - max_deviation_all: maximum of the above three

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the network.
    time_steady : float
        Time after which the transient is considered over.
    device : str, optional
        Device for computation ('cpu', 'cuda', etc.).
    """

    def __init__(
        self,
        n_nodes: int,
        time_steady: float = 1000.0,
        device: str | None = None,
    ):
        super().__init__(time_steady=time_steady)
        self.n_nodes = n_nodes
        self.device = torch.device(device if device else "cpu")
        self._feature_names = [
            "max_deviation_x",
            "max_deviation_y",
            "max_deviation_z",
            "max_deviation_all",
        ]

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """
        Extract synchronization features from trajectories.

        Parameters
        ----------
        solution : Solution
            Solution containing trajectory data with shape (n_times, n_samples, 3*N)

        Returns
        -------
        torch.Tensor
            Feature matrix with shape (n_samples, 4)
        """
        y_filtered = self.filter_time(solution)

        y_filtered = y_filtered.to(self.device)

        y_transposed = y_filtered.transpose(0, 1)

        features = self._compute_sync_features(y_transposed)

        return features

    def _compute_sync_features(self, y_steady: torch.Tensor) -> torch.Tensor:
        """
        Compute synchronization features for all trajectories.

        Uses only the FINAL time step to measure synchronization state,
        avoiding penalization of trajectories still converging during the window.

        Parameters
        ----------
        y_steady : torch.Tensor
            Steady-state trajectories with shape (n_samples, n_steady_times, 3*N)

        Returns
        -------
        torch.Tensor
            Features with shape (n_samples, 4)
        """
        N = self.n_nodes

        y_final = y_steady[:, -1, :]

        x = y_final[:, :N]
        y = y_final[:, N : 2 * N]
        z = y_final[:, 2 * N :]

        max_dev_x = torch.max(x, dim=1).values - torch.min(x, dim=1).values
        max_dev_y = torch.max(y, dim=1).values - torch.min(y, dim=1).values
        max_dev_z = torch.max(z, dim=1).values - torch.min(z, dim=1).values

        max_dev_all = torch.maximum(torch.maximum(max_dev_x, max_dev_y), max_dev_z)

        features = torch.stack([max_dev_x, max_dev_y, max_dev_z, max_dev_all], dim=1)

        return features
