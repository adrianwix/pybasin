from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from pybasin.utils import OrbitData


class Solution:
    """
    Solution: Represents the time integration result for a single initial condition.

    This class stores:

    - The initial condition used for integration.
    - The time series result from integration.
    - Features extracted from the trajectory.
    - Optional labels/classification for each trajectory.
    - Optional model parameters that were used in the integration.
    - Optional orbit data for orbit diagram plotting.

    :ivar initial_condition: The initial condition used for integration (shape: B, S).
    :ivar time: Time points of the solution (shape: N).
    :ivar y: State values over time (shape: B, S, N). The last axis must be
        contiguous (``stride(-1) == 1``), required by downstream tstorch kernels.
    :ivar features: Filtered features used for classification.
    :ivar extracted_features: Original extracted features before filtering.
    :ivar extracted_feature_names: Names of extracted features.
    :ivar feature_names: Names of filtered features.
    :ivar labels: Labels assigned to each solution in the batch.
    :ivar model_params: Parameters of the ODE model.
    :ivar orbit_data: Peak amplitude data for orbit diagram plotting.
    """

    def __init__(
        self,
        initial_condition: torch.Tensor,
        time: torch.Tensor,
        y: torch.Tensor,
        features: torch.Tensor | None = None,
        labels: np.ndarray | None = None,
        model_params: dict[str, float] | None = None,
    ):
        """
        Initialize the Solution object.

        :param initial_condition: shape: (B, S) => B batches, S state variables
        :param time: shape: (N,) => N time points
        :param y: shape: (B, S, N) => B batches, S state variables, N time points.
            Last axis (time) must be contiguous (``stride(-1) == 1``).
        :param features: Optional features describing the trajectory.
        :param labels: Optional classification labels for the solutions.
        :param model_params: Optional dictionary of model parameters used in the simulation.
        """
        # Assertions for shape checks
        assert initial_condition.ndim == 2, "initial_condition must be 2D (B, S)"
        assert time.ndim == 1, "time must be 1D (N,)"
        assert y.ndim == 3, "y must be 3D (B, S, N)"
        assert y.shape[0] == initial_condition.shape[0], (
            f"Initial conditions dimension mismatch: y.shape[0] ({y.shape[0]}) != initial_condition.shape[0] ({initial_condition.shape[0]})"
        )
        assert y.shape[1] == initial_condition.shape[1], (
            f"State dimension mismatch: y.shape[1] ({y.shape[1]}) != initial_condition.shape[1] ({initial_condition.shape[1]})"
        )
        assert y.shape[2] == time.shape[0], (
            f"Time dimension mismatch: y.shape[2] ({y.shape[2]}) != len(time) ({time.shape[0]})"
        )
        assert y.stride(-1) == 1, (
            f"y's time axis (last dim) must be contiguous (stride(-1) == 1), got stride {y.stride()}"
        )

        self.initial_condition: torch.Tensor = initial_condition
        self.time: torch.Tensor = time
        self.y: torch.Tensor = y
        self.features: torch.Tensor | None = features if features is not None else None
        self.extracted_features: torch.Tensor | None = None
        self.extracted_feature_names: list[str] | None = None
        self.feature_names: list[str] | None = None
        self.labels: np.ndarray | None = labels
        self.model_params: dict[str, Any] | None = model_params
        self.orbit_data: OrbitData | None = None

    def set_labels(self, labels: np.ndarray):
        """
        Assign a label to this solution.

        :param labels: The label to assign from the classification results.
        """
        assert len(labels) == self.y.shape[0], (
            f"Initial conditions dimension mismatch: len(labels) ({len(labels)}) != y.shape[0] ({self.y.shape[0]})"
        )

        self.labels = labels

    def set_extracted_features(self, features: torch.Tensor, names: list[str]):
        """
        Store extracted features before filtering.

        :param features: Extracted feature tensor.
        :param names: List of feature names.
        """
        self.extracted_features = features
        self.extracted_feature_names = names

    def set_features(self, features: torch.Tensor, names: list[str] | None = None):
        """
        Store features extracted from the trajectory.

        :param features: A feature vector describing the solution (typically filtered).
        :param names: Optional list of feature names (for filtered features).
        """
        self.features = features
        if names is not None:
            self.feature_names = names

    def get_summary(self) -> dict[str, Any]:
        """
        Return a summary of the solution.

        :return: Dictionary with key information about the solution.
        """
        initial_condition_list: list[Any] = self.initial_condition.tolist()  # type: ignore[assignment]
        features_list: list[Any] | None = None
        if self.features is not None:
            features_list = self.features.tolist()  # type: ignore[assignment]
        labels_list: list[Any] | None = self.labels.tolist() if self.labels is not None else None

        return {
            "initial_condition": initial_condition_list,
            "num_time_steps": len(self.time),
            "trajectory_shape": self.y.shape,
            "features": features_list,
            "labels": labels_list,
            "model_params": self.model_params,
        }
