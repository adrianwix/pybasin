"""Synchronization classifier for network basin stability.

This module provides a classifier that determines whether a network of oscillators
has synchronized based on the synchronization features (max deviation).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin


class SynchronizationClassifier(BaseEstimator, ClusterMixin):
    """
    Classifier that labels trajectories as 'synchronized' or 'desynchronized'.

    Works with features from SynchronizationFeatureExtractor, which computes
    the max deviation across all node pairs. Synchronization is achieved when:
        max_deviation_all < epsilon

    :param epsilon: Synchronization threshold. States are synchronized if max deviation < epsilon.
    :param feature_index: Index of the feature to use for thresholding. Default is 3 (max_deviation_all).
        Options: 0=max_deviation_x, 1=max_deviation_y, 2=max_deviation_z, 3=max_deviation_all
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        feature_index: int = 3,
    ):
        self.epsilon = epsilon
        self.feature_index = feature_index

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:
        """
        Classify each trajectory as synchronized or desynchronized.

        :param X: Feature matrix from SynchronizationFeatureExtractor.
            Shape: (n_samples, 4) with columns [max_dev_x, max_dev_y, max_dev_z, max_dev_all]
        :param y: Ignored. Present for API compatibility.
        :return: Labels: 'synchronized' or 'desynchronized' for each trajectory.
        """
        X_arr = np.asarray(X)
        max_deviation = X_arr[:, self.feature_index]

        labels = np.where(
            max_deviation < self.epsilon,
            "synchronized",
            "desynchronized",
        )

        return labels
