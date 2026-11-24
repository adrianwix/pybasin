"""Conversion between Attractors.jl's 1-based group labels and pybasin's 0-based ones."""

import numpy as np


def julia_labels_to_pybasin(labels_julia: np.ndarray) -> np.ndarray:
    """Convert Attractors.jl's 1-based group labels to pybasin's 0-based convention.

    ``-1`` (Attractors.jl's out-of-range/unclassified sentinel) is left unchanged.

    :param labels_julia: 1-based group labels as returned by Attractors.jl's
        ``group_features``.
    :return: 0-based labels matching what pybasin's clusterers assign.
    """
    labels = np.asarray(labels_julia)
    return np.where(labels == -1, -1, labels - 1)
