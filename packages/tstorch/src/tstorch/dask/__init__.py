"""Distributed feature extraction over Dask CPU clusters and one Dask-CUDA worker.

``dask[array]``, ``distributed``, and ``numpy`` are core tstorch dependencies. Importing
``tstorch`` itself never imports Dask; only importing this subpackage does, so a caller
who never touches distributed extraction pays no Dask import cost.
"""

from tstorch.dask.extractor import DaskFeatureExtractor

__all__ = ["DaskFeatureExtractor"]
