"""Independent preprocessing: extractors return raw features only.

``ExtremeImputer``, ``StatisticalImputer``, and ``StandardScaler`` have no reference to
:class:`tstorch.torch_feature_extractor.TorchFeatureExtractor`,
:class:`tstorch.dask.DaskFeatureExtractor`, a Dask ``Client``, or a worker cache, and are
usable in a pipeline that never touches tstorch feature extraction at all. ``ExtremeImputer``
and ``StandardScaler`` accept a ``torch.Tensor`` (eager) or a NumPy- or CuPy-backed
``dask.array.Array`` (lazy) - a CuPy-backed array stays GPU-resident through every block;
``StatisticalImputer`` needs a whole feature's statistics at once, so it only accepts an
eager ``torch.Tensor``.
"""

from tstorch.preprocessing.extreme_imputer import ExtremeImputer
from tstorch.preprocessing.standard_scaler import ScalingState, StandardScaler
from tstorch.preprocessing.statistical_imputer import StatisticalImputer

__all__ = ["ExtremeImputer", "StatisticalImputer", "StandardScaler", "ScalingState"]
