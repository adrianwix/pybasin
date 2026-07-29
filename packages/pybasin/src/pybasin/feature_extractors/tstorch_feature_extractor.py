"""Feature extractor backed by the `tstorch` workspace package.

Requires the ``tstorch`` optional dependency, available only inside this workspace
(``uv sync --group complete``) - it is not published to PyPI, so this extractor is
usable only in development, not by ``pip install pybasin`` users.
"""

from typing import TYPE_CHECKING, Literal

import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution

if TYPE_CHECKING:
    from tstorch.settings import FCParameters
    from tstorch.torch_feature_extractor import TorchFeatureExtractor

try:
    from tstorch.settings import FCParameters
    from tstorch.torch_feature_extractor import TorchFeatureExtractor

    _tstorch_available = True
except ImportError:
    _tstorch_available = False


class TorchFeatureExtractorAdapter(FeatureExtractor):
    """A pybasin `FeatureExtractor` backed by `tstorch.TorchFeatureExtractor`.

    `tstorch.TorchFeatureExtractor` has no dependency on pybasin: it doesn't subclass
    `FeatureExtractor` and doesn't accept a `pybasin.solution.Solution`. This adapter
    exists purely to bridge that gap - both sides already agree on the (B, S, T) axis
    convention, so `filter_time`'s output can be passed straight through.

    Takes the same configuration as `tstorch.TorchFeatureExtractor` and forwards it
    unchanged, except for `time_steady`: that's consumed here rather than passed through,
    because this adapter always hands the wrapped extractor an already-filtered raw
    tensor, and a raw-tensor input makes `tstorch.TorchFeatureExtractor` skip its own
    time filtering entirely.

    :raises ImportError: If the ``tstorch`` package is not installed.
    """

    def __init__(
        self,
        time_steady: float | None = None,
        features: 'Literal["minimal"] | FCParameters | None' = "minimal",
        features_per_state: "dict[int, FCParameters | None] | None" = None,
        normalize: bool = True,
        impute_method: Literal["extreme", "tsfresh"] = "extreme",
        use_compile: bool = True,
    ) -> None:
        if not _tstorch_available:
            raise ImportError(
                "tstorch is required for TorchFeatureExtractorAdapter. "
                "It is a workspace-only package: install it with `uv sync --group complete`."
            )

        super().__init__(time_steady=time_steady)
        self._torch_extractor = TorchFeatureExtractor(
            features=features,
            features_per_state=features_per_state,
            normalize=normalize,
            impute_method=impute_method,
            use_compile=use_compile,
        )

    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered = self.filter_time(solution)
        features = self._torch_extractor.extract_features(y_filtered)
        self._feature_names = self._torch_extractor.feature_names
        return features
