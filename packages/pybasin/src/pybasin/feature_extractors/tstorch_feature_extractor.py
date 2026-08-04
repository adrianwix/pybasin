"""Feature extractor backed by the `tstorch` workspace package.

Requires the ``tstorch`` optional dependency, available only inside this workspace
(``uv sync``) - it is not published to PyPI, so this extractor is
usable only in development, not by ``pip install pybasin`` users.
"""

from typing import TYPE_CHECKING, Literal

import torch

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution

if TYPE_CHECKING:
    from tstorch.preprocessing import ExtremeImputer, StandardScaler, StatisticalImputer
    from tstorch.settings import FeaturesConfig
    from tstorch.torch_feature_extractor import TorchFeatureExtractor

try:
    from tstorch.preprocessing import ExtremeImputer, StandardScaler, StatisticalImputer
    from tstorch.settings import FeaturesConfig
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

    `tstorch.TorchFeatureExtractor` itself only produces raw features; this adapter
    composes that raw extraction with `tstorch.preprocessing` imputation, controlled by
    `impute_method`, and, when `normalize` is set, a standalone
    `tstorch.preprocessing.StandardScaler`. The scaler fits on the first batch it sees
    and stays fitted across subsequent calls, standardizing with population variance
    (``ddof=0``).

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
        features: 'Literal["minimal"] | FeaturesConfig | None' = "minimal",
        features_per_state: "dict[int, FeaturesConfig | None] | None" = None,
        normalize: bool = True,
        impute_method: Literal["extreme", "statistical"] = "extreme",
        use_compile: bool = True,
    ) -> None:
        if not _tstorch_available:
            raise ImportError(
                "tstorch is required for TorchFeatureExtractorAdapter. "
                "It is a workspace-only package: install it with `uv sync`."
            )

        super().__init__(time_steady=time_steady)
        self._torch_extractor = TorchFeatureExtractor(
            features=features,
            features_per_state=features_per_state,
            use_compile=use_compile,
        )
        self._normalize = normalize
        self._impute_method: Literal["extreme", "statistical"] = impute_method
        self._extreme_imputer = ExtremeImputer()
        self._statistical_imputer = StatisticalImputer()
        self._scaler = StandardScaler() if normalize else None
        self._scaler_fitted = False

    def extract_features(self, solution: Solution) -> torch.Tensor:
        y_filtered = self.filter_time(solution)
        features = self._torch_extractor.extract_features(y_filtered)
        self._feature_names = self._torch_extractor.feature_names

        features = (
            self._extreme_imputer.transform(features)
            if self._impute_method == "extreme"
            else self._statistical_imputer.transform(features)
        )

        if self._scaler is not None:
            if not self._scaler_fitted:
                self._scaler.fit(features)
                self._scaler_fitted = True
            features = self._scaler.transform(features)

        return features
