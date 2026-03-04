"""Feature extractor using tsfresh library for time series feature extraction.

Requires the ``tsfresh`` optional dependency: ``pip install pybasin[tsfresh]``
"""

import multiprocessing
import threading
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from sklearn.preprocessing import StandardScaler

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution

if TYPE_CHECKING:
    from tsfresh import extract_features  # type: ignore[import-untyped]
    from tsfresh.feature_extraction import MinimalFCParameters  # type: ignore[import-untyped]
    from tsfresh.utilities.dataframe_functions import impute  # type: ignore[import-untyped]

try:
    from tsfresh import extract_features  # type: ignore[import-untyped]
    from tsfresh.feature_extraction import MinimalFCParameters  # type: ignore[import-untyped]
    from tsfresh.utilities.dataframe_functions import impute  # type: ignore[import-untyped]

    _tsfresh_available = True
except ImportError:
    _tsfresh_available = False


class TsfreshFeatureExtractor(FeatureExtractor):
    """Feature extractor using tsfresh for comprehensive time series analysis.

    This extractor uses the tsfresh library to automatically extract a large number
    of time series features from ODE solutions. It converts PyTorch/JAX tensors to
    pandas DataFrames for tsfresh processing, then converts the results back to tensors.

    Supports per-state variable feature configuration using tsfresh's kind_to_fc_parameters
    mechanism, allowing you to apply different feature sets to different state variables
    based on domain knowledge.

    Internally, the solution tensor is converted to tsfresh's wide/flat DataFrame
    format where each state variable becomes a column (``state_0``, ``state_1``, etc.).
    The ``kind_to_fc_parameters`` accepts integer state indices and maps them to these
    column names automatically.

    ```python
    # Same minimal features for all states
    extractor = TsfreshFeatureExtractor(
        time_steady=9.0, default_fc_parameters=MinimalFCParameters(), n_jobs=-1, normalize=True
    )

    # Specific features for all states
    extractor = TsfreshFeatureExtractor(
        time_steady=950.0,
        default_fc_parameters={"mean": None, "std": None, "maximum": None},
        n_jobs=-1,
    )

    # Different features per state (e.g., pendulum: position vs velocity)
    from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters

    extractor = TsfreshFeatureExtractor(
        time_steady=950.0,
        kind_to_fc_parameters={
            0: {"mean": None, "maximum": None, "minimum": None},
            1: ComprehensiveFCParameters(),
        },
        n_jobs=1,  # Use n_jobs=1 for deterministic results
    )
    ```

    Note on parallelism:
        Setting n_jobs > 1 enables parallel feature extraction but introduces
        non-determinism due to floating-point arithmetic order. This can cause
        inconsistent classification results. Use n_jobs=1 for reproducible results.

    Note on normalization:
        When normalize=True, the scaler is fitted on the FIRST dataset that calls
        extract_features(). For best results with supervised classifiers:
        - Either set normalize=False (recommended for KNN with few templates)
        - Or call fit_scaler() explicitly with representative data before extraction

    :param time_steady: Time threshold for filtering transients. If ``None`` (default),
        uses 85% of the integration time span. Set to ``0.0`` to use the entire series.
    :param default_fc_parameters: Default feature extraction parameters for all states.
        Can be one of:
        - MinimalFCParameters() - Fast extraction with ~20 features
        - ComprehensiveFCParameters() - Full extraction with ~800 features
        - Custom dict like {"mean": None, "maximum": None} for specific features
        - None - must provide kind_to_fc_parameters
        Default is MinimalFCParameters().
    :param kind_to_fc_parameters: Optional dict mapping state indices (e.g. ``0``, ``1``)
        to FCParameters. Indices correspond to the state dimension of the solution
        tensor. If provided, overrides ``default_fc_parameters`` for those states.
    :param n_jobs: Number of parallel jobs for feature extraction. Default is 1.
        Set to -1 to use all available cores.
    :param normalize: Whether to apply StandardScaler normalization to features.
        Highly recommended for distance-based classifiers like KNN.
        Default is True.
    """

    def __init__(
        self,
        time_steady: float | None = None,
        default_fc_parameters: dict[str, Any] | Any | None = None,
        kind_to_fc_parameters: dict[int, dict[str, Any] | Any] | None = None,
        n_jobs: int = 1,
        normalize: bool = True,
    ):
        if not _tsfresh_available:
            raise ImportError(
                "tsfresh is required for TsfreshFeatureExtractor. "
                "Install it with: pip install pybasin[tsfresh]"
            )
        super().__init__(time_steady=time_steady)
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self._is_fitted = False
        self._fit_lock = threading.Lock()
        self._feature_columns: list[str] | None = None

        # tsfresh doesn't handle n_jobs=-1 well, convert to actual number
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif n_jobs < 1:
            self.n_jobs = 1
        else:
            self.n_jobs = n_jobs

        # Set default feature parameters
        if default_fc_parameters is None and kind_to_fc_parameters is None:
            self.default_fc_parameters: dict[str, Any] | Any | None = MinimalFCParameters()
        else:
            self.default_fc_parameters: dict[str, Any] | Any | None = default_fc_parameters

        self.kind_to_fc_parameters: dict[int, dict[str, Any] | Any] | None = kind_to_fc_parameters

    def reset_scaler(self) -> None:
        """Reset the scaler to unfitted state.

        Call this if you need to refit the scaler on different data.
        """
        if self.normalize:
            self.scaler = StandardScaler()
            self._is_fitted = False

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract features from an ODE solution using tsfresh.

        Converts the solution tensor to pandas DataFrame format expected by tsfresh,
        extracts features for each trajectory and state variable, then converts back
        to PyTorch tensor.

        :param solution: ODE solution with y tensor of shape (N, B, S) where N is
            time steps, B is batch size, and S is number of state variables.
        :return: Feature tensor of shape (B, F) where B is the batch size and F is
            the total number of features extracted by tsfresh across all state
            variables.
        """
        # Apply time filtering
        y_filtered = self.filter_time(solution)

        # Get dimensions
        n_timesteps, n_batch, n_states = y_filtered.shape

        # Convert to numpy for pandas compatibility
        y_np = y_filtered.cpu().numpy()

        # Build tsfresh wide-format DataFrame: [id, time, state_0, ..., state_S]
        # Transpose (N, B, S) -> (B, N, S) then flatten to (B*N, S)
        y_flat = y_np.transpose(1, 0, 2).reshape(-1, n_states)

        df_data: dict[str, Any] = {
            "id": np.repeat(np.arange(n_batch), n_timesteps),
            "time": np.tile(np.arange(n_timesteps), n_batch),
        }
        for state_idx in range(n_states):
            df_data[f"state_{state_idx}"] = y_flat[:, state_idx]

        df_wide = pd.DataFrame(df_data)

        kind_to_fc_params_mapped: dict[str, dict[str, Any] | Any] | None = None
        if self.kind_to_fc_parameters is not None:
            kind_to_fc_params_mapped = {
                f"state_{idx}": fc_params for idx, fc_params in self.kind_to_fc_parameters.items()
            }

        features_df: pd.DataFrame = cast(
            pd.DataFrame,
            extract_features(
                df_wide,
                column_id="id",
                column_sort="time",
                default_fc_parameters=self.default_fc_parameters,
                kind_to_fc_parameters=kind_to_fc_params_mapped,
                n_jobs=self.n_jobs,
                disable_progressbar=True,
            ),
        )

        # Handle NaN and inf values using tsfresh's impute function
        # This replaces NaN with 0 and inf with large finite values
        impute(features_df)

        # Store feature column names
        self._feature_columns = list(features_df.columns)

        # Convert to numpy array
        features_array = features_df.values

        # Apply normalization if enabled (thread-safe)
        if self.normalize and self.scaler is not None:
            with self._fit_lock:
                if not self._is_fitted:
                    # Fit and transform on first call
                    features_array = self.scaler.fit_transform(features_array)  # type: ignore[reportUnknownMemberType]
                    self._is_fitted = True
                else:
                    features_array = self.scaler.transform(features_array)

        # Convert back to tensor
        features_tensor = torch.tensor(features_array, dtype=torch.float32)

        return features_tensor

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature names.

        :return: List of feature names from tsfresh extraction.
        :raises RuntimeError: If extract_features has not been called yet.
        """
        if self._feature_columns is None:
            raise RuntimeError("Feature columns not initialized. Call extract_features first.")
        return self._feature_columns
