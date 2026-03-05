# pyright: basic
"""Feature extractor using nolds library for nonlinear dynamics analysis.

Provides extraction of nonlinear dynamics features from ODE solution trajectories
with multiprocessing parallelization.
"""

import multiprocessing as mp
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from numpy.typing import NDArray

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.solution import Solution

if TYPE_CHECKING:
    import nolds  # pyright: ignore[reportMissingTypeStubs]

import warnings as _warnings

try:
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", category=SyntaxWarning)
        import nolds  # pyright: ignore[reportMissingTypeStubs]

    _nolds_available = True
except (ImportError, SyntaxError):
    _nolds_available = False

NoldsFCParameters = Mapping[str, list[dict[str, Any]] | None]


def _get_nolds_feature_functions() -> dict[str, Any]:
    """Get mapping of feature names to nolds functions.

    :return: Dictionary mapping feature names to callable nolds functions.
    :raises ImportError: If nolds is not available.
    """
    if not _nolds_available:
        return {}

    return {
        "lyap_r": nolds.lyap_r,  # pyright: ignore[reportPossiblyUnbound]
        "lyap_e": nolds.lyap_e,  # pyright: ignore[reportPossiblyUnbound]
        "sampen": nolds.sampen,  # pyright: ignore[reportPossiblyUnbound]
        "hurst_rs": nolds.hurst_rs,  # pyright: ignore[reportPossiblyUnbound]
        "corr_dim": nolds.corr_dim,  # pyright: ignore[reportPossiblyUnbound]
        "dfa": nolds.dfa,  # pyright: ignore[reportPossiblyUnbound]
        "mfhurst_b": nolds.mfhurst_b,  # pyright: ignore[reportPossiblyUnbound]
        "mfhurst_dm": nolds.mfhurst_dm,  # pyright: ignore[reportPossiblyUnbound]
    }


NOLDS_DEFAULT_FC_PARAMETERS: NoldsFCParameters = {
    "lyap_r": None,
    "corr_dim": None,
}

WorkerTaskArgs = tuple[int, int, str, dict[str, Any], NDArray[np.floating]]
WorkerTaskResult = tuple[int, int, str, float]


def _worker_task(args: WorkerTaskArgs) -> WorkerTaskResult:
    """Worker function for multiprocessing feature extraction.

    :param args: Tuple of (batch_idx, state_idx, feature_key, params, time_series).
    :return: Tuple of (batch_idx, state_idx, feature_key, result).
    """
    batch_idx, state_idx, feature_key, params, time_series = args

    feature_name = feature_key.split("__")[0]
    feature_functions = _get_nolds_feature_functions()
    func = feature_functions[feature_name]

    try:
        result = float(func(time_series, **params))  # type: ignore[misc]
    except Exception:
        result = float("nan")

    return (batch_idx, state_idx, feature_key, result)


def _impute_torch(features: torch.Tensor) -> torch.Tensor:
    """Columnwise replace NaNs and infs with average/extreme values.

    Replacement strategy per column:
        * -inf -> min (of finite values in that column)
        * +inf -> max (of finite values in that column)
        * NaN -> median (of finite values in that column)

    If a column has no finite values, it is filled with zeros.

    :param features: Feature tensor of shape (B, F).
    :return: Imputed tensor with no NaN or inf values.
    """
    result = features.clone()

    for col in range(features.shape[1]):
        col_data = features[:, col]
        finite_mask = torch.isfinite(col_data)

        if finite_mask.any():
            finite_values = col_data[finite_mask]
            col_min = finite_values.min()
            col_max = finite_values.max()
            col_median = finite_values.median()
        else:
            col_min = torch.tensor(0.0)
            col_max = torch.tensor(0.0)
            col_median = torch.tensor(0.0)

        result[:, col] = torch.where(torch.isneginf(col_data), col_min, result[:, col])
        result[:, col] = torch.where(torch.isposinf(col_data), col_max, result[:, col])
        result[:, col] = torch.where(torch.isnan(col_data), col_median, result[:, col])

    return result


def _expand_fc_parameters(fc_params: NoldsFCParameters) -> list[tuple[str, dict[str, Any]]]:
    """Expand FCParameters into a list of (feature_key, params) tuples.

    :param fc_params: Feature configuration mapping.
    :return: List of (feature_key, params) where feature_key encodes the parameters.
    """
    feature_functions = _get_nolds_feature_functions()
    expanded: list[tuple[str, dict[str, Any]]] = []

    for feature_name, param_list in fc_params.items():
        if feature_name not in feature_functions:
            raise ValueError(
                f"Unknown feature '{feature_name}'. Available: {list(feature_functions.keys())}"
            )

        if param_list is None:
            expanded.append((feature_name, {}))
        else:
            for params in param_list:
                param_suffix = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
                feature_key = f"{feature_name}__{param_suffix}" if param_suffix else feature_name
                expanded.append((feature_key, params))

    return expanded


class NoldsFeatureExtractor(FeatureExtractor):
    """Feature extractor using nolds for nonlinear dynamics analysis.

    Computes nonlinear dynamics features for each trajectory with multiprocessing
    parallelization. Uses tsfresh-style FCParameters configuration for specifying
    which features to extract and with what parameters.

    Available features (passed directly to nolds):
        * ``lyap_r``: Largest Lyapunov exponent (Rosenstein's algorithm)
        * ``lyap_e``: Largest Lyapunov exponent (Eckmann's algorithm)
        * ``sampen``: Sample entropy
        * ``hurst_rs``: Hurst exponent (R/S analysis)
        * ``corr_dim``: Correlation dimension
        * ``dfa``: Detrended fluctuation analysis
        * ``mfhurst_b``: Multifractal Hurst exponent (basic method)
        * ``mfhurst_dm``: Multifractal Hurst exponent (DM method)

    ```python
    # Default: extract lyap_r and corr_dim from all states
    extractor = NoldsFeatureExtractor(time_steady=9.0)

    # Only extract Lyapunov exponents with custom parameters
    extractor = NoldsFeatureExtractor(
        time_steady=9.0,
        features={"lyap_r": [{"emb_dim": 15}]},
    )

    # Per-state configuration
    extractor = NoldsFeatureExtractor(
        time_steady=9.0,
        features=None,
        features_per_state={
            0: {"lyap_r": None},
            1: {"corr_dim": [{"emb_dim": 10}]},
        },
    )

    # Multiple parameter sets for same feature
    extractor = NoldsFeatureExtractor(
        time_steady=9.0,
        features={
            "lyap_r": [
                {"emb_dim": 5},
                {"emb_dim": 10},
            ],
        },
    )
    ```

    :param time_steady: Time threshold for filtering transients. Default 0.0.
    :param features: Feature configuration for all states. Can be:
        * NoldsFCParameters dict: Feature names mapped to parameter lists
        * None: Skip states not in features_per_state
        Default extracts both lyap_r and corr_dim with nolds defaults.
    :param features_per_state: Optional dict mapping state indices to FCParameters.
        Overrides `features` for specified states. Use None to skip a state.
    :param n_jobs: Number of worker processes. If None, uses all CPU cores.

    :raises ImportError: If nolds library is not installed.
    """

    def __init__(
        self,
        time_steady: float = 0.0,
        features: NoldsFCParameters | None = None,
        features_per_state: dict[int, NoldsFCParameters | None] | None = None,
        n_jobs: int | None = None,
    ):
        if not _nolds_available:
            raise ImportError(
                "nolds library is required for NoldsFeatureExtractor. "
                "Install it with: pip install nolds"
            )

        super().__init__(time_steady=time_steady)

        self.n_jobs = n_jobs if n_jobs is not None else (os.cpu_count() or 1)

        if features is None:
            self.features: NoldsFCParameters | None = NOLDS_DEFAULT_FC_PARAMETERS
        else:
            self.features = features

        self.features_per_state = features_per_state or {}

        self._resulting_features_config: dict[int, NoldsFCParameters] | None = None
        self._num_states: int | None = None
        self._extracted_feature_names: list[str] | None = None

    def _configure_resulting_features_config(self, num_states: int) -> None:
        """Configure which features to compute for each state."""
        if self._resulting_features_config is not None and self._num_states == num_states:
            return

        self._num_states = num_states
        self._resulting_features_config = {}

        for state_idx in range(num_states):
            if state_idx in self.features_per_state:
                fc_params = self.features_per_state[state_idx]
                if fc_params is not None:
                    self._resulting_features_config[state_idx] = fc_params
            elif self.features is not None:
                self._resulting_features_config[state_idx] = self.features

    def _build_task_list(
        self,
        y_np: NDArray[np.floating],
        batch_size: int,
    ) -> tuple[list[WorkerTaskArgs], list[str]]:
        """Build list of tasks for parallel processing and feature names."""
        assert self._resulting_features_config is not None

        tasks: list[WorkerTaskArgs] = []
        feature_names: list[str] = []

        for state_idx in sorted(self._resulting_features_config.keys()):
            fc_params = self._resulting_features_config[state_idx]
            expanded = _expand_fc_parameters(fc_params)

            for feature_key, params in expanded:
                full_feature_name = f"state_{state_idx}__{feature_key}"
                feature_names.append(full_feature_name)

                for batch_idx in range(batch_size):
                    time_series = y_np[:, batch_idx, state_idx]
                    tasks.append((batch_idx, state_idx, feature_key, params, time_series))

        return tasks, feature_names

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract nolds features from an ODE solution.

        :param solution: ODE solution with shape (N, B, S).
        :return: Features tensor of shape (B, F) where F depends on configuration.
        """
        y_filtered = self.filter_time(solution)
        _n, batch_size, num_states = y_filtered.shape

        self._configure_resulting_features_config(num_states)
        assert self._resulting_features_config is not None

        if not self._resulting_features_config:
            self._extracted_feature_names = []
            return torch.zeros((batch_size, 0), dtype=torch.float32)

        y_np = y_filtered.cpu().numpy()
        tasks, feature_names = self._build_task_list(y_np, batch_size)
        self._extracted_feature_names = feature_names

        with mp.Pool(processes=self.n_jobs) as pool:
            results = pool.map(_worker_task, tasks)

        feature_name_to_col: dict[str, int] = {name: i for i, name in enumerate(feature_names)}

        features = torch.full((batch_size, len(feature_names)), float("nan"), dtype=torch.float32)

        for batch_idx, state_idx, feature_key, value in results:
            col_name = f"state_{state_idx}__{feature_key}"
            col_idx = feature_name_to_col[col_name]
            features[batch_idx, col_idx] = value

        features = _impute_torch(features)

        return features

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature names in format 'state_X__feature__params'.

        :raises RuntimeError: If extract_features has not been called yet.
        """
        if self._extracted_feature_names is None:
            raise RuntimeError("Feature names not available. Call extract_features first.")
        return self._extracted_feature_names
