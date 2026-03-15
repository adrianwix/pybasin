# pyright: basic
"""PyTorch-based feature extractor for ODE solution trajectories.

This module provides a high-performance feature extractor using PyTorch for
time series feature extraction from ODE solutions, supporting both CPU (with
multiprocessing) and GPU (with batched CUDA operations).
"""

import os
import threading
from typing import Literal

import torch
from torch import Tensor

from pybasin.feature_extractors.feature_extractor import FeatureExtractor
from pybasin.feature_extractors.utils import format_feature_name
from pybasin.solution import Solution
from pybasin.ts_torch.settings import (
    TORCH_COMPREHENSIVE_FC_PARAMETERS,
    TORCH_MINIMAL_FC_PARAMETERS,
    FCParameters,
)
from pybasin.ts_torch.torch_feature_processors import (
    extract_features_gpu_batched,
    extract_features_parallel,
)
from pybasin.ts_torch.torch_feature_utilities import impute, impute_extreme
from pybasin.ts_torch.utils import get_feature_names_from_config


class TorchFeatureExtractor(FeatureExtractor):
    """PyTorch-based feature extractor for time series features.

    Supports per-state variable feature configuration using tsfresh-style FCParameters
    dictionaries, allowing different feature sets for different state variables.

    For CPU extraction, uses multiprocessing to parallelize across batches.
    For GPU extraction, uses batched CUDA operations for optimal performance.

    ```python
    # Default: use comprehensive features for all states on CPU
    extractor = TorchFeatureExtractor(time_steady=9.0)

    # GPU extraction with default features
    extractor = TorchFeatureExtractor(time_steady=9.0, device="gpu")

    # Custom features for specific states, skip others
    extractor = TorchFeatureExtractor(
        time_steady=9.0,
        features=None,  # Don't extract features by default
        features_per_state={
            1: {"maximum": None, "minimum": None},  # Only extract for state 1
        },
    )

    # Global features with per-state override
    extractor = TorchFeatureExtractor(
        time_steady=9.0,
        features_per_state={
            0: {"maximum": None},  # Override state 0
            1: None,  # Skip state 1
        },
    )
    ```

    :param time_steady: Time threshold for filtering transients. If ``None`` (default),
        uses 85% of the integration time span. Set to ``0.0`` to use the entire series.
    :param features: Default feature configuration to apply to all states. Can be:

        - 'comprehensive': Use TORCH_COMPREHENSIVE_FC_PARAMETERS (default)
        - 'minimal': Use TORCH_MINIMAL_FC_PARAMETERS (10 basic features)
        - FCParameters dict: Custom feature configuration
        - None: Skip states not explicitly configured in features_per_state

    :param features_per_state: Optional dict mapping state indices to FCParameters.
        Overrides `features` for specified states. Use None as value to skip
        a state. States not in this dict use the global `features` config.
    :param normalize: Whether to apply z-score normalization. Default True.
    :param device: Execution device ('cpu' or 'gpu'). Default 'cpu'.
    :param n_jobs: Number of worker processes for CPU extraction. If None, uses all
        available CPU cores. Ignored when device='gpu'.
    :param impute_method: Method for handling NaN/inf values in features:

        - 'extreme': Replace with extreme values (1e10) to distinguish unbounded trajectories. Best for systems with divergent solutions. (default)
        - 'tsfresh': Replace using tsfresh-style imputation (inf->max/min, NaN->median). Better when all trajectories are bounded.

    :raises RuntimeError: If device='gpu' but CUDA is not available.
    """

    def __init__(
        self,
        time_steady: float | None = None,
        features: Literal["comprehensive", "minimal"] | FCParameters | None = "comprehensive",
        features_per_state: dict[int, FCParameters | None] | None = None,
        normalize: bool = True,
        device: Literal["cpu", "gpu"] = "cpu",
        n_jobs: int | None = None,
        impute_method: Literal["extreme", "tsfresh"] = "extreme",
    ):
        super().__init__(time_steady=time_steady)

        if device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available but device='gpu' was requested")

        self.normalize = normalize
        self.device = device
        self.n_jobs = n_jobs if n_jobs is not None else (os.cpu_count() or 1)
        self.impute_method = impute_method
        self._is_fitted = False
        # Review what is this
        self._fit_lock = threading.Lock()

        self._feature_mean: Tensor | None = None
        self._feature_std: Tensor | None = None

        if features == "comprehensive":
            self.features: FCParameters | None = TORCH_COMPREHENSIVE_FC_PARAMETERS
        elif features == "minimal":
            self.features = TORCH_MINIMAL_FC_PARAMETERS
        else:
            self.features = features
        self.features_per_state = features_per_state or {}

        self._resulting_features_config: dict[int, FCParameters] | None = None
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

    def _is_uniform_config(self) -> bool:
        """Check if all states use the same feature configuration."""
        if not self._resulting_features_config:
            return True
        configs = list(self._resulting_features_config.values())
        if len(configs) <= 1:
            return True
        first = configs[0]
        return all(c is first for c in configs[1:])

    def _extract_all_states(self, y: Tensor, fc_params: FCParameters) -> dict[str, Tensor]:
        """Extract features for all states together (optimized path).

        :param y: Tensor of shape (N, B, S) where N=timesteps, B=batch size, S=states.
        :param fc_params: Feature configuration for all states.
        :return: Dictionary mapping feature names to result tensors of shape (B, S).
        """
        if self.device == "gpu":
            y = y.cuda() if not y.is_cuda else y
            return extract_features_gpu_batched(y, fc_params, use_gpu_friendly=False)
        else:
            y = y.cpu() if y.is_cuda else y
            return extract_features_parallel(y, fc_params, n_workers=self.n_jobs)

    def _extract_for_state(self, y_state: Tensor, fc_params: FCParameters) -> dict[str, Tensor]:
        """Extract features for a single state variable.

        :param y_state: Tensor of shape (N, B, 1) where N=timesteps, B=batch size, for one state.
        :param fc_params: Feature configuration for this state.
        :return: Dictionary mapping feature names to result tensors of shape (B,).
        """
        if self.device == "gpu":
            y_state = y_state.cuda() if not y_state.is_cuda else y_state
            results = extract_features_gpu_batched(y_state, fc_params, use_gpu_friendly=False)
        else:
            y_state = y_state.cpu() if y_state.is_cuda else y_state
            results = extract_features_parallel(y_state, fc_params, n_workers=self.n_jobs)

        flattened: dict[str, Tensor] = {}
        for fname, tensor in results.items():
            if tensor.dim() == 2:
                flattened[fname] = tensor[:, 0]
            else:
                flattened[fname] = tensor

        return flattened

    def extract_features(self, solution: Solution) -> torch.Tensor:
        """Extract features from an ODE solution using PyTorch.

        :param solution: ODE solution containing time series data with y tensor of shape
            (N, B, S) where N=timesteps, B=batch size, S=state variables.
        :return: Feature tensor of shape (B, F) where F is the total number of features.
        """
        y = self.filter_time(solution)

        num_states = y.shape[2]
        if self._resulting_features_config is None:
            self._configure_resulting_features_config(num_states)

        assert self._resulting_features_config is not None

        if not self._resulting_features_config:
            n_batches = y.shape[1]
            return torch.zeros((n_batches, 0), dtype=y.dtype, device=y.device)

        all_features: list[Tensor] = []
        all_feature_names: list[str] = []

        if self._is_uniform_config() and self._resulting_features_config:
            fc_params = next(iter(self._resulting_features_config.values()))
            state_indices = list(self._resulting_features_config.keys())
            y_selected = y[:, :, state_indices]

            results = self._extract_all_states(y_selected, fc_params)

            feature_names = get_feature_names_from_config(fc_params, include_custom=False)
            for state_pos, state_idx in enumerate(state_indices):
                for fname in feature_names:
                    if fname in results:
                        feature_tensor = results[fname][:, state_pos]
                        if feature_tensor.dim() > 1:
                            for i in range(feature_tensor.shape[-1]):
                                all_features.append(feature_tensor[:, i])
                                all_feature_names.append(
                                    format_feature_name(f"{fname}__{i}", state_index=state_idx)
                                )
                        else:
                            all_features.append(feature_tensor)
                            all_feature_names.append(
                                format_feature_name(fname, state_index=state_idx)
                            )
        else:
            for state_idx, fc_params in self._resulting_features_config.items():
                y_state = y[:, :, state_idx : state_idx + 1]

                state_results = self._extract_for_state(y_state, fc_params)

                feature_names = get_feature_names_from_config(fc_params, include_custom=False)
                for fname in feature_names:
                    if fname in state_results:
                        feature_tensor = state_results[fname]
                        if feature_tensor.dim() > 1:
                            for i in range(feature_tensor.shape[-1]):
                                all_features.append(feature_tensor[:, i])
                                all_feature_names.append(
                                    format_feature_name(f"{fname}__{i}", state_index=state_idx)
                                )
                        else:
                            all_features.append(feature_tensor)
                            all_feature_names.append(
                                format_feature_name(fname, state_index=state_idx)
                            )

        if not all_features:
            n_batches = y.shape[1]
            return torch.zeros((n_batches, 0), dtype=y.dtype, device=y.device)

        self._extracted_feature_names = all_feature_names
        features = torch.stack(all_features, dim=1)

        if self.device == "gpu":
            features = features.cpu()

        features = impute_extreme(features) if self.impute_method == "extreme" else impute(features)

        if self.normalize:
            with self._fit_lock:
                if not self._is_fitted:
                    self._feature_mean = features.mean(dim=0, keepdim=True)
                    self._feature_std = features.std(dim=0, keepdim=True)
                    self._feature_std = torch.where(
                        self._feature_std == 0,
                        torch.ones_like(self._feature_std),
                        self._feature_std,
                    )
                    self._is_fitted = True

                assert self._feature_mean is not None
                assert self._feature_std is not None
                features = (features - self._feature_mean) / self._feature_std

        return features

    def reset_scaler(self) -> None:
        """Reset the normalization parameters."""
        with self._fit_lock:
            self._feature_mean = None
            self._feature_std = None
            self._is_fitted = False

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature names in the format 'state_X__feature_name'."""
        if self._extracted_feature_names is None:
            raise RuntimeError("Feature names not available. Call extract_features first.")
        return self._extracted_feature_names
