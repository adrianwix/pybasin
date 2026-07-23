# pyright: basic
"""Compiled PyTorch feature extractor for BST time series tensors.

Every requested feature - parameter-free or parametrized - goes through one uniform
mechanism: a feature class (see `tstorch.feature_factory`) is constructed directly - with
the distinct param dicts requested for it (deduped across every state) for a `Feature`
subclass, with no arguments for a `ParameterlessFeature` subclass - and the constructed
instance itself is the closure; `_compile()` composes every requested feature's closure
into a single `torch.cat([f(y) for f in closures], dim=-1)` kernel - no factory/adapter
layer in between.

This file does not build any features itself - each calculator file that contributes an
extractor-eligible feature exposes its own feature class, and `SUPPORTED_FEATURES` below
maps feature name to class directly. Requesting anything else raises instead of silently
skipping it.
"""

import threading
from collections.abc import Callable, Hashable
from typing import Any, Literal, NamedTuple, Protocol

import torch
from torch import Tensor

from tstorch.calculators.torch_features_advanced import (
    C3,
    BenfordCorrelation,
    EnergyRatioByChunks,
    TimeReversalAsymmetryStatistic,
)
from tstorch.calculators.torch_features_autocorrelation import (
    AggAutocorrelation,
    Autocorrelation,
    AutocorrelationPeriodicity,
    PartialAutocorrelation,
)
from tstorch.calculators.torch_features_boolean import (
    HasDuplicate,
    HasDuplicateMax,
    HasDuplicateMin,
    HasVarianceLargerThanStandardDeviation,
    LargeStandardDeviation,
)
from tstorch.calculators.torch_features_change import (
    AbsoluteSumOfChanges,
    ChangeQuantiles,
    MeanAbsChange,
    MeanChange,
    MeanSecondDerivativeCentral,
)
from tstorch.calculators.torch_features_count import (
    CountAbove,
    CountAboveMean,
    CountBelow,
    CountBelowMean,
    CountInRange,
    CountValue,
)
from tstorch.calculators.torch_features_entropy_complexity import (
    ApproximateEntropy,
    BinnedEntropy,
    CidCe,
    FourierEntropy,
    LempelZivComplexity,
    PermutationEntropy,
    SampleEntropy,
)
from tstorch.calculators.torch_features_frequency import (
    CwtCoefficients,
    FftAggregated,
    FftCoefficient,
    SpectralFrequencyRatio,
    SpktWelchDensity,
)
from tstorch.calculators.torch_features_location import (
    FirstLocationOfMaximum,
    FirstLocationOfMinimum,
    IndexMassQuantile,
    LastLocationOfMaximum,
    LastLocationOfMinimum,
)
from tstorch.calculators.torch_features_pattern import (
    LongestStrikeAboveMean,
    LongestStrikeBelowMean,
    NumberCrossingM,
    NumberCwtPeaks,
    NumberPeaks,
)
from tstorch.calculators.torch_features_reocurrance import (
    PercentageOfReoccurringDatapointsToAllDatapoints,
    PercentageOfReoccurringValuesToAllValues,
    RatioValueNumberToTimeSeriesLength,
    SumOfReoccurringDataPoints,
    SumOfReoccurringValues,
)
from tstorch.calculators.torch_features_statistical import (
    AbsEnergy,
    AbsoluteMaximum,
    Amplitude,
    Delta,
    Kurtosis,
    Length,
    LogDelta,
    Maximum,
    Mean,
    MeanNAbsoluteMax,
    Median,
    Minimum,
    Quantile,
    RatioBeyondRSigma,
    RootMeanSquare,
    Skewness,
    StandardDeviation,
    SumValues,
    SymmetryLooking,
    Variance,
    VariationCoefficient,
)
from tstorch.calculators.torch_features_trend import (
    AggLinearTrend,
    ArCoefficient,
    AugmentedDickeyFuller,
    LinearTrend,
    LinearTrendTimewise,
)
from tstorch.feature_factory import (
    Feature,
    ParameterlessFeature,
    construct_feature,
    is_parameterless,
    param_key,
)
from tstorch.settings import MINIMAL_FEATURE_NAMES, FCParameters
from tstorch.torch_feature_utilities import impute, impute_extreme

DEFAULT_STEADY_FRACTION = 0.85

# feature_name -> Feature class. Every feature, parameter-free or parametrized, goes
# through the same `_compile()`/`extract_features()` path; `is_parameterless()` picks
# `feature_cls()` vs `feature_cls(params)` since the two protocols construct differently.
SUPPORTED_FEATURES: dict[str, type[Feature[Any]] | type[ParameterlessFeature]] = {
    "benford_correlation": BenfordCorrelation,
    "c3": C3,
    "energy_ratio_by_chunks": EnergyRatioByChunks,
    "time_reversal_asymmetry_statistic": TimeReversalAsymmetryStatistic,
    "autocorrelation": Autocorrelation,
    "partial_autocorrelation": PartialAutocorrelation,
    "agg_autocorrelation": AggAutocorrelation,
    "autocorrelation_periodicity": AutocorrelationPeriodicity,
    "has_duplicate": HasDuplicate,
    "has_duplicate_max": HasDuplicateMax,
    "has_duplicate_min": HasDuplicateMin,
    "has_variance_larger_than_standard_deviation": HasVarianceLargerThanStandardDeviation,
    "large_standard_deviation": LargeStandardDeviation,
    "absolute_sum_of_changes": AbsoluteSumOfChanges,
    "mean_abs_change": MeanAbsChange,
    "mean_change": MeanChange,
    "mean_second_derivative_central": MeanSecondDerivativeCentral,
    "change_quantiles": ChangeQuantiles,
    "count_above": CountAbove,
    "count_above_mean": CountAboveMean,
    "count_below": CountBelow,
    "count_below_mean": CountBelowMean,
    "count_in_range": CountInRange,
    "count_value": CountValue,
    "permutation_entropy": PermutationEntropy,
    "binned_entropy": BinnedEntropy,
    "fourier_entropy": FourierEntropy,
    "lempel_ziv_complexity": LempelZivComplexity,
    "cid_ce": CidCe,
    "approximate_entropy": ApproximateEntropy,
    "sample_entropy": SampleEntropy,
    "fft_coefficient": FftCoefficient,
    "fft_aggregated": FftAggregated,
    "spkt_welch_density": SpktWelchDensity,
    "cwt_coefficients": CwtCoefficients,
    "spectral_frequency_ratio": SpectralFrequencyRatio,
    "first_location_of_maximum": FirstLocationOfMaximum,
    "first_location_of_minimum": FirstLocationOfMinimum,
    "last_location_of_maximum": LastLocationOfMaximum,
    "last_location_of_minimum": LastLocationOfMinimum,
    "index_mass_quantile": IndexMassQuantile,
    "longest_strike_above_mean": LongestStrikeAboveMean,
    "longest_strike_below_mean": LongestStrikeBelowMean,
    "number_crossing_m": NumberCrossingM,
    "number_cwt_peaks": NumberCwtPeaks,
    "number_peaks": NumberPeaks,
    "percentage_of_reoccurring_datapoints_to_all_datapoints": (
        PercentageOfReoccurringDatapointsToAllDatapoints
    ),
    "percentage_of_reoccurring_values_to_all_values": (PercentageOfReoccurringValuesToAllValues),
    "sum_of_reoccurring_data_points": SumOfReoccurringDataPoints,
    "sum_of_reoccurring_values": SumOfReoccurringValues,
    "ratio_value_number_to_time_series_length": RatioValueNumberToTimeSeriesLength,
    "sum_values": SumValues,
    "median": Median,
    "mean": Mean,
    "length": Length,
    "standard_deviation": StandardDeviation,
    "variance": Variance,
    "root_mean_square": RootMeanSquare,
    "maximum": Maximum,
    "absolute_maximum": AbsoluteMaximum,
    "minimum": Minimum,
    "delta": Delta,
    "log_delta": LogDelta,
    "abs_energy": AbsEnergy,
    "kurtosis": Kurtosis,
    "skewness": Skewness,
    "quantile": Quantile,
    "variation_coefficient": VariationCoefficient,
    "mean_n_absolute_max": MeanNAbsoluteMax,
    "ratio_beyond_r_sigma": RatioBeyondRSigma,
    "symmetry_looking": SymmetryLooking,
    "amplitude": Amplitude,
    "linear_trend": LinearTrend,
    "linear_trend_timewise": LinearTrendTimewise,
    "agg_linear_trend": AggLinearTrend,
    "ar_coefficient": ArCoefficient,
    "augmented_dickey_fuller": AugmentedDickeyFuller,
}


class TimeSeriesSolution(Protocol):
    """Protocol for objects accepted by :class:`TorchFeatureExtractor`.

    :ivar time: Time values of shape (T,).
    :ivar y: Time series values of shape (B, S, T).
    """

    time: Tensor
    y: Tensor


class _FeatureSpec(NamedTuple):
    """One output column requested by the compiled feature plan."""

    state_idx: int
    feature_name: str
    param_key: Hashable
    output_name: str


class _CompiledFeaturePlan(NamedTuple):
    """Feature plan specialized to a state count and tensor device."""

    specs: list[_FeatureSpec]
    num_states: int
    device: torch.device
    # (feature_name, param_key) -> absolute column index in kernel(y)'s (B, S, F) output.
    column_of: dict[tuple[str, Hashable], int]
    # `torch.cat([f(y) for f in closures], dim=-1)`, closed over exactly the requested
    # closures, optionally `torch.compile`'d as one traced graph.
    kernel: Callable[[Tensor], Tensor]


def _normalize_fc_parameters(features: FCParameters) -> dict[str, list[dict[str, Any]] | None]:
    """Validate and freeze a per-state feature configuration.

    :raises ValueError: If a requested feature isn't in ``SUPPORTED_FEATURES``.
    """
    unsupported = [name for name in features if name not in SUPPORTED_FEATURES]
    if unsupported:
        raise ValueError(
            f"Unsupported feature(s) {unsupported}. Only these are wired in right now: "
            f"{sorted(SUPPORTED_FEATURES)}"
        )
    result: dict[str, list[dict[str, Any]] | None] = {}
    for name, params in features.items():
        if params is None:
            result[name] = None
        elif isinstance(params, list):
            result[name] = params
        else:
            result[name] = [dict(params)]
    return result


class TorchFeatureExtractor:
    """Configurable compiled feature extractor for ``(B, S, T)`` tensors.

    Compiles a state-major feature plan once per input shape/device: every requested
    (feature, params) combo is resolved to a closure by constructing its ``Feature``
    class directly (see ``SUPPORTED_FEATURES``) - the instance itself is the closure -
    and every closure is composed into a single ``kernel(y)`` call
    (``torch.cat([f(y) for f in closures], dim=-1)``), optionally wrapped with
    ``torch.compile`` (``use_compile``, default True) as one traced graph - no fallback:
    if ``torch.compile`` can't handle it, it raises, so the failure is visible instead of
    silently paying for both a compiled and an eager path. ``extract_features`` then does
    exactly one kernel call plus plain per-spec indexing (each spec selects one
    ``(state, feature, params)`` cell from the kernel's output with two Python ints - a
    view, not a copy - no gather needed). Reusing an instance for a different
    ``(num_states, device)`` than it was first compiled for is also an error - construct a
    new extractor instead of silently recompiling.

    Only the calculators in ``SUPPORTED_FEATURES`` are supported; requesting anything else
    raises at construction time.

    :param time_steady: Time threshold for filtering transients when ``extract_features`` receives
        a solution object. If ``None``, uses 85% of the integration time span. Ignored when passing
        a raw BST tensor directly.
    :param features: Default feature configuration to apply to all states. ``"minimal"`` uses the
        tsfresh Minimal-equivalent 10 features, an ``FCParameters``-shaped mapping
        (``{name: None}`` for parameter-free, ``{name: [params, ...]}`` for parametrized) uses
        exactly that, and ``None`` skips states not configured in ``features_per_state``.
    :param features_per_state: Optional mapping from state index to feature configuration. Values
        override ``features`` for that state. Use ``None`` to skip a state.
    :param normalize: Whether to apply z-score normalization after imputation.
    :param impute_method: Imputation strategy for non-finite features. ``"extreme"`` replaces
        non-finite values with large sentinel values; ``"tsfresh"`` uses columnwise finite
        extrema/median imputation.
    :param use_compile: Whether to wrap the composed kernel with ``torch.compile``.
    """

    def __init__(
        self,
        time_steady: float | None = None,
        features: Literal["minimal"] | FCParameters | None = "minimal",
        features_per_state: dict[int, FCParameters | None] | None = None,
        normalize: bool = True,
        impute_method: Literal["extreme", "tsfresh"] = "extreme",
        use_compile: bool = True,
    ):
        self.time_steady: float | None = time_steady
        self.normalize: bool = normalize
        self.impute_method: Literal["extreme", "tsfresh"] = impute_method
        self.use_compile: bool = use_compile
        self._is_fitted: bool = False
        self._fit_lock = threading.Lock()

        self._feature_mean: Tensor | None = None
        self._feature_std: Tensor | None = None

        self.features: dict[str, list[dict[str, Any]] | None] | None = (
            _normalize_fc_parameters(dict.fromkeys(MINIMAL_FEATURE_NAMES))
            if features == "minimal"
            else (_normalize_fc_parameters(features) if features is not None else None)
        )
        self.features_per_state: dict[int, dict[str, list[dict[str, Any]] | None] | None] = {
            state_idx: (_normalize_fc_parameters(fc_params) if fc_params is not None else None)
            for state_idx, fc_params in (features_per_state or {}).items()
        }

        self._compiled_plan: _CompiledFeaturePlan | None = None
        self._extracted_feature_names: list[str] | None = None

    def _resolve_time_steady(self, solution: TimeSeriesSolution) -> float:
        """Resolve the effective steady-state time threshold.

        :param solution: Object with ``time`` of shape (T,).
        :return: Time threshold used to filter transients.
        """
        if self.time_steady is not None:
            return self.time_steady
        t = solution.time
        t0 = float(t[0])
        t1 = float(t[-1])
        return t0 + DEFAULT_STEADY_FRACTION * (t1 - t0)

    def _coerce_input(self, data: Tensor | TimeSeriesSolution) -> Tensor:
        """Return a BST tensor from a raw tensor or solution object.

        :param data: Raw tensor of shape ``(B, S, T)`` or an object with ``time`` and ``y``.
        :return: Tensor of shape ``(B, S, T)``.
        :raises ValueError: If the tensor is not three-dimensional.
        """
        if isinstance(data, Tensor):
            y = data
        else:
            effective_steady = self._resolve_time_steady(data)
            idx_steady = torch.where(data.time >= effective_steady)[0]
            start_idx = idx_steady[0] if len(idx_steady) > 0 else 0
            y = data.y[:, :, start_idx:]

        if y.ndim != 3:
            raise ValueError(f"Expected a BST tensor with shape (B, S, T), got {tuple(y.shape)}")
        if y.stride(-1) != 1:
            raise ValueError("Expected the time dimension to be contiguous with stride 1")
        return y

    def _state_feature_config(
        self, num_states: int
    ) -> dict[int, dict[str, list[dict[str, Any]] | None]]:
        """Resolve the configured feature/params mapping for each state."""
        config: dict[int, dict[str, list[dict[str, Any]] | None]] = {}
        for state_idx in range(num_states):
            state_features = self.features_per_state.get(state_idx, self.features)
            if state_features is not None:
                config[state_idx] = state_features
        return config

    def _compile(self, num_states: int, device: torch.device) -> None:
        """Compile the configured states/features for the current input shape/device."""
        state_config = self._state_feature_config(num_states)

        # Per feature name: the distinct param dicts requested for it anywhere, deduped
        # via the factory's param_key, first-seen order - this is what lets a Type B
        # feature's shared computation (e.g. agg_autocorrelation's ACF) run once for
        # every distinct combo instead of once per requesting state.
        unique_params_by_feature: dict[str, list[dict[str, Any]]] = {}
        seen_keys_by_feature: dict[str, set[Hashable]] = {}
        specs: list[_FeatureSpec] = []

        for state_idx in range(num_states):
            fc_params = state_config.get(state_idx)
            if fc_params is None:
                continue
            for feature_name, param_list in fc_params.items():
                feature_cls = SUPPORTED_FEATURES[feature_name]
                parameterless = is_parameterless(feature_cls)
                if parameterless:
                    if param_list is not None:
                        raise ValueError(f"{feature_name} takes no parameters; got {param_list}")
                    entries: list[dict[str, Any]] = [{}]
                else:
                    entries = param_list if param_list is not None else [{}]
                for params in entries:
                    key = param_key(params)
                    seen_keys = seen_keys_by_feature.setdefault(feature_name, set())
                    if key not in seen_keys:
                        seen_keys.add(key)
                        unique_params_by_feature.setdefault(feature_name, []).append(params)
                    # One-param throwaway instance just to read .names[0] - cheap relative
                    # to this per-(state, param) naming pass, and never part of the
                    # compiled/hot path (that's the shared closure built below). .names[0]
                    # is already the feature's complete, self-sufficient output name -
                    # this only adds the state prefix, uniformly for every class kind.
                    ctor_params = None if parameterless else [params]
                    name = construct_feature(feature_cls, ctor_params).names[0]
                    output_name = f"state_{state_idx}__{name}"
                    specs.append(_FeatureSpec(state_idx, feature_name, key, output_name))

        closures: list[Callable[[Tensor], Tensor]] = []
        column_of: dict[tuple[str, Hashable], int] = {}
        offset = 0
        for feature_name, params_list in unique_params_by_feature.items():
            feature_cls = SUPPORTED_FEATURES[feature_name]
            # The instance itself is the closure - constructed with no args for a
            # ParameterlessFeature, with the deduped params list otherwise.
            ctor_params = None if is_parameterless(feature_cls) else params_list
            closures.append(construct_feature(feature_cls, ctor_params))
            for params in params_list:
                column_of[(feature_name, param_key(params))] = offset
                offset += 1

        def kernel_eager(
            y: Tensor, closures: tuple[Callable[[Tensor], Tensor], ...] = tuple(closures)
        ) -> Tensor:
            return torch.cat([f(y) for f in closures], dim=-1)

        kernel = torch.compile(kernel_eager, dynamic=True) if self.use_compile else kernel_eager

        self._compiled_plan = _CompiledFeaturePlan(
            specs=specs,
            num_states=num_states,
            device=device,
            column_of=column_of,
            kernel=kernel,
        )

    def extract_features(self, data: Tensor | TimeSeriesSolution) -> Tensor:
        """Extract features from a raw BST tensor or a solution object.

        :param data: Tensor with shape ``(B, S, T)`` or an object whose ``y`` tensor has that shape.
        :return: Feature tensor of shape ``(B, F)`` on the same device as the input tensor.
        """
        y = self._coerce_input(data)
        num_states = y.shape[1]

        if self._compiled_plan is None:
            self._compile(num_states, y.device)
        elif self._compiled_plan.num_states != num_states or self._compiled_plan.device != y.device:
            raise ValueError(
                "This extractor was compiled for "
                f"(num_states={self._compiled_plan.num_states}, device={self._compiled_plan.device}); "
                f"got (num_states={num_states}, device={y.device}). Construct a new extractor "
                "instead of silently recompiling for a different shape/device."
            )
        assert self._compiled_plan is not None
        plan = self._compiled_plan

        if not plan.specs:
            self._extracted_feature_names = []
            return torch.zeros((y.shape[0], 0), dtype=y.dtype, device=y.device)

        with torch.no_grad():
            # torch.compile failures raise straight through - no silent eager fallback.
            cube = plan.kernel(y)  # (B, S, total distinct (feature, params) columns)

            all_features = [
                cube[:, spec.state_idx, plan.column_of[(spec.feature_name, spec.param_key)]]
                for spec in plan.specs
            ]
            all_feature_names = [spec.output_name for spec in plan.specs]

            features = torch.stack(all_features, dim=1)
            features = (
                impute_extreme(features) if self.impute_method == "extreme" else impute(features)
            )

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
                    features = (features - self._feature_mean.to(features.device)) / (
                        self._feature_std.to(features.device)
                    )

        self._extracted_feature_names = all_feature_names
        return features

    def reset_scaler(self) -> None:
        """Reset the normalization parameters."""
        with self._fit_lock:
            self._feature_mean = None
            self._feature_std = None
            self._is_fitted = False

    @property
    def feature_names(self) -> list[str]:
        """Return feature names in the format ``state_X__feature_name``."""
        if self._extracted_feature_names is None:
            raise RuntimeError("Feature names not available. Call extract_features first.")
        return self._extracted_feature_names
