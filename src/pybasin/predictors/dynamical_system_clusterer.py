# pyright: basic
"""Dynamical system clusterer for two-stage attractor type classification."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin  # type: ignore[import-untyped]

from pybasin.feature_extractors.utils import (
    get_feature_indices_by_base_name,
    validate_feature_names,
)
from pybasin.predictors._device_backend import (
    DeviceBackend,
    resolve_backend,
    standardize,
    to_native,
    to_numpy,
)
from pybasin.predictors.hdbscan_clusterer import HDBSCANClusterer
from pybasin.predictors.tuned_clusterer import MinClusterSizeSearch, TunedClusterer
from pybasin.utils import DisplayNameMixin

_CHAOS, _FP, _LC = 0, 1, 2
_TYPE_NAMES = {_CHAOS: "chaos", _FP: "FP", _LC: "LC"}
_TYPE_CODES = {"chaos": _CHAOS, "FP": _FP, "LC": _LC}


class DynamicalSystemClusterer(DisplayNameMixin, ClusterMixin, BaseEstimator):
    """Two-stage hierarchical clustering for dynamical systems.

    This clusterer uses physics-based heuristics to classify trajectories into
    attractor types (Stage 1) and then sub-classifies within each type (Stage 2).

    Stage 1: Attractor Type Classification
    =======================================

    **Fixed Point (FP) Detection:**
        Heuristic: variance < fp_variance_threshold

        A trajectory is classified as converging to a fixed point if the variance
        of its steady-state values is extremely low. The threshold should be set
        based on the expected numerical precision of your integration.

        IMPORTANT: If features are normalized/scaled (e.g., StandardScaler), the
        variance values will be transformed. For normalized features with unit
        variance, use a threshold relative to 1.0 (e.g., 1e-4). For unnormalized
        features, use absolute thresholds based on your system's scale.

    **Limit Cycle (LC) Detection:**
        Heuristic: (periodicity_strength > lc_periodicity_threshold AND
                   variance < chaos_variance_threshold) OR has_drift

        A trajectory is classified as a limit cycle if:
        1. It shows strong periodic behavior (high autocorrelation periodicity)
           AND has bounded variance (not chaotic), OR
        2. It shows monotonic drift (rotating solutions like pendulum rotations)

        The periodicity_strength comes from autocorrelation analysis and ranges
        from 0 (no periodicity) to 1 (perfect periodicity). Values above 0.5
        typically indicate clear periodic behavior.

    **Chaos Detection:**
        Heuristic: NOT FP AND NOT LC (default fallback)

        Trajectories that don't meet FP or LC criteria are classified as chaotic.
        High variance combined with low periodicity strength indicates chaos.

    Stage 2: Sub-classification
    ===========================

    Within each attractor type, trajectories are further clustered:
    - FP: Clustered by steady-state location (mean values)
    - LC: Hierarchically clustered by period number, then amplitude/mean
    - Chaos: Clustered by spatial mean location

    Required Features
    =================
    Feature names must follow the convention: state_X__feature_name

    Required base features:
        - variance: Steady-state variance (FP detection)
        - amplitude: Peak-to-peak amplitude (LC sub-classification)
        - mean: Steady-state mean (FP/chaos sub-classification)
        - linear_trend__attr_slope: Linear drift rate (rotating LC detection)
        - autocorrelation_periodicity__output_strength: Periodicity measure [0-1]
        - autocorrelation_periodicity__output_period: Detected period
        - spectral_frequency_ratio: Ratio for period-n detection

    Note: This clusterer requires feature names to be set via set_feature_names()
    before calling fit_predict(). The BasinStabilityEstimator handles this
    automatically during the estimation process.
    """

    REQUIRED_FEATURES = [
        "variance",
        "amplitude",
        "mean",
        "linear_trend__attr_slope",
        "autocorrelation_periodicity__output_strength",
        "autocorrelation_periodicity__output_period",
        "spectral_frequency_ratio",
    ]

    def __init__(
        self,
        # General settings
        drift_threshold: float = 0.1,
        drift_fraction: float = 0.3,
        tiers: list[str] | None = None,
        # Fixed Point (FP) settings
        fp_variance_threshold: float = 1e-6,
        fp_sub_classifier: Any = None,
        # Limit Cycle (LC) settings
        lc_periodicity_threshold: float = 0.5,
        lc_sub_classifier: Any = None,
        # Chaos settings
        chaos_variance_threshold: float = 5.0,
        chaos_sub_classifier: Any = None,
    ):
        """Initialize the dynamical system clusterer.

        :param drift_threshold: Minimum |slope| to consider a dimension as drifting.
            Drifting dimensions (e.g., pendulum angle during rotation) are
            excluded from variance/mean calculations for FP and chaos
            sub-classification to avoid spurious splits. Also used to detect
            rotating limit cycles. Units: [state_units / time_units]. Default: 0.1.
        :param drift_fraction: Minimum fraction of trajectories with
            |slope| > drift_threshold for a dimension to be flagged as drifting.
            Default: 0.3 (i.e., 30% of trajectories must show drift).
        :param tiers: List of attractor types to detect, in priority order.
            First matching tier wins. Options: "FP", "LC", "chaos".
            Default: ["FP", "LC", "chaos"].
        :param fp_variance_threshold: Maximum variance to classify as fixed point.
            For unnormalized features, set based on expected steady-state
            fluctuations (e.g., 1e-6 for well-converged integrations).
            For normalized features (unit variance), use relative threshold
            (e.g., 1e-4 meaning 0.01% of typical variance). Default: 1e-6.
        :param fp_sub_classifier: Custom sub-classifier for fixed points.
            Input: mean values per non-drifting dimension. Default: HDBSCAN with
            min_cluster_size=50.
        :param lc_periodicity_threshold: Minimum periodicity strength [0-1] to
            classify as limit cycle. The periodicity strength measures how
            well the autocorrelation matches periodic behavior (0.0 = no periodic
            pattern, 0.3-0.5 = weak/noisy, 0.5-0.8 = clear periodic, 0.8-1.0 =
            strong/clean limit cycle). Default: 0.5.
        :param lc_sub_classifier: Custom sub-classifier for limit cycles.
            Input: [freq_ratio, amplitude, mean] features. Default: Hierarchical
            period-based clustering.
        :param chaos_variance_threshold: Maximum variance for limit cycle.
            Trajectories with variance above this AND low periodicity are
            classified as chaotic. Set based on expected LC amplitude range.
            For normalized features, typical LC variance is ~0.5-2.0. Default: 5.0.
        :param chaos_sub_classifier: Custom sub-classifier for chaotic attractors.
            Input: mean values per dimension. Default: HDBSCAN, auto-tuned.
        """
        self.feature_names: list[str] | None = None

        # General settings
        self.drift_threshold = drift_threshold
        self.drift_fraction = drift_fraction
        self.tiers = tiers or ["FP", "LC", "chaos"]

        # FP settings
        self.fp_variance_threshold = fp_variance_threshold
        self.fp_sub_classifier = fp_sub_classifier

        # LC settings
        self.lc_periodicity_threshold = lc_periodicity_threshold
        self.lc_sub_classifier = lc_sub_classifier

        # Chaos settings
        self.chaos_variance_threshold = chaos_variance_threshold
        self.chaos_sub_classifier = chaos_sub_classifier

        self._feature_indices: dict[str, list[int]] = {}
        self._drifting_dims: list[int] = []
        self._non_drifting_dims: list[int] = []
        self._initialized = False

    def needs_feature_names(self) -> bool:
        """This clusterer requires feature names to parse physics-based features."""
        return True

    def set_feature_names(self, feature_names: list[str]) -> None:
        """Set feature names and build feature indices.

        :param feature_names: List of feature names matching the feature array columns.
        """
        self.feature_names = feature_names
        self._build_feature_indices(feature_names)
        self._initialized = True

    def _build_feature_indices(self, feature_names: list[str]) -> None:
        """Build mapping from feature base names to column indices.

        Uses the pybasin.utils feature name parsing utilities.

        :param feature_names: List of feature names.
        :raises ValueError: If required features are missing or names are invalid.
        """
        all_valid, invalid_names = validate_feature_names(feature_names)
        if not all_valid:
            raise ValueError(
                f"Feature names do not follow the 'state_X__feature_name' convention. "
                f"Invalid names: {invalid_names[:5]}{'...' if len(invalid_names) > 5 else ''}"
            )

        self._feature_indices = {}

        for base_feature in self.REQUIRED_FEATURES:
            matching_indices = get_feature_indices_by_base_name(feature_names, base_feature)

            if not matching_indices:
                raise ValueError(
                    f"Required feature '{base_feature}' not found in feature_names. "
                    f"Available: {feature_names}"
                )

            self._feature_indices[base_feature] = matching_indices

    def _detect_drifting_dims(self, features: Any, xp: Any) -> None:
        """Detect dimensions with linear drift (rotating variables).

        Uses linear_trend__attr_slope to identify dimensions where the majority
        of trajectories show monotonic drift (e.g., pendulum angle for rotating solutions).

        Sets self._drifting_dims and self._non_drifting_dims.

        :param features: Full feature array (n_samples, n_features).
        :param xp: Array module (numpy or cupy) matching ``features``.
        """
        slope_indices = self._feature_indices["linear_trend__attr_slope"]
        n_dims = len(slope_indices)

        self._drifting_dims = []
        for dim_idx, slope_idx in enumerate(slope_indices):
            slopes = features[:, slope_idx]
            high_slope_fraction = float(xp.mean(xp.abs(slopes) > self.drift_threshold))
            if high_slope_fraction > self.drift_fraction:
                self._drifting_dims.append(dim_idx)

        self._non_drifting_dims = [d for d in range(n_dims) if d not in self._drifting_dims]
        if not self._non_drifting_dims:
            self._non_drifting_dims = list(range(n_dims))

    def _get_feature_values(self, features: Any, base_feature: str) -> Any:
        """Get feature values from the first non-drifting dimension.

        If all dimensions are drifting, falls back to the first dimension.

        :param features: Full feature array (n_samples, n_features).
        :param base_feature: Base feature name (e.g., "variance").
        :return: Feature values array of shape (n_samples,).
        """
        indices = self._feature_indices[base_feature]

        for dim in self._non_drifting_dims:
            if dim < len(indices):
                return features[:, indices[dim]]

        return features[:, indices[0]]

    def _get_non_drifting_feature_indices(self, base_feature: str) -> list[int]:
        """Get feature column indices for non-drifting dimensions only.

        :param base_feature: Base feature name (e.g., "mean", "variance").
        :return: List of column indices for non-drifting dimensions.
            Falls back to all indices if no non-drifting dimensions exist.
        """
        indices = self._feature_indices[base_feature]
        non_drifting_indices = [indices[d] for d in self._non_drifting_dims if d < len(indices)]
        return non_drifting_indices if non_drifting_indices else indices

    def _classify_attractor_type(self, features: Any, xp: Any) -> Any:
        """Classify trajectories by attractor type (Stage 1).

        Uses only non-drifting dimensions for classification to avoid
        noise from rotating/drifting state variables.

        :param features: Feature array (n_samples, n_features).
        :param xp: Array module (numpy or cupy) matching ``features``.
        :return: Integer array of type codes (see ``_TYPE_CODES``).
        """
        n_samples = features.shape[0]
        type_codes = xp.zeros(n_samples, dtype=xp.int64)

        non_drifting_var_indices = self._get_non_drifting_feature_indices("variance")
        variance = xp.mean(features[:, non_drifting_var_indices], axis=1)

        periodicity_strength = self._get_feature_values(
            features, "autocorrelation_periodicity__output_strength"
        )

        slope_indices = self._feature_indices["linear_trend__attr_slope"]
        max_abs_slope = xp.max(xp.abs(features[:, slope_indices]), axis=1)

        for tier in self.tiers:
            if tier == "FP":
                fp_mask = (variance < self.fp_variance_threshold) & (type_codes == _CHAOS)
                type_codes = xp.where(fp_mask, _FP, type_codes)

            elif tier == "LC":
                has_periodicity = periodicity_strength > self.lc_periodicity_threshold
                not_high_variance = variance < self.chaos_variance_threshold
                has_drift = max_abs_slope > self.drift_threshold
                lc_periodic = has_periodicity & not_high_variance
                lc_rotating = has_drift & (variance > self.fp_variance_threshold)
                lc_mask = (lc_periodic | lc_rotating) & (type_codes == _CHAOS)
                type_codes = xp.where(lc_mask, _LC, type_codes)

        return type_codes

    def _sub_classify_fixed_points(
        self,
        features: Any,
        indices: Any,
        xp: Any,
        single_cluster_range_threshold: float = 0.01,
    ) -> Any:
        """Sub-classify fixed points by steady-state location.

        Excludes drifting dimensions from clustering to avoid spurious splits.

        :param features: Full feature array.
        :param indices: Indices of FP trajectories.
        :param xp: Array module (numpy or cupy) matching ``features``.
        :param single_cluster_range_threshold: If range (max-min) of mean values
            is below this threshold for all dimensions, treat as single cluster.
        :return: Sub-cluster labels for FP trajectories.
        """
        if indices.shape[0] == 0:
            return xp.array([], dtype=xp.int64)

        non_drifting_mean_indices = self._get_non_drifting_feature_indices("mean")
        fp_features = features[indices][:, non_drifting_mean_indices]

        if self.fp_sub_classifier is not None:
            return self.fp_sub_classifier.fit_predict(fp_features)

        if indices.shape[0] < 2:
            return xp.zeros(indices.shape[0], dtype=xp.int64)

        data_range = xp.max(fp_features, axis=0) - xp.min(fp_features, axis=0)
        if bool(xp.all(data_range < single_cluster_range_threshold)):
            return xp.zeros(indices.shape[0], dtype=xp.int64)

        fp_scaled = standardize(fp_features, xp)

        min_cluster_size = max(50, int(indices.shape[0]) // 10)
        clusterer = HDBSCANClusterer(
            min_cluster_size=min_cluster_size,
            cpu_options={"copy": True},
            assign_noise=True,
        )
        return clusterer.fit_predict(fp_scaled)

    def _sub_classify_limit_cycles(
        self,
        features: Any,
        indices: Any,
        xp: Any,
        amp_cv_threshold: float = 0.1,
        mean_range_threshold: float = 0.05,
    ) -> Any:
        """Sub-classify limit cycles using hierarchical period-based approach.

        Two-level hierarchical clustering:

        1. First level: Group by period-n (freq_ratio rounded to nearest integer)
        2. Second level: Within each period group, cluster by amplitude and mean

        Uses non-drifting dimensions for amplitude and mean features.

        :param features: Full feature array.
        :param indices: Indices of LC trajectories.
        :param xp: Array module (numpy or cupy) matching ``features``.
        :param amp_cv_threshold: Min coefficient of variation for amplitude to cluster.
        :param mean_range_threshold: Min range for mean to cluster.
        :return: Sub-cluster labels for LC trajectories.
        """
        if indices.shape[0] == 0:
            return xp.array([], dtype=xp.int64)

        analysis_dim = self._non_drifting_dims[0] if self._non_drifting_dims else 0

        freq_ratio_indices = self._feature_indices["spectral_frequency_ratio"]
        amp_indices = self._feature_indices["amplitude"]
        mean_indices = self._feature_indices["mean"]

        freq_ratio_idx = (
            freq_ratio_indices[analysis_dim]
            if analysis_dim < len(freq_ratio_indices)
            else freq_ratio_indices[0]
        )
        amp_idx = amp_indices[analysis_dim] if analysis_dim < len(amp_indices) else amp_indices[0]
        mean_idx = (
            mean_indices[analysis_dim] if analysis_dim < len(mean_indices) else mean_indices[0]
        )

        freq_ratios = features[indices, freq_ratio_idx]
        amplitudes = features[indices, amp_idx]
        means = features[indices, mean_idx]

        if self.lc_sub_classifier is not None:
            lc_features = features[indices][:, [freq_ratio_idx, amp_idx, mean_idx]]
            return self.lc_sub_classifier.fit_predict(lc_features)

        finite_mask = xp.isfinite(freq_ratios) & xp.isfinite(amplitudes) & xp.isfinite(means)
        if not bool(xp.any(finite_mask)):
            return xp.zeros(indices.shape[0], dtype=xp.int64)

        period_n = xp.clip(xp.round(freq_ratios).astype(xp.int64), 1, 10)

        unique_periods = xp.unique(period_n[finite_mask])
        labels = xp.zeros(indices.shape[0], dtype=xp.int64)
        current_label = 0

        for period in [p.item() if hasattr(p, "item") else p for p in unique_periods]:
            period_mask = (period_n == period) & finite_mask
            period_indices = xp.where(period_mask)[0]

            if period_indices.shape[0] == 0:
                continue

            period_amps = amplitudes[period_mask]
            period_means = means[period_mask]

            amp_cv = float(xp.std(period_amps)) / (float(xp.mean(period_amps)) + 1e-10)
            mean_range = float(xp.max(period_means)) - float(xp.min(period_means))

            amp_varies = amp_cv > amp_cv_threshold
            mean_varies = mean_range > mean_range_threshold

            if not amp_varies and not mean_varies:
                labels = xp.where(period_mask, current_label, labels)
                current_label += 1
            else:
                if amp_varies and mean_varies:
                    sub_features = xp.column_stack([period_amps, period_means])
                elif amp_varies:
                    sub_features = period_amps.reshape(-1, 1)
                else:
                    sub_features = period_means.reshape(-1, 1)

                sub_labels = self._cluster_1d_or_2d(sub_features, int(period_indices.shape[0]), xp)

                for sub_label in [
                    s.item() if hasattr(s, "item") else s for s in xp.unique(sub_labels)
                ]:
                    sub_mask = sub_labels == sub_label
                    full_mask = xp.zeros(indices.shape[0], dtype=xp.bool_)
                    full_mask[period_indices[sub_mask]] = True
                    labels = xp.where(full_mask, current_label, labels)
                    current_label += 1

        labels = xp.where(finite_mask, labels, 0)

        return labels

    def _cluster_1d_or_2d(self, data: Any, n_samples: int, xp: Any) -> Any:
        """Cluster 1D or 2D data using appropriate method.

        For 1D data: detect natural gaps/modes.
        For 2D data: use HDBSCAN.

        :param data: Feature array of shape (n_samples, 1) or (n_samples, 2).
        :param n_samples: Number of samples.
        :param xp: Array module (numpy or cupy) matching ``data``.
        :return: Cluster labels.
        """
        if data.shape[1] == 1:
            return self._cluster_1d_with_gaps(data.ravel(), xp)

        scaled = standardize(data, xp)

        min_cluster_size = max(15, n_samples // 20)
        if n_samples >= 500:
            clusterer: Any = TunedClusterer(
                estimator=HDBSCANClusterer(assign_noise=True),
                search=MinClusterSizeSearch(),
            )
        else:
            clusterer = HDBSCANClusterer(
                min_cluster_size=min_cluster_size,
                cpu_options={"copy": True},
                assign_noise=True,
            )
        return clusterer.fit_predict(scaled)

    def _cluster_1d_with_gaps(
        self, data: Any, xp: Any, min_gap_ratio: float = 5.0, max_clusters: int = 5
    ) -> Any:
        """Cluster 1D data by detecting the largest natural gap(s).

        Finds gaps in sorted data that are significantly larger than typical spacing.
        Only creates up to max_clusters to avoid over-splitting. Vectorized (no
        per-element Python loop) so it never syncs a GPU-resident array per iteration.

        :param data: 1D array of values.
        :param xp: Array module (numpy or cupy) matching ``data``.
        :param min_gap_ratio: Minimum ratio of gap to median gap to be significant.
        :param max_clusters: Maximum number of clusters to create.
        :return: Cluster labels.
        """
        n = data.shape[0]
        if n < 4:
            return xp.zeros(n, dtype=xp.int64)

        sorted_indices = xp.argsort(data)
        sorted_data = data[sorted_indices]

        gaps = xp.diff(sorted_data)

        if float(xp.max(gaps)) < 1e-6:
            return xp.zeros(n, dtype=xp.int64)

        gap_threshold = float(xp.median(gaps)) * min_gap_ratio
        gap_threshold = max(gap_threshold, float(xp.max(gaps)) * 0.3)

        significant_gap_indices = xp.where(gaps > gap_threshold)[0]

        if significant_gap_indices.shape[0] == 0:
            return xp.zeros(n, dtype=xp.int64)

        if significant_gap_indices.shape[0] >= max_clusters:
            gap_values = gaps[significant_gap_indices]
            top_gaps = xp.argsort(gap_values)[-(max_clusters - 1) :]
            significant_gap_indices = xp.sort(significant_gap_indices[top_gaps])

        gap_starts = xp.zeros(n, dtype=xp.int64)
        gap_starts[significant_gap_indices + 1] = 1
        labels_sorted = xp.cumsum(gap_starts)

        labels = xp.zeros(n, dtype=xp.int64)
        labels[sorted_indices] = labels_sorted

        return labels

    def _sub_classify_chaos(self, features: Any, indices: Any, xp: Any) -> Any:
        """Sub-classify chaotic trajectories by spatial mean.

        Excludes drifting dimensions from clustering to avoid spurious splits.

        :param features: Full feature array.
        :param indices: Indices of chaos trajectories.
        :param xp: Array module (numpy or cupy) matching ``features``.
        :return: Sub-cluster labels for chaos trajectories.
        """
        if indices.shape[0] == 0:
            return xp.array([], dtype=xp.int64)

        non_drifting_mean_indices = self._get_non_drifting_feature_indices("mean")
        chaos_features = features[indices][:, non_drifting_mean_indices]

        if self.chaos_sub_classifier is not None:
            return self.chaos_sub_classifier.fit_predict(chaos_features)

        finite_mask = xp.all(xp.isfinite(chaos_features), axis=1)
        if int(xp.sum(finite_mask)) < 2:
            return xp.zeros(indices.shape[0], dtype=xp.int64)

        chaos_features_clean = xp.where(finite_mask[:, None], chaos_features, 0)
        chaos_scaled = standardize(chaos_features_clean, xp)

        clusterer = TunedClusterer(
            estimator=HDBSCANClusterer(assign_noise=True),
            search=MinClusterSizeSearch(),
        )
        return clusterer.fit_predict(chaos_scaled)

    def fit_predict(self, X: Any, y: Any = None) -> np.ndarray:  # type: ignore[override]
        """Predict labels using two-stage hierarchical clustering.

        :param X: Feature array of shape (n_samples, n_features).
        :param y: Ignored (present for sklearn API compatibility).
        :return: Array of predicted labels with format "TYPE_subcluster", always a
            host ``dtype=object`` array regardless of ``X``'s device (Finding F/C:
            string labels have no torch/cupy equivalent).
        :raises RuntimeError: If set_feature_names() was not called before prediction.
        """
        if not self._initialized:
            raise RuntimeError(
                f"{type(self).__name__} requires feature names before prediction. "
                "Call set_feature_names() first, or use with BasinStabilityEstimator "
                "which handles this automatically."
            )

        backend: DeviceBackend = resolve_backend(X)
        xp = backend.xp
        features = to_native(X, backend=backend)

        self._detect_drifting_dims(features, xp)

        n_samples = features.shape[0]
        type_codes = self._classify_attractor_type(features, xp)

        final_labels = np.empty(n_samples, dtype=object)
        current_label = 0

        for attractor_type in ["FP", "LC", "chaos"]:
            if attractor_type not in self.tiers:
                continue

            type_indices = xp.where(type_codes == _TYPE_CODES[attractor_type])[0]

            if type_indices.shape[0] == 0:
                continue

            if attractor_type == "FP":
                sub_labels = self._sub_classify_fixed_points(features, type_indices, xp)
            elif attractor_type == "LC":
                sub_labels = self._sub_classify_limit_cycles(features, type_indices, xp)
            else:
                sub_labels = self._sub_classify_chaos(features, type_indices, xp)

            type_indices_np = to_numpy(type_indices, backend=backend)
            sub_labels_np = to_numpy(sub_labels, backend=backend)

            for sub_label in np.unique(sub_labels_np):
                mask = sub_labels_np == sub_label
                final_labels[type_indices_np[mask]] = f"{attractor_type}_{current_label}"
                current_label += 1

        return final_labels
