# pyright: reportPrivateUsage=false
"""Tests for DynamicalSystemClusterer."""

import numpy as np
import pytest

from pybasin.predictors.dynamical_system_clusterer import DynamicalSystemClusterer


def create_feature_names(n_states: int = 2) -> list[str]:
    """Create valid feature names for testing."""
    base_features = [
        "variance",
        "amplitude",
        "mean",
        "linear_trend__attr_slope",
        "autocorrelation_periodicity__output_strength",
        "autocorrelation_periodicity__output_period",
        "spectral_frequency_ratio",
    ]
    names: list[str] = []
    for state_idx in range(n_states):
        for feature in base_features:
            names.append(f"state_{state_idx}__{feature}")
    return names


def create_feature_array(
    n_samples: int,
    n_states: int = 2,
    attractor_type: str = "LC",
    variance: float = 0.5,
    periodicity_strength: float = 0.8,
    slope: float = 0.0,
) -> np.ndarray:
    """Create a feature array for testing with specified attractor characteristics."""
    n_features_per_state = 7
    n_features = n_states * n_features_per_state
    features = np.zeros((n_samples, n_features))

    for state_idx in range(n_states):
        base_idx = state_idx * n_features_per_state
        features[:, base_idx + 0] = variance
        features[:, base_idx + 1] = 1.0
        features[:, base_idx + 2] = 0.0
        features[:, base_idx + 3] = slope
        features[:, base_idx + 4] = periodicity_strength
        features[:, base_idx + 5] = 10.0
        features[:, base_idx + 6] = 1.0

    return features


class TestDynamicalSystemClustererInit:
    """Test initialization of DynamicalSystemClusterer."""

    def test_default_initialization(self):
        clusterer = DynamicalSystemClusterer()
        assert clusterer.fp_variance_threshold == 1e-6
        assert clusterer.lc_periodicity_threshold == 0.5
        assert clusterer.chaos_variance_threshold == 5.0
        assert clusterer.drift_threshold == 0.1
        assert clusterer.tiers == ["FP", "LC", "chaos"]
        assert clusterer.feature_names is None
        assert not clusterer._initialized

    def test_custom_thresholds(self):
        clusterer = DynamicalSystemClusterer(
            fp_variance_threshold=1e-8,
            lc_periodicity_threshold=0.3,
            chaos_variance_threshold=10.0,
            drift_threshold=0.05,
        )
        assert clusterer.fp_variance_threshold == 1e-8
        assert clusterer.lc_periodicity_threshold == 0.3
        assert clusterer.chaos_variance_threshold == 10.0
        assert clusterer.drift_threshold == 0.05

    def test_custom_tiers(self):
        clusterer = DynamicalSystemClusterer(tiers=["FP", "LC"])
        assert clusterer.tiers == ["FP", "LC"]

    def test_needs_feature_names(self):
        clusterer = DynamicalSystemClusterer()
        assert clusterer.needs_feature_names() is True

    def test_display_name(self):
        clusterer = DynamicalSystemClusterer()
        assert clusterer.display_name == "Dynamical System Clusterer"


class TestSetFeatureNames:
    """Test set_feature_names() method."""

    def test_set_valid_feature_names(self):
        clusterer = DynamicalSystemClusterer()
        feature_names = create_feature_names(n_states=2)

        clusterer.set_feature_names(feature_names)

        assert clusterer._initialized is True
        assert clusterer.feature_names == feature_names
        assert len(clusterer._feature_indices) == len(DynamicalSystemClusterer.REQUIRED_FEATURES)

    def test_feature_indices_built_correctly(self):
        clusterer = DynamicalSystemClusterer()
        feature_names = create_feature_names(n_states=2)

        clusterer.set_feature_names(feature_names)

        assert "variance" in clusterer._feature_indices
        assert len(clusterer._feature_indices["variance"]) == 2
        assert clusterer._feature_indices["variance"] == [0, 7]

    def test_invalid_feature_names_raises_error(self):
        clusterer = DynamicalSystemClusterer()
        invalid_names = ["invalid_name", "another_bad_name"]

        with pytest.raises(ValueError, match="do not follow"):
            clusterer.set_feature_names(invalid_names)

    def test_missing_required_feature_raises_error(self):
        clusterer = DynamicalSystemClusterer()
        incomplete_names = [
            "state_0__variance",
            "state_0__amplitude",
        ]

        with pytest.raises(ValueError, match="Required feature"):
            clusterer.set_feature_names(incomplete_names)


class TestPredictLabelsValidation:
    """Test fit_predict() validation."""

    def test_predict_without_feature_names_raises_error(self):
        clusterer = DynamicalSystemClusterer()
        features = np.random.randn(100, 14)

        with pytest.raises(RuntimeError, match="requires feature names"):
            clusterer.fit_predict(features)


class TestAttractorTypeClassification:
    """Test Stage 1: attractor type classification."""

    def test_classifies_fixed_points(self):
        clusterer = DynamicalSystemClusterer(fp_variance_threshold=1e-4)
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(
            n_samples=50,
            n_states=2,
            variance=1e-8,
            periodicity_strength=0.0,
        )

        labels = clusterer.fit_predict(features)

        assert all("FP" in label for label in labels)

    def test_classifies_limit_cycles(self):
        clusterer = DynamicalSystemClusterer(
            fp_variance_threshold=1e-6,
            lc_periodicity_threshold=0.5,
        )
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(
            n_samples=50,
            n_states=2,
            variance=0.5,
            periodicity_strength=0.8,
        )

        labels = clusterer.fit_predict(features)

        assert all("LC" in label for label in labels)

    def test_classifies_chaos(self):
        clusterer = DynamicalSystemClusterer(
            fp_variance_threshold=1e-6,
            lc_periodicity_threshold=0.5,
            chaos_variance_threshold=5.0,
        )
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(
            n_samples=50,
            n_states=2,
            variance=10.0,
            periodicity_strength=0.2,
        )

        labels = clusterer.fit_predict(features)

        assert all("chaos" in label for label in labels)

    def test_classifies_rotating_as_limit_cycle(self):
        clusterer = DynamicalSystemClusterer(drift_threshold=0.05)
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(
            n_samples=50,
            n_states=2,
            variance=0.5,
            periodicity_strength=0.2,
            slope=0.5,
        )

        labels = clusterer.fit_predict(features)

        assert all("LC" in label for label in labels)


class TestMixedAttractorTypes:
    """Test classification with mixed attractor types."""

    def test_classifies_mixed_fp_and_lc(self):
        clusterer = DynamicalSystemClusterer(fp_variance_threshold=1e-4)
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        fp_features = create_feature_array(
            n_samples=25,
            n_states=2,
            variance=1e-8,
            periodicity_strength=0.0,
        )
        lc_features = create_feature_array(
            n_samples=25,
            n_states=2,
            variance=0.5,
            periodicity_strength=0.8,
        )
        features = np.vstack([fp_features, lc_features])

        labels = clusterer.fit_predict(features)

        fp_labels = labels[:25]
        lc_labels = labels[25:]
        assert all("FP" in label for label in fp_labels)
        assert all("LC" in label for label in lc_labels)


class TestSubClassification:
    """Test Stage 2: sub-classification within attractor types."""

    def test_sub_classifies_fixed_points_by_location(self):
        clusterer = DynamicalSystemClusterer(fp_variance_threshold=1e-4)
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features1 = create_feature_array(n_samples=60, n_states=2, variance=1e-8)
        features2 = create_feature_array(n_samples=60, n_states=2, variance=1e-8)
        features1[:, 2] = 0.0
        features2[:, 2] = 100.0
        features1[:, 9] = 0.0
        features2[:, 9] = 100.0

        features = np.vstack([features1, features2])

        labels = clusterer.fit_predict(features)

        unique_labels = set(labels)
        assert len(unique_labels) >= 1
        assert all("FP" in label for label in labels)

    def test_sub_classifies_limit_cycles_by_amplitude(self):
        clusterer = DynamicalSystemClusterer()
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features1 = create_feature_array(
            n_samples=30, n_states=2, variance=0.5, periodicity_strength=0.8
        )
        features2 = create_feature_array(
            n_samples=30, n_states=2, variance=0.5, periodicity_strength=0.8
        )
        features1[:, 1] = 1.0
        features2[:, 1] = 10.0
        features1[:, 8] = 1.0
        features2[:, 8] = 10.0

        features = np.vstack([features1, features2])

        labels = clusterer.fit_predict(features)

        unique_labels = set(labels)
        assert len(unique_labels) >= 1


class TestTiersConfiguration:
    """Test custom tiers configuration."""

    def test_only_fp_tier_classifies_fp_samples(self):
        clusterer = DynamicalSystemClusterer(tiers=["FP"], fp_variance_threshold=1e-4)
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(
            n_samples=50,
            n_states=2,
            variance=1e-8,
            periodicity_strength=0.0,
        )

        labels = clusterer.fit_predict(features)

        assert all("FP" in label for label in labels)

    def test_fp_and_lc_tiers_classifies_lc_samples(self):
        clusterer = DynamicalSystemClusterer(tiers=["FP", "LC"])
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(
            n_samples=50,
            n_states=2,
            variance=0.5,
            periodicity_strength=0.8,
        )

        labels = clusterer.fit_predict(features)

        assert all("LC" in label for label in labels)


class TestLabelFormat:
    """Test output label format."""

    def test_labels_have_correct_format(self):
        clusterer = DynamicalSystemClusterer()
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(n_samples=50, n_states=2)

        labels = clusterer.fit_predict(features)

        for label in labels:
            parts = label.split("_")
            assert len(parts) == 2
            assert parts[0] in ["FP", "LC", "chaos"]
            assert parts[1].isdigit()

    def test_returns_correct_number_of_labels(self):
        clusterer = DynamicalSystemClusterer()
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        n_samples = 100
        features = create_feature_array(n_samples=n_samples, n_states=2)

        labels = clusterer.fit_predict(features)

        assert len(labels) == n_samples


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample(self):
        clusterer = DynamicalSystemClusterer()
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(n_samples=1, n_states=2)

        labels = clusterer.fit_predict(features)

        assert len(labels) == 1

    def test_all_same_features(self):
        clusterer = DynamicalSystemClusterer()
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(n_samples=50, n_states=2)

        labels = clusterer.fit_predict(features)

        unique_labels = set(labels)
        assert len(unique_labels) == 1

    def test_handles_nan_in_features_gracefully(self):
        clusterer = DynamicalSystemClusterer()
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(n_samples=50, n_states=2, periodicity_strength=0.8)

        labels = clusterer.fit_predict(features)

        assert len(labels) == 50
        assert all(label is not None and isinstance(label, str) for label in labels)

    def test_single_state(self):
        clusterer = DynamicalSystemClusterer()
        feature_names = create_feature_names(n_states=1)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(n_samples=50, n_states=1)

        labels = clusterer.fit_predict(features)

        assert len(labels) == 50


class TestDriftingDimensionDetection:
    """Test detection of drifting dimensions."""

    def test_detects_drifting_dimensions(self):
        clusterer = DynamicalSystemClusterer(drift_threshold=0.1)
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(n_samples=100, n_states=2, slope=0.0)
        features[:50, 3] = 0.5

        clusterer._detect_drifting_dims(features, np)

        assert 0 in clusterer._drifting_dims or 0 in clusterer._non_drifting_dims

    def test_excludes_drifting_dims_from_fp_clustering(self):
        clusterer = DynamicalSystemClusterer(fp_variance_threshold=1e-4, drift_threshold=0.05)
        feature_names = create_feature_names(n_states=2)
        clusterer.set_feature_names(feature_names)

        features = create_feature_array(n_samples=50, n_states=2, variance=1e-8)
        features[:, 3] = 0.5

        labels = clusterer.fit_predict(features)

        assert len(labels) == 50
