import pytest
import torch
from jax_features.jax_feature_calculators import MINIMAL_FEATURE_NAMES
from jax_features.jax_feature_extractor import JaxFeatureExtractor

from pybasin.solution import Solution


def test_jax_feature_extractor_output_shape():
    """Test that JaxFeatureExtractor returns correct output shape with minimal features."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(time_steady=0.0, normalize=False)
    features = extractor.extract_features(solution)

    n_minimal = len(MINIMAL_FEATURE_NAMES)
    assert features.shape == (n_batch, n_minimal * n_states)


def test_jax_feature_extractor_feature_count():
    """Test that extractor has correct feature count after extraction with minimal features."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor()
    extractor.extract_features(solution)

    n_minimal = len(MINIMAL_FEATURE_NAMES)
    expected = n_minimal * n_states
    assert len(extractor.feature_names) == expected


def test_jax_feature_extractor_feature_names():
    """Test that feature names use per-state naming convention with minimal features."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor()
    extractor.extract_features(solution)

    feature_names = extractor.feature_names
    n_minimal = len(MINIMAL_FEATURE_NAMES)
    assert len(feature_names) == n_minimal * n_states
    assert feature_names[0] == "state_0__median"
    assert feature_names[n_minimal] == "state_1__median"
    assert all("state_0__" in name or "state_1__" in name for name in feature_names)


def test_jax_feature_extractor_time_filtering():
    """Test that time filtering reduces the time dimension."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(time_steady=5.0, normalize=False)
    features = extractor.extract_features(solution)

    n_minimal = len(MINIMAL_FEATURE_NAMES)
    assert features.shape == (n_batch, n_minimal * n_states)


def test_jax_feature_extractor_per_state_features():
    """Test that per-state feature configuration works correctly."""
    n_steps, n_batch, n_states = 100, 5, 3
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(
        features_per_state={
            0: {"maximum": None, "standard_deviation": None},
            1: {"mean": None},
        },
        normalize=False,
    )
    features = extractor.extract_features(solution)

    n_minimal = len(MINIMAL_FEATURE_NAMES)
    assert features.shape == (n_batch, 2 + 1 + n_minimal)
    assert len(extractor.feature_names) == 2 + 1 + n_minimal


def test_jax_feature_extractor_normalization():
    """Test that normalization produces zero mean and unit variance."""
    n_steps, n_batch, n_states = 100, 50, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(normalize=True)
    features = extractor.extract_features(solution)

    assert torch.abs(features.mean()).item() < 0.2
    assert torch.abs(features.std() - 1.0).item() < 0.2


def test_jax_feature_extractor_reset_scaler():
    """Test that reset_scaler resets normalization state."""
    extractor = JaxFeatureExtractor(normalize=True)

    n_steps, n_batch, n_states = 100, 5, 2
    solution = Solution(
        initial_condition=torch.randn(n_batch, n_states),
        time=torch.linspace(0, 10, n_steps),
        y=torch.randn(n_steps, n_batch, n_states),
    )

    extractor.extract_features(solution)
    assert extractor._is_fitted  # pyright: ignore[reportPrivateUsage]

    extractor.reset_scaler()
    assert not extractor._is_fitted  # pyright: ignore[reportPrivateUsage]
    assert extractor._feature_mean is None  # pyright: ignore[reportPrivateUsage]
    assert extractor._feature_std is None  # pyright: ignore[reportPrivateUsage]


def test_jax_feature_extractor_minimal_includes_delta():
    """Test that minimal feature set includes delta and log_delta."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(normalize=False)
    features = extractor.extract_features(solution)

    n_minimal = len(MINIMAL_FEATURE_NAMES)
    expected_features = n_minimal * n_states
    assert features.shape == (n_batch, expected_features)
    assert len(extractor.feature_names) == expected_features

    feature_names = extractor.feature_names
    assert "state_0__delta" in feature_names
    assert "state_0__log_delta" in feature_names
    assert "state_1__delta" in feature_names
    assert "state_1__log_delta" in feature_names


def test_jax_feature_extractor_skip_state_with_features_none():
    """Test that setting features=None skips states not in features_per_state."""
    n_steps, n_batch, n_states = 100, 5, 2
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(
        features=None,
        features_per_state={
            1: {"log_delta": None},
        },
        normalize=False,
    )
    features = extractor.extract_features(solution)

    assert features.shape == (n_batch, 1)
    assert len(extractor.feature_names) == 1
    assert extractor.feature_names == ["state_1__log_delta"]


def test_jax_feature_extractor_feature_names_format():
    """Test that feature_names property returns correctly formatted names."""
    n_steps, n_batch, n_states = 100, 5, 3
    ic = torch.randn(n_batch, n_states)
    time = torch.linspace(0, 10, n_steps)
    y = torch.randn(n_steps, n_batch, n_states)
    solution = Solution(initial_condition=ic, time=time, y=y)

    extractor = JaxFeatureExtractor(
        features_per_state={
            0: {"maximum": None, "minimum": None},
            1: {"mean": None, "variance": None, "standard_deviation": None},
            2: {"delta": None},
        },
        normalize=False,
    )
    extractor.extract_features(solution)

    expected_names = [
        "state_0__maximum",
        "state_0__minimum",
        "state_1__mean",
        "state_1__variance",
        "state_1__standard_deviation",
        "state_2__delta",
    ]
    assert extractor.feature_names == expected_names
    assert len(extractor.feature_names) == 6


def test_jax_feature_extractor_feature_names_raises_before_extraction():
    """Test that accessing feature_names before extraction raises RuntimeError."""
    extractor = JaxFeatureExtractor()

    with pytest.raises(RuntimeError, match="Feature configuration not initialized"):
        _ = extractor.feature_names
