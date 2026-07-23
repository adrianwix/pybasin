import numpy as np

from pybasin.feature_selector.correlation_selector import CorrelationSelector


def _make_correlated_features() -> np.ndarray:
    """Create two highly correlated features plus two independent features (4 total)."""
    rng: np.random.Generator = np.random.default_rng(0)
    n: int = 500
    base: np.ndarray = rng.standard_normal(n)
    f1: np.ndarray = base + rng.standard_normal(n) * 0.01
    f2: np.ndarray = base + rng.standard_normal(n) * 0.01
    f3: np.ndarray = rng.standard_normal(n)
    f4: np.ndarray = rng.standard_normal(n)
    return np.column_stack([f1, f2, f3, f4])


def test_removes_one_of_correlated_pair() -> None:
    """When two features are nearly identical, exactly one is removed."""
    X: np.ndarray = _make_correlated_features()
    selector = CorrelationSelector(threshold=0.95, min_features=2)
    X_out: np.ndarray = selector.fit_transform(X)  # type: ignore[reportUnknownMemberType]

    assert X_out.shape[1] == 3
    support: np.ndarray = selector.get_support(indices=False)
    assert support.sum() == 3
    assert support[2], "Independent feature f3 should be kept"
    assert support[3], "Independent feature f4 should be kept"


def test_keeps_all_when_below_threshold() -> None:
    """No features removed when all correlations are below threshold."""
    rng: np.random.Generator = np.random.default_rng(1)
    X: np.ndarray = np.column_stack([
        rng.standard_normal(200),
        rng.standard_normal(200),
        rng.standard_normal(200),
        rng.standard_normal(200),
    ])
    selector = CorrelationSelector(threshold=0.9, min_features=2)
    X_out: np.ndarray = selector.fit_transform(X)  # type: ignore[reportUnknownMemberType]

    assert X_out.shape[1] == 4


def test_drops_more_globally_redundant_feature() -> None:
    """The feature with higher mean absolute correlation is dropped.

    Setup: f0 correlates with f1 and f2 (high mean abs corr).
    f1 only correlates with f0 (lower mean abs corr).
    f2 only correlates with f0 (lower mean abs corr).
    f3 is independent.
    f0 should be dropped first because it's globally more redundant.
    """
    rng: np.random.Generator = np.random.default_rng(2)
    n: int = 1000
    base: np.ndarray = rng.standard_normal(n)
    f0: np.ndarray = base
    f1: np.ndarray = base + rng.standard_normal(n) * 0.01
    f2: np.ndarray = base + rng.standard_normal(n) * 0.01
    f3: np.ndarray = rng.standard_normal(n)
    X: np.ndarray = np.column_stack([f0, f1, f2, f3])

    selector = CorrelationSelector(threshold=0.95, min_features=2)
    selector.fit(X)

    support: np.ndarray = selector.get_support(indices=False)

    assert not support[0], "f0 (highest mean abs corr) should be dropped first"
    assert support[3], "f3 (independent) should be kept"


def test_min_features_respected() -> None:
    """Removal stops when min_features is reached."""
    rng: np.random.Generator = np.random.default_rng(3)
    base: np.ndarray = rng.standard_normal(300)
    X: np.ndarray = np.column_stack([
        base + rng.standard_normal(300) * 0.001,
        base + rng.standard_normal(300) * 0.001,
        base + rng.standard_normal(300) * 0.001,
    ])

    selector = CorrelationSelector(threshold=0.5, min_features=2)
    X_out: np.ndarray = selector.fit_transform(X)  # type: ignore[reportUnknownMemberType]

    assert X_out.shape[1] == 2


def test_min_features_with_fewer_input_features() -> None:
    """When input has fewer features than min_features, all are kept."""
    X: np.ndarray = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    selector = CorrelationSelector(threshold=0.5, min_features=3)
    X_out: np.ndarray = selector.fit_transform(X)  # type: ignore[reportUnknownMemberType]

    assert X_out.shape[1] == 2
    assert selector.get_support(indices=False).all()


def test_get_support_indices() -> None:
    """get_support(indices=True) returns integer indices."""
    X: np.ndarray = _make_correlated_features()

    selector = CorrelationSelector(threshold=0.95, min_features=2)
    selector.fit(X)

    indices: np.ndarray = selector.get_support(indices=True)
    mask: np.ndarray = selector.get_support(indices=False)

    assert len(indices) == mask.sum()
    np.testing.assert_array_equal(np.where(mask)[0], indices)


def test_transform_is_consistent_with_support() -> None:
    """transform output columns match get_support mask."""
    X: np.ndarray = _make_correlated_features()
    selector = CorrelationSelector(threshold=0.95, min_features=2)
    X_out: np.ndarray = selector.fit_transform(X)  # type: ignore[reportUnknownMemberType]

    support: np.ndarray = selector.get_support(indices=False)
    np.testing.assert_array_equal(X_out, X[:, support])


def test_order_independent_of_column_position() -> None:
    """Removal decision depends on global redundancy, not column order.

    Shuffling columns should not change which underlying feature is dropped.
    """
    rng: np.random.Generator = np.random.default_rng(5)
    n: int = 1000
    base: np.ndarray = rng.standard_normal(n)
    f_redundant: np.ndarray = base
    f_copy: np.ndarray = base + rng.standard_normal(n) * 0.005
    f_independent_a: np.ndarray = rng.standard_normal(n)
    f_independent_b: np.ndarray = rng.standard_normal(n)

    X_original: np.ndarray = np.column_stack([
        f_redundant,
        f_copy,
        f_independent_a,
        f_independent_b,
    ])
    X_shuffled: np.ndarray = np.column_stack([
        f_independent_a,
        f_copy,
        f_independent_b,
        f_redundant,
    ])

    sel_orig = CorrelationSelector(threshold=0.95, min_features=2)
    sel_orig.fit(X_original)

    sel_shuf = CorrelationSelector(threshold=0.95, min_features=2)
    sel_shuf.fit(X_shuffled)

    assert sel_orig.get_support(indices=False).sum() == sel_shuf.get_support(indices=False).sum()
    assert sel_orig.get_support(indices=False)[2], "Independent feature kept in original"
    assert sel_orig.get_support(indices=False)[3], "Independent feature kept in original"
    assert sel_shuf.get_support(indices=False)[0], "Independent feature kept in shuffled"
    assert sel_shuf.get_support(indices=False)[2], "Independent feature kept in shuffled"
