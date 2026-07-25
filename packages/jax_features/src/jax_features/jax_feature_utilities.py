# pyright: basic

"""JAX-based utility functions for feature extraction.

This module provides utility functions for handling NaN, inf, and other
edge cases in feature arrays, similar to tsfresh's dataframe_functions.
"""

import jax.numpy as jnp
from jax import Array


def impute(features: Array) -> Array:
    """Columnwise replaces all NaNs and infs from the feature array with average/extreme values.

    This is done as follows for each column:

    - -inf -> min (of finite values in that column)
    - +inf -> max (of finite values in that column)
    - NaN -> median (of finite values in that column)

    If a column does not contain any finite values at all, it is filled with zeros.

    This function is the JAX equivalent of tsfresh's impute function.

    ```python
    import jax.numpy as jnp
    from pybasin.jax_feature_utilities import impute

    features = jnp.array([[1.0, jnp.nan, jnp.inf], [2.0, 3.0, -jnp.inf], [3.0, 4.0, 5.0]])
    imputed = impute(features)
    # NaN replaced with median, inf with max, -inf with min
    ```

    :param features: Feature array of shape (B, F) where B is batch size and F is number of features.
    :return: Imputed feature array with the same shape, guaranteed to contain no NaN or inf values.
    """
    # Create mask for finite values
    finite_mask = jnp.isfinite(features)

    # For each column, compute min, max, and median of finite values
    # We need to handle columns that have no finite values

    # Replace non-finite values with nan temporarily for nanmin/nanmax/nanmedian
    features_masked = jnp.where(finite_mask, features, jnp.nan)

    # Compute column-wise statistics (ignoring NaN)
    col_min = jnp.nanmin(features_masked, axis=0)
    col_max = jnp.nanmax(features_masked, axis=0)
    col_median = jnp.nanmedian(features_masked, axis=0)

    # Handle columns with all non-finite values (nanmin/nanmax/nanmedian return nan)
    # Replace those with 0
    col_min = jnp.where(jnp.isnan(col_min), 0.0, col_min)
    col_max = jnp.where(jnp.isnan(col_max), 0.0, col_max)
    col_median = jnp.where(jnp.isnan(col_median), 0.0, col_median)

    # Create replacement arrays (broadcast to full shape)
    result = features

    # Replace -inf with column min
    result = jnp.where(jnp.isneginf(features), col_min, result)

    # Replace +inf with column max
    result = jnp.where(jnp.isposinf(features), col_max, result)

    # Replace NaN with column median
    result = jnp.where(jnp.isnan(features), col_median, result)

    return result


def impute_extreme(features: Array, extreme_value: float = 1e10) -> Array:
    """Replaces all NaNs and infs with extreme values to make them distinguishable.

    This is useful when you want samples with non-finite features to be
    classified separately (e.g., unbounded trajectories in dynamical systems).

    - +inf -> +extreme_value
    - -inf -> -extreme_value
    - NaN -> +extreme_value (to cluster with +inf)

    :param features: Feature array of shape (B, F) where B is batch size and F is number of features.
    :param extreme_value: The extreme value to use for replacement. Default is 1e10.
    :return: Imputed feature array with the same shape.
    """
    result = features
    result = jnp.where(jnp.isposinf(features), extreme_value, result)
    result = jnp.where(jnp.isneginf(features), -extreme_value, result)
    result = jnp.where(jnp.isnan(features), extreme_value, result)
    return result


def delay_embedding(data: Array, emb_dim: int, lag: int = 1) -> Array:
    """Create delay embedding of a 1D time series.

    :param data: 1D time series of shape (N,).
    :param emb_dim: Embedding dimension.
    :param lag: Lag between elements in embedded vectors.
    :return: Embedded orbit matrix of shape (M, emb_dim) where M = N - (emb_dim-1)*lag.
    """
    n = data.shape[0]
    m = n - (emb_dim - 1) * lag

    indices = jnp.arange(emb_dim) * lag
    indices = indices + jnp.arange(m)[:, None]

    return data[indices]


def rowwise_euclidean(x: Array, y: Array) -> Array:
    """Compute Euclidean distance from each row of x to vector y.

    :param x: Matrix of shape (M, D).
    :param y: Vector of shape (D,).
    :return: Distances of shape (M,).
    """
    return jnp.sqrt(jnp.sum((x - y) ** 2, axis=1))
