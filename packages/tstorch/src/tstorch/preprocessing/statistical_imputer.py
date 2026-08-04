# pyright: basic
"""Per-feature finite-extrema/median imputation, independent of any extractor."""

import torch
from torch import Tensor


class StatisticalImputer:
    """Replace non-finite feature values using each feature's own finite statistics.

    For every feature (column): ``-inf`` is replaced by that feature's finite minimum,
    ``+inf`` by its finite maximum, and ``NaN`` by its finite median. A feature with no
    finite values at all is replaced entirely with zeros.

    Needs the whole feature column at once to compute its statistics, so unlike
    :class:`tstorch.preprocessing.ExtremeImputer` this only accepts an eager
    ``torch.Tensor`` - there's no local, block-by-block way to compute a feature's true
    min/max/median that would agree with the eager result.

    :raises TypeError: If ``transform`` receives a non-floating or complex tensor.
    """

    def transform(self, features: Tensor) -> Tensor:
        """Impute ``features`` per feature, preserving dtype and device.

        :param features: Eager ``torch.Tensor`` of shape ``(B, F)``.
        :return: Tensor of the same shape and dtype, with every NaN and infinite value in
            a feature replaced by that feature's finite median/max/min respectively.
        """
        if features.is_complex() or not features.is_floating_point():
            raise TypeError(
                f"StatisticalImputer requires a real floating tensor, got dtype={features.dtype}"
            )

        result = features.clone()
        n_features = features.shape[1]

        for feature_idx in range(n_features):
            col = features[:, feature_idx]

            finite_mask = torch.isfinite(col)
            finite_vals = col[finite_mask]

            if finite_vals.numel() == 0:
                result[:, feature_idx] = 0.0
                continue

            col_min = finite_vals.min()
            col_max = finite_vals.max()
            col_median = finite_vals.median()

            neg_inf_mask = torch.isneginf(col)
            pos_inf_mask = torch.isposinf(col)
            nan_mask = torch.isnan(col)

            result[neg_inf_mask, feature_idx] = col_min
            result[pos_inf_mask, feature_idx] = col_max
            result[nan_mask, feature_idx] = col_median

        return result
