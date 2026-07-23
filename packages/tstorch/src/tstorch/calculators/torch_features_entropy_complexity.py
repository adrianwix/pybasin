"""Entropy and complexity feature calculators for BST time-series tensors."""

from typing import TypedDict

import torch
from torch import Tensor

from tstorch._ops.lempel_ziv import lempel_ziv_phrase_count
from tstorch.feature_factory import Feature, ParameterlessFeature


class PermutationEntropyParams(TypedDict):
    """Parameters for one `PermutationEntropy` output column.

    Attributes:
        tau: Spacing between successive window start positions.
        dimension: Number of consecutive samples in each ordinal-pattern window.
    """

    tau: int
    dimension: int


_PERMUTATION_ENTROPY_DEFAULT_PARAMS: list[PermutationEntropyParams] = [
    {"tau": 1, "dimension": dimension} for dimension in range(3, 8)
]


class PermutationEntropy(Feature[PermutationEntropyParams]):
    r"""Shannon entropy of the series' ordinal-pattern distribution.

    The series is embedded into windows of ``dimension`` consecutive samples whose
    starting positions are separated by ``tau``. Each window is replaced by its rank
    permutation, and the natural-log entropy is

    \[
    H = -\sum_{\pi} p(\pi)\log p(\pi).
    \]

    The result is not normalized by \(\log(\mathrm{dimension}!)\). A parameter
    combination returns NaN when the series is too short to form a window.

    Args:
        params: One dict per requested output. Defaults to tsfresh's dimensions 3-7
            with ``tau=1``. An explicit empty list is rejected.

    """

    def __init__(self, params: list[PermutationEntropyParams] | None = None) -> None:
        if params is None:
            params = list(_PERMUTATION_ENTROPY_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "PermutationEntropy requires at least one param; omit `params` to use "
                f"the default {_PERMUTATION_ENTROPY_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[PermutationEntropyParams] = params
        self._taus = torch.tensor([param["tau"] for param in params], dtype=torch.long)
        self._dimensions = torch.tensor([param["dimension"] for param in params], dtype=torch.long)
        self._max_dimension = max(param["dimension"] for param in params)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Padded windows broadcast every structural parameter combination through one kernel.
        batch_size, n_states, T = x.shape
        if T == 0:
            return torch.full(
                (batch_size, n_states, len(self.params)),
                float("nan"),
                dtype=x.dtype,
                device=x.device,
            )
        taus = self._taus.to(device=x.device)
        dimensions = self._dimensions.to(device=x.device)
        starts = taus[:, None] * torch.arange(T, device=x.device)[None, :]
        offsets = torch.arange(self._max_dimension, device=x.device)
        indices = starts[:, :, None] + offsets[None, None, :]
        valid_dimensions = offsets[None, :] < dimensions[:, None]
        num_windows = torch.div(T - dimensions, taus, rounding_mode="floor") + 1
        valid_windows = torch.arange(T, device=x.device)[None, :] < num_windows[:, None]

        indices = indices.clamp(min=0, max=max(T - 1, 0))
        embedded = x[:, :, indices]
        embedded = embedded.masked_fill(
            ~valid_dimensions.view(1, 1, len(self.params), 1, self._max_dimension),
            float("inf"),
        )
        ranks = embedded.argsort(dim=-1)

        powers = dimensions[:, None] - 1 - offsets[None, :]
        multipliers = torch.where(
            valid_dimensions,
            dimensions[:, None] ** powers.clamp(min=0),
            torch.zeros_like(powers),
        )
        pattern_ids = (ranks * multipliers.view(1, 1, len(self.params), 1, -1)).sum(dim=-1)
        pattern_ids = torch.where(
            valid_windows.view(1, 1, len(self.params), T),
            pattern_ids,
            torch.full_like(pattern_ids, -1),
        )

        sorted_ids = pattern_ids.sort(dim=-1).values.contiguous()
        lower = torch.searchsorted(sorted_ids, sorted_ids, right=False)
        upper = torch.searchsorted(sorted_ids, sorted_ids, right=True)
        counts = (upper - lower).to(dtype=x.dtype)
        sorted_valid = sorted_ids >= 0
        safe_num_windows = num_windows.clamp(min=1).to(dtype=x.dtype)
        contributions = torch.where(
            sorted_valid,
            (counts / safe_num_windows.view(1, 1, -1, 1)).log(),
            torch.zeros_like(counts),
        )
        entropy = -contributions.sum(dim=-1) / safe_num_windows.view(1, 1, -1)
        return torch.where(
            (num_windows > 0).view(1, 1, -1),
            entropy,
            torch.full(
                (batch_size, n_states, len(self.params)),
                float("nan"),
                dtype=x.dtype,
                device=x.device,
            ),
        )

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format:
            ``permutation_entropy__tau_<tau>__dimension_<dimension>``.
        """
        return [
            f"permutation_entropy__tau_{param['tau']}__dimension_{param['dimension']}"
            for param in self.params
        ]


def _binned_entropy_multi(x: Tensor, bins: list[int]) -> Tensor:
    """Binned entropy at every requested bin count.

    Bin edges are computed exactly like ``numpy.histogram``'s ``linspace(min, max,
    k + 1)`` and assigned via ``searchsorted`` against those explicit per-series edges,
    rather than via ``floor((x - min) / range * k)``: the two formulas are
    mathematically equivalent but round differently in floating point, close enough to
    a bin boundary to matter.
    """
    batch_size, n_states, T = x.shape
    min_vals = x.min(dim=2, keepdim=True).values  # (B, S, 1)
    max_vals = x.max(dim=2, keepdim=True).values
    ranges = max_vals - min_vals
    ranges = torch.where(ranges == 0, torch.ones_like(ranges), ranges)
    is_constant = max_vals == min_vals
    has_nan = torch.isnan(x).any(dim=2, keepdim=True)  # matches tsfresh's `if isnan(x).any(): nan`

    columns: list[Tensor] = []
    for k in bins:
        step = ranges / k
        edges = min_vals + step * torch.arange(k + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        bin_idx = torch.searchsorted(edges.contiguous(), x.contiguous(), right=True) - 1
        bin_idx = bin_idx.clamp(min=0, max=k - 1)

        counts = torch.zeros(batch_size, n_states, k, dtype=x.dtype, device=x.device)
        counts.scatter_add_(dim=2, index=bin_idx, src=torch.ones_like(bin_idx, dtype=x.dtype))

        probs = (counts / T).clamp(min=1e-10)
        log_probs = torch.where(counts > 0, probs.log(), torch.zeros_like(probs))
        columns.append(-(probs * log_probs).sum(dim=2, keepdim=True))  # (B, S, 1)

    entropy = torch.cat(columns, dim=-1)  # (B, S, K)
    entropy = torch.where(is_constant, torch.zeros_like(entropy), entropy)
    entropy = torch.where(has_nan, torch.full_like(entropy, float("nan")), entropy)
    return entropy


class BinnedEntropyParams(TypedDict):
    """Parameters for one `BinnedEntropy` output column.

    Attributes:
        max_bins: Number of equal-width bins.
    """

    max_bins: int


_BINNED_ENTROPY_DEFAULT_PARAMS: list[BinnedEntropyParams] = [{"max_bins": 10}]


class BinnedEntropy(Feature[BinnedEntropyParams]):
    r"""Entropy of the equal-width binned value distribution.

    Values are assigned to ``max_bins`` bins spanning
    \([\min(x), \max(x)]\), matching NumPy histogram edges, and the natural-log
    Shannon entropy is

    \[
    H = -\sum_k p_k\log p_k.
    \]

    A constant series returns zero. A series containing NaN returns NaN.

    Args:
        params: One dict per requested output. Defaults to ``max_bins=10``. An
            explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[BinnedEntropyParams] | None = None) -> None:
        if params is None:
            params = list(_BINNED_ENTROPY_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "BinnedEntropy requires at least one param; omit `params` to use the "
                f"default {_BINNED_ENTROPY_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[BinnedEntropyParams] = params
        self._bins: list[int] = [param["max_bins"] for param in params]

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Different bin counts require differently shaped edge/count tensors, while
        # min/max and all time and batch/state work remain shared and tensorized.
        return _binned_entropy_multi(x, self._bins)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format: ``binned_entropy__max_bins_<max_bins>``.
        """
        return [f"binned_entropy__max_bins_{param['max_bins']}" for param in self.params]


class FourierEntropyParams(TypedDict):
    """Parameters for one `FourierEntropy` output column.

    Attributes:
        bins: Number of equal-width bins applied to the power spectrum.
    """

    bins: int


_FOURIER_ENTROPY_DEFAULT_PARAMS: list[FourierEntropyParams] = [
    {"bins": bins} for bins in [2, 3, 5, 10, 100]
]


class FourierEntropy(Feature[FourierEntropyParams]):
    """Binned entropy of the Welch power spectral density.

    The Welch spectrum is computed once for all requested bin counts, then each
    spectrum is summarized with the same equal-width Shannon entropy as
    `BinnedEntropy`. Multiplying a spectrum by a positive per-series constant does not
    change its bin occupancy, so the unscaled Welch output is sufficient.

    Args:
        params: One dict per requested output. Defaults to tsfresh's bin counts
            2, 3, 5, 10, and 100. An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[FourierEntropyParams] | None = None) -> None:
        if params is None:
            params = list(_FOURIER_ENTROPY_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "FourierEntropy requires at least one param; omit `params` to use the "
                f"default {_FOURIER_ENTROPY_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[FourierEntropyParams] = params
        self._bins: list[int] = [param["bins"] for param in params]

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # The expensive Welch transform is shared across the complete parameter axis.
        batch_size, n_states, T = x.shape
        nperseg = min(T, 256)
        step = nperseg - nperseg // 2
        segments = x.reshape(batch_size * n_states, T).unfold(1, nperseg, step)
        segments = segments - segments.mean(dim=-1, keepdim=True)
        window = torch.hann_window(nperseg, periodic=True, dtype=x.dtype, device=x.device)
        segments = segments * window

        # At nperseg <= 256, a direct DFT avoids torch.fft's dispatch floor.
        samples = torch.arange(nperseg, dtype=x.dtype, device=x.device)
        frequencies = torch.arange(nperseg // 2 + 1, dtype=x.dtype, device=x.device)
        angles = (2 * torch.pi / nperseg) * frequencies[:, None] * samples[None, :]
        real = segments @ torch.cos(angles).T
        imag = -(segments @ torch.sin(angles).T)
        power = real.square() + imag.square()

        if nperseg % 2 == 0:
            power[..., 1:-1] *= 2
        else:
            power[..., 1:] *= 2
        pxx = power.mean(dim=1).reshape(batch_size, n_states, -1)
        return _binned_entropy_multi(pxx, self._bins)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format: ``fourier_entropy__bins_<bins>``.
        """
        return [f"fourier_entropy__bins_{param['bins']}" for param in self.params]


class LempelZivComplexityParams(TypedDict):
    """Parameters for one `LempelZivComplexity` output column.

    Attributes:
        bins: Number of equal-width levels used to symbolize the series.
    """

    bins: int


_LEMPEL_ZIV_COMPLEXITY_DEFAULT_PARAMS: list[LempelZivComplexityParams] = [
    {"bins": bins} for bins in [2, 3, 5, 10, 100]
]


class LempelZivComplexity(Feature[LempelZivComplexityParams]):
    """Normalized Lempel-Ziv phrase count after equal-width symbolization.

    Each series is discretized into ``bins`` equal-width levels. Scanning the
    resulting symbols from left to right, the parser adds the shortest phrase not
    already present in its phrase dictionary. The returned complexity is the number
    of phrases divided by the series length.

    All requested bin counts are symbolized together. Phrase parsing remains
    sequential within one symbolic row and runs independently across the combined
    batch, state, and parameter axes.

    Args:
        params: One dict per requested output. Defaults to tsfresh's bin counts
            2, 3, 5, 10, and 100. An explicit empty list is rejected.

    References:
        Ratsaby, J., & Timashkov, A. (2023). Accelerating the LZ-complexity
        algorithm. 2023 IEEE 29th International Conference on Parallel and
        Distributed Systems. https://doi.org/10.1109/ICPADS60453.2023.00038
    """

    def __init__(self, params: list[LempelZivComplexityParams] | None = None) -> None:
        if params is None:
            params = list(_LEMPEL_ZIV_COMPLEXITY_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "LempelZivComplexity requires at least one param; omit `params` to use "
                f"the default {_LEMPEL_ZIV_COMPLEXITY_DEFAULT_PARAMS!r} instead of passing []."
            )
        if any(param["bins"] < 1 for param in params):
            raise ValueError("Parameter bins must be positive.")
        self.params: list[LempelZivComplexityParams] = params
        self._bins = torch.tensor([param["bins"] for param in params], dtype=torch.long)
        self._max_bins = max(param["bins"] for param in params)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Symbolization broadcasts bins; the registered operator owns the combined
        # batch, state, and parameter rows.
        batch_size, n_states, sequence_length = x.shape
        if sequence_length == 0:
            return torch.zeros(
                batch_size,
                n_states,
                len(self.params),
                dtype=x.dtype,
                device=x.device,
            )

        bins = self._bins.to(device=x.device)
        minimum = x.min(dim=2).values
        maximum = x.max(dim=2).values
        step = (maximum - minimum).unsqueeze(-1) / bins.view(1, 1, -1)
        levels = torch.arange(1, self._max_bins + 1, dtype=x.dtype, device=x.device)
        edges = minimum.unsqueeze(-1).unsqueeze(-1) + step.unsqueeze(-1) * levels
        valid_edges = levels.view(1, 1, 1, -1) <= bins.view(1, 1, -1, 1)
        last_edges = levels.view(1, 1, 1, -1) == bins.view(1, 1, -1, 1)
        edges = torch.where(last_edges, maximum.unsqueeze(-1).unsqueeze(-1), edges)
        edges = torch.where(valid_edges, edges, torch.full_like(edges, float("inf")))

        values = x.unsqueeze(2).expand(-1, -1, len(self.params), -1)
        symbols = torch.searchsorted(edges.contiguous(), values.contiguous(), right=False)
        finite_symbols = torch.minimum(symbols, bins.view(1, 1, -1, 1) - 1)
        has_nan = torch.isnan(x).any(dim=2, keepdim=True).unsqueeze(2)
        symbols = torch.where(has_nan, torch.zeros_like(finite_symbols), finite_symbols)
        symbol_rows = symbols.reshape(-1, sequence_length)
        phrase_counts = lempel_ziv_phrase_count(symbol_rows)
        return (
            phrase_counts.to(dtype=x.dtype).reshape(batch_size, n_states, len(self.params))
            / sequence_length
        )

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format: ``lempel_ziv_complexity__bins_<bins>``.
        """
        return [f"lempel_ziv_complexity__bins_{param['bins']}" for param in self.params]


class CidCeParams(TypedDict):
    """Parameters for one `CidCe` output column.

    Attributes:
        normalize: Whether to divide the series by its population standard deviation.
    """

    normalize: bool


_CID_CE_DEFAULT_PARAMS: list[CidCeParams] = [{"normalize": True}, {"normalize": False}]


class CidCe(Feature[CidCeParams]):
    r"""Complexity estimate based on the Euclidean norm of first differences.

    \[
    CE = \sqrt{\sum_{i=1}^{T-1}(x_{i+1}-x_i)^2}.
    \]

    With ``normalize=True``, the series is first divided by its population standard
    deviation. A constant series then returns zero.

    Args:
        params: One dict per requested output. Defaults to both normalization states.
            An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[CidCeParams] | None = None) -> None:
        if params is None:
            params = list(_CID_CE_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "CidCe requires at least one param; omit `params` to use the "
                f"default {_CID_CE_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[CidCeParams] = params
        self._normalize_flags = torch.tensor(
            [param["normalize"] for param in params], dtype=torch.bool
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Both normalization variants select from one shared CE/std computation.
        diff = x[:, :, 1:] - x[:, :, :-1]
        ce = torch.sqrt((diff**2).sum(dim=2, keepdim=True))
        std = x.std(dim=2, correction=0, keepdim=True)
        std_is_zero = std == 0
        safe_std = torch.where(std_is_zero, torch.ones_like(std), std)
        ce_normalized = torch.where(std_is_zero, torch.zeros_like(ce), ce / safe_std)
        flags = self._normalize_flags.to(device=x.device).view(1, 1, -1)
        return torch.where(flags, ce_normalized, ce)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format: ``cid_ce__normalize_<normalize>``.
        """
        return [f"cid_ce__normalize_{param['normalize']}" for param in self.params]


class ApproximateEntropyParams(TypedDict):
    """Parameters for one `ApproximateEntropy` output column.

    Attributes:
        m: Length of the compared sample windows.
        r: Nonnegative tolerance multiplier applied to the population standard deviation.
    """

    m: int
    r: float


_APPROXIMATE_ENTROPY_DEFAULT_PARAMS: list[ApproximateEntropyParams] = [
    {"m": 2, "r": r} for r in [0.1, 0.3, 0.5, 0.7, 0.9]
]


class ApproximateEntropy(Feature[ApproximateEntropyParams]):
    r"""Likelihood that nearby runs remain nearby after one-sample extension.

    \[
    \operatorname{ApEn}(m,r)=\left|\Phi^m(r)-\Phi^{m+1}(r)\right|,
    \]

    where each \(\Phi^k\) averages the log fraction of length-\(k\) windows within
    Chebyshev distance \(r\,\operatorname{std}(x)\). Results are zero when
    ``T <= m + 1``.

    Args:
        params: One dict per requested output. Defaults to ``m=2`` with tsfresh's
            five tolerance values. ``r`` must be nonnegative. An explicit empty list
            is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[ApproximateEntropyParams] | None = None) -> None:
        if params is None:
            params = list(_APPROXIMATE_ENTROPY_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "ApproximateEntropy requires at least one param; omit `params` to use "
                f"the default {_APPROXIMATE_ENTROPY_DEFAULT_PARAMS!r} instead of passing []."
            )
        if any(param["r"] < 0 for param in params):
            raise ValueError("Parameter r must be positive.")
        self.params: list[ApproximateEntropyParams] = params
        self._window_lengths = torch.tensor([param["m"] for param in params], dtype=torch.long)
        self._rs = torch.tensor([param["r"] for param in params], dtype=torch.float64)
        self._max_window_length = max(param["m"] for param in params) + 1

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Padded windows broadcast all requested (m, r) pairs through each distance pass.
        batch_size, n_states, T = x.shape
        if T == 0:
            return torch.zeros(
                batch_size, n_states, len(self.params), dtype=x.dtype, device=x.device
            )
        window_lengths = self._window_lengths.to(device=x.device)
        rs = self._rs.to(device=x.device, dtype=x.dtype)
        tolerances = rs.view(1, 1, -1) * x.std(dim=2, correction=0, keepdim=True)
        starts = torch.arange(T, device=x.device)

        phi_values: list[Tensor] = []
        for extension in (0, 1):
            lengths = window_lengths + extension
            num_windows = T - lengths + 1
            valid_windows = starts[None, :] < num_windows[:, None]
            cheb = torch.zeros(
                batch_size,
                n_states,
                T,
                T,
                dtype=x.dtype,
                device=x.device,
            )
            within = torch.zeros(
                batch_size,
                n_states,
                len(self.params),
                T,
                T,
                dtype=torch.bool,
                device=x.device,
            )
            for offset in range(self._max_window_length):
                indices = (starts + offset).clamp(max=T - 1)
                column = x[:, :, indices]
                distance = (column.unsqueeze(-1) - column.unsqueeze(-2)).abs()
                cheb = torch.maximum(cheb, distance)
                candidate = cheb.unsqueeze(2) <= tolerances.view(batch_size, n_states, -1, 1, 1)
                active = (lengths == offset + 1).view(1, 1, -1, 1, 1)
                within = torch.where(active, candidate, within)

            valid_pairs = valid_windows[:, :, None] & valid_windows[:, None, :]
            within = within & valid_pairs.view(1, 1, len(self.params), T, T)
            safe_num_windows = num_windows.clamp(min=1).to(dtype=x.dtype)
            counts = within.sum(dim=-1).to(dtype=x.dtype) / safe_num_windows.view(1, 1, -1, 1)
            log_counts = torch.where(
                valid_windows.view(1, 1, len(self.params), T),
                counts.clamp(min=torch.finfo(x.dtype).tiny).log(),
                torch.zeros_like(counts),
            )
            phi_values.append(log_counts.sum(dim=-1) / safe_num_windows.view(1, 1, -1))

        entropy = (phi_values[0] - phi_values[1]).abs()
        return torch.where(
            (window_lengths + 1 < T).view(1, 1, -1),
            entropy,
            torch.zeros_like(entropy),
        )

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in parameter order. Format: ``approximate_entropy__m_<m>__r_<r>``.
        """
        return [f"approximate_entropy__m_{param['m']}__r_{param['r']}" for param in self.params]


class SampleEntropy(ParameterlessFeature):
    r"""Negative log probability that matching length-2 runs still match at length 3.

    \[
    \operatorname{SampEn}=-\log(A/B),
    \]

    where \(B\) counts distinct matching pairs of length-2 windows and \(A\) counts
    distinct matching pairs of length-3 windows at tolerance
    \(0.2\,\operatorname{std}(x)\). A series containing NaN returns NaN.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, 1).
        """
        batch_size, n_states, T = x.shape
        tolerance = 0.2 * x.std(dim=2, correction=0, keepdim=True)
        counts: list[Tensor] = []
        m = 2
        for window_len in (m, m + 1):
            num_windows = T - window_len + 1
            windows = x.unfold(dimension=2, size=window_len, step=1)
            cheb = torch.zeros(
                batch_size,
                n_states,
                num_windows,
                num_windows,
                dtype=x.dtype,
                device=x.device,
            )
            for offset in range(window_len):
                column = windows[..., offset]
                cheb = torch.maximum(cheb, (column.unsqueeze(3) - column.unsqueeze(2)).abs())
            within = cheb <= tolerance.unsqueeze(-1)
            counts.append(within.sum(dim=(2, 3)).to(x.dtype) - num_windows)

        entropy = -torch.log(counts[1] / counts[0])
        has_nan = torch.isnan(x).any(dim=2)
        entropy = torch.where(has_nan, torch.full_like(entropy, float("nan")), entropy)
        return entropy.unsqueeze(-1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``sample_entropy``.
        """
        return ["sample_entropy"]
