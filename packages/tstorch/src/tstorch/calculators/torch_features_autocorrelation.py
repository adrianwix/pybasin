r"""Autocorrelation-based feature calculators for time series.

All features follow a consistent tensor shape convention:
- Input: (B, S, T) where B=batch size, S=state variables, T=timesteps
- Output: (B, S, F) always, where F=1 for single-output features and F=len(params)
  for multi-output features (e.g. `AggAutocorrelation`).

Features are computed along the time dimension (dim=2), preserving batch and state
dimensions. Each class below inherits ``tstorch.feature_factory.Feature``: direct calls
and ``TorchFeatureExtractor`` both construct the same class the same way, there is no
separate public/internal function pair.
"""

from typing import NotRequired, Protocol, TypedDict, cast

import torch
from torch import Tensor

from tstorch.feature_factory import Feature
from tstorch.settings import DYNAMICAL_SYSTEM_FEATURES_CONFIG
from tstorch.torch_feature_utilities import local_maxima_1d


class _FFTNamespace(Protocol):
    """Typed boundary for PyTorch's dynamically exposed FFT namespace."""

    def rfft(self, input: Tensor, n: int | None = None, dim: int = -1) -> Tensor: ...

    def irfft(self, input: Tensor, n: int | None = None, dim: int = -1) -> Tensor: ...


_fft = cast(_FFTNamespace, torch.fft)

# =============================================================================
# AUTOCORRELATION FEATURES (4 features)
# =============================================================================


class AutocorrelationParams(TypedDict):
    """Parameters for one `Autocorrelation` output column.

    Attributes:
        lag: Lag index to evaluate. ``0`` requests the zero-lag autocorrelation;
            values greater than or equal to the time dimension ``T`` return 0.
    """

    lag: int


# Direct-sum branch applies for len(distinct requested lags) <= this; FFT branch above it.
# Keyed on lag *count*, not (unlike `_DIRECT_SUM_MAXLAG_THRESHOLD` below, used by
# `AggAutocorrelation`) on the max lag value: Autocorrelation only computes the distinct
# lags actually requested, so its cost is O(count * T) regardless of how large any individual
# lag is. Starting value carried over from `_DIRECT_SUM_MAXLAG_THRESHOLD`'s measurement, not
# independently re-measured for this criterion - see references/performance-notes.md.
_AUTOCORRELATION_DIRECT_SUM_LAG_COUNT_THRESHOLD = 128

# Matches tsfresh's `ComprehensiveFCParameters` default for `autocorrelation`.
_AUTOCORRELATION_DEFAULT_PARAMS: list[AutocorrelationParams] = [{"lag": lag} for lag in range(10)]


class Autocorrelation(Feature[AutocorrelationParams]):
    r"""Autocorrelation at every requested lag.

    \[
    r_k =
      \frac{\sum_{t=1}^{n-k} (x_t - \bar{x})(x_{t+k} - \bar{x})}
           {(n - k)\,\mathrm{Var}(x)}
    \]

    the adjusted estimator (matches tsfresh's ``autocorrelation``, per the Wikipedia
    formula tsfresh's own docstring cites; equivalently
    ``statsmodels.tsa.stattools.acf(x, adjusted=True)``). Lag 0 is *not* unconditionally
    1: for a degenerate series (``np.isclose(np.var(x), 0)``), the result is NaN at
    every lag including 0, since the formula's value at lag 0
    (``Var(x) / Var(x)``) is as undefined there as at any other lag.

    tsfresh itself calls this once *per* lag (``fctype="simple"``), but this class shares
    one autocovariance pass across every requested lag, so a caller requesting several lags
    at once gets the shared-pass speedup instead of paying per-lag overhead in a Python-level
    loop - see references/performance-notes.md for the measured win.

    Args:
        params: One dict per requested output, with a ``lag`` (int) key. May contain 0
            or repeated values. Defaults to tsfresh's own default (lags 0 through 9) when
            omitted; an explicit empty list is rejected rather than silently producing a
            feature with no output columns.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[AutocorrelationParams] | None = None) -> None:
        if params is None:
            params = list(_AUTOCORRELATION_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "Autocorrelation requires at least one param; omit `params` to use the "
                f"default {_AUTOCORRELATION_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[AutocorrelationParams] = params
        self._lags: list[int] = [p["lag"] for p in params]

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute autocorrelation values.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)). 0 at any column whose
            ``lag >= T``.
        """
        batch_size, n_states, T = x.shape
        lags = self._lags
        K = len(lags)

        x_centered = x - x.mean(dim=2, keepdim=True)
        var_full = (x_centered**2).sum(dim=2, keepdim=True)
        var = var_full / T

        # Only lags >= 1 need a computed autocovariance - lag 0 and lag >= T are both fully
        # determined by the masks below.
        compute_lags = list(dict.fromkeys(lag for lag in lags if lag >= 1))
        L = len(compute_lags)

        if L == 0:
            acf = torch.zeros(batch_size, n_states, 0, dtype=x.dtype, device=x.device)
        else:
            if L <= _AUTOCORRELATION_DIRECT_SUM_LAG_COUNT_THRESHOLD:
                autocov_by_lag = torch.stack(
                    [
                        (x_centered[..., :-lag] * x_centered[..., lag:]).sum(dim=2)
                        for lag in compute_lags
                    ],
                    dim=-1,
                )
            else:
                n_fft = 2 * T
                fft_x = _fft.rfft(x_centered, n=n_fft, dim=2)
                power_spectrum = fft_x.real**2 + fft_x.imag**2
                autocorr_full = _fft.irfft(power_spectrum, n=n_fft, dim=2)
                lag_index = torch.tensor(compute_lags, device=x.device, dtype=torch.long)
                autocov_by_lag = autocorr_full.index_select(dim=2, index=lag_index)

            lags_f = torch.tensor(compute_lags, device=x.device, dtype=x.dtype).view(1, 1, L)
            denom = (T - lags_f) * var
            acf = autocov_by_lag / (denom + 1e-10)

        col_of_lag = {lag: idx for idx, lag in enumerate(compute_lags)}
        gather_idx = (
            torch
            .tensor([col_of_lag.get(lag, 0) for lag in lags], device=x.device, dtype=torch.long)
            .view(1, 1, K)
            .expand(batch_size, n_states, K)
        )
        acf_full = (
            acf.gather(dim=2, index=gather_idx)
            if L > 0
            else torch.zeros(batch_size, n_states, K, dtype=x.dtype, device=x.device)
        )

        lags_t = torch.tensor(lags, device=x.device, dtype=torch.long).view(1, 1, K)
        is_zero_lag = lags_t == 0
        is_ge_T = lags_t >= T
        is_degenerate = (var.abs() <= 1e-8).expand(batch_size, n_states, K)

        result = torch.where(is_zero_lag, torch.ones_like(acf_full), acf_full)
        result = torch.where(is_ge_T, torch.zeros_like(result), result)
        return torch.where(is_degenerate & ~is_ge_T, torch.full_like(result, float("nan")), result)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in the same order as ``params``. Format: ``autocorrelation__lag_<lag>``.
        """
        return [f"autocorrelation__lag_{p['lag']}" for p in self.params]


class PartialAutocorrelationParams(TypedDict):
    """Parameters for one `PartialAutocorrelation` output column.

    Attributes:
        lag: Lag index to evaluate. ``0`` returns 1; values beyond the recursion's
            computable range return 0.
    """

    lag: int


# Matches tsfresh's `ComprehensiveFCParameters` default for `partial_autocorrelation`.
_PARTIAL_AUTOCORRELATION_DEFAULT_PARAMS: list[PartialAutocorrelationParams] = [
    {"lag": lag} for lag in range(10)
]


class PartialAutocorrelation(Feature[PartialAutocorrelationParams]):
    r"""Partial autocorrelation at every requested lag.

    The partial autocorrelation at lag \(k\) is the coefficient \(\phi_{k,k}\)
    obtained by running the Durbin-Levinson recursion

    \[
    \begin{aligned}
    \phi_{k,k}
      &= \frac{r_k - \sum_{j=1}^{k-1} \phi_{k-1,j}\, r_{k-j}}
              {1 - \sum_{j=1}^{k-1} \phi_{k-1,j}\, r_j}, \\
    \phi_{k,j}
      &= \phi_{k-1,j} - \phi_{k,k}\, \phi_{k-1,k-j}.
    \end{aligned}
    \]

    on the adjusted autocorrelation function \(r\) (the same estimator
    `Autocorrelation` computes), up to order \(\min(k, T - 1)\). Matches
    tsfresh's ``partial_autocorrelation`` (``statsmodels.tsa.stattools.pacf(x,
    method="ld")``).

    The recursion is inherently sequential (order ``k`` depends on every order
    ``< k``), so every requested lag shares one recursion run up to
    ``max(requested lags)``. A single-lag request costs the same as computing every
    order up to it, so no efficiency is lost by batching several lags in one call.

    Args:
        params: One dict per requested output, with a ``lag`` (int) key. Defaults to
            tsfresh's own default (lags 0 through 9) when omitted; an explicit empty list
            is rejected rather than silently producing a feature with no output columns.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[PartialAutocorrelationParams] | None = None) -> None:
        if params is None:
            params = list(_PARTIAL_AUTOCORRELATION_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "PartialAutocorrelation requires at least one param; omit `params` to use "
                f"the default {_PARTIAL_AUTOCORRELATION_DEFAULT_PARAMS!r} instead of "
                "passing []."
            )
        self.params: list[PartialAutocorrelationParams] = params
        self._lags: Tensor = torch.tensor([p["lag"] for p in params], dtype=torch.long)
        self._global_maxlag: int = max(p["lag"] for p in params)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute partial autocorrelation values.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)). 1 at lag 0. 0 when
            ``lag > T - 1`` (beyond the recursion's computable range).
        """
        batch_size, n_states, T = x.shape
        lags = self._lags.to(device=x.device, dtype=torch.long)
        K = lags.shape[0]

        maxlag = min(self._global_maxlag, T - 1)

        if maxlag < 1:
            is_zero_lag = (lags == 0).view(1, 1, K)
            ones = torch.ones(batch_size, n_states, K, dtype=x.dtype, device=x.device)
            zeros = torch.zeros(batch_size, n_states, K, dtype=x.dtype, device=x.device)
            return torch.where(is_zero_lag, ones, zeros)

        # Durbin-Levinson recursion accumulates `phi[1:k] = phi[1:k] - phi_k * phi_prev`
        # repeated `maxlag` times. Running it in float64 avoids compounding float32 error on
        # near-singular Toeplitz systems from real periodic/oscillatory trajectories.
        x64 = x.double()
        x_centered = x64 - x64.mean(dim=2, keepdim=True)
        n_fft = 2 * T
        fft_x = _fft.rfft(x_centered, n=n_fft, dim=2)
        power_spectrum = fft_x.real**2 + fft_x.imag**2
        autocorr_full = _fft.irfft(power_spectrum, n=n_fft, dim=2)

        raw = autocorr_full[..., : maxlag + 1]
        lag_idx = torch.arange(maxlag + 1, device=x.device, dtype=torch.float64).view(1, 1, -1)
        var0 = raw[..., 0:1]
        acf = (raw * T / ((T - lag_idx) * var0)).permute(2, 0, 1)

        phi = torch.zeros(maxlag + 1, batch_size, n_states, dtype=torch.float64, device=x.device)
        pacf = torch.zeros(maxlag + 1, batch_size, n_states, dtype=torch.float64, device=x.device)
        pacf[0] = 1.0
        phi[1] = acf[1]
        pacf[1] = acf[1]

        for k in range(2, maxlag + 1):
            phi_prev = phi[1:k].flip(dims=[0])
            acf_prev = acf[1:k]
            num = acf[k] - (phi[1:k] * acf_prev.flip(dims=[0])).sum(dim=0)
            denom = 1.0 - (phi[1:k] * acf_prev).sum(dim=0)
            phi_k = num / denom
            pacf[k] = phi_k
            phi[k] = phi_k
            phi[1:k] = phi[1:k] - phi_k.unsqueeze(0) * phi_prev

        pacf_bsk = pacf.permute(1, 2, 0)
        valid = lags <= maxlag
        gather_idx = lags.clamp(min=0, max=maxlag).view(1, 1, K).expand(batch_size, n_states, K)
        result = pacf_bsk.gather(dim=2, index=gather_idx).to(x.dtype)
        return torch.where(valid.view(1, 1, K), result, torch.zeros_like(result))

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in the same order as ``params``. Format:
            ``partial_autocorrelation__lag_<lag>``.
        """
        return [f"partial_autocorrelation__lag_{p['lag']}" for p in self.params]


class AggAutocorrelationParams(TypedDict):
    """Parameters for one `AggAutocorrelation` output column.

    Attributes:
        maxlag: Largest autocorrelation lag included in the aggregation. ``0``
            gives NaN because the aggregated lag range is empty.
        f_agg: Aggregation function. Must be one of ``"mean"``, ``"median"``, or
            ``"var"``.
    """

    maxlag: int
    f_agg: str


# Exported so both AggAutocorrelation and torch_feature_extractor use the same
# string->index table - never a second copy.
AGG_AUTOCORRELATION_F_AGG_INDEX = {"mean": 0, "median": 1, "var": 2}

# Direct-sum branch applies for global_maxlag <= this; FFT branch above it. See
# references/performance-notes.md ("Small-batch dispatch overhead") for the measurement
# behind this threshold and the crossover.
_DIRECT_SUM_MAXLAG_THRESHOLD = 128

# Matches tsfresh's `ComprehensiveFCParameters` default for `agg_autocorrelation`.
_AGG_AUTOCORRELATION_DEFAULT_PARAMS: list[AggAutocorrelationParams] = [
    {"f_agg": f_agg, "maxlag": 40} for f_agg in ["mean", "median", "var"]
]


class AggAutocorrelation(Feature[AggAutocorrelationParams]):
    r"""Aggregated autocorrelation: for each requested ``(maxlag, f_agg)``, the mean,
    median, or variance of the autocorrelation function evaluated at lags ``1..maxlag``.

    Matches tsfresh's ``agg_autocorrelation`` (``fctype=combiner``). The autocorrelation
    at lag \(k\) is

    \[
    r_k =
      \frac{\sum_{t=1}^{n-k} (x_t - \bar{x})(x_{t+k} - \bar{x})}
           {(n - k)\,\mathrm{Var}(x)}
    \]

    the adjusted estimator (``statsmodels.tsa.stattools.acf(x, adjusted=True)``). For a
    degenerate series \(|\mathrm{Var}(x)| < 10^{-10}\), \(r_k = 0\) for every
    lag. A requested ``maxlag`` of 0 gives NaN for that column (an empty lag range,
    matching tsfresh's ``getattr(np, f_agg)(a[:0])``), independently of what any other
    requested column's ``maxlag`` is.

    Both ``maxlag`` and ``f_agg`` are genuinely batchable: tsfresh itself is
    ``fctype="combiner"`` here and shares one ACF pass across every requested ``maxlag``,
    read via a masked prefix per request - this class shares that same one pass, parity
    with tsfresh's own approach rather than an additional algorithmic win on its own (see
    migration-logs/autocorrelation.md).

    Args:
        params: One dict per requested output, with ``maxlag`` (int) and ``f_agg``
            (one of ``"mean"``, ``"median"``, ``"var"``) keys. Defaults to tsfresh's own
            default (``mean``/``median``/``var`` at ``maxlag=40``) when omitted; an explicit
            empty list is rejected rather than silently producing a feature with no output
            columns.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[AggAutocorrelationParams] | None = None) -> None:
        if params is None:
            params = list(_AGG_AUTOCORRELATION_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "AggAutocorrelation requires at least one param; omit `params` to use the "
                f"default {_AGG_AUTOCORRELATION_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[AggAutocorrelationParams] = params
        self._params_tensor: Tensor = torch.tensor(
            [[p["maxlag"], AGG_AUTOCORRELATION_F_AGG_INDEX[p["f_agg"]]] for p in params],
            dtype=torch.long,
        )
        self._global_maxlag: int = max(p["maxlag"] for p in params)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute aggregated autocorrelation values.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        batch_size, n_states, T = x.shape
        params = self._params_tensor.to(device=x.device, dtype=torch.long)
        maxlags = params[:, 0].clamp(max=T - 1)
        f_agg_idx = params[:, 1]
        K = params.shape[0]
        global_maxlag = self._global_maxlag

        if global_maxlag == 0:
            return torch.full(
                (batch_size, n_states, K), float("nan"), dtype=x.dtype, device=x.device
            )

        x_centered = x - x.mean(dim=2, keepdim=True)
        var_full = (x_centered**2).sum(dim=2, keepdim=True)

        if 1 <= global_maxlag <= _DIRECT_SUM_MAXLAG_THRESHOLD:
            autocov_by_lag = torch.stack(
                [
                    (x_centered[..., :-lag] * x_centered[..., lag:]).sum(dim=2)
                    for lag in range(1, global_maxlag + 1)
                ],
                dim=-1,
            )
        else:
            n_fft = 2 * T
            fft_x = _fft.rfft(x_centered, n=n_fft, dim=2)
            power_spectrum = fft_x.real**2 + fft_x.imag**2
            autocorr_full = _fft.irfft(power_spectrum, n=n_fft, dim=2)
            autocov_by_lag = autocorr_full[..., 1 : global_maxlag + 1]

        lags = torch.arange(1, global_maxlag + 1, device=x.device, dtype=x.dtype)
        denom = (T - lags) * (var_full / T)
        is_degenerate = (var_full / T).abs() < 1e-10
        safe_denom = torch.where(is_degenerate, torch.ones_like(denom), denom)
        acf = autocov_by_lag / safe_denom
        acf = torch.where(is_degenerate, torch.zeros_like(acf), acf)

        lag_idx = torch.arange(global_maxlag, device=x.device).view(1, -1)
        mask = lag_idx < maxlags.view(-1, 1)
        counts_raw = mask.sum(dim=1)
        counts = counts_raw.clamp(min=1)

        acf_exp = acf.unsqueeze(2)
        mask_exp = mask.view(1, 1, K, global_maxlag)

        masked = torch.where(mask_exp, acf_exp, torch.zeros_like(acf_exp))
        mean_val = masked.sum(dim=-1) / counts.view(1, 1, K)
        mean_sq = torch.where(mask_exp, acf_exp**2, torch.zeros_like(acf_exp)).sum(
            dim=-1
        ) / counts.view(1, 1, K)
        var_val = mean_sq - mean_val**2

        sort_source = torch.where(
            mask_exp,
            acf_exp.expand(batch_size, n_states, K, global_maxlag),
            torch.full(
                (batch_size, n_states, K, global_maxlag),
                float("inf"),
                dtype=x.dtype,
                device=x.device,
            ),
        )
        sorted_vals, _ = sort_source.sort(dim=-1)
        c = counts.view(1, 1, K, 1)
        lo_idx = ((c - 1) // 2).expand(batch_size, n_states, K, 1)
        hi_idx = (c // 2).expand(batch_size, n_states, K, 1)
        median_val = (sorted_vals.gather(-1, lo_idx) + sorted_vals.gather(-1, hi_idx)).squeeze(
            -1
        ) / 2

        stacked = torch.stack([mean_val, median_val, var_val], dim=-1)
        gather_idx = f_agg_idx.view(1, 1, K, 1).expand(batch_size, n_states, K, 1)
        result = stacked.gather(-1, gather_idx).squeeze(-1)
        is_empty = (counts_raw == 0).view(1, 1, K).expand(batch_size, n_states, K)
        return torch.where(is_empty, torch.full_like(result, float("nan")), result)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in the same order as ``params``. Format:
            ``agg_autocorrelation__f_agg_"<f_agg>"__maxlag_<maxlag>``.
        """
        return [
            f'agg_autocorrelation__f_agg_"{p["f_agg"]}"__maxlag_{p["maxlag"]}' for p in self.params
        ]


class AutocorrelationPeriodicityParams(TypedDict):
    """Parameters for one `AutocorrelationPeriodicity` output column.

    Attributes:
        min_lag: First lag eligible to be considered as a periodicity peak. Defaults
            to 2.
        peak_threshold: Minimum normalized autocorrelation value for an eligible
            peak. Defaults to 0.3.
        output: Which measure to return. ``"strength"`` returns the first peak
            height; ``"period"`` returns the first peak lag. Defaults to
            ``"strength"``.
    """

    min_lag: NotRequired[int]
    peak_threshold: NotRequired[float]
    output: NotRequired[str]


_AUTOCORRELATION_PERIODICITY_DEFAULT_MIN_LAG = 2
_AUTOCORRELATION_PERIODICITY_DEFAULT_PEAK_THRESHOLD = 0.3
_AUTOCORRELATION_PERIODICITY_DEFAULT_OUTPUT = "strength"

_AUTOCORRELATION_PERIODICITY_DEFAULT_PARAMS = cast(
    "list[AutocorrelationPeriodicityParams]",
    DYNAMICAL_SYSTEM_FEATURES_CONFIG["autocorrelation_periodicity"],
)


class AutocorrelationPeriodicity(Feature[AutocorrelationPeriodicityParams]):
    r"""Autocorrelation-based periodicity measures: the height and lag of the first
    significant peak of the normalized autocorrelation function.

    Multi-output feature: each requested param dict selects one measure (see ``output``).

    The normalized autocorrelation is

    \[
    r_k = \frac{\mathrm{acf}(k)}{\mathrm{acf}(0)}, \qquad k = 0, \dots, T - 1
    \]

    computed via FFT, where ``acf`` is the autocovariance of the mean-centered,
    std-normalized series. A lag \(k \ge\) ``min_lag`` is a peak candidate when
    \(r_k\) is a local maximum and \(r_k >\) ``peak_threshold``; the *first*
    such \(k\) (in increasing order) determines both outputs: ``strength`` is
    \(r_k\) there, ``period`` is \(k\) itself. Both default to 0 when no lag
    satisfies both conditions. Has no tsfresh equivalent - useful for distinguishing
    limit cycles, chaos, and fixed points in a dynamical system's trajectory.

    ``min_lag``/``peak_threshold`` are structural (they change what counts as a peak, not
    just which final value is read): requested params are grouped by ``(min_lag,
    peak_threshold)`` and share one FFT + peak-detection pass per group - in practice
    almost always one group, since real configs only vary ``output`` (see
    ``DYNAMICAL_SYSTEM_FEATURES_CONFIG`` in settings.py). ``strength``/``period`` are already
    both computed by that one pass regardless of which is requested, so batching several
    ``output`` values within a group is free.

    Args:
        params: One dict per requested output, optionally with ``min_lag`` (default
            2), ``peak_threshold`` (default 0.3), and ``output`` (``"strength"`` or
            ``"period"``, default ``"strength"``) keys. Defaults to
            ``DYNAMICAL_SYSTEM_FEATURES_CONFIG``'s preset (``strength`` and ``period``) when
            omitted; an explicit empty list is rejected rather than silently producing a
            feature with no output columns.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for
        partial autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[AutocorrelationPeriodicityParams] | None = None) -> None:
        if params is None:
            params = list(_AUTOCORRELATION_PERIODICITY_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "AutocorrelationPeriodicity requires at least one param; omit `params` to "
                f"use the default {_AUTOCORRELATION_PERIODICITY_DEFAULT_PARAMS!r} instead "
                "of passing []."
            )
        self.params: list[AutocorrelationPeriodicityParams] = params
        self._group_of: list[tuple[int, float]] = [
            (
                p.get("min_lag", _AUTOCORRELATION_PERIODICITY_DEFAULT_MIN_LAG),
                p.get("peak_threshold", _AUTOCORRELATION_PERIODICITY_DEFAULT_PEAK_THRESHOLD),
            )
            for p in params
        ]
        self._wants_period: list[bool] = [
            p.get("output", _AUTOCORRELATION_PERIODICITY_DEFAULT_OUTPUT) == "period" for p in params
        ]
        self._distinct_groups: list[tuple[int, float]] = list(
            dict.fromkeys(self._group_of)
        )  # first-seen order dedup

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Compute autocorrelation periodicity values.

        Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        T = x.shape[2]
        core_by_group: dict[tuple[int, float], tuple[Tensor, Tensor]] = {}
        for min_lag, peak_threshold in self._distinct_groups:
            x_centered = x - x.mean(dim=2, keepdim=True)
            x_std = x.std(dim=2, keepdim=True)
            x_normalized = x_centered / (x_std + 1e-10)

            n_fft = 2 * T
            fft_x = _fft.rfft(x_normalized, n=n_fft, dim=2)
            power_spectrum = fft_x.real**2 + fft_x.imag**2
            autocorr_full = _fft.irfft(power_spectrum, n=n_fft, dim=2)

            autocorr = autocorr_full[..., :T] / (autocorr_full[..., 0:1] + 1e-10)

            peaks_mask = local_maxima_1d(autocorr)
            peaks_mask[..., :min_lag] = False

            above_threshold = autocorr > peak_threshold
            valid_peaks = peaks_mask & above_threshold

            # First True index along T per (B, S): non-peaks become a sentinel >= T.
            time_idx = torch.arange(T, device=x.device, dtype=torch.long).view(1, 1, T)
            sentinel = torch.full_like(time_idx, T)
            candidate_idx = torch.where(valid_peaks, time_idx, sentinel)
            first_peak_idx = candidate_idx.amin(dim=2)
            found_any = first_peak_idx < T

            gather_idx = first_peak_idx.clamp(max=T - 1).unsqueeze(-1)
            periodicity_strength = torch.where(
                found_any,
                autocorr.gather(dim=2, index=gather_idx).squeeze(-1),
                torch.zeros_like(x[:, :, 0]),
            )
            period_estimate = torch.where(
                found_any, first_peak_idx.to(x.dtype), torch.zeros_like(x[:, :, 0])
            )
            core_by_group[(min_lag, peak_threshold)] = (
                periodicity_strength.unsqueeze(-1),
                period_estimate.unsqueeze(-1),
            )

        columns = [
            core_by_group[group][1] if want_period else core_by_group[group][0]
            for group, want_period in zip(self._group_of, self._wants_period, strict=True)
        ]
        return torch.cat(columns, dim=-1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in the same order as ``params``. Format:
            ``autocorrelation_periodicity__min_lag_<min_lag>__peak_threshold_<peak_threshold>__output_<output>``.
        """
        return [
            f"autocorrelation_periodicity__min_lag_"
            f"{p.get('min_lag', _AUTOCORRELATION_PERIODICITY_DEFAULT_MIN_LAG)}"
            f"__peak_threshold_"
            f"{p.get('peak_threshold', _AUTOCORRELATION_PERIODICITY_DEFAULT_PEAK_THRESHOLD)}"
            f"__output_{p.get('output', _AUTOCORRELATION_PERIODICITY_DEFAULT_OUTPUT)}"
            for p in self.params
        ]
