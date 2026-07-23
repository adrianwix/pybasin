"""Frequency-domain feature calculators."""

from __future__ import annotations

import math
from functools import cache
from typing import Protocol, TypedDict

import torch
import torch.nn.functional as functional
from torch import Tensor

from tstorch.feature_factory import Feature, ParameterlessFeature


class _RealFft(Protocol):
    def __call__(
        self,
        input: Tensor,
        n: int | None = None,
        dim: int = -1,
        norm: str | None = None,
    ) -> Tensor: ...


_rfft: _RealFft = vars(torch.fft)["rfft"]


class FftCoefficientParams(TypedDict):
    """Parameters for one `FftCoefficient` output column.

    Attributes:
        coeff: Real-FFT coefficient index.
        attr: Attribute to return: ``real``, ``imag``, ``abs``, or ``angle``.
    """

    coeff: int
    attr: str


_FFT_COEFFICIENT_DEFAULT_PARAMS: list[FftCoefficientParams] = [
    {"coeff": coeff, "attr": attr}
    for attr in ("real", "imag", "abs", "angle")
    for coeff in range(100)
]
_FFT_COEFFICIENT_ATTR_INDEX = {"real": 0, "imag": 1, "abs": 2, "angle": 3}


class FftCoefficient(Feature[FftCoefficientParams]):
    r"""Attributes of requested real discrete Fourier transform coefficients.

    For \(X_k=\operatorname{rfft}(x)_k\), each output is one of
    \(\operatorname{Re}(X_k)\), \(\operatorname{Im}(X_k)\), \(|X_k|\), or
    \(\arg(X_k)\). Angles are measured in degrees. Coefficients outside the available
    FFT bins produce NaN, matching tsfresh.

    Args:
        params: One dict per requested output, with ``coeff`` and ``attr`` keys.
            Defaults to tsfresh's comprehensive set of coefficients 0 through 99 for
            all four attributes. An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[FftCoefficientParams] | None = None) -> None:
        if params is None:
            params = list(_FFT_COEFFICIENT_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "FftCoefficient requires at least one param; omit `params` to use the "
                f"default {_FFT_COEFFICIENT_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[FftCoefficientParams] = params
        self._coeffs = torch.tensor([param["coeff"] for param in params], dtype=torch.long)
        self._attr_indices = torch.tensor(
            [_FFT_COEFFICIENT_ATTR_INDEX[param["attr"]] for param in params],
            dtype=torch.long,
        )
        unique_attr_indices = {_FFT_COEFFICIENT_ATTR_INDEX[param["attr"]] for param in params}
        self._single_attr_index = (
            next(iter(unique_attr_indices)) if len(unique_attr_indices) == 1 else None
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Every requested attribute and coefficient reads from one shared FFT.
        coeffs = self._coeffs.to(device=x.device)
        attr_indices = self._attr_indices.to(device=x.device)
        fft_result = _rfft(x, dim=2)
        n_coeffs = fft_result.shape[2]
        valid = (coeffs >= -n_coeffs) & (coeffs < n_coeffs)
        normalized = torch.where(coeffs < 0, coeffs + n_coeffs, coeffs)
        safe_coeffs = normalized.clamp(min=0, max=n_coeffs - 1)
        selected_fft = fft_result.index_select(dim=2, index=safe_coeffs)
        if self._single_attr_index == 2:
            gathered = torch.abs(selected_fft)
        else:
            selected_components = torch.view_as_real(selected_fft)
            selected_real = selected_components[..., 0]
            selected_imag = selected_components[..., 1]
            if self._single_attr_index == 0:
                gathered = selected_real
            elif self._single_attr_index == 1:
                gathered = selected_imag
            elif self._single_attr_index == 3:
                gathered = torch.rad2deg(torch.atan2(selected_imag, selected_real))
            else:
                attributes = torch.stack(
                    (
                        selected_real,
                        selected_imag,
                        torch.abs(selected_fft),
                        torch.rad2deg(torch.atan2(selected_imag, selected_real)),
                    ),
                    dim=-1,
                )
                output_indices = torch.arange(coeffs.shape[0], device=x.device)
                gathered = attributes[:, :, output_indices, attr_indices]
        return torch.where(
            valid.view(1, 1, -1),
            gathered,
            torch.full_like(gathered, float("nan")),
        )

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format:
            ``fft_coefficient__coeff_<coeff>__attr_<attr>``.
        """
        return [
            f"fft_coefficient__coeff_{param['coeff']}__attr_{param['attr']}"
            for param in self.params
        ]


class FftAggregatedParams(TypedDict):
    """Parameters for one `FftAggregated` output column.

    Attributes:
        aggtype: Statistic to return: ``centroid``, ``variance``, ``skew``, or
            ``kurtosis``.
    """

    aggtype: str


_FFT_AGGREGATED_DEFAULT_PARAMS: list[FftAggregatedParams] = [
    {"aggtype": aggtype} for aggtype in ("centroid", "variance", "skew", "kurtosis")
]
_FFT_AGGREGATED_AGGTYPE_INDEX = {
    "centroid": 0,
    "variance": 1,
    "skew": 2,
    "kurtosis": 3,
}


class FftAggregated(Feature[FftAggregatedParams]):
    r"""Statistics of the FFT magnitude spectrum.

    Treating \(|X_k|\) as unnormalized weights over frequency-bin indices, this
    feature returns the centroid, variance, skew, or kurtosis requested by each
    parameter. Skew and kurtosis are NaN when the spectral variance is below 0.5.
    The kurtosis expression follows tsfresh's raw-moment implementation exactly.

    Args:
        params: One dict per requested output, each with an ``aggtype`` key. Defaults
            to all four supported statistics. An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[FftAggregatedParams] | None = None) -> None:
        if params is None:
            params = list(_FFT_AGGREGATED_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "FftAggregated requires at least one param; omit `params` to use the "
                f"default {_FFT_AGGREGATED_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[FftAggregatedParams] = params
        self._aggtype_indices = torch.tensor(
            [_FFT_AGGREGATED_AGGTYPE_INDEX[param["aggtype"]] for param in params],
            dtype=torch.long,
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # All requested moments share one magnitude spectrum and its raw moments.
        fft_result = _rfft(x, dim=2)
        magnitude = torch.abs(fft_result)
        frequencies = torch.arange(magnitude.shape[2], dtype=x.dtype, device=x.device).view(
            1, 1, -1
        )
        weights = magnitude / magnitude.sum(dim=2, keepdim=True)
        m1 = (frequencies * weights).sum(dim=2)
        m2 = (frequencies.square() * weights).sum(dim=2)
        m3 = (frequencies.pow(3) * weights).sum(dim=2)
        m4 = (frequencies.pow(4) * weights).sum(dim=2)
        variance = m2 - m1.square()
        degenerate = variance < 0.5
        skew_raw = (m3 - 3 * m1 * variance - m1.pow(3)) / variance.pow(1.5)
        kurtosis_raw = (m4 - 4 * m1 * m3 + 6 * m2 * m1.square() - 3 * m1) / variance.square()
        nan = torch.full_like(variance, float("nan"))
        values = torch.stack(
            (
                m1,
                variance,
                torch.where(degenerate, nan, skew_raw),
                torch.where(degenerate, nan, kurtosis_raw),
            ),
            dim=-1,
        )
        return values.index_select(dim=-1, index=self._aggtype_indices.to(device=x.device))

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format:
            ``fft_aggregated__aggtype_<aggtype>``.
        """
        return [f"fft_aggregated__aggtype_{param['aggtype']}" for param in self.params]


class SpktWelchDensityParams(TypedDict):
    """Parameters for one `SpktWelchDensity` output column.

    Attributes:
        coeff: Welch power-spectral-density coefficient index.
    """

    coeff: int


_SPKT_WELCH_DENSITY_DEFAULT_PARAMS: list[SpktWelchDensityParams] = [
    {"coeff": coeff} for coeff in (2, 5, 8)
]


class SpktWelchDensity(Feature[SpktWelchDensityParams]):
    """Welch power spectral density at requested frequency coefficients.

    The spectrum uses Hann-windowed segments of at most 256 samples, half-window
    overlap, detrending by each segment mean, and SciPy-compatible density scaling.
    Coefficients outside the available one-sided spectrum produce NaN.

    Args:
        params: One dict per requested output, each with a ``coeff`` key. Defaults to
            coefficients 2, 5, and 8. An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[SpktWelchDensityParams] | None = None) -> None:
        if params is None:
            params = list(_SPKT_WELCH_DENSITY_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "SpktWelchDensity requires at least one param; omit `params` to use the "
                f"default {_SPKT_WELCH_DENSITY_DEFAULT_PARAMS!r} instead of passing []."
            )
        self.params: list[SpktWelchDensityParams] = params
        self._coeffs = torch.tensor([param["coeff"] for param in params], dtype=torch.long)

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Selected coefficients share detrending, windowing, and segment construction.
        batch_size, n_states, n_time = x.shape
        segment_length = min(n_time, 256)
        step = segment_length - segment_length // 2
        segments = x.reshape(batch_size * n_states, n_time).unfold(
            dimension=1, size=segment_length, step=step
        )
        segments = segments - segments.mean(dim=-1, keepdim=True)
        window = torch.hann_window(segment_length, periodic=True, dtype=x.dtype, device=x.device)
        segments = segments * window
        coeffs = self._coeffs.to(device=x.device)
        n_coeffs = segment_length // 2 + 1
        valid = (coeffs >= -n_coeffs) & (coeffs < n_coeffs)
        normalized = torch.where(coeffs < 0, coeffs + n_coeffs, coeffs)
        safe_coeffs = normalized.clamp(min=0, max=n_coeffs - 1)
        sample_indices = torch.arange(segment_length, dtype=x.dtype, device=x.device)
        angles = (
            2
            * math.pi
            * safe_coeffs.to(dtype=x.dtype).unsqueeze(-1)
            * sample_indices
            / segment_length
        )
        real = torch.matmul(segments, torch.cos(angles).transpose(0, 1))
        imag = torch.matmul(segments, -torch.sin(angles).transpose(0, 1))
        power = real.square() + imag.square()
        doubled = (safe_coeffs != 0) & (
            (segment_length % 2 != 0) | (safe_coeffs != segment_length // 2)
        )
        power = torch.where(doubled.view(1, 1, -1), 2 * power, power)
        power = power / window.square().sum()
        gathered = power.mean(dim=1).reshape(batch_size, n_states, -1)
        return torch.where(
            valid.view(1, 1, -1),
            gathered,
            torch.full_like(gathered, float("nan")),
        )

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format:
            ``spkt_welch_density__coeff_<coeff>``.
        """
        return [f"spkt_welch_density__coeff_{param['coeff']}" for param in self.params]


class CwtCoefficientsParams(TypedDict):
    """Parameters for one `CwtCoefficients` output column.

    Attributes:
        widths: Mexican-hat wavelet scales available to this parameter.
        coeff: Time coefficient index to return.
        w: Scale selected from ``widths``.
    """

    widths: tuple[int, ...]
    coeff: int
    w: int


_CWT_COEFFICIENTS_DEFAULT_PARAMS: list[CwtCoefficientsParams] = [
    {"widths": (width,), "coeff": coeff, "w": width}
    for width in (2, 5, 10, 20)
    for coeff in range(15)
]


@cache
def _integrated_mexican_hat_kernel(width: int) -> Tensor:
    precision = 12
    sample_count = 2**precision
    grid = torch.linspace(-8.0, 8.0, sample_count, dtype=torch.float64)
    step = grid[1] - grid[0]
    squared = grid.square()
    wavelet = 2.0 / (math.sqrt(3.0) * math.pi**0.25) * (1.0 - squared) * torch.exp(-squared / 2.0)
    integrated = torch.cumsum(wavelet, dim=0) * step
    sample_indices = (torch.arange(width * 16 + 1, dtype=torch.float64) / (width * step)).to(
        torch.long
    )
    return integrated[sample_indices[sample_indices < sample_count]]


class CwtCoefficients(Feature[CwtCoefficientsParams]):
    r"""Mexican-hat continuous wavelet-transform coefficients.

    For each requested scale \(w\), this reproduces ``pywt.cwt`` with the ``mexh``
    wavelet at precision 12: it convolves the series with the sampled integrated
    wavelet, differentiates the result, scales it by \(-\sqrt{w}\), and applies the
    same center crop. Distinct outputs that select the same scale share one
    convolution. A coefficient outside the series length produces NaN.

    Args:
        params: One dict per output, containing ``widths``, ``coeff``, and ``w``.
            ``w`` must occur in ``widths``. Defaults to tsfresh's comprehensive
            combinations for scales 2, 5, 10, and 20 and coefficients 0 through 14.
            An explicit empty list is rejected.

    References:
        Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
        autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.
    """

    def __init__(self, params: list[CwtCoefficientsParams] | None = None) -> None:
        if params is None:
            params = list(_CWT_COEFFICIENTS_DEFAULT_PARAMS)
        elif not params:
            raise ValueError(
                "CwtCoefficients requires at least one param; omit `params` to use the "
                f"default {_CWT_COEFFICIENTS_DEFAULT_PARAMS!r} instead of passing []."
            )
        invalid = [param for param in params if param["w"] not in param["widths"]]
        if invalid:
            raise ValueError(
                "CwtCoefficients requires each `w` to occur in its parameter's "
                f"`widths`; invalid params: {invalid!r}."
            )
        self.params: list[CwtCoefficientsParams] = params
        self._unique_widths = tuple(sorted({param["w"] for param in params}))
        width_indices = {width: index for index, width in enumerate(self._unique_widths)}
        self._width_indices = torch.tensor(
            [width_indices[param["w"]] for param in params], dtype=torch.long
        )
        self._coeffs = torch.tensor([param["coeff"] for param in params], dtype=torch.long)
        self._kernels = tuple(
            _integrated_mexican_hat_kernel(width) for width in self._unique_widths
        )

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Args:
            x: Tensor of shape (B, S, T).

        Returns:
            Tensor of shape (B, S, len(self.params)).
        """
        # Parameters share one convolution per distinct structural wavelet scale.
        batch_size, n_states, n_time = x.shape
        x_flat = x.reshape(batch_size * n_states, 1, n_time)
        coeffs = self._coeffs.to(device=x.device)
        valid = (coeffs >= -n_time) & (coeffs < n_time)
        normalized = torch.where(coeffs < 0, coeffs + n_time, coeffs)
        safe_coeffs = normalized.clamp(min=0, max=n_time - 1)
        transformed: list[Tensor] = []
        for width, stored_kernel in zip(self._unique_widths, self._kernels, strict=True):
            kernel = stored_kernel.to(dtype=x.dtype, device=x.device).view(1, 1, -1)
            kernel_size = kernel.shape[-1]
            padded = functional.pad(x_flat, (kernel_size - 1, kernel_size - 1))
            crop_left = (kernel_size - 2) // 2
            offsets = torch.arange(kernel_size, device=x.device).view(1, -1)
            starts = crop_left + safe_coeffs.view(-1, 1)
            first_windows = padded[:, :, starts + offsets]
            second_windows = padded[:, :, starts + offsets + 1]
            shaped_kernel = kernel.view(1, 1, 1, -1)
            first = (first_windows * shaped_kernel).sum(dim=-1)
            second = (second_windows * shaped_kernel).sum(dim=-1)
            transformed.append((-math.sqrt(width) * (second - first)).squeeze(1))
        stacked = torch.stack(transformed, dim=0)
        output_indices = torch.arange(coeffs.shape[0], device=x.device)
        selected = stacked[self._width_indices.to(device=x.device), :, output_indices]
        selected = selected.transpose(0, 1).reshape(batch_size, n_states, -1)
        return torch.where(
            valid.view(1, 1, -1),
            selected,
            torch.full_like(selected, float("nan")),
        )

    @property
    def names(self) -> list[str]:
        """
        Returns:
            Names in input order. Format:
            ``cwt_coefficients__coeff_<coeff>__w_<w>__widths_<widths>``.
        """
        return [
            f"cwt_coefficients__coeff_{param['coeff']}__w_{param['w']}__widths_{param['widths']}"
            for param in self.params
        ]


class SpectralFrequencyRatio(ParameterlessFeature):
    """Ratio of the second dominant frequency to the dominant frequency.

    The mean-centered power spectrum supplies the dominant non-DC bin. The second
    frequency is the strongest bin at least two indices away, which excludes the
    dominant peak's immediate neighbors. The result is zero for a spectrum without a
    nonzero peak or without an eligible second bin.

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
        centered = x - x.mean(dim=2, keepdim=True)
        fft_result = _rfft(centered, dim=2)
        components = torch.view_as_real(fft_result)
        power = components[..., 0].square() + components[..., 1].square()
        frequency_indices = torch.arange(power.shape[2], device=x.device).view(1, 1, -1)
        power_without_dc = torch.where(frequency_indices == 0, torch.zeros_like(power), power)
        has_signal = power_without_dc.amax(dim=2) >= 1e-10
        first_index = power_without_dc.argmax(dim=2)
        near_first = (frequency_indices - first_index.unsqueeze(-1)).abs() < 2
        masked_power = torch.where(
            near_first,
            torch.full_like(power_without_dc, float("-inf")),
            power_without_dc,
        )
        second_value, second_index = masked_power.max(dim=2)
        first_frequency = first_index.to(x.dtype)
        second_frequency = second_index.to(x.dtype)
        ratio = torch.where(
            first_frequency > 0,
            second_frequency / first_frequency,
            torch.zeros_like(first_frequency),
        )
        result = torch.where(
            has_signal & torch.isfinite(second_value),
            ratio,
            torch.zeros_like(ratio),
        )
        return result.unsqueeze(-1)

    @property
    def names(self) -> list[str]:
        """
        Returns:
            A single-element list containing ``spectral_frequency_ratio``.
        """
        return ["spectral_frequency_ratio"]
