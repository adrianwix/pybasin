# pyright: basic
import torch
from torch import Tensor

# =============================================================================
# FREQUENCY DOMAIN FEATURES (4 features)
# =============================================================================


@torch.no_grad()
def fft_coefficient(x: Tensor, coeff: int = 0, attr: str = "abs") -> Tensor:
    """FFT coefficient attributes."""
    fft_result = torch.fft.rfft(x, dim=0)
    n_coeffs = fft_result.shape[0]
    if coeff >= n_coeffs:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)

    c = fft_result[coeff]
    if attr == "real":
        return c.real
    elif attr == "imag":
        return c.imag
    elif attr == "abs":
        return c.abs()
    elif attr == "angle":
        return torch.atan2(c.imag, c.real)
    else:
        return c.abs()


@torch.no_grad()
def fft_aggregated(x: Tensor, aggtype: str = "centroid") -> Tensor:
    """Aggregated FFT spectral statistics."""
    fft_result = torch.fft.rfft(x, dim=0)
    psd = fft_result.real**2 + fft_result.imag**2
    n_coeffs = psd.shape[0]
    freqs = torch.arange(n_coeffs, dtype=x.dtype, device=x.device).unsqueeze(-1).unsqueeze(-1)

    psd_sum = psd.sum(dim=0, keepdim=True)
    psd_norm = psd / (psd_sum + 1e-10)

    if aggtype == "centroid":
        return (freqs * psd_norm).sum(dim=0)
    elif aggtype == "variance":
        centroid = (freqs * psd_norm).sum(dim=0, keepdim=True)
        return ((freqs - centroid) ** 2 * psd_norm).sum(dim=0)
    elif aggtype == "skew":
        centroid = (freqs * psd_norm).sum(dim=0, keepdim=True)
        var = ((freqs - centroid) ** 2 * psd_norm).sum(dim=0, keepdim=True)
        return (((freqs - centroid) ** 3 * psd_norm).sum(dim=0)) / (var**1.5 + 1e-10).squeeze(0)
    elif aggtype == "kurtosis":
        centroid = (freqs * psd_norm).sum(dim=0, keepdim=True)
        var = ((freqs - centroid) ** 2 * psd_norm).sum(dim=0, keepdim=True)
        return (((freqs - centroid) ** 4 * psd_norm).sum(dim=0)) / (var**2 + 1e-10).squeeze(0)
    else:
        return (freqs * psd_norm).sum(dim=0)


@torch.no_grad()
def spkt_welch_density(x: Tensor, coeff: int = 0) -> Tensor:
    """Simplified Welch power spectral density at coefficient."""
    fft_result = torch.fft.rfft(x, dim=0)
    psd = (fft_result.real**2 + fft_result.imag**2) / x.shape[0]
    n_coeffs = psd.shape[0]
    if coeff >= n_coeffs:
        return torch.zeros(x.shape[1:], dtype=x.dtype, device=x.device)
    return psd[coeff]


@torch.no_grad()
def cwt_coefficients(
    x: Tensor, widths: tuple[int, ...] = (2,), coeff: int = 0, w: int = 2
) -> Tensor:
    """CWT coefficients using Ricker wavelet (vectorized).

    This matches tsfresh's cwt_coefficients interface:
    - widths: tuple of scale values to compute CWT for
    - coeff: coefficient index to extract from the convolution result
    - w: which width from the widths tuple to use for the result

    Note: This implementation uses a direct Ricker wavelet convolution which differs
    from tsfresh's pywt.cwt in normalization. Results have the same sign but different
    scaling. This is acceptable for feature extraction where relative patterns matter.

    Args:
        x: Input tensor of shape (N, B, S)
        widths: Tuple of wavelet width (scale) parameters
        coeff: Coefficient index to extract
        w: Which width from widths to use (must be in widths)

    Returns:
        Tensor of shape (B, S) with the CWT coefficient
    """
    n, batch_size, n_states = x.shape

    if w not in widths:
        return torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    width = w

    t = torch.arange(-width * 4, width * 4 + 1, dtype=x.dtype, device=x.device)
    sigma = float(width)
    wavelet = (
        (2 / (torch.sqrt(torch.tensor(3.0 * sigma, device=x.device)) * torch.pi**0.25))
        * (1 - (t / sigma) ** 2)
        * torch.exp(-(t**2) / (2 * sigma**2))
    )
    wavelet_len = len(wavelet)
    pad_size = wavelet_len // 2

    x_reshaped = x.permute(1, 2, 0).reshape(batch_size * n_states, 1, n)

    if n >= pad_size:
        padded = torch.nn.functional.pad(x_reshaped, (pad_size, pad_size), mode="reflect")
    else:
        padded = torch.nn.functional.pad(x_reshaped, (pad_size, pad_size), mode="replicate")

    kernel = wavelet.unsqueeze(0).unsqueeze(0)

    conv = torch.nn.functional.conv1d(padded, kernel, padding=0)

    if coeff < conv.shape[2]:
        result = conv[:, 0, coeff].reshape(batch_size, n_states)
    else:
        result = torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    return result


# =============================================================================
# BATCHED FREQUENCY DOMAIN FEATURES
# =============================================================================


@torch.no_grad()
def fft_coefficient_batched(x: Tensor, coeffs: list[int], attr: str = "abs") -> Tensor:
    """Compute FFT coefficients for multiple indices at once.

    Args:
        x: Input tensor of shape (N, B, S)
        coeffs: List of coefficient indices to extract
        attr: Attribute to extract ("real", "imag", "abs", "angle")

    Returns:
        Tensor of shape (len(coeffs), B, S)
    """
    fft_result = torch.fft.rfft(x, dim=0)
    n_coeffs = fft_result.shape[0]
    batch_size, n_states = x.shape[1], x.shape[2]

    results = []
    for coeff in coeffs:
        if coeff >= n_coeffs:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))
        else:
            c = fft_result[coeff]
            if attr == "real":
                results.append(c.real)
            elif attr == "imag":
                results.append(c.imag)
            elif attr == "abs":
                results.append(c.abs())
            elif attr == "angle":
                results.append(torch.atan2(c.imag, c.real))
            else:
                results.append(c.abs())

    return torch.stack(results, dim=0)


@torch.no_grad()
def fft_coefficient_all_attrs_batched(x: Tensor, coeffs: list[int]) -> Tensor:
    """Compute all FFT attributes for multiple coefficients at once.

    Args:
        x: Input tensor of shape (N, B, S)
        coeffs: List of coefficient indices

    Returns:
        Tensor of shape (len(coeffs) * 4, B, S) ordered as:
        [coeff0_real, coeff0_imag, coeff0_abs, coeff0_angle, coeff1_real, ...]
    """
    fft_result = torch.fft.rfft(x, dim=0)
    n_coeffs = fft_result.shape[0]
    batch_size, n_states = x.shape[1], x.shape[2]

    results = []
    for coeff in coeffs:
        if coeff >= n_coeffs:
            results.extend([
                torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device) for _ in range(4)
            ])
        else:
            c = fft_result[coeff]
            results.append(c.real)
            results.append(c.imag)
            results.append(c.abs())
            results.append(torch.atan2(c.imag, c.real))

    return torch.stack(results, dim=0)


@torch.no_grad()
def fft_aggregated_batched(x: Tensor, aggtypes: list[str]) -> Tensor:
    """Compute fft_aggregated for all aggregation types at once.

    Computes FFT and PSD once, then returns all requested aggregations.

    Args:
        x: Input tensor of shape (N, B, S)
        aggtypes: List of aggregation types ("centroid", "variance", "skew", "kurtosis")

    Returns:
        Tensor of shape (len(aggtypes), B, S)
    """
    if not aggtypes:
        raise ValueError("aggtypes cannot be empty")

    fft_result = torch.fft.rfft(x, dim=0)
    psd = fft_result.real**2 + fft_result.imag**2
    n_coeffs = psd.shape[0]
    freqs = torch.arange(n_coeffs, dtype=x.dtype, device=x.device).unsqueeze(-1).unsqueeze(-1)

    psd_sum = psd.sum(dim=0, keepdim=True)
    psd_norm = psd / (psd_sum + 1e-10)

    centroid = (freqs * psd_norm).sum(dim=0)
    variance = ((freqs - centroid.unsqueeze(0)) ** 2 * psd_norm).sum(dim=0)
    skew = ((freqs - centroid.unsqueeze(0)) ** 3 * psd_norm).sum(dim=0) / (variance**1.5 + 1e-10)
    kurtosis = ((freqs - centroid.unsqueeze(0)) ** 4 * psd_norm).sum(dim=0) / (variance**2 + 1e-10)

    agg_map = {
        "centroid": centroid,
        "variance": variance,
        "skew": skew,
        "kurtosis": kurtosis,
    }

    results = []
    for aggtype in aggtypes:
        if aggtype in agg_map:
            results.append(agg_map[aggtype])
        else:
            results.append(centroid)

    return torch.stack(results, dim=0)


@torch.no_grad()
def spkt_welch_density_batched(x: Tensor, coeffs: list[int]) -> Tensor:
    """Compute spkt_welch_density for multiple coefficient indices at once.

    Computes Welch PSD once, then extracts multiple coefficients.

    Args:
        x: Input tensor of shape (N, B, S)
        coeffs: List of coefficient indices to extract

    Returns:
        Tensor of shape (len(coeffs), B, S)
    """
    fft_result = torch.fft.rfft(x, dim=0)
    psd = (fft_result.real**2 + fft_result.imag**2) / x.shape[0]
    n_coeffs = psd.shape[0]
    batch_size, n_states = x.shape[1], x.shape[2]

    results = []
    for coeff in coeffs:
        if coeff >= n_coeffs:
            results.append(torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device))
        else:
            results.append(psd[coeff])

    return torch.stack(results, dim=0)


@torch.no_grad()
def cwt_coefficients_batched(x: Tensor, params: list[dict]) -> Tensor:
    """Compute CWT coefficients for all parameter combinations at once (GPU-optimized).

    This function computes CWT once per unique width value, then uses vectorized
    advanced indexing to extract all requested coefficients in a single operation.
    The extraction step is fully vectorized with no Python loops.

    Note: This implementation uses a direct Ricker wavelet convolution which differs
    from tsfresh's pywt.cwt in normalization. Results have the same sign but different
    scaling.

    Args:
        x: Input tensor of shape (N, B, S)
        params: List of parameter dicts, each with keys:
            - "widths": tuple of int (e.g., (2, 5, 10, 20))
            - "coeff": int (coefficient index to extract)
            - "w": int (which width from widths to use)

    Returns:
        Tensor of shape (len(params), B, S) with CWT coefficient for each param set

    Example:
        params = [
            {"widths": (2, 5, 10, 20), "coeff": c, "w": w}
            for c in range(15) for w in (2, 5, 10, 20)
        ]
        result = cwt_coefficients_batched(x, params)  # shape: (60, B, S)
    """
    n, batch_size, n_states = x.shape
    flat_size = batch_size * n_states

    if not params:
        return torch.zeros(0, batch_size, n_states, dtype=x.dtype, device=x.device)

    unique_widths_set: set[int] = set()
    for p in params:
        w = p.get("w", 2)
        widths = p.get("widths", (2,))
        if w in widths:
            unique_widths_set.add(w)

    if not unique_widths_set:
        return torch.zeros(len(params), batch_size, n_states, dtype=x.dtype, device=x.device)

    unique_widths = sorted(unique_widths_set)
    width_to_idx = {w: i for i, w in enumerate(unique_widths)}
    num_widths = len(unique_widths)

    x_flat = x.permute(1, 2, 0).reshape(flat_size, 1, n)

    cwt_results = []
    for width in unique_widths:
        t = torch.arange(-width * 4, width * 4 + 1, dtype=x.dtype, device=x.device)
        sigma = float(width)
        wavelet = (
            (2 / (torch.sqrt(torch.tensor(3.0 * sigma, device=x.device)) * torch.pi**0.25))
            * (1 - (t / sigma) ** 2)
            * torch.exp(-(t**2) / (2 * sigma**2))
        )
        kernel = wavelet.unsqueeze(0).unsqueeze(0)
        pad_size = len(t) // 2

        if n >= pad_size:
            x_padded = torch.nn.functional.pad(x_flat, (pad_size, pad_size), mode="reflect")
        else:
            x_padded = torch.nn.functional.pad(x_flat, (pad_size, pad_size), mode="replicate")

        conv = torch.nn.functional.conv1d(x_padded, kernel, padding=0)
        cwt_results.append(conv.squeeze(1))

    max_len = max(c.shape[1] for c in cwt_results)
    cwt_stacked = torch.zeros(num_widths, flat_size, max_len, dtype=x.dtype, device=x.device)
    for i, c in enumerate(cwt_results):
        cwt_stacked[i, :, : c.shape[1]] = c

    param_w_indices = []
    param_coeff_indices = []
    valid_mask = []

    for p in params:
        w = p.get("w", 2)
        coeff = p.get("coeff", 0)
        widths = p.get("widths", (2,))

        if w in widths and w in width_to_idx and coeff < max_len:
            param_w_indices.append(width_to_idx[w])
            param_coeff_indices.append(coeff)
            valid_mask.append(True)
        else:
            param_w_indices.append(0)
            param_coeff_indices.append(0)
            valid_mask.append(False)

    w_idx = torch.tensor(param_w_indices, dtype=torch.long, device=x.device)
    coeff_idx = torch.tensor(param_coeff_indices, dtype=torch.long, device=x.device)
    valid = torch.tensor(valid_mask, dtype=torch.bool, device=x.device)

    results = cwt_stacked[w_idx, :, coeff_idx]

    results = results.reshape(len(params), batch_size, n_states)

    invalid_mask = ~valid
    if invalid_mask.any():
        results[invalid_mask] = 0.0

    return results


@torch.no_grad()
def spectral_frequency_ratio(x: Tensor) -> Tensor:
    """Compute the ratio of 2nd to 1st dominant frequency.

    This feature is critical for distinguishing period-doubling bifurcations:
    - Period-1 limit cycle: ratio ≈ 2.0 (2nd peak is harmonic at 2f)
    - Period-2 limit cycle: ratio ≈ 0.5 (2nd peak is subharmonic at f/2)
    - Period-3 limit cycle: ratio ≈ 0.33 (2nd peak is subharmonic at f/3)

    The function finds the two highest peaks in the power spectrum and returns
    the ratio of the 2nd dominant frequency to the 1st dominant frequency.

    Args:
        x: Input tensor of shape (N, B, S) where N is timesteps, B is batch, S is states.

    Returns:
        Tensor of shape (B, S) with the frequency ratio. Returns 0 if only one peak found.
    """
    n, batch_size, n_states = x.shape

    x_centered = x - x.mean(dim=0, keepdim=True)

    fft_result = torch.fft.rfft(x_centered, dim=0)
    power = fft_result.real**2 + fft_result.imag**2

    power[0] = 0

    result = torch.zeros(batch_size, n_states, dtype=x.dtype, device=x.device)

    for b in range(batch_size):
        for s in range(n_states):
            psd = power[:, b, s]

            if psd.max() < 1e-10:
                continue

            sorted_indices = torch.argsort(psd, descending=True)

            first_idx = sorted_indices[0]

            second_idx = None
            for idx in sorted_indices[1:]:
                if abs(idx - first_idx) >= 2:
                    second_idx = idx
                    break

            if second_idx is not None:
                freq_1st = float(first_idx)
                freq_2nd = float(second_idx)
                if freq_1st > 0:
                    result[b, s] = freq_2nd / freq_1st

    return result
