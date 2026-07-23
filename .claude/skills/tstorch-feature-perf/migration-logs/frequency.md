# Frequency domain migration log

The frequency calculator now exposes five feature classes. Direct callers and
`TorchFeatureExtractor` construct the same class, use the same parameter list, and read
the same complete names. The old public functions, factory builders, parameter mappers,
and `FREQUENCY_FEATURE_FACTORIES` registry are gone.

All four parametrized classes reject an explicit empty parameter list and default to the
corresponding comprehensive tstorch settings when parameters are omitted. Their
constructors translate strings and indices into closed-over tensors once. Runtime kernels
therefore receive BST tensors and numeric parameter data only.

## FftCoefficient

`FftCoefficient` computes one real FFT and gathers every requested coefficient before
deriving attributes. A request containing one attribute type follows a specialized branch
for `real`, `imag`, `abs`, or degree-valued `angle`, so it does not pay for unused
attributes. Mixed requests stack the four variants after coefficient selection.

Negative coefficients follow NumPy indexing when they remain in range. Indices outside
the available real-FFT bins return NaN. Angle accuracy uses a wider absolute tolerance
because phase becomes unstable when both real and imaginary parts lie near the float32
noise floor; the underlying complex coefficient remains accurate.

## FftAggregated

`FftAggregated` treats FFT magnitudes as weights over bin indices. Centroid, variance,
skew, and kurtosis share one spectrum and one set of raw moments. Skew and kurtosis return
NaN below tsfresh's variance threshold.

The kurtosis numerator deliberately follows tsfresh's raw-moment expression, including
its final `-3 * centroid` term. Replacing that term with the textbook central-moment
formula would be mathematically conventional but numerically incompatible with the
reference calculator. A zero spectrum also propagates NaN through the weight
normalization instead of being hidden by an epsilon.

## SpktWelchDensity

`SpktWelchDensity` matches SciPy Welch defaults used by tsfresh: segments are at most 256
samples long, overlap by half a segment, remove their means, use a periodic Hann window,
apply one-sided density scaling, and average across segments.

The class shares segment construction and windowing across parameters, but projects only
the requested frequency bins. This avoids calculating a complete spectrum when callers
ask for a few coefficients. Out-of-range coefficients return NaN, while valid negative
indices retain NumPy-style indexing. CPU and GPU projections use a slightly wider
cross-device tolerance because their long dot products reduce in different orders.

## CwtCoefficients

`CwtCoefficients` replaces the former approximate Ricker convolution with the algorithm
used by `pywt.cwt(..., wavelet="mexh", precision=12)`. It samples and integrates the
Mexican-hat wavelet over the same support, scales the sampled kernel for each requested
width, evaluates the first difference of the full convolution, and applies PyWavelets'
center alignment.

Only requested time positions are evaluated. Parameters sharing a scale reuse one cached
integrated-wavelet kernel, and the structural scale loop is fixed by the constructor so
Dynamo unrolls it during tracing. A missing `w` in its parameter's `widths` tuple is
rejected at construction, matching the reference's invalid-configuration behavior.
Coefficient indices outside the series return NaN.

This correction changes CWT from a shape-only approximation to a value-tested tsfresh
implementation. The accuracy suite covers several widths tuples, selected scales,
coefficient positions, and an out-of-range position across every canonical trajectory.

## SpectralFrequencyRatio

`SpectralFrequencyRatio` remains a tstorch-only, parameter-free feature. It mean-centers
the series, removes the DC bin from consideration, finds the dominant power-spectrum bin,
and then selects the strongest bin at least two indices away. The result is the second
index divided by the first. Constant series and spectra without an eligible second peak
return zero.

The implementation avoids in-place spectrum mutation. Its masking and selection remain
traceable under both fullgraph compilation and export.

## Integration and verification

All five classes are registered directly in `SUPPORTED_FEATURES`. The frequency
`FeatureSpec` entries record that tsfresh's combiner calculators receive parameter lists
and share their FFT, Welch, or CWT work. The complete benchmark module covers cold and
compiled direct calls, batched direct and extractor calls, compile cost, steady-state
calls, and argument-count sweeps where at least ten distinct valid parameters exist.
`FftAggregated` is absent from the 1/5/10 sweep because it has only four distinct valid
parameter values; repeating deduplicated parameters would misstate the argument count.

The unit suite checks numerical parity, output names, empty-parameter rejection,
fullgraph compilation, export, custom spectral-ratio behavior, and CPU/GPU agreement.
The shared accuracy helper now materializes iterator-valued tsfresh combiner rows before
indexing them, which supports the `zip` results returned by this tsfresh version.

The API page documents every class and parameter schema with Google-style docstrings.
Current precision, compile, export, and performance results live only in
`packages/tstorch/reports/report.md`.
