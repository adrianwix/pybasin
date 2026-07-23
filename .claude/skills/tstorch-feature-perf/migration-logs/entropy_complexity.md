# entropy_complexity domain migration log

Relative to `origin/main`, this branch supplies the entropy and complexity calculator
domain as seven traceable feature classes. They share one contract: typed constructor
parameters, complete output names, BST input, BSF output, and clean tracing through both
`torch.compile(fullgraph=True)` and `torch.export`.

## permutation_entropy

`PermutationEntropy` accepts `tau` and `dimension`. Padded index and dimension tensors
broadcast every requested combination through one kernel despite their different window
counts and ordinal-pattern lengths. Sorting the pattern ids and finding each run's
boundaries computes empirical frequencies without allocating a `dimension**dimension`
histogram.

Window elements are consecutive and window starts advance by `tau`, matching tsfresh's
subchunk convention. This distinction matters when `tau > 1`; spacing elements inside
the window instead would compute a different time-delay embedding. Too-short series
return NaN, and base-`dimension` positional encoding gives every rank permutation a
unique integer id.

## binned_entropy

`BinnedEntropy` shares the series minimum, maximum, constant-series mask, and NaN mask
across all requested `max_bins` values. Each bin count still needs its own edge and
count tensor because those shapes differ.

Explicit linspace-equivalent edges and `torch.searchsorted` reproduce NumPy histogram
boundary assignment. This avoids epsilon-adjusted arithmetic near an interior boundary,
where a mathematically equivalent floor formula can round a sample into the wrong bin.
Constant series return zero; a series containing NaN returns NaN.

## fourier_entropy

`FourierEntropy` computes one Welch power spectrum and reuses it for every requested
bin count before delegating to the shared binned-entropy kernel. Welch segments use
constant detrending, a periodic Hann window, half overlap, one-sided power doubling,
and the same 256-sample segment cap as tsfresh.

For this bounded transform size, direct real DFT matrix multiplies avoid the fixed eager
dispatch cost of `torch.fft.rfft`. The replacement preserves the canonical entropy bins
and makes the small CPU direct-call path competitive without a NumPy branch. Absolute
PSD scaling is omitted because multiplying a spectrum by a positive constant does not
change equal-width bin occupancy.

## cid_ce

`CidCe` computes the first-difference norm and population standard deviation once.
Requested `normalize` values select either the raw norm or normalized result with
`torch.where`, so both Boolean variants share the expensive work.

An exact zero-standard-deviation mask handles constant series. It also preserves
legitimate tiny nonzero standard deviations, which an additive denominator epsilon
would distort.

## approximate_entropy

`ApproximateEntropy` broadcasts every requested `(m, r)` pair. A running maximum builds
the Chebyshev distances by window offset; Boolean selection captures the appropriate
distance pass for each requested `m`, and every `r` is applied as a broadcast tolerance.
This avoids independent calls or Python iteration over parameters while keeping only one
floating-point distance matrix live.

It returns zero when the series cannot form both required window lengths and rejects
negative tolerances during construction.

## sample_entropy

`SampleEntropy` is parameter-free because tsfresh fixes both `m=2` and the tolerance
multiplier. It uses the same batched Chebyshev-distance structure as approximate entropy
for length-2 and length-3 windows, excludes self-matches, and returns NaN when the input
contains NaN.

Both entropy classes vectorize the quadratic comparison across all batch and state rows.
That is the main computational advantage over tsfresh's one-series-at-a-time calls.

## lempel_ziv_complexity

`LempelZivComplexity` broadcasts equal-width symbolization across every requested bin
count, then flattens the batch, state, and parameter axes into independent symbolic
rows. One registered operator parses those rows. The CPU dispatch uses the exact
phrase-dictionary procedure, while CUDA assigns independent rows to Triton programs.

The phrase boundary within a row remains sequential. Moving that state machine behind a
functional PyTorch operator gives Dynamo and export one stable operation with a fixed
output shape; a FakeTensor implementation supplies its metadata. The class therefore
uses the same direct-call and extractor paths as the rest of the domain.
