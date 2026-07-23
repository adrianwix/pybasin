# pattern domain migration log

The pattern calculator module now exposes five feature classes through the same direct-call
and `TorchFeatureExtractor` API. Each class returns `(B, S, F)`, owns its output names, rejects
invalid constructor arguments, and traces without graph breaks under both `torch.compile` and
`torch.export`. The deprecated factory registry, builder closures, public calculator
functions, and NumPy dispatch path are gone.

## LongestStrikeAboveMean and LongestStrikeBelowMean

Both streak features use `_longest_consecutive_run`. The helper records the latest false
position with `cummax`; subtracting that position from the current index gives the active run
length at every sample. A final maximum yields the longest run without a Python loop over time.

The comparison remains strict, so a value equal to the series mean ends a run. Constant and
empty series return zero. The two public classes differ only in the comparison that builds
their masks, while their constructor, naming, output shape, and tracing contracts remain
identical.

## NumberCrossingM

`NumberCrossingM` accepts every requested level in one typed parameter list. Its kernel
broadcasts those levels against the input, compares adjacent boolean masks, and reduces the
time axis once. Equality belongs to the lower side because the mask uses `x > m`, matching
tsfresh at the boundary.

This is a broadcast migration. tsfresh's simple-calculator dispatcher invokes
`number_crossing_m` once per parameter, but tstorch evaluates the full level axis in one tensor
operation. The old small-CPU NumPy branch was removed because a NumPy round trip cannot satisfy
the shared `torch.export` contract.

## NumberPeaks

`NumberPeaks` shares the expensive work across supports. It pads and unfolds once at the
largest requested `n`, compares the centre with neighbours ordered from nearest to farthest,
and applies cumulative conjunctions along that distance axis. Since a peak that beats all
neighbours through distance `d` also beats every smaller support, each requested output is a
gather from the shared result.

Position masks exclude the first and last `n` samples, preserving tsfresh's edge semantics.
Plateaus are not peaks because every comparison is strict. Non-positive supports are rejected
at construction: tsfresh's raw `n=0` implementation returns the Python object `None`, which is
not a meaningful tensor feature contract.

## NumberCwtPeaks

`NumberCwtPeaks` builds normalized Ricker wavelets at every integer width up to the largest
requested `n` and applies them as one `conv1d` filter bank. That transform is shared across
the batch, state, and width axes. The remaining ridge-linking pass is tensorized over every
series, but each distinct `n` needs its own pass over the corresponding width prefix because
adding a scale changes the ridge topology.

Ridge storage uses the strict-maxima bound instead of reserving one slot per time sample and
width. Interior strict maxima cannot be adjacent, so at most half of a coefficient row can
start new ridges. This cuts the dominant state tensors roughly in half without changing which
ridges can be represented.

The implementation follows SciPy's local-maxima, ridge-gap, minimum-length, and local-noise
rules. It remains sensitive to convolution roundoff because ridge candidates use strict
comparisons. PyTorch and SciPy can therefore differ by an integer peak count when adjacent CWT
coefficients differ only at the last floating-point bit. The canonical numerical test records
the actual per-parameter differences in the generated report and uses an absolute tolerance
of three counts; a fixed random and sinusoidal sweep found exact agreement in 792 outputs.

## Tests, benchmarks, and documentation

The domain unit suite checks tsfresh parity across canonical trajectories, distinct-name
generation, invalid parameters, boundary behavior, CPU/GPU agreement, full-graph compilation,
and export. `NumberCrossingM` and `NumberPeaks` each cover eight distinct parameter values.

The benchmark module supplies the complete direct-call and proxy-call inventory, including
compiled and uncompiled paths plus argument counts of 1, 5, and 10. Raw tsfresh source confirms
that both parametrized calculators use its simple dispatcher, which loops once per argument.
Current measurements and precision records live only in `packages/tstorch/reports/report.md`.
The API page documents each owning class and places its typed parameter schema directly below
the class.
