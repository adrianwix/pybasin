# Reocurrance migration

The reoccurrence domain now exposes five parameter-free feature classes:
`PercentageOfReoccurringDatapointsToAllDatapoints`,
`PercentageOfReoccurringValuesToAllValues`, `SumOfReoccurringDataPoints`,
`SumOfReoccurringValues`, and `RatioValueNumberToTimeSeriesLength`. Direct callers and
`TorchFeatureExtractor` construct these same classes, and each call returns `(B, S, 1)`.
The deprecated factory registry and public function facades are gone.

All five classes use the same export-safe grouping strategy. Each `(B, S, T)` tensor is viewed
as `(B*S, T)`, sorted along contiguous time, and adjacent equal values identify duplicate
groups. Neighbor masks distinguish every member of a repeated group from its first member, so
the two sum features can respectively count every occurrence or one representative per group.
The ratio features derive their numerators and denominators from the same group boundaries.

NaN behavior cannot be uniform across the five features because tsfresh itself uses two
different grouping APIs. `PercentageOfReoccurringDatapointsToAllDatapoints` follows pandas
`Series.value_counts`, so NaNs never count as repeated but remain in the total datapoint count.
`PercentageOfReoccurringValuesToAllValues` and
`RatioValueNumberToTimeSeriesLength` follow NumPy `unique`, so all NaNs form one distinct group.
Both sum features return NaN whenever the input contains a NaN because tsfresh multiplies the
NaN unique value by its group count before summing. Empty-series results also match tsfresh:
the percentage and ratio features return NaN, while both sum features return zero.

The shared implementation deliberately stays in PyTorch. A NumPy sort is faster for the tiny
CPU tensors in the direct-call benchmark, but a NumPy round trip cannot pass
`torch.export.export`; the repository-wide export contract therefore rules it out. The
generated report shows this fixed `torch.sort` dispatch cost in some single-trajectory CPU
rows. At the production-scale 10,000-trajectory row, every feature's tstorch CPU implementation
is substantially faster than tsfresh. The same kernels pass `torch.compile(fullgraph=True)`,
`torch.export`, and a composed-extractor check with one graph and no graph breaks.

The migration adds numerical, name, compile, export, NaN/multiplicity, empty-series, and
CPU/GPU parity tests; the complete six-row benchmark matrix applicable to this
parameter-free domain; direct extractor registration; and a Google-style API page for all five
classes.
