# tstorch

tstorch provides PyTorch-native time-series feature calculators for batched CPU
and GPU workloads. Its calculators are tested for numerical agreement,
`torch.compile(fullgraph=True)`, dynamic-batch `torch.export`, and performance
against tsfresh where a reference implementation exists.

The package is being developed toward array and dataframe agnostic front ends,
distributed execution with Dask and RAPIDS, and features from recent
publications.

See the [tstorch documentation](https://adrianwix.github.io/pybasin/tstorch/)
for API details and the current evaluation record.
