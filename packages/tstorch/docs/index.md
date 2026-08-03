# TSTorch

tstorch provides PyTorch-native time-series feature calculators. Features accept tensors directly, preserve batching, and are tested as building blocks for compiled and exported extraction pipelines.

The project began with compatibility work for tsfresh features, but its scope is broader: an array and dataframe agnostic interface, efficient CPU and GPU execution, distributed workloads with Dask and RAPIDS, and features drawn from current research.

## Evidence

The evaluation pages answer separate questions:

- [numerical verification](evaluation/numerical-verification.md) compares supported calculators with tsfresh;
- [compile and export](evaluation/compile-export.md) records full-graph compilation and dynamic-batch export;
- [performance](evaluation/performance.md) compares cold and warmed execution on single and batched trajectories;
- [domain details](evaluation/index.md#domain-details) keep the complete evidence for each calculator in one place.

The tables are rendered from `packages/tstorch/reports/store.json`. Updating that store updates every view without copying results into Markdown.
