# TorrStack

TorrStack is a scientific computing ecosystem for studying dynamical systems and extracting useful structure from time series. Its public packages share infrastructure and case studies, but each package keeps its own API reference, verification record, and performance evidence.

## Packages

### [PyBasin](pybasin/)

Estimate basin stability, run parameter studies, integrate large trajectory ensembles, and classify attractors. The package supports CPU and accelerator workflows through several ODE solver backends.

[Open the PyBasin documentation](pybasin/){ .md-button .md-button--primary }

### [TSTorch](tstorch/)

Compute time-series features with PyTorch tensors. The project is designed for batched execution, compilation, export, and eventual use across array and dataframe front ends.

[Open the TSTorch documentation](tstorch/){ .md-button .md-button--primary }

## Shared case studies

The [case studies](case-studies/overview.md) exercise the packages together on physical and canonical dynamical systems. Some reproduce published reference results; others document systems for which no previous ground truth is available.

Package-specific evaluation pages separate three different questions:

- numerical verification: whether results agree with a reference or a known invariant;
- reproducibility: how stored results can be regenerated;
- performance and design studies: what an implementation choice costs and when it is useful.
