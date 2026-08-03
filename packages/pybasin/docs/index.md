# PyBasin

pybasin estimates basin stability for dynamical systems. It combines trajectory sampling, ODE integration, feature extraction, attractor classification, parameter studies, and plotting behind a composable Python API.

Use the [quick start](getting-started/quickstart.md) for a first calculation or browse the [user guides](user-guide/basin-stability-estimator.md) for the individual parts of a study.

## Evaluation

The evaluation section keeps distinct kinds of evidence separate:

- [numerical verification](evaluation/numerical-verification.md) records agreement with bSTAB and other reference results;
- [performance measurements](evaluation/performance/index.md) compare complete workflows and solver backends;
- [design studies](evaluation/design-studies/index.md) preserve experiments that informed implementation choices, including Diffrax batching and event stopping.

The ecosystem-level [case studies](https://adrianwix.github.io/pybasin/case-studies/overview/) show these components working together on complete systems.
