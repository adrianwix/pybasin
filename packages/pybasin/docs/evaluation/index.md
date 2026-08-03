# Evaluation

pybasin uses several kinds of evidence, each with a different purpose.

## Numerical verification

[Numerical verification](numerical-verification.md) compares basin-stability results with bSTAB and checks known results from published systems. These checks run with the integration test suite and produce the artifacts used by the shared case studies.

## Reproducibility

[Reproducibility](reproducibility.md) documents the commands, inputs, environments, and stored outputs needed to repeat a result without making every documentation build rerun an expensive calculation.

## Performance

[Performance measurements](performance/index.md) compare solver backends and complete basin-stability workflows. They are measurements of a stated machine and configuration, not universal rankings.

## Design studies

[Design studies](design-studies/index.md) record focused experiments used to choose an implementation strategy. The Diffrax studies cover batch size, warm-up, precision, and event stopping.
