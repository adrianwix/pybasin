# pybasin

pybasin estimates basin stability for multistable dynamical systems. It combines
batched ODE solvers, automated feature extraction, attractor classification, and
parameter studies in a scikit-learn-compatible workflow.

Install the published package with:

```bash
pip install pybasin
```

Optional extras add JAX, interactive plotting, alternative feature extraction,
and TorchODE support. The complete user guide and API reference are available in
the [pybasin documentation](https://adrianwix.github.io/pybasin/pybasin/).

Development takes place in the
[pybasin monorepo](https://github.com/adrianwix/pybasin), where this distribution
lives under `packages/pybasin`.
