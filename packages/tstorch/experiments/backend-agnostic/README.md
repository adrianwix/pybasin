# Backend-Agnostic Feature Experiment

This folder is an isolated tstorch experiment. It keeps the current feature class API:

```python
feature = PartialAutocorrelation([{"lag": 1}, {"lag": 5}])
values = feature(x_bst)
names = feature.names
```

The experiment adds one Keras-like idea: `set_backend(...)` configures a global backend.
NumPy arrays use the NumPy backend, PyTorch tensors use the PyTorch backend, and JAX arrays
use the JAX backend. The visible feature contract stays identical for every backend:

- Input shape: `(B, S, T)`
- Output shape: `(B, S, len(params))`
- Names: complete `partial_autocorrelation__lag_<lag>` names, independent of backend
- Params: one list of typed parameter dicts, independent of backend

The feature kernel is `partial_autocorrelation(...)`, which is written once against an
`Ops` protocol. Backend-specific code is limited to primitive operations such as `zeros`,
`stack`, `flip`, and indexed updates.

The code is split by responsibility:

- `backend_config.py`: global backend state, `backend()`, and `set_backend(...)`
- `ops.py`: backend-specific primitive operation namespaces
- `partial_autocorrelation.py`: feature class and backend-agnostic PACF kernel
- `backend_agnostic_features.py`: public facade for the experiment

The tests encode the current success criteria:

```bash
uv run pytest packages/tstorch/experiments/backend-agnostic/test_backend_agnostic_features.py -q
```

They check PyTorch `torch.compile(fullgraph=True)`, `torch.export`, JAX `jit`, exact
NumPy/PyTorch agreement at tight tolerance, and JAX agreement at `1e-2` tolerance for the
recursive PACF kernel.
