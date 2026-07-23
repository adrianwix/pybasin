# tstorch Experiments

These experiments probe deployment and code-generation paths without committing the public API
to any one runtime.

The user-facing target should stay stable:

```python
extractor = TorchFeatureExtractor(features={"mean": None, "maximum": None, "minimum": None})
features = extractor.extract_features(y_bst)
```

Export backends should hang off that same configured feature plan. A caller should choose a
deployment backend by configuration or an explicit export helper, not by rewriting feature
definitions.

## Current Threads

- `run_basic_stats_report.py`: regenerates the AOTI and standalone basic-statistics experiments
  and writes one tracked comparison report at `basic_stats_report.md`.
- `aoti_basic_stats/`: PyTorch-native `torch.export` + AOTInductor packaging, including C++
  loading and generated-kernel inspection.
- `executorch_basic_stats/`: PyTorch Edge / ExecuTorch experiment for users who want a
  PyTorch-managed edge runtime.
- `standalone_codegen_basic_stats/`: backend-agnostic generated C/C++ feature kernel experiment
  for users who need a small runtime-free artifact.

## API Direction

Keep feature classes as the semantic source of truth. Backends should lower an extractor plan:

1. Resolve the same configured feature plan used by `TorchFeatureExtractor`.
2. Preserve the same input contract: `(B, S, T)` contiguous in time.
3. Preserve the same feature order and names.
4. Lower to the selected backend: PyTorch AOTI, ExecuTorch, generated C/C++, Zig, or another
   future backend.

This keeps tstorch closer to a Keras-like model: the API describes what to compute, and a backend
decides how to compile or execute it.
