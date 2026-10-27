# Shared feature tests

The feature unit tests default to tstorch's native PyTorch calculators. A pytest provider can
run the numerical, naming, batching, and backend-specific compiler assertions through another
implementation while keeping the tsfresh reference cases and tolerances in tstorch.

Providers implement the structural `FeatureTestProvider` protocol declared in
`packages/tstorch/tests/utils/feature_test_provider.py` and supply an object with `name`, `compile_mode`,
`export_mode`, `supports(feature_name)`, and `create(feature_name, params, native_factory)`.
`create` returns a callable with `names` that accepts a Torch tensor and returns a Torch tensor;
the provider owns any input and output conversion. `native_factory` is the explicit native
constructor used by the default provider. An external provider must create its own feature and
must return `False` from `supports` when it has no implementation. Unsupported features become
pytest skips with the provider and feature named.

`compile_mode` and `export_mode` identify the exact check performed by the provider. The shared
Torch assertions record only `torch.compile(fullgraph=True)` and `torch.export` outcomes. A
provider with no corresponding check records a skip. Other compiler modes can be added by a
provider-specific assertion without labeling that result as a Torch check.

The UFeatures repository includes an opt-in proof provider at
`packages/ufeatures/tests/tstorch_provider.py`. Its supported Python version is 3.13 or newer;
run it from a pybasin environment using Python 3.13, with both workspaces' source packages
available on `PYTHONPATH`:

```bash
PYTHONPATH="/path/to/solvers/packages/ufeatures/src:/path/to/solvers/packages/ufeatures/tests:/path/to/solvers/packages/utensor/src" \
  uv run pytest packages/tstorch/tests/unit/test_autocorrelation.py \
  --feature-provider tstorch_provider:UFeaturesTorchProvider \
  -m 'shared_provider or native_only' -ra
```

The selected suite runs the same four autocorrelation calculators through UFeatures on Torch.
Its numerical assertions use the same canonical trajectories, tsfresh calls, and tolerances as
the native run. It also runs the same Torch fullgraph compilation and dynamic batch export
checks. Empty/default constructor behavior remains native-only and is reported as skipped. The
marker selection and pytest summary make that omission visible.

The Python 3.13 proof run on the pinned canonical cache produced 9 passed, 8 failed, and 8
native-only skips. The failures include the same three tsfresh discrepancies present in the
native profile. UFeatures also fails dynamic-batch export for autocorrelation and aggregate
autocorrelation because tracing specializes the batch size to 2; the tests retain the required
dynamic batch contract. Periodicity's zero-length FFT failure appears in both implementations.
These results show the provider can expose backend-specific failures while keeping the test
selection and comparison contract shared.

External-provider runs reject `--precision-json`, `--compile-json`, and `--export-json`; those
files have native report schemas and are not suitable for storing external-provider results.
