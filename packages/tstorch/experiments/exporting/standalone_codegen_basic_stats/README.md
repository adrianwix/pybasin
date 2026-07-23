# Standalone Codegen Basic Statistics Experiment

Goal: generate a small C/C++ feature-extraction artifact with no PyTorch runtime dependency.

This is the path for factory devices, embedded Linux targets, and very constrained systems where
shipping LibTorch or an ExecuTorch runtime is too heavy.

## Target Question

Can the same feature selection API:

```python
TorchFeatureExtractor(
    features={"mean": None, "maximum": None, "minimum": None},
    normalize=False,
)
```

be lowered to standalone code such as:

```c
void tstorch_basic_stats_f32(
    const float* x,
    float* out,
    int batch,
    int states,
    int timesteps
);
```

without changing how the user asks for features?

## Implemented Baseline

Generate a combined one-pass kernel for:

- mean
- maximum
- minimum

The kernel computes all three features for each `(batch, state)` series in one pass over time. It
exposes a plain C ABI first:

```c
void tstorch_basic_stats_f32(
    const float* x,
    float* out,
    int batch,
    int states,
    int timesteps
);
```

Run from the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python packages/tstorch/experiments/exporting/standalone_codegen_basic_stats/run_standalone_codegen.py
```

This writes generated source to `generated/`, compiles a dependency-light benchmark executable
under `artifacts/`, and writes `artifacts/report.md`.

Current report:

- `artifacts/report.md`

## CWT Peak Feasibility

`run_cwt_peaks_experiment.py` tests whether SciPy's `find_peaks_cwt` algorithm can cross the
same static boundary without changing its peak counts. The boundary fixes the time dimension,
requested widths, Ricker kernels, ridge capacity, and percentile-window indices. Signal values
remain inside the tensor graph.

SciPy stores ridge lines in data-dependent Python lists. The tensor kernel instead reserves one
slot for every possible `(width, time)` maximum. At each width it builds a position-to-ridge map
with `scatter_reduce`, then finds the nearest live ridge through left and right cumulative scans.
This keeps the association step linear in the number of ridge slots. All requested CWT widths are
evaluated together, so the comprehensive `n=1` and `n=5` outputs share one wavelet transform.

Two CWT implementations feed the same ridge kernel. `conv1d` applies the zero-padded Ricker bank
in one convolution call. `direct` exposes a static loop over the 50 kernel taps and accumulates
every width through batched tensor arithmetic. The loop is specialized and unrolled; it never
iterates over batches, states, or time positions in Python.

Run the experiment from the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache TORCHINDUCTOR_COMPILE_THREADS=1 \
  uv run python \
  packages/tstorch/experiments/exporting/standalone_codegen_basic_stats/run_cwt_peaks_experiment.py
```

The runner checks exact tsfresh parity on seeded random inputs, full-graph Inductor compilation,
`torch.export`, dynamic batch export, and AOTInductor packaging for both implementations. A
separate paired parity sweep covered 792 random and sinusoidal outputs across time dimensions from
21 through 1000 without a count mismatch.

On the July 23, 2026 development environment, the single-threaded `(16, 2, 1000)` comparison
produced these median times:

| CWT implementation | Eager [ms] | Inductor [ms] | AOTI [ms] |
| --- | ---: | ---: | ---: |
| `conv1d` | 275.46 | 195.55 | 208.34 |
| `direct` | 219.64 | 167.72 | 191.54 |

At batch 16, exposing the tap accumulation improved the compiled path by 14 percent and AOTI by 8
percent. Batch 1 slightly favored `conv1d`: 10.69 ms through AOTI versus 11.07 ms for `direct`.
The direct graph contained 990 export nodes instead of 692, added roughly 43 KB to the package,
and took 21.08 seconds to package instead of 18.75 seconds.

These measurements use `torch._inductor.config.cpp.simdlen = 1`. PyTorch 2.13 with GCC 15
currently emits invalid mixed mask-vector C++ for this graph at the default SIMD length; scalar
code generation avoids that backend defect and still beats SciPy. Removing the workaround is a
compiler-backend task, not a change to the feature algorithm.

Bitwise equivalence of the CWT coefficients is still unresolved. SciPy's direct convolution and
PyTorch arithmetic can differ at roundoff scale; strict relative-maximum tests can expose that
difference on degenerate data. For example, SciPy reports one `n=5` peak for a length-31 all-ones
series while both tensor implementations report zero. Forward and reverse tap traversal,
sequential accumulation, tensor reduction, and SciPy-generated Ricker coefficients all reproduced
the tensor count rather than SciPy's count. An all-zero series reports zero in every path. The
static direct loop is therefore a useful performance option, but it does not solve literal
tsfresh parity.

The `.pt2` result is native code, but it links against LibTorch. It is not the runtime-free C ABI
produced by `run_standalone_codegen.py`. Reaching that smaller target requires a C/C++ lowering for
the fixed ridge-state operations, or a compiler capable of emitting those operations without the
PyTorch runtime. The experiment establishes a compilable semantic source and a faster native path;
it does not claim that the runtime-free lowering already exists.

## Design Direction

Separate feature semantics from backend lowering:

- Feature classes remain the semantic implementation and the test oracle.
- A backend-specific lowering layer maps supported feature combinations to generated code.
- Unsupported feature/backend combinations fail explicitly with a useful message.

For the first prototype, hard-code a tiny lowering rule for `mean + maximum + minimum`. Later,
generalize to a registry such as:

```python
CodegenBackend("c99").supports(feature_plan)
CodegenBackend("c99").emit(feature_plan, output_dir)
```

## Artifact Goals

Measure:

- generated source size
- compiled object/shared-library size
- runtime dependency list from `ldd`
- latency for 1 trajectory and 10000 trajectories

For ESP32-class targets, the ideal artifact is source that can be compiled by the platform
toolchain with no dynamic runtime dependencies beyond the C/C++ standard support available on the
device.
