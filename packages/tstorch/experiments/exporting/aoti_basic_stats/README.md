# AOTInductor Basic Statistics Experiment

This experiment packages the `Mean`, `Maximum`, and `Minimum` tstorch feature classes with
`torch.export` + AOTInductor, then compares CPU execution paths for combined and separated
feature computation.

The combined paths return one tensor with shape `(B, S, 3)`:

1. mean
2. maximum
3. minimum

The separated paths return one tensor with shape `(B, S, 1)` per feature.

The Python script exports four AOTI packages:

- combined `TorchFeatureExtractor` generated kernel: the internal single kernel built by
  `_compile()`
- separated `Mean` feature class
- separated `Maximum` feature class
- separated `Minimum` feature class

It also compares:

- `TorchFeatureExtractor`, eager kernel generation (`use_compile=False`)
- `TorchFeatureExtractor`, compiled kernel generation (`use_compile=True`)

The C++ runner compares:

- combined `TorchFeatureExtractor` generated-kernel AOTI package loaded through C++
- combined handwritten C++ one-pass loop
- separated `Mean` / `Maximum` / `Minimum` feature-class AOTI packages loaded through C++
- separated handwritten C++ loops for `Mean` / `Maximum` / `Minimum`

## Runtime Dependency Size

The `.pt2` packages are small because they contain the generated wrapper/source and compiled
model shared object. On this machine the combined extractor package is about 388 KB.

The C++ runner is not standalone: it dynamically links against the PyTorch wheel's LibTorch
runtime in `.venv/lib/python3.13/site-packages/torch/lib`.

Current measured runtime split on this machine:

```text
AOTI combined .pt2 package: ~388 KB
C++ runner executable:      ~90 KB
libtorch.so:                ~336 KB
libc10.so:                  ~1.4 MB
libtorch_cpu.so:            ~417 MB
libtorch_cuda.so:           ~976 MB
full torch package:         ~1.7 GB
```

This makes AOTI useful for Linux/server/factory deployments that can ship LibTorch, but not for
ESP32-class deployment. For very small targets, see `../standalone_codegen_basic_stats/`.

## Python package and benchmark

From the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python packages/tstorch/experiments/exporting/aoti_basic_stats/run_aoti_basic_stats.py
```

Useful options:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python packages/tstorch/experiments/exporting/aoti_basic_stats/run_aoti_basic_stats.py \
  --large-batch 10000 \
  --states 2 \
  --timesteps 1000 \
  --threads 1
```

The script writes generated packages and the compiled runner under
`packages/tstorch/experiments/exporting/aoti_basic_stats/artifacts/` by default, and writes a markdown report to
`packages/tstorch/experiments/exporting/aoti_basic_stats/artifacts/report.md`.

## C++ runner

After running the Python script, build and run the C++ loader command it prints. It uses the
LibTorch headers and libraries bundled in the active PyTorch wheel, so CMake is not required.

The C++ runner is compiled into the experiment's `artifacts/` directory by the Python script.
It accepts the combined extractor-kernel package and the three separated feature-class packages:

```bash
packages/tstorch/experiments/exporting/aoti_basic_stats/artifacts/aoti_basic_stats_runner \
  packages/tstorch/experiments/exporting/aoti_basic_stats/artifacts/extractor_generated_kernel.pt2 \
  packages/tstorch/experiments/exporting/aoti_basic_stats/artifacts/mean.pt2 \
  packages/tstorch/experiments/exporting/aoti_basic_stats/artifacts/maximum.pt2 \
  packages/tstorch/experiments/exporting/aoti_basic_stats/artifacts/minimum.pt2 \
  10000 2 1000 100 24
```

## Zig

Zig can be tried later as a C++ driver if a local Zig toolchain is installed, but this experiment
uses PyTorch's C++ API and therefore still needs a C++-compatible compilation path plus the
LibTorch headers and libraries from the PyTorch wheel.
