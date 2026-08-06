# pybasin → ExecuTorch → WASM interactive demo

Goal: an in-browser, illustrative demo of a subset of pybasin's pipeline (predefined ODE
case studies, sampling, integration, supervised/unsupervised classification) running
entirely client-side via an ExecuTorch `.pte` model executed through Emscripten/WASM — no
server, no Python at request time.

See the approved plan for the full phase breakdown. This README tracks findings per phase,
following the `report.md` convention used in `packages/tstorch/experiments/exporting/`.

## Phase 0 — De-risking the WASM toolchain

**Question**: does ExecuTorch's browser/WASM path (XNNPACK backend, built from source via
Emscripten) actually work, given it's explicitly experimental upstream and untested in this
repo? This matters because the existing `packages/tstorch/experiments/exporting/executorch_basic_stats/`
experiment only exercises the pip-wheel Python runtime binding (which fails to import here)
and its C++ runner (which still links full LibTorch/CUDA) — neither result says anything about
the from-source Emscripten build, which is a different toolchain entirely.

**Result: it works.** Environment: Python 3.12, PyTorch 2.13.0+cu130, ExecuTorch source at
`main` (shallow clone, commit as of 2026-08-05), Emscripten SDK 4.0.10 (via
`.ci/scripts/setup-emscripten.sh`), Node 24.19.0 (Emscripten-bundled), CMake 4.4.2 / Ninja
1.13.0 (installed into an isolated scratch venv — this machine has no system `cmake`/`buck2`
and no passwordless `sudo`, so these were installed via `pip install cmake ninja` into a
throwaway venv rather than touching the system or the pybasin project venv).

Recipe (following `examples/wasm/test_build_wasm.sh`, ExecuTorch's own CI test for this
feature):

1. `git submodule update --init --recursive --depth 1` (~20 third-party deps: XNNPACK,
   flatbuffers, cpuinfo, pthreadpool, etc.)
2. Emscripten + Node via `.ci/scripts/setup-emscripten.sh`.
3. `cmake_install_executorch_lib` — host Release build of `libexecutorch.a`/`libportable_kernels.a`.
4. Export a test model to `.pte` via `examples.portable.scripts.export` (reusing the pybasin
   project's own venv as `PYTHON_EXECUTABLE`, since it already has `torch`+`executorch`
   installed for the existing `executorch_basic_stats` experiment — no separate Python install
   needed for this step).
5. `emcmake cmake -DWASM_MODEL_DIR=<dir> -B<build_dir> .` then `cmake --build <build_dir>
   --target executor_runner`.
6. Run via Node: `$EMSDK_NODE <build_dir>/executor_runner.js --model_path=<model>.pte`.

**`add_mul.pte`** (ExecuTorch's minimal built-in test model) built and ran correctly end to
end through the from-source WASM build:

```
Model file add_mul.pte is loaded.
Model loaded in 4.742449 ms.
Model executed successfully 1 time(s) in 1.143340 ms.
1 outputs:
OutputX 0: tensor(sizes=[2, 2], [3., 3., 3., 3.])
```

The build also produces `executor_runner.html` + `.js` + `.wasm`, servable via a plain static
file server (`python3 -m http.server`) with correct MIME types (`text/html`,
`application/wasm`) — confirmed by fetching both directly. A literal browser click-through
wasn't performed in this sandboxed environment (no headless browser available), but since the
Node run and the browser page load the *same* `.wasm`/`.js` pair produced by the same build,
the Node success is strong evidence the browser path works too; this should still get a real
browser smoke-test in Phase 4 before considering the demo done.

The **`mv2`** (MobileNetV2) second test case in the same script failed — but only because the
pybasin venv doesn't have `torchvision` installed, which that unrelated stock example needs.
This isn't a blocker for anything in this plan (we don't use MobileNetV2); it's noted here only
so the failure isn't mistaken for a WASM/ExecuTorch problem when re-running the spike.

**Concrete finding to carry into Phase 3**: the default build's `executor_runner.wasm` is
**~92 MB**. The build log shows `portable_kernels` compiling essentially every CPU op kernel
unconditionally (`op_add`, `op_where`, `op_upsample_bilinear2d`, `op_topk`, ... dozens more),
none of which our case-study graphs need. For a "download this to run a browser demo"
experience, Phase 3's build must use ExecuTorch's selective-build mechanism
(`EXECUTORCH_SELECT_OPS_YAML`/selective op list) scoped to only the handful of ops our 6
exported graphs actually use, plus a size-optimized (`-Os`) build — not the default CMake
config used for this spike.

**Toolchain note for reproducing this**: this machine has no system `cmake`, no `buck2`, and no
passwordless `sudo`. `cmake`/`ninja` were installed via `pip install cmake ninja` into an
isolated venv (not the project's own `.venv`, not system-wide) and prepended to `PATH` after
the project venv's `bin` directory, so `python3` still resolves to the project venv (with
`torch`/`executorch` already installed) while `cmake`/`ninja` resolve to the scratch venv.
`buck2` was never actually needed for this build path — its absence only produced harmless
warnings from an unrelated cleanup/retry code path in `.ci/scripts/utils.sh`.

**Conclusion**: no fallback to onnxruntime-web needed. Proceeding with ExecuTorch for
Phases 1-4.

## Phase 1 — A torch-export-safe fixed-step solver, validated against diffrax

**Question**: pybasin's default integrator (`JaxSolver`, wrapping diffrax's adaptive Dopri5)
can't go through `torch.export` at all (JAX, not torch). `TorchDiffEqSolver` (torchdiffeq) is
pure-torch and already in the pipeline — can its fixed-step `rk4`/`euler` methods be reused
directly instead of writing new integration code?

**Finding 1 — a real, independent bug**: `TorchDiffEqSolver._integrate()` never passed
`options={"step_size": ...}` to `odeint`. Diffrax's `t_span`/`t_eval` split (integrate over the
full span, save only a narrower window) has no equivalent in torchdiffeq — `TorchDiffEqSolver`
instead prepends a `t_span[0]` anchor to the save-region time array before calling `odeint`
(e.g. `ts = [0, 900, 900.1, …, 1000]` for Duffing), which correctly tells `odeint` to start
integrating at `t=0` while only needing to look up the `900–1000` window afterwards — *that*
part is correct and necessary. But without `step_size`, `FixedGridODESolver` (torchdiffeq's
fixed-step base class) defaults to stepping directly between whatever points are in `ts` — one
absurd single RK4 step across the entire `0→900` gap. Fixed by adding a `step_size` constructor
param, threaded through to `options={"step_size": step_size}`. This is a correctness fix
independent of the export experiment — fixed-step methods were silently unusable before it.

**Finding 2 — torchdiffeq's `odeint` does not trace under `torch.export`, fixed-step or not.**
Isolated with a minimal harmonic-oscillator spike
(`phase1_export_spike.py`) in three variants — dynamic batch dim, static batch dim, time
tensor as a plain attribute instead of a registered buffer — all three failed identically with
`torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode` inside torchdiffeq's own
input validation (`_check_timelike`'s `assert diff.all() or (~diff).all()`, a boolean-tensor
`or` that `torch.export`'s tracer can't resolve to a concrete Python bool). This is a
`torchdiffeq`-internal incompatibility, not fixable from pybasin's side short of forking the
library.

**Response**: `TorchRK4Solver` (`pybasin/solvers/torch_rk4_solver.py`) — a ~130-line hand-rolled
fixed-step RK4, numerically identical to torchdiffeq's own `RK4._step_func`, built specifically
to avoid what broke torchdiffeq under export: every step count is a plain Python `int` computed
from `t_span`/`t_eval`/`step_size` at construction time (never from a traced tensor), and
trajectory states accumulate functionally (a Python list, stacked once) rather than via in-place
writes into a preallocated tensor. `tests/solvers/test_torch_rk4_solver.py::test_traces_under_torch_export`
confirms it exports cleanly with a dynamic batch dimension and matches the eager output.

**Finding 3 — a fixed-step integrator can silently produce `nan`, not just be less accurate,
around a divergent/unbounded regime.** The Lorenz case study's own `LorenzODE.ode()` freezes
the derivative to zero once `|y| > 200` (mirroring `JaxSolver`'s `event_fn`, which halts
integration entirely at that threshold) — but that mask only takes effect on the *next* step.
A fixed-step RK4 can overshoot straight past the threshold in a single step, landing on `inf`;
`0 * inf = nan` from then on, corrupting every subsequent state. Confirmed empirically: the
unclamped solver produced `inf` trajectory values for Lorenz even though downstream
classification still happened to land on the right label (NaN propagation through
`cdist`/`argmin` is not a reliable classifier — it isn't safe to depend on that). Fixed with an
opt-in `state_bounds` clamp applied after every RK4 step (`TorchRK4Solver(..., state_bounds=(-200,
200))` for Lorenz only — Duffing/Pendulum are naturally bounded and don't need it). Since a
bounded, finite input to a polynomial ODE RHS stays finite, this prevents `inf`/`nan` from ever
arising in the first place, deterministically, and plots as a clean plateau at the bound rather
than a spike to infinity.

**Finding 4 — literally reproducing each case study's `t_span=(0, 1000)` is not viable for
export.** Total RK4 step count is `simulated_time / step_size`, independent of batch size: a
naive attempt at `dt=0.001` over the reference `t_span=1000` needs **900,000+ unrolled steps**
for the anchor phase alone. That's not just slow to run — `torch.export` would need to trace
every one of those steps as a static unrolled graph, which is infeasible to trace in reasonable
time and would produce a `.pte` far too large for a browser download. The reference case
studies use `t_span=1000` specifically because diffrax's adaptive Dopri5 can blow through a
long, dynamically boring transient almost for free; a fixed-step method pays the same tiny `dt`
for every one of those 1000 time units regardless of whether the dynamics need it.

Swept `(transient length, observation-window length, step size)` per case study
(`phase1_case_study_accuracy.py`) against each one's full-length `JaxSolver` reference, at the
real target scale (`n=1000` samples). The transient decays in far fewer than 900 units for all
three systems — a short, tuned window reproduces the same classification labels as the
full-length reference at a small fraction of the step count:

| Case study | `t_span` | `t_eval` | `step_size` | Unrolled steps | Wall time (n=1000, CPU) | Label agreement vs. full-length reference |
|---|---|---|---|---:|---:|---:|
| Duffing | (0, 110) | (80, 110) | 0.05 | ~2,200 | 0.19s | 99.8% |
| Lorenz | (0, 120) | (100, 120) | 0.02 | ~6,000 | 1.43s | 99.7% |
| Pendulum | (0, 200) | (150, 200) | 0.02 | ~10,000 | 0.78s | 100.0% |

All three: zero non-finite trajectory values (state clamp confirmed working for Lorenz). The
residual ~0.2-0.3% disagreement on Duffing/Lorenz is consistent with basin-boundary noise
(points near a fractal/thin basin boundary can flip label under any tiny numerical
perturbation, including between two runs of the *same* adaptive solver) rather than integration
error — pushing `step_size` smaller on Lorenz did not improve agreement further, which is the
signature of having already hit that noise floor rather than a step-size-limited error.

**Finding 5 — a missing `features=None` silently widened the feature space being validated,
masking a real accuracy problem for Pendulum.** `phase1_case_study_accuracy.py`'s
`TorchFeatureExtractorAdapter` calls initially omitted `features=None`; the real case-study
setups (`setup_pendulum_system.py`, `setup_duffing_oscillator_system.py`) pass it explicitly so
that any state *not* listed in `features_per_state` is skipped, rather than falling back to the
"minimal" 10-feature default. Omitting it meant Pendulum's validation was silently computing 11
features (10 default stats on state 0 plus the intended `log_delta` on state 1) instead of the
1 feature the real classifier actually uses — self-consistent on both sides of that script's
comparison, so it didn't crash or look wrong, just quietly validated a different, easier
classification problem than the demo will actually run. Fixing it dropped Pendulum's reported
agreement from a false 100% to a real 84.6% at the original `(0, 80)`/`(60, 80)` window.

That regression traced to a genuine tuning gap, not an integration error: `dt` made no
difference (86.9% identical at `step_size` 0.05, 0.02, and 0.01), but lengthening the transient
from 60 to 150 time units (with `step_size=0.02`) restored 100.0% agreement. `log_delta`
(`log(|max - mean| + 1e-10)` on the angular velocity) is extremely sensitive to whether the
transient has *fully* decayed — Pendulum's damping constant is `1/alpha = 10` time units, so 60
units (6 time constants) left enough residual oscillation to occasionally confuse the
near-zero-variance signature of a fixed point with a limit cycle, while 150 units (15 time
constants) did not. Duffing and Lorenz were unaffected by the same bug — both configs already
list every state explicitly (Lorenz maps unused states to `None`, which correctly skips them
even without `features=None`), so there was no unlisted state to silently backfill. The table
above reflects the corrected numbers.

**Conclusion**: `TorchRK4Solver` with these tuned per-case-study windows is the Phase 2 export
target — not the literal reference `t_span`/`t_eval` values, which exist for an adaptive
solver's convenience and are not required for the resulting classification/basin-stability
outcome the demo needs to reproduce. Given Finding 5, any future change to a case study's
feature spec should be re-validated through `phase1_case_study_accuracy.py` with the *exact*
`TorchFeatureExtractorAdapter`/`TorchFeatureExtractor` construction the real graph uses —
matching configuration between the validation script and the production code is itself part of
what needs validating, not a given.

## Phase 2 — Independent per-stage export graphs, and `scan` instead of unrolling

**UX requirement**: the demo needs each pipeline control to be independently interactive —
changing the sampling region, the feature selection, or the classifier/clustering method
should each re-run only the affected stage, not the whole pipeline from scratch. This rules out
one fused `.pte` per case study; each stage needs its own separately-callable graph.

**Finding 1 — unrolling the RK4 loop into the exported graph does not scale, even for the
cheapest case study.** A first attempt fused sampling + integration + feature extraction +
classification into one graph per case study/mode, unrolling `TorchRK4Solver`'s step loop as
plain Python `for` (matching how it already worked in Phase 1). For Pendulum at its
*original* tuned config (~1,600 steps): `torch.export.export` took 124.8s, `to_edge` took
611.6s, `to_executorch` took 1930.9s — **~44.5 minutes total** — for an **18.9 MB** `.pte`.
Each unrolled step adds a fixed number of FX graph nodes, and `torch.export`/`to_edge`/
`to_executorch` all scale worse than linearly with node count; Lorenz's ~6,000-10,000 steps
would likely take hours and produce a far larger artifact. This doesn't scale regardless of how
the pipeline is split into stages.

**Finding 2 — `torch._higher_order_ops.scan` solves this completely.** `scan(combine_fn, init,
xs)` represents a fixed-length loop as *one* graph op with a body subgraph, independent of step
count — unlike a Python `for` loop, which the tracer unrolls into N copies. A 400-step spike
exported into a **7-node graph** in under a second, producing a ~10 KB `.pte`. It's an explicit
PyTorch prototype feature ("you may run into miscompiles"), so `scan_rk4.py`'s
`scan_rk4_integrate` was validated against `TorchRK4Solver`'s already-diffrax-validated output
before trusting it for any case study (`test_scan_rk4.py`) — first attempt used one flat step
size for the whole span and was off by 0.127 max-abs-diff against diffrax for Pendulum (vs.
0.00195 for `TorchRK4Solver`'s own two-phase schedule); matching `TorchRK4Solver`'s exact
two-phase step-size scheme (anchor-phase `dt` and save-phase `dt` computed independently, each
rounded up to its own whole step count) via two separate `scan` calls brought it back to
`TorchRK4Solver` parity (float32-rounding-level agreement, ~4e-6).

**Response**: each case study now exports as (at least) three independent graphs —
`IntegrateGraph` (sample + `scan`-based integrate; takes runtime `min_limits`/`max_limits`
tensors so the sampling-region control doesn't require re-export), `FeatureExtractGraph`
(trajectory → features), and `ClassifyGraph` (features → label/cluster index, supervised or
unsupervised). All four Pendulum stages combined: ~4.7s total build time, ~61 KB total `.pte`
size — matching the scale already proven by the pre-existing `tstorch` feature-extraction-only
export experiment, now for the whole pipeline.

`scan_rk4_integrate` is time-autonomous only (`ode_fn(y) -> dy/dt`, no explicit `t` argument)
so far — sufficient for Lorenz and Pendulum, whose right-hand sides don't depend on absolute
time. Duffing's forcing term (`A*cos(t)`) does, and needs time threaded through `scan`'s carry
as a second pytree leaf (`scan`'s `init`/carry can be a pytree, not just one tensor) — not yet
implemented.

**Known correctness pitfall**: `scan_rk4_integrate`'s `.permute(...)` output is not contiguous
in the time axis by construction (permute changes strides, not memory layout), but
`TorchFeatureExtractor` requires `stride(-1) == 1`. Missed initially — caught only once
`FeatureExtractGraph` was chained after `IntegrateGraph` on real data, not by the solver-level
unit tests alone (which never round-tripped through feature extraction). Fixed with an explicit
`.contiguous()` on both return paths; `test_scan_rk4.py` now asserts `stride(-1) == 1` directly
so this can't regress silently.

**End-to-end chain validation (Pendulum)**: with the corrected window/`dt` from Finding 5,
running the actual three exported-graph modules in sequence (`IntegrateGraph` →
`FeatureExtractGraph` → `ClassifyGraph`, `n=1000`, fresh `u01` input exactly as JS would
generate it) reproduces the full diffrax reference's classification **exactly** — 100.0%
agreement, identical FP/LC label counts (144/856). This is the first check that exercises the
whole split pipeline together rather than each stage in isolation, and it's clean.

## Phase 3 — `scan` fails at the real ExecuTorch runtime; pivot to zigode for integration

**Finding 6 — `scan` traces and exports cleanly, but the compiled `.pte` fails inside the actual
ExecuTorch C++ runtime.** `IntegrateGraph` passed `torch.export`, `to_edge`, `to_executorch`,
and eager-vs-exported numerical parity (`exported.module()`, i.e. the FX interpreter) without
any issue. Running the resulting `.pte` through the real runtime (`runtime::Program`/`Method`,
not the Python-side interpreter) failed with `[op_copy.cpp:83] Check failed
(tensor_is_broadcastable_to(src, in))`. A control test — identical tensor shapes and I/O, but no
`scan` — ran correctly through the same runtime, isolating the failure specifically to `scan`'s
lowering to the runtime's `aten::copy` kernel, not to marshaling, dynamic shapes, or anything
else in the pipeline. This matches `scan`'s documented prototype status and means it cannot be
trusted for the deployed demo, only for the eager/export-time validation already done in Phase 2.

**Response — move integration out of ExecuTorch entirely.** `packages/zigode` already ships a
Zig Dopri5 adaptive solver (`Dopri5Generic.solve_into`) with its own accuracy/performance tests,
compiles to a freestanding WASM module directly (no ExecuTorch involved), and needs no unrolling
or `scan` at all since it's real native code, not a traced graph — adaptive step count is just a
runtime loop, unaffected by any of Phase 2's tracing constraints. The demo's ExecuTorch `.pte`
graphs are now `FeatureExtractGraph` and `ClassifyGraph` only; `IntegrateGraph`/`scan_rk4.py`
remain on disk (validated, still useful as an export-time numerics reference) but are not part
of the deployed pipeline. `zigode`'s `wasm/pendulum_wasm.zig` exports `solve_batch_pendulum`
(loops `Dopri5.solve_into` per initial condition) and was checked against the reference values
already documented in zigode's own README (e.g. IC `[0.4, 0]` → `0.523617`, exact match) before
being wired into the browser demo.

Building for Zig's `wasm32-freestanding` target surfaced a few Zig-0.16-specific issues, none
numerically interesting but all blocking: `Compile.linkLibC()` was removed in 0.16 (now
`root_module.link_libc = true`); `zig build-lib -target wasm32-freestanding` produces a static
archive, not an instantiable module (`WebAssembly.instantiate` rejects it — wrong magic bytes),
so the WASM target must be `build-exe -fno-entry -rdynamic` instead; and `build.zig`'s module
system rejects a relative `@import("../src/...")` reaching outside the current module's root
even though raw `zig build-exe` on the CLI doesn't enforce this — fixed by declaring named
module imports (`b.createModule()` + `.addImport(name, module)`) instead of relative paths.

**Finding 7 — the ExecuTorch WASM build's real stall was a wrong Python interpreter, not disk
I/O.** The `executorch_wasm` CMake target (ExecuTorch's official Emscripten/embind JS binding,
`extension/wasm/wasm_bindings.cpp` — a cleaner `Module.load()`/`Tensor.fromArray()` API than
hand-rolling a C ABI runner) repeatedly stalled for 30+ minutes at "Generating kernel bindings".
WSL2 mounts Windows drives (`/mnt/c/...`) over 9P, and a PATH full of `/mnt/c/...` entries did
measurably slow CMake/Emscripten's toolchain probing — worth stripping (`grep -v '^/mnt/'`) —
but doing so alone only bought a handful more log lines before re-stalling at the same point.
Running the exact logged `gen_oplist.py` codegen command by hand, with the exact interpreter
path CMake had chosen, failed *instantly* with `ModuleNotFoundError: No module named 'yaml'` —
CMake's `find_package(Python3)` had silently picked a bare system Python
(`/home/adrian/.local/bin/python3.12`, no `pyyaml`) instead of the project's `.venv` (which has
it), because the actual cache variable `FindPython3` reads is `Python3_EXECUTABLE`, not the
`PYTHON_EXECUTABLE` env var this build's scripts had been setting. Passing both
`-DPython3_EXECUTABLE=... -DPYTHON_EXECUTABLE=...` to `emcmake cmake` fixed it outright — the
build that previously never got past kernel-binding generation completed end to end.

**Finding 8 — `EXECUTORCH_SELECT_OPS_MODEL` only accepts a single `.pte` path.** With the real
stall fixed, `gen_oplist.py` still failed, this time with `AssertionError: ... needs to be a
valid file, got <path1> <path2> <path3>` — a single string with all three of our `.pte` paths
space-joined into one non-existent "file". `--model_file_path` (unlike `--root_ops`, which
explicitly supports a comma/space-delimited list) takes exactly one path; passing a
semicolon-joined CMake list through `EXECUTORCH_SELECT_OPS_MODEL` doesn't do what its
`.pte`-file-list-shaped name suggests. Worked around by extracting the operator set ourselves —
`executorch.codegen.tools.selective_build._get_program_operators` on each of the three `.pte`
files, unioned, comma-joined — and passing that through `EXECUTORCH_SELECT_OPS_LIST` instead
(which *does* support a delimited list). This selective build produced a **592 KB** `.wasm`
(vs. the ~92 MB unselective build from Phase 0), confirming selective ops was worth the trouble.

**Finding 9 — the selective build silently drops primitive-op registration, breaking only the
unsupervised graph at runtime.** With the library built, the demo's supervised path worked, but
switching to unsupervised failed with `Failed to execute method forward, error: 0x14`
(`Error::OperatorMissing`) — and, since `App.tsx`'s mode-switch handler had no error handling,
this failure was silent in the UI: the stale supervised result stayed on screen relabeled with
unsupervised cluster names, showing identical percentages that looked plausible but weren't
actually recomputed. Root cause: `extension/wasm/CMakeLists.txt` links `executorch_core` (the
`Program`/`Method` runtime) but not `executorch` (the separate target that adds primitive-op
registration — `kernels/prim_ops/register_prim_ops.cpp`, covering ops like
`executorch_prim::floordiv.Scalar` and `aten::sym_size.int`). The supervised graph (`cdist` +
`argmin`) never touches a prim op, so it worked regardless; `torch_kmeans.py`'s dynamic-shape
arithmetic does, so only the unsupervised graph was affected. Fixed by adding `executorch` to
`extension/wasm/CMakeLists.txt`'s `link_libraries`.

**Finding 10 — Bun's HTML bundler tries to statically resolve a `<script src="/absolute/path">`
tag as a build-time module reference.** Putting the Emscripten glue script directly in
`index.html` (`<script src="/wasm/executorch/pybasin_wasm_lib.js">`) failed the whole page build
with `Could not resolve: "/wasm/executorch/pybasin_wasm_lib.js"` — Bun's bundler treats any
`<script src>` in an HTML entrypoint as something to bundle, not a runtime-fetched asset, even
for an absolute path pointing at a route served dynamically by `index.ts`. Fixed by loading it
at runtime instead — a `document.createElement("script")` inserted into `<head>` from
`executorchRunner.ts`'s `loadExecuTorchScript()` — which Bun's static HTML scan never sees.
(Building `pybasin_wasm_lib.js` without `-sEXPORT_ES6=1`, i.e. plain Emscripten `MODULARIZE`
UMD output, means this runtime-inserted classic script attaches `window.createExecuTorchModule`
as a plain global once loaded, with no ES-module resolution involved at all.)

**End-to-end browser validation**: with `bun run src/index.ts` serving the app and the `.pte`/
`.wasm` artifacts copied into `demo-app/static/`, driving the page with a real headless Chromium
(Puppeteer, since no browser-automation package was already installed in this repo) confirmed
the full pipeline runs client-side with zero console/page errors: sampling → zigode WASM
integration → ExecuTorch WASM feature extraction → ExecuTorch WASM classification, at `n=1000`
(the documented cap), for both supervised and unsupervised mode, including the reclassify-only
path (switching modes re-runs only `ClassifyGraph`, not integration or feature extraction).
Supervised (1-NN) and unsupervised (k-means) basin-stability fractions agreed closely in every
run (e.g. 16.4%/83.6% vs. 83.6%/16.4% — same split, cluster indices swapped), consistent with
Pendulum's single-feature (`log_delta`) FP/LC split being cleanly bimodal.

## Phase 4 — Generic extract/classify graphs, for genuine runtime feature selection

**The problem with Phase 2's per-case-study `.pte`s**: `FeatureExtractGraph` was originally
constructed with a fixed `feature_spec` baked in at export time (e.g. Pendulum's
`{1: {"log_delta": None}}`), matching each case study's real `setup_*.py`. That works for a
demo that only *runs* a case study, but not for the UX requirement that a user can pick which
features to use, for which state, and see results update live — `torch.export` bakes shapes
and config into the graph, so "any feature combination" would mean exporting one `.pte` per
combination, which is combinatorially infeasible. The same problem applied one level up: it's
also unnecessary to export one `.pte` *per case study* at all — Duffing and Pendulum share
`state_dim=2`, and nothing about feature extraction or classification is actually specific to
one case study's dynamics.

**Response — compute a fixed superset, select client-side.** `packages/pybasin`'s
`DefaultFeatureExtractor` already defines exactly this: a fixed 10-statistic set (`median,
mean, standard_deviation, variance, root_mean_square, maximum, absolute_maximum, minimum,
delta, log_delta`) computed identically for *every* state, unconditionally — no per-case-study
configuration at all. `FeatureExtractGraph` now reimplements this set directly (reusing
`DefaultFeatureExtractor`'s own `_FEATURE_NAMES`/`_delta` for exact numeric parity), so its
output shape depends only on `state_dim`. "Choosing features" in the UI becomes a pure
client-side column slice of that fixed `(n, state_dim * 10)` output — free, instant, no
re-export, and no re-run of feature extraction at all (see `browserBasinStability.ts`'s
`classify()`, which slices already-cached full features). One `extract_features_dim{N}.pte`
per distinct `state_dim` now serves every case study with that state count.

`ClassifyGraph`'s supervised mode had the same problem one level further: `reference_features`
was a `register_buffer`, fixed to one case study's specific feature count and template values
at export time. Changed to a runtime input tensor instead (`forward(features,
reference_features)`), with both the feature-count dimension and the template-count dimension
declared dynamic (`torch.export.Dim`) alongside the existing dynamic batch dim. This makes
`classify_supervised.pte` a single graph shared by *every* case study — each just computes its
own template features offline (same graphs, same config) and passes them in at call time, kept
in sync with the live batch by construction rather than by re-exporting anything.
`classify_unsupervised_k{N}.pte` keeps `n_clusters` fixed at export time (one `.pte` per
distinct cluster count, still shared across case studies with that attractor count) but also
gained a dynamic feature-count dimension for the same reason.

**Finding 11 — ExecuTorch's portable CPU kernel library has no `sort` implementation at all.**
`DefaultFeatureExtractor`'s `median` is computed via `.sort(dim=-1).values`, which traces and
exports fine (`torch.export`/`to_edge`/`to_executorch` all succeeded, and the eager-vs-exported
check passed) but failed at actual runtime with `Error::OperatorMissing` — the same failure
mode as Finding 9, but this time because `kernels/portable/cpu/` genuinely has no `op_sort.cpp`
at all (confirmed by `find`), not because a needed library wasn't linked. Rebuilding with
`EXECUTORCH_ENABLE_LOGGING=ON` surfaced the exact missing kernel
(`executorch:operator_registry.cpp: kernel 'aten::sort.values' not found`), which is what made
this diagnosable in under a minute instead of another guessing round — worth reaching for
immediately next time a `.pte` traces and exports cleanly but fails only at runtime. Fixed by
computing median via `topk` instead (`kernels/portable/cpu/op_topk.cpp` exists and supports
`largest=False, sorted=True`): take the `n // 2 + 1` smallest values and average the last two
for even `n`, take the last for odd `n` — numerically exact, verified bit-identical to
`DefaultFeatureExtractor`'s own `sort`-based `_median` (`max abs diff: 0.0`) since both are
exact order-statistic selections, just via a different algorithm.

**Validation**: `validate_shared_graphs.py` checks that selecting the same named features the
real production classifier already uses, out of the new generic per-state superset, reproduces
each case study's diffrax reference labels: Duffing 99.3%, Pendulum 100.0% (threshold lowered
to 0.97 from Phase 1's 0.99 for this check specifically, since it compounds two independent,
already-separately-validated small error sources — `TorchRK4Solver`-vs-diffrax integration
error, and generic-feature-selection-vs-originally-tuned-feature choice — rather than
isolating one). End-to-end browser validation (Puppeteer) confirmed the full pipeline still
works with the new shared graphs, including toggling feature-selection checkboxes live: the
selected column indices and sliced feature values verifiably change per selection (confirmed
via direct inspection, not just outcome), though two different single-feature selections
(`log_delta` on omega vs. `maximum` on theta) can legitimately produce the *same*
basin-stability split — both are strong, nearly-perfect separators for Pendulum's bounded-FP
vs. unbounded-rotation-LC distinction, so agreement there is expected physics, not a sign the
selection control is a no-op.
