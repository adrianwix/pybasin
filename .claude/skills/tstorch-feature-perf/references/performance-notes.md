# tstorch performance notes

Accumulated *algorithmic* performance findings only — confirmed slow patterns and their fixes,
keyed by pattern. Not workflow (see `SKILL.md`), not report.md usage (see SKILL.md's "Reading
report.md"), not design (see `references/api-design.md`), not one feature's own history (see
`migration-logs/<domain>.md`). Numbers themselves live in `report.md`, never here — this file
records the *shape* of a problem ("torch.fft pays a large fixed dispatch cost below N rows"),
which stays true as exact milliseconds drift.

## Small-batch dispatch overhead

Single-feature benchmarks use tiny per-case tensors (`B*S` roughly 4-20 rows). Several ops pay a
large *fixed* per-call cost (thread-pool dispatch, allocator setup) that dominates when there's
too little work to spread across threads — cheap per-element, expensive per-call.

- **`torch.sort`**: use `x.sort(dim=-1).values` directly, never a numpy round-trip
  (`np.sort(x.numpy(), ...)` then `torch.from_numpy`), even though the round-trip measures
  faster on CPU at small row counts (5-21x; re-confirmed at `(rows, T)` layouts identical to
  what a calculator actually receives, so this isn't a memory-layout artifact - the overhead is
  per-call thread-pool dispatch, not strided access). Every feature must be
  `torch.export`-compatible, and `.numpy()`/`torch.from_numpy()` calls raise `.numpy() is not
  supported for tensor subclasses` under `torch.export.export()`'s dynamic-shape tracing
  (re-confirmed on torch 2.9.0) even though they trace fine under `torch.compile(fullgraph=True)`
  - export-safety is non-negotiable, so no numpy dispatch branch is used, full stop.
  `torch.compile` does not close the CPU gap either: benchmarked `torch.compile(fullgraph=True,
  dynamic=True)` (compiled once, warmed across every tested shape, then timed - a fresh
  compile/warm per shape is needed to avoid measuring recompilation instead of steady state)
  against plain numpy at `rows` 4-20000 and `T` 50-2000. Compiled CPU torch is still 3-37x slower
  than numpy at `rows` 4-100 (the scale single-trajectory benchmarks actually use), reaching
  parity only around `rows` ~8,000-20,000. Compiled GPU torch has its own fixed floor from
  kernel-launch overhead (~0.07-0.09ms/call, independent of data size) that dominates below
  roughly `rows * T` ~ 200,000 elements, but crosses over quickly above that and reaches 25-27x
  faster than numpy by `rows=20,000, T=2000` - GPU is worth dispatching to for genuinely large
  batched-extractor workloads, just not for single-trajectory calls.
- **`torch.fft.rfft`/`irfft`**: same tax, confirmed for `agg_autocorrelation` — the ACF pass
  always went through a zero-padded FFT pair even for small requested `maxlag`, making it slower
  than `tsfresh` at single-feature scale. Fixed with a direct multiply-and-sum loop bounded by
  `_DIRECT_SUM_MAXLAG_THRESHOLD`, falling back to FFT only above the threshold. Crossover
  measured empirically around a few hundred lags on canonical benchmark shapes — not re-tuned
  per workload. `fourier_entropy` showed the same fixed cost with Welch segments capped at 256
  samples. A direct real DFT expressed as two matrix multiplies preserved every canonical
  entropy bin while making the uncompiled single-trajectory CPU path faster than tsfresh; at
  this bounded transform size, the extra arithmetic costs less than FFT dispatch.
- **`torch.sum`/`torch.linalg.vector_norm` over the last dimension**: confirmed for
  `absolute_sum_of_changes` after removing the old numpy CPU fast path to make the Feature class
  export-safe. On tiny single-trajectory CPU cases, native torch reductions can still show
  large cold-call outliers from dispatch/thread setup; replacing the sum with `cumsum`, 2-D/3-D
  matmul against ones, or a shape-static pairwise tree sum did not improve the benchmark and
  often made compiled/cold rows worse. Keep the simple torch reduction unless a future
  export-safe reduction kernel actually measures better; do not reintroduce a numpy branch in
  `__call__`.
- **`torch.argmax`/`argmin` and `cumsum` on short row batches**: confirmed for the location
  domain on a 24-thread CPU host. The export-safe kernels beat tsfresh consistently when the
  benchmark process uses one intra-op thread, but default-thread cold calls pay millisecond-scale
  dispatch outliers that can reverse the uncompiled single-trajectory comparison. Replacing the
  cumulative-mass crossing mask with `torch.searchsorted` removes the `(B, S, T, K)` temporary
  and improves the kernel, but it cannot remove `cumsum`'s thread-pool cost. Do not set PyTorch's
  process-global thread count from a feature class or restore the old numpy reduction path;
  either change benchmark/runtime thread policy explicitly at the application boundary or accept
  the fixed dispatch floor.
- General suspects when a feature is flagged slow with no Python loop: `sort`/`argsort`, `fft`,
  `one_hot` (materializes a large intermediate on CPU — prefer `bincount`/`scatter_add_`),
  repeated `cat` for boundary padding (prefer `nn.functional.pad` or share one padded tensor),
  and `.cpu().numpy()` round-trips in traceable feature code. A registered backend operator can
  own non-tensor CPU implementation details without introducing that boundary into `__call__`.

## `torch.compile(fullgraph=True)` compile-safety: diagnosing a graph break

Diagnostic technique for *investigating* a break, not a substitute for the persisted regression
test (`assert_compiles_fullgraph` — see SKILL.md's "Recording torch.compile(fullgraph=True)
safety").

Diagnostic: `torch.compile(fn, dynamic=True, fullgraph=True, backend="eager")(y)`, plus
`torch._dynamo.explain(fn)(y)` checked for `graph_count == 1, graph_break_count == 0` across a
couple of different `(B, S, T)` shapes.

General rule: `.item()` (or any tensor→Python-scalar extraction) that drives control flow (a
branch or a loop bound) breaks the graph — it needs a device→host sync Dynamo cannot trace into
one graph. Check first whether the value is already known in Python *before* compilation (e.g.
it comes from the caller's config, not from data inside the input tensor) — if so, resolve it in
plain Python at kernel-build time and close over it as a constant instead of deriving it via
`.item()`.

Confirmed cases:

- `median` used `torch.quantile(x, 0.5)`, whose CPU kernel cannot trace with symbolic shapes.
  Reimplemented via sort + midpoint average (mathematically identical, verified for odd and even
  `T`), compiles cleanly.
- `agg_autocorrelation` derived `global_maxlag = int(maxlags.max().item())` from tensor data,
  driving both a Python `if` (direct-sum vs FFT) and a `range()` loop bound. Fixed by passing
  `global_maxlag` through explicitly as a plain Python int (already known before it becomes a
  tensor) instead of deriving it via `.item()`.

## Cross-state waste for Type B parametrized features

One shared call covering every distinct param combo requested for a feature anywhere in an
extractor config preserves a Type B feature's "compute once, read many" property across states,
not just within one state — but if two states request *different* params for the same feature,
both states get *every* requested combo's column computed, including ones only the other state
asked for. Accepted deliberately: the wasted part is the cheap read (a masked
mean/median/var over an already-computed array), the expensive shared part (the pass over `T`)
is unavoidable regardless of which state asked. Slicing states apart before the call to avoid
this measured ~7x slower in eager mode (`index_select`-before-reduce). Small bounded waste beats
a large unbounded one.

Not yet empirically measured for a feature where the shared computation is cheap relative to the
per-combo read (unlike `agg_autocorrelation`'s ACF) — re-derive before relying on this for such
a feature.

## Batched small-matrix least squares on CUDA

`torch.linalg.qr` can be a poor CUDA kernel shape when a feature fits thousands of independent
regressions with short observation axes and small design matrices. Confirmed for the ADF and
autoregressive trend features: the batched CUDA QR path spent far more time in solver dispatch
than in arithmetic.

Normal equations are not a safe shortcut. Even after column scaling, exact singular systems make
the inverse fail, while regularization large enough to guarantee Cholesky completion noticeably
biases ill-conditioned float32 coefficients. `torch.linalg.lstsq` is fast and stable in eager
CUDA execution, but its data-dependent output metadata prevents `torch.compile(fullgraph=True)`.

For fixed, small parameter counts, a re-orthogonalized modified Gram-Schmidt loop expressed in
ordinary tensor operations avoids the solver-dispatch bottleneck and remains full-graph
compilable. Keep the loop bound structural, use no tensor-to-Python scalar extraction, and
validate CPU/CUDA agreement on well-conditioned inputs in every supported dtype. This is not a
blanket replacement for vendor QR; remeasure when matrices are large enough to amortize the
library call.

## GPU benchmark noise

Zero warmup by design (`warmup_rounds=0`) — a cold CUDA-context/kernel-launch round can dominate
`mean`/`stddev` for a small GPU workload. Observed for `agg_autocorrelation`: `stddev > mean`,
`max` 4-8x `min`, on both single-feature and 10,000-trajectory GPU benchmarks. Read `min`
alongside `mean`/`stddev` for GPU rows and state the noise explicitly — don't average it away.
In that case `min` still showed GPU slower than CPU (kernel-launch/PCIe overhead dominating
actual compute at that batch size) — a real result, not just noise.

## tsfresh parallelism fairness

`execute_tsfresh_extract_features` (the tsfresh side of every B*S-scale benchmark) must not
force `n_jobs=1`. tsfresh's own default is one worker process per CPU core, which is what any
real user gets for free batching many trajectories through `extract_features` in one call — an
earlier version of this benchmark hardcoded `n_jobs=1`, silently single-threading tsfresh for
every recorded number, an indefensible comparison. Fixed by dropping the override.

Empirically this made little practical difference for `agg_autocorrelation` — multiprocessing
pool startup/IPC cost for one lightweight feature over 10,000 short series isn't amortized
within a single call, so parallelism doesn't pay off here. That's a real characteristic of
tsfresh's batching model (per-call process-pool spin-up, not a persistent pool), not something
to "fix" by reintroducing `n_jobs=1`.

Single-feature benchmarks are unaffected: that path calls the raw per-series calculator directly,
never goes through `extract_features`, has no `n_jobs` concept.
