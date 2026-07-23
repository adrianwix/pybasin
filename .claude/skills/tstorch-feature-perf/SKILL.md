---
name: tstorch-feature-perf
description: "Rework a tstorch calculator domain as one coherent migration - make every feature in that domain faster than tsfresh's CPU implementation, keep numerical output identical, confirm torch.compile(fullgraph=True)/torch.export safety, lock all of that in with regression tests and Google-style docs, and wire the domain into TorchFeatureExtractor's SUPPORTED_FEATURES in a clean way. Use when fixing a calculator file under packages/tstorch/src/tstorch/calculators/, adding a new domain, or working through batched-migration-wip.md's 'Not started' list."
---

# tstorch Feature Performance

## Skill structure

- **`references/api-design.md`** — the target API shape and its hard constraints (what a
  `Feature` class must satisfy, including tracing cleanly under both `torch.compile` and
  `torch.export`). Design, not process. **Read this file in full before writing or modifying any
  function — not just this summary, not just the parts a neighboring function in the same file
  seems to imply.** Existing code in a not-yet-migrated file (still on the old `FeatureFactory`
  pattern) is not a reference for new decisions; it may predate rules stated in this doc, most
  importantly "Decision tree: is this feature batchable?" — check every parametrized/batchable
  function against that section by name, don't infer it from what's already in the file.
- **This file (SKILL.md)** — the process: "Instructions" below is the spine (11 numbered steps);
  every other section is detail consulted from a specific step, never read standalone.
- **`references/performance-notes.md`** — accumulated *algorithmic* performance findings only
  (dispatch overhead, GPU noise, confirmed slow patterns and their fixes). Not design, not
  process, not compile/export tooling notes.
- **`references/benchmarking-patterns.md`** — the canonical benchmark inventory: function names,
  `direct_call`/`proxy_call` meanings, backend coverage, compile state, parameter-count axes, and
  tsfresh report-time reuse rules.
- **`migration-logs/<domain>.md`** — one final branch-level record per migrated calculator
  domain.

## Files structure

All paths below (and every command elsewhere in this skill) are relative to the `pybasin` git
root.

Layout (all paths relative to `pybasin/`):

```
scripts/benchmark/
  benchmark-feature.sh        # wrapper: runs unit tests + benchmarks, upserts, regenerates report.md
  tstorch_bench_store.py      # store upsert/render logic (performance, precision, compile sections)
packages/tstorch/
  src/tstorch/calculators/torch_features_<domain>.py   # the calculator functions/Feature classes themselves
  src/tstorch/torch_feature_extractor.py               # TorchFeatureExtractor, SUPPORTED_FEATURES: dict[str, type[Feature[Any]]]
  src/tstorch/feature_factory.py                       # Feature protocol; DEPRECATED_-prefixed FeatureFactory/adapter machinery only for calculator files not yet migrated
  src/tstorch/settings.py                              # TorchFeatureExtractor config presets (FCParameters) only - no calculator imports
  tests/
    unit/test_<domain>.py                  # numerical-accuracy + torch.compile/export-safety tests
    unit/conftest.py                       # --precision-json / --compile-json, cached-trajectory fixtures
    benchmarks/
      feature_specs.py                     # FeatureSpec registry (shared — don't edit in a parallel per-file pass)
      calculators/test_<domain>.py         # pytest-benchmark performance suites
    utils/
      benchmark_utils.py                   # execute_*/run_*_benchmark performance helpers only, see "Unit test utilities vs benchmark utilities"
      unit_test_utils.py                   # assert_feature_matches_tsfresh, assert_feature_compiles_fullgraph, assert_feature_exports_cleanly, assert_feature_names_are_valid
      trajectory_builders.py               # shared cached-trajectory builders
      precision_recording.py               # in-memory tsfresh-diff records, dumped to JSON at session end
      compile_recording.py                 # in-memory torch.compile(fullgraph=True) pass/fail records
      export_recording.py                  # in-memory torch.export pass/fail records - not yet dumped by conftest.py/upserted into store.json
      test_benchmark_utils.py              # tests for benchmark_utils.py
      test_unit_test_utils.py              # tests for unit_test_utils.py
  reports/
    store.json                             # persistent store, keyed by feature — the only source of truth
    report.md                              # generated from store.json — read this, never hand-type numbers
docs/tstorch/api/
  <domain>.md                            # one TSTorch API page per calculator domain
benchmarks/tstorch/cache/*.pt              # cached (B, S, T) trajectory tensors used by both unit tests and benchmarks
references/                                # this skill's reference docs (api-design.md, benchmarking-patterns.md, performance-notes.md, batched-migration-wip.md)
migration-logs/                            # one final branch-level record per migrated calculator domain
```

`torch_compile_debug/` (created at `pybasin/`'s root by `options={"trace.enabled": True}`, used
in "Recording torch.compile(fullgraph=True) safety" below) is gitignored — don't commit it, and
don't worry about deleting it between runs, it's overwritten each time.

### Unit test utilities vs benchmark utilities

`tests/utils/unit_test_utils.py` and `tests/utils/benchmark_utils.py` are deliberately
separate modules, split by *what question the helper answers*, not by which directory happens to
import it:

- **`unit_test_utils.py`** -- is the result correct, and does it trace cleanly?
  `assert_feature_matches_tsfresh`, `assert_feature_compiles_fullgraph`,
  `assert_feature_exports_cleanly`, `assert_feature_names_are_valid`. These are unit-test concerns even though a couple of them
  reuse tsfresh-calling infrastructure that also serves the benchmark suite - they import
  `execute_tsfresh_direct`/`FeatureSpec` from `benchmark_utils.py`, never the reverse.
- **`benchmark_utils.py`** — how fast is it? `execute_tstorch_direct`/`execute_tstorch_extractor`/
  `execute_tsfresh_direct`/`execute_tsfresh_extract_features`, every `run_*_benchmark` helper,
  `FeatureSpec`, `BenchmarkSeriesData`. Performance-benchmark concerns only.

When adding a new shared test helper, put it in whichever module matches what question it
answers - don't add an accuracy or compile-safety assertion to `benchmark_utils.py` just because
a benchmark file happens to sit nearby.

## Instructions

Treat one calculator domain as the atomic migration unit. Inventory every public calculator in
`torch_features_<domain>.py`, then migrate that complete inventory through implementation, tests,
extractor wiring, feature specs, benchmarks, documentation, and reporting in one change. Finish
the domain with one uniform `Feature`/`ParameterlessFeature` class API before running its final
test and benchmark suite. If several agents are assigned different calculator domains, read
"Scope for parallel agents" before starting.

Before running *any* test or benchmark command anywhere in this workflow (steps 3-5, 8, "Benchmark
one domain"), read "Running tests and formatting during migration" below.

**Before step 2, read `references/api-design.md`'s "API design" and "Decision tree: is this
feature batchable?" sections in full** — every time, even for a feature that looks like a small
tweak to something already working, and even if neighboring code in the same file suggests a
pattern (it may predate the doc — see "Skill structure" above).

1. Baseline: `scripts/benchmark/benchmark-feature.sh --match <domain>` (informal - see
   "Benchmark one domain"; read the result per "Reading report.md" and "Understand what the
   benchmark actually measures"). If the command fails before producing performance JSON, record
   the failure as the baseline state and continue with the same domain selection and backend
   matrix.
2. Replace the domain's public calculator surface with one class per feature, per
   `references/api-design.md`
   (`ParamsT` TypedDict, `__init__`, `__call__`, `.names`) — faster than tsfresh, identical
   numerical output.
   - Export parameter-free calculators as `ParameterlessFeature` subclasses and parametrized
     calculators as `Feature[ParamsT]` subclasses. Callers instantiate the class and invoke the
     instance. The completed module's public calculator symbols are feature classes. Keep
     reusable implementation kernels private with underscore-prefixed names.
   - Complete this class conversion for every feature named by the domain's `FeatureSpec`
     inventory in the same migration.
   - If step 1 showed a feature slower than tsfresh with no obvious single cause, see "Fix the
     actual bottleneck" to diagnose why, before rewriting blind.
   - For every feature that takes params, check it against `api-design.md`'s "Decision tree" by
     name (already read above) - shared computation, broadcast, or (rare, must be flagged
     explicitly) neither.
   - Preserve the `(B, S, T)` input layout and `(B, S, F)` output shape — see "Layout invariant"
     and "Output shape convention".
   - For *what* the feature computes (formula, edge cases), see "Consult tsfresh for semantics,
     not performance".
   - Docstring goes on the class, per `api-design.md`'s "Where the documentation goes".
   - Until real paper citations are available, use this exact placeholder in the
     class docstring:
     `Doe, J., Smith, A., & Example, R. (2026). A placeholder reference for partial
     autocorrelation recursions. Journal of Example Time Series, 1(1), 1-10.`

   Every check below constructs each feature class via
   `construct_feature(FeatureClass, params_or_none)` — never through `SUPPORTED_FEATURES[name]`
   or any `DEPRECATED_`-prefixed machinery in `feature_factory.py`. Steps 3-5 need no wiring and
   run before step 6.

3. **Numerical accuracy and naming.** Add `test_<feature>_matches_tsfresh` in
   `packages/tstorch/tests/unit/test_<domain>.py` and call
   `assert_feature_matches_tsfresh(FeatureClass, spec, canonical_trajectories, params)` from
   `unit_test_utils.py`. Omit `params` for a `ParameterlessFeature`; pass the full list of distinct
   params for a parametrized `Feature`. The utility constructs the class, checks `(B, S, F)`,
   evaluates every parameter against every canonical trajectory, handles tsfresh simple and
   combiner calculators, and records the aggregate precision rows consumed by `report.md` and the
   website docs. See "Recording numerical accuracy" for the exact contract. Apply this test to
   every tsfresh-backed feature in the domain.

   Also add `test_<feature>_has_valid_names`, calling `assert_feature_names_are_valid(feature_name,
   FeatureClass, distinct_params)` from `unit_test_utils.py` — mandatory for every feature,
   including parameter-free ones (omit `distinct_params`, or pass `None`, for a
   `ParameterlessFeature`). This is the shared utility that proves every requested param produces
   a non-empty name and that distinct params produce distinct names; see "Recording feature name
   validity" for the exact call shape. `test_autocorrelation.py`'s four features carry this test
   too, not just newly migrated ones, since the naming contract applies codebase-wide.
4. **Compile safety.** Add a test named `test_<feature>_compiles_fullgraph` in
   `test_<domain>.py`, calling `assert_feature_compiles_fullgraph(feature_name, FeatureClass,
   params, y_bst)` — `params` is `None` for a `ParameterlessFeature` — see "Recording
   torch.compile(fullgraph=True) safety" for the exact call shape. Mandatory for every feature in
   the domain; needs no wiring.
5. **Export safety.** Add a test named `test_<feature>_exports_cleanly` in `test_<domain>.py`,
   right after step 4's test, calling `assert_feature_exports_cleanly(feature_name, FeatureClass,
   params, y_bst)` (same `None`-for-parameterless convention) — see "Recording
   torch.compile(fullgraph=True) safety" for the exact call shape. Mandatory for every feature in
   the domain; needs no wiring.
6. Wire the whole domain in — see "Wire it into TorchFeatureExtractor" for the full procedure; in
   short:
   - In `torch_feature_extractor.py`, import every feature class from its calculator file and add
     each `"feature_name": NewFeatureClass` entry to the `SUPPORTED_FEATURES` dict literal (or
     replace existing entries, if migrating features already present under an older shape).
   - In `packages/tstorch/tests/utils/feature_specs.py`, set the feature's `FeatureSpec`
     `params: list[dict[str, Any]]` if a single `kwargs` dict isn't enough to express what you
     want benchmarked (`params` defaults to `[kwargs or {}]`, which covers most cases as-is), and
     remove `supports_batched_extractor=False` if this is the first time the feature is wired in.
     If two states might request different params for a parametrized feature, note whether the
     cross-state waste tradeoff (`references/performance-notes.md`) applies.
   - Remove the domain's deprecated factory registry, builders, name adapters, and public
     module-level calculator functions after all class entries are present. Retain private
     underscore-prefixed kernels when classes share implementation.
   - Compile-check the composed kernel too: `torch.compile(instance, dynamic=True,
     fullgraph=True, backend="eager")` plus `torch._dynamo.explain(instance)` on an extractor
     mixing this feature with existing ones, checked for `graph_count == 1, graph_break_count ==
     0` (see `references/performance-notes.md`). Quick and throwaway, not a persisted test.
7. **Benchmarks.** Before creating or renaming benchmark functions, read
   `references/benchmarking-patterns.md` in full. It is the canonical inventory for the eight
   benchmark rows, their exact function names, `direct_call`/`proxy_call` semantics, compile
   state, parameter-count axes, the tsfresh argument-batching research requirement, and the
   tsfresh report-time reuse rule. Keep the benchmark file
   `parametrize_domain(FEATURES)`-driven and backend-parametrized via `BACKENDS`
   (`tstorch_cpu`/`tstorch_gpu`/`tsfresh`) for the complete domain. Preserve the canonical
   backend axis and let the shared benchmark utilities apply each row's backend semantics and
   report-time tsfresh reuse.
8. Re-run `scripts/benchmark/benchmark-feature.sh --match <domain>` (the "after" run —
   covers steps 1, 3, and the benchmark rows wired into that wrapper; upserts into `report.md`).
   Steps 4 and 5's compile/export-safety tests and any benchmark rows not wired into
   `benchmark-feature.sh` (currently the compiled single-trajectory direct-call row and the two
   batched-argument rows) must be run directly:
   `uv run pytest packages/tstorch/tests/unit/test_<domain>.py -k "compiles_fullgraph or
   exports_cleanly" -v` and `uv run pytest
   packages/tstorch/tests/benchmarks/calculators/test_<domain>.py -k "batched_arguments or
   single_trajectory_direct_call_compiled" --benchmark-only`; read their numbers/results from
   pytest's own output, not `report.md`. Read the rest per "Reading report.md" and "Understand
   what the benchmark actually measures"; confirm the
   `test_single_trajectory_direct_call_uncompiled_<domain>` and
   `test_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_<domain>`
   `tstorch_cpu`
   rows beat `tsfresh` (GPU rows are not required to). A failing compile- or export-safety test
   is a real result to record and investigate, not something to silently ignore.
9. Update `docs/tstorch/api/<domain>.md` so every feature class is documented on its calculator
   domain page, with that page using `docstring_style: google`. If the domain page does
   not exist yet, create it and add it to `mkdocs.yml` under `TSTorch > API Reference`;
   do not dump all calculator domains into one flat page. Within the domain page, group
   docs by owning feature: render `api_object_heading("Human Title", "<dotted.path.Feature>")`
   first, then the feature's mkdocstrings block with `show_root_heading: false`, then the compact
   signature heading for that feature's params `TypedDict`, rather than rendering every `TypedDict`
   at the top or at the same hierarchy level as feature classes. Do not use mkdocstrings'
   class renderer for params `TypedDict`s; it adds noisy "Bases: TypedDict" and
   "Attributes" boilerplate. Use the reusable `typed_dict_schema(...)` MkDocs macro
   instead, e.g.
   `{{ typed_dict_schema("tstorch.calculators.torch_features_autocorrelation.PartialAutocorrelationParams") }}`.
   Do not hand-write anchors, headings, or field tables. Do not add an extra heading such
   as "Partial Autocorrelation Parameters" above it. Run `uv run mkdocs build --strict`,
   then spot-check the
   generated page for leaked literal `Args:`, `:param`, `:return`, `.. math::`,
   `.. rubric::`, `math::::`, or `childfieldmarker` text.
10. Check the domain's features off in `references/batched-migration-wip.md`'s checklist — no
    numbers go in that file, `report.md` is the only source of truth. After the complete domain
    implementation, tests, benchmarks, registry wiring, and documentation are in their final
    state, inspect `git diff origin/main -- <all files changed for the domain>`. Write
    `migration-logs/<domain>.md` from that comparison: describe every feature in the domain as
    the final branch contribution relative to `origin/main`, including the implemented
    semantics, numerical corrections, edge cases, shared-computation choices, and final design
    rationale represented by the diff. Use the final symbols and behavior in the working tree.
    Keep domain-specific findings in this log; `references/api-design.md` holds only rules that
    apply to every feature.
11. Use `--match <domain>` for the final verification. Treat the domain benchmark file and all
    of its features as one correctness and performance result.

## Scope for parallel agents

*Consulted before starting, only if multiple agents are fixing different calculator files at the
same time — see the "Instructions" preamble.*

Only edit your assigned file and its `tests/unit/test_<domain>.py` — `settings.py`,
`utils/feature_specs.py`, `torch_feature_extractor.py`, `benchmark_utils.py`, `unit_test_utils.py`, and
every other calculator file are shared, and concurrent edits will clobber each other. Writing the
`Feature` class itself and its accuracy/compile/export tests (steps 2-5) is safe in parallel — it
only touches your assigned calculator file and its own `test_<domain>.py`. **Wiring it in (step
6) is not**: unlike the old adapter-based design, adding a feature to `SUPPORTED_FEATURES` means
editing a dict literal directly in the shared `torch_feature_extractor.py` (see "Wire it into
TorchFeatureExtractor") — one agent at a time, after every parallel agent doing steps 2-5 has
finished, not concurrently. `feature_specs.py` stays off-limits the same way -- update
`supports_batched_extractor` centrally for every feature in each completed domain after the
parallel calculator and unit-test edits have finished.

## Running tests and formatting during migration

*Consulted before every test/benchmark command this skill runs — see the "Instructions"
preamble.*

**Format and lint the files you touched before running any test, every time**, not just at the
end of a feature:

```bash
uv run ruff format <touched files>
uv run ruff check --fix <touched files>
```

A formatting/lint issue that surfaces as a test failure (or as noise in a diff) wastes a cycle
figuring out it was never a real test problem. This is cheap and has no reason to be skipped or
batched until later.

### Full typing is required

Every Python file created or modified by a domain migration must be fully typed. This includes
calculator implementations, unit tests, benchmark tests, and shared helpers changed to support
the domain. Annotate every function parameter and return type, including pytest fixtures and
benchmark parameters; annotate module-level collections whose element type is part of the test
contract. Use the concrete `BenchmarkFixture`, `BenchmarkSeriesData`, `FeatureSpec`, and
`Backend` types in benchmark signatures. Do not use an unqualified `Any`, a file-level Pyright
mode override, `cast`, or `# type: ignore` to conceal a type mismatch. At a genuinely dynamic
third-party boundary, keep `Any` confined to that boundary and expose a precise type to the
migration code.

Shared APIs must accept the least restrictive read-only abstraction they consume. In
particular, helpers that only iterate feature parameters take
`Sequence[Mapping[str, Any]]`, not invariant `list[dict[str, Any]]`; this allows a feature's
`list[ParamsT]` to pass without a cast while preserving its `TypedDict` contract.

After formatting and before running tests, run Pyright over the complete migrated surface and
every shared Python file touched by the migration. Zero errors and zero warnings are required:

```bash
uv run pyright \
  packages/tstorch/src/tstorch/calculators/torch_features_<domain>.py \
  packages/tstorch/tests/unit/test_<domain>.py \
  packages/tstorch/tests/benchmarks/calculators/test_<domain>.py \
  <touched shared Python files>
```

This codebase is mid-migration: most calculator files are still on the pre-`Feature`-class shape,
and `SUPPORTED_FEATURES` only holds the domains that have actually been migrated (see
`references/batched-migration-wip.md`). **Never run a broad/whole-tree test command** —
`uv run pytest packages/tstorch/tests/` or `.../tests/unit/` with no path — it collects tests for
domains you didn't touch and haven't been migrated yet, so a broad run mixes real regressions
with expected, unrelated failures and tells you nothing useful. Scope every test command to the
complete domain you're migrating:

```bash
uv run pytest packages/tstorch/tests/unit/test_<domain>.py
uv run pytest packages/tstorch/tests/benchmarks/calculators/test_<domain>.py
scripts/benchmark/benchmark-feature.sh --match <domain>
```

If you edited a file shared across domains (`benchmark_utils.py`, `unit_test_utils.py`,
`utils/feature_specs.py`, `synthetic_feature.py`), also run the colocated utility test with the
matching filename, e.g. `packages/tstorch/tests/utils/test_unit_test_utils.py` for
`unit_test_utils.py` or `packages/tstorch/tests/utils/test_benchmark_utils.py` for
`benchmark_utils.py` — still by explicit file path, never the whole tree.

## Benchmark one domain

*Consulted during step 1 (baseline) and step 8 (confirm the fix) of "Instructions".*

Use the wrapper script so benchmark data is captured and rendered through the shared store:

```bash
scripts/benchmark/benchmark-feature.sh --match <domain>
```

This runs the numerical-accuracy unit tests, their `torch.compile(fullgraph=True)` safety checks
(both in `packages/tstorch/tests/unit/test_<domain>.py`), and every performance benchmark
defined for every feature in the domain (all backends -- `tstorch_cpu`, `tstorch_gpu` when CUDA
is available, and `tsfresh`; every variant that exists for the domain, following
`references/benchmarking-patterns.md`'s row names and tsfresh reuse rules), upserts all three into
`packages/tstorch/reports/store.json` keyed by feature (and within a feature, by variant/backend
for performance or by param combo for precision/compile), and regenerates
`packages/tstorch/reports/report.md` from the full store. Read that report for the numbers.
`--match <domain>` runs every feature in that domain's file. Treat
`benchmarks/tstorch/single_feature_benchmark_report.ipynb` as historical single-feature material;
the generated domain report is the source of truth.

Run this once before touching the domain (baseline, informal -- the store only keeps the
latest value per key, so there's no permanent "before" record) and once after (the number that
matters, now in `report.md`).

GPU rows are frequently noise-dominated (a cold CUDA-context/kernel-launch round can make
`stddev > mean` with zero warmup) — `report.md` includes `min`/`max` alongside `mean`/`stddev`
for exactly this reason; read the full row, not just the mean.

## Reading `report.md`

*Consulted whenever "Instructions" or another section below says "read report.md" — mainly
steps 1 and 8.*

`report.md` is the single source of truth for every number this skill produces — regenerated
from `store.json` by the command above, never hand-edited, never re-typed into a doc or a PR
description. If a claim needs a number, the claim points at `report.md`; it doesn't embed one
(numbers drift as code changes, prose that repeats them silently goes stale).

It's ~2,000+ lines (one `## \`feature_name\`` section per feature, each with `### Numerical
accuracy`/`### torch.compile(fullgraph=True) safety`/`### Performance` subsections) — don't
`Read` the whole file for one feature. Get the line range first, then read just that:

```bash
grep -n "^## \`<feature_name>\`$" packages/tstorch/reports/report.md   # e.g. `agg_autocorrelation`
```

That gives you the start line; the section ends at the next `^## ` line (`grep -n "^## " | ...`
to find it, or just read ~60 lines from the start — one feature's section is rarely longer).

Each feature's "Performance" table should match `references/benchmarking-patterns.md`'s current
inventory. The rows are not interchangeable:

- **`single_trajectory_direct_call_uncompiled`** — raw `Feature` class call for tstorch and raw
  calculator call for tsfresh. This is the fair single-trajectory direct-call comparison.
- **`single_trajectory_direct_call_compiled`** — raw `Feature` class call for tstorch, compiled
  and warmed. tsfresh should be displayed by reusing row 1 at render time, not by writing a
  duplicate `store.json` record.
- **`batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call`** — raw `Feature`
  class call for tstorch and `extract_features` for tsfresh. The name is deliberately verbose
  because the mechanisms differ.
- **`batched_trajectories_proxy_call_compiled_cold_run`** — a fresh `TorchFeatureExtractor` built and
  compiled every round, 1 round only. This is the one-time cost a pipeline pays once, ever, per
  input shape — not steady state. (Building a fresh extractor and re-calling it in a loop does
  *not* give independent cold samples either: `torch.compile` caches by code structure and FX
  graph hash, not by object identity, so round 2+ of an in-process "fresh extractor" loop
  silently hits that cache. 1 round is the honest number, not an average that quietly
  understates the real cost.)
- **`batched_trajectories_proxy_call_uncompiled`** — one extractor built once, then timed rounds
  reusing it with `use_compile=False` and no untimed warmup. Isolates extractor-call overhead with
  no `torch.compile` step at all.
- **`batched_trajectories_proxy_call_compiled`** — same as above but compiled. The steady-state per-call
  cost a pipeline that builds one extractor and reuses it across many batches actually pays, once
  compilation is already paid for. Use this row (or
  `batched_trajectories_proxy_call_uncompiled`) for real steady-state comparisons — never the
  `_compiled_cold_run` row, which measures compile cost plus one execution.
- **`batched_trajectories_proxy_call_uncompiled_batched_arguments`** — same uncompiled proxy-call
  comparison as above, parametrized over argument counts and run without untimed warmup.
- **`batched_trajectories_proxy_call_compiled_batched_arguments`** — compiled proxy-call
  comparison, parametrized over argument counts. Row 8 reuses row 7's tsfresh value for the same
  count at render time.

## Understand what the benchmark actually measures

*Consulted alongside "Benchmark one domain"/"Reading report.md" (steps 1 and 8) when a number
looks surprising and you need to know whether it's noise or real.*

The single-trajectory direct-call uncompiled row should call `benchmark.pedantic(func, rounds=5,
iterations=1, warmup_rounds=0)` — 5 independent cold calls, no artificial warmup, on tiny
per-case tensors (`benchmarks/tstorch/cache/canonical_supervised_bounded.pt`, shapes like
`(2, 2, 1000)`-`(5, 2, 5000)`). Still noisy (10-60ms swings from allocator/OS jitter) — check
`stddev`, not just `mean`; re-run the whole command if `stddev` is large relative to the
difference you're trying to confirm.

A genuine Python-level `for` loop over `T` (time) or `B*S` (batch×state) is never noise — it's
O(T) or O(B*S) Python overhead per call. Grep for `for .* in range(` in the target file before
assuming a gap is dispatch overhead instead — if it's not a loop, see "Fix the actual
bottleneck" for other suspects.

## Fix the actual bottleneck

*Consulted during step 2 of "Instructions", when step 1's benchmark showed the function slower
than tsfresh and the cause isn't already obvious (e.g. not an explicit Python loop, already
ruled out by "Understand what the benchmark actually measures"). After identifying and fixing
the bottleneck, go back to step 1's benchmark command to confirm it actually improved, then
continue to step 3 — don't assume a fix worked without re-measuring.*

See `references/performance-notes.md` for confirmed findings: specific files/functions with
known Python-loop bugs, the general dispatch-overhead pattern (many small ops beat few fused
ones, when each op is individually "vectorized"), and measured `torch.sort`/`torch.fft`
small-batch costs with their fixes. Treat every entry there as a hypothesis to check against the
*current* source, not a guaranteed diagnosis — re-read the target function before assuming which
pattern applies. Add a new entry when you confirm something there with a measurement; keep this
file (the workflow) free of specific findings — they belong in `performance-notes.md`, or in
`migration-logs/<domain>.md` if they're specific to the one feature you're fixing.

## Layout invariant: (B, S, T), T contiguous

*Consulted during step 2 of "Instructions", while writing or rewriting the function body.*

All calculators receive `x: Tensor` of shape `(Batch, State, Time)`. The benchmark fixtures
build this via `y_steady_tsb.permute(1, 2, 0).contiguous()` (see
`packages/tstorch/tests/benchmarks/conftest.py`), so `T` is the fastest-varying (contiguous)
stride at the call boundary. Preserve that inside the calculator:

- Reduce/scan/sort/unfold along `dim=2` (or `dim=-1`), not after a transpose that makes `T`
  non-contiguous.
- If you must reshape to `(B*S, T)`, use `.reshape(-1, T)` on a tensor that is already
  `(B, S, T)`-contiguous — this is a view, not a copy. Adding a `.permute(...)` before it turns
  the reshape into a copy.
- Never permute `T` to a non-last dimension and then run per-timestep Python loops "for clarity"
  — see "Fix the actual bottleneck" for what that anti-pattern costs.

## Output shape convention: always `(B, S, F)`, even when `F == 1`

*Consulted during step 2 of "Instructions", while writing or rewriting the function body, and
during step 3 when writing the accuracy test's shape assertion.*

Every calculator returns `(B, S, F)` with `F` last — never a bare `(B, S)`, even for one
scalar output. Assert the exact shape in tests, not just values, since a wrong-axis bug can
still pass a values-only check when two dimensions happen to match in the test fixtures:

```python
result = fn(y_bst, **kwargs)
expected_shape = (*y_bst.shape[:2], expected_f)  # expected_f = 1 for single-output features
assert result.shape == expected_shape, f"{feature_name}: expected {expected_shape}, got {result.shape}"
```

## Parametrized features: one `Feature` class per feature

*Consulted during step 2 of "Instructions", to decide how to structure the function itself, and
during step 6 when wiring it in (see "Wire it into TorchFeatureExtractor" for the wiring
procedure this class shape enables).*

Every feature — parameter-free or parametrized — is one class implementing the `Feature`
protocol: `references/api-design.md`'s "API design" section has the full spec, "Example" has a
complete worked one (`Autocorrelation`). In short: direct calls and `TorchFeatureExtractor` are
two doors into the same class — `Feature(params)` constructs it, `feature(x)` runs it, `.names`
gives per-param output-name suffixes — and both doors must go through the exact same class,
fully typed via a per-feature `TypedDict` (that page's "Hard rule: the two doors must be
symmetric" is a hard rule, not a suggestion — read it before touching any parametrized feature).

## Consult tsfresh for semantics, not performance

*Consulted during step 2 of "Instructions", to confirm exactly what the function should
compute before or while rewriting it.*

`.venv/lib/python3.13/site-packages/tsfresh/feature_extraction/feature_calculators.py` is the
source of truth for *what* a feature computes (edge cases: empty series, all-constant series,
NaN handling, off-by-one in windowing). It is row-at-a-time numpy/pandas and its performance
characteristics don't transfer — do not port its loop structure, only its semantics. Cross-check
tricky cases (ties in sorting, `T <= 1`, all-equal-value series) against the tsfresh docstring
example before calling a rewrite done.

## Recording numerical accuracy

*Consulted during step 3 of "Instructions". If `test_<domain>.py` doesn't exist yet for this
domain, see "Add a unit test using cached trajectories" first for how to set one up (fixture
usage, cache file), then come back here.*

Call `assert_feature_matches_tsfresh` for every tsfresh-backed feature. Its uniform signature is:

```python
assert_feature_matches_tsfresh(
    FeatureClass,
    spec,
    canonical_trajectories,
    params,  # omit for ParameterlessFeature
    tsfresh_params=tsfresh_params,  # include only when tsfresh uses different parameter names
)
```

The utility owns feature construction, `(B, S, F)` shape validation, tsfresh calculator
resolution, simple-versus-combiner dispatch, comparison, and precision recording. For a
parametrized feature, pass all distinct params in one call. The utility checks each parameter
against every tensor in `canonical_trajectories.case_tensors` and records one aggregate row per
parameter. Parameter-free features follow the same path with one implicit output parameter.

The pytest session writes these records through `--precision-json`; `benchmark-feature.sh`
upserts them into `store.json` and regenerates `report.md`. That generated numerical-accuracy
table is the data source for the website docs.

**Test at least ~8 distinct, meaningfully different param values where feasible** — not
one. A single param (e.g. one `lag`) only proves the formula is right at that one point;
it's cheap for cross-checking to miss an off-by-one, an edge case (0, a value near a
series' length, a value straddling an internal branch threshold), or a bug that only
shows up at some params and not others — exactly what happened migrating
`partial_autocorrelation` (see `migration-logs/autocorrelation.md`): a single-lag test
had passed for a long time, and only testing 8 lags together (including 0) surfaced two
real bugs and a lag-0-degenerate-variance regression in `autocorrelation`'s sibling
kernel. `assert_feature_matches_tsfresh` evaluates the full parameter list across every
canonical trajectory in one assertion. The point is param *coverage*. Skip this
only when a feature has no meaningful param space (parameter-free) or when tsfresh itself
cannot compute more than a handful of valid values for the fixture data (rare).

Apply the recording helpers to every feature in the domain migration so `report.md` captures one
complete, consistent numerical-accuracy result for the domain.

### Noting when tstorch batches something tsfresh itself does not

*Consulted from within step 3, after confirming the accuracy test passes — not a separate step.*

If a feature is "truly batchable" in tstorch (see `references/api-design.md`) but
tsfresh's own calling convention does *not* share that computation — either tsfresh calls it
`fctype="simple"` (one independent call per param, e.g. `autocorrelation`) or the feature has
no tsfresh equivalent at all — that's a genuine algorithmic improvement over tsfresh, not just
a faster port of the same computation, and it's easy to lose once the migration checklist just
shows `[x]`. Add a one-entry note to `FEATURE_NOTES` in `scripts/benchmark/tstorch_bench_store.py`
(a plain hand-maintained dict — this is stable prose, not a measurement, same exception as the
other fixed preamble text in that file's `render_report`; see "Reading report.md" above for why
measurements themselves still can't be hand-typed) so it renders as a `**Note:** ...` line under
that feature's `report.md` section on the next `benchmark-feature.sh` run. Don't use a markdown
blockquote (`> ...`) for this or anything else in the report — it renders poorly in some themes;
a plain `**Note:**`-prefixed paragraph reads correctly everywhere.

## Recording feature name validity

*Consulted during step 3 of "Instructions", alongside "Recording numerical accuracy" — a
separate mandatory test, not part of the accuracy test itself.*

Every feature, parameter-free or parametrized, needs `test_<feature>_has_valid_names` in
`test_<domain>.py`, calling `assert_feature_names_are_valid(feature_name, FeatureClass,
distinct_params)` from `utils/unit_test_utils.py`:

```python
def test_sum_values_has_valid_names() -> None:
    assert_feature_names_are_valid("sum_values", SumValues)


def test_quantile_has_valid_names() -> None:
    params: list[QuantileParams] = [{"q": q} for q in _QUANTILE_QS]
    assert_feature_names_are_valid("quantile", Quantile, params)
```

For a parametrized feature, reuse the same distinct-params list the accuracy test already
defines rather than inventing a second one. For a `ParameterlessFeature`, omit `distinct_params`
entirely (it defaults to `None`) — no `y_bst` fixture needed either, since this test never calls
the feature, only constructs it and reads `.names`.

The assertion itself, via `tstorch.feature_factory.construct_feature`: constructs
`FeatureClass(distinct_params)` (the batched call) and `FeatureClass([p])` per entry (the
single-param call `TorchFeatureExtractor._compile()` uses to name one column) for a
`Feature[ParamsT]`, or just `FeatureClass()` for a `ParameterlessFeature`, then checks every name
is non-empty, that both call shapes agree (parametrized only), and — when `distinct_params` has
more than one entry — that every name is distinct. A parameter-free feature only exercises the
non-empty check - there is exactly one name, so there's nothing to prove distinct.

## Add a unit test using cached trajectories

*Consulted from step 3 of "Instructions", only when `test_<domain>.py` doesn't already exist for
the domain you're touching — otherwise skip straight to "Recording numerical accuracy".*

Write the test in `packages/tstorch/tests/unit/test_<domain>.py`, mirroring the calculator
filename minus the `torch_features_` prefix — most domains already have one (see "Files
structure" above), so check whether the file exists before creating it; either way, use the
`canonical_trajectories` session fixture from `unit/conftest.py` (built once, shared across the
whole file) rather than loading the cache tensor by hand — it gives you the same realistic BST
data the benchmarks measure against, not synthetic edge cases only:

```python
from tstorch.calculators.torch_features_statistical import Median, Quantile, QuantileParams

from ..utils.benchmark_utils import BenchmarkSeriesData
from ..utils.feature_specs import DOMAIN_FEATURES
from ..utils.unit_test_utils import assert_feature_matches_tsfresh

FEATURES = {spec.name: spec for spec in DOMAIN_FEATURES["statistical"]}


def test_median_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    assert_feature_matches_tsfresh(Median, FEATURES["median"], canonical_trajectories)


def test_quantile_matches_tsfresh(
    canonical_trajectories: BenchmarkSeriesData,
) -> None:
    params: list[QuantileParams] = [{"q": q} for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]]
    assert_feature_matches_tsfresh(
        Quantile, FEATURES["quantile"], canonical_trajectories, params
    )
```

If the cache file the fixture loads doesn't exist yet, running the test (or the relevant
benchmark) once builds and persists it (`conftest.py`'s `load_or_build_cache`) — don't hand-roll
a second trajectory generator or load the `.pt` file directly.

Test both CPU and GPU (skip GPU with `pytest.mark.skipif(not torch.cuda.is_available())`) since
the calculator must produce identical results on both — GPU reductions can differ in tie-breaking
or floating-point summation order from CPU, so use `torch.allclose(..., atol=1e-5)` for
cross-device comparisons rather than exact equality. If a specific feature is ill-conditioned in
a way that genuinely needs a looser cross-device tolerance (e.g. dividing by a near-zero `std` on
a near-constant state amplifies ordinary reduction-order differences), override the tolerance for
that one feature name with a comment explaining *why* — don't loosen the shared default, and
don't loosen it to hide an actual bug (first confirm with a double-precision CPU reference that
the "true" value sits between the two, not off to one side).

`assert_feature_matches_tsfresh` handles NaN values, output shape checks, every parameter column,
and every canonical trajectory for both feature base classes.

## Recording torch.compile(fullgraph=True) safety

*Consulted during steps 4 and 5 of "Instructions" — both construct `FeatureClass(params)`
directly, so neither needs the feature wired into `TorchFeatureExtractor` first (unlike
benchmarks in step 7, which do).*

Every feature you rework (or add) needs a unit test in `packages/tstorch/tests/unit/test_<domain>.py`
that confirms its calculator is `torch.compile(fullgraph=True)`-safe — that Dynamo can trace the
*whole* function into one graph with no graph breaks, not just that eager-mode output is correct.
`fullgraph=True` turns a graph break into a hard error instead of Dynamo silently falling back to
a partial-graph/eager mix — see
[`torch.compile`'s docs](https://docs.pytorch.org/docs/2.12/generated/torch.compile.html) for
`fullgraph` and the `options={"trace.enabled": True}` debug flag (dumps the Dynamo/Inductor trace
under `pybasin/torch_compile_debug/` — gitignored — so a break can be diagnosed from the dumped
graph/guards instead of just the exception message).

Call `assert_feature_compiles_fullgraph(feature_name, FeatureClass, params, y_bst)` from
`utils/unit_test_utils.py` (see "Unit test utilities vs benchmark utilities" for why it lives
there, not in `benchmark_utils.py`). It constructs `FeatureClass(params)` directly — the exact
instance `TorchFeatureExtractor` will compile once the feature is wired into `SUPPORTED_FEATURES`
(see "Wire it into TorchFeatureExtractor" below) — and needs no registry lookup, so it works
before that wiring exists at all. It builds `torch.compile(instance, fullgraph=True,
options={"trace.enabled": True})`, runs it once on `y_bst`, records pass/fail plus the exception
message via `compile_recording.record_compile` (so it shows up in `report.md`'s
"torch.compile(fullgraph=True) safety" table even on failure -- see "Benchmark one domain" for
how that recording reaches `report.md`), resets Dynamo's cache afterwards (so this attempt can't
leak guards into an unrelated test), then asserts. One call per feature is enough — unlike
precision, this isn't checking numerical values row-by-row, just whether Dynamo can trace the
function at all, so there's no need to loop over every cached trajectory case the way the
precision test does:

```python
from ..utils.synthetic_feature import Scale  # replace with the real Feature class under test
from ..utils.unit_test_utils import (
    assert_feature_compiles_fullgraph,
    assert_feature_exports_cleanly,
)

def test_agg_autocorrelation_compiles_fullgraph(canonical_trajectories) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_compiles_fullgraph(
        "agg_autocorrelation", AggAutocorrelation, [{"maxlag": 10, "f_agg": "mean"}], y_bst
    )

def test_agg_autocorrelation_exports_cleanly(canonical_trajectories) -> None:
    y_bst = next(iter(canonical_trajectories.case_tensors.values()))
    assert_feature_exports_cleanly(
        "agg_autocorrelation", AggAutocorrelation, [{"maxlag": 10, "f_agg": "mean"}], y_bst
    )
```

`assert_feature_exports_cleanly(feature_name, FeatureClass, params, y_bst)` constructs
`FeatureClass(params)` the same way, wrapped in a thin `nn.Module` shim (`torch.export` requires
`nn.Module` at the top level; the `Feature` class itself does not become one) and calls
`torch.export.export(shim, (y_bst,))`. On failure, the raised `AssertionError` carries a
`torch.export.draft_export()` report listing every graph break in one pass instead of just the
first; if `draft_export` itself can't produce a report, the message falls back to the original
`torch.export.export()` error instead of losing it.

`scripts/benchmark/benchmark-feature.sh` wires `assert_feature_compiles_fullgraph`'s recording up
end-to-end: it passes `--compile-json=...` alongside `--precision-json=...` to the same unit-test
run (both are dumped by `unit/conftest.py`'s `pytest_sessionfinish`, no separate invocation
needed), then upserts the compile-safety JSON into `store.json` via `tstorch_bench_store.py
upsert-compile` before regenerating `report.md`. `assert_feature_exports_cleanly` is not wired
into that script — its result is visible only in pytest's own output (see "Instructions" step 8
for the exact command to run it directly).

Apply compile and export checks to every feature in the domain migration. Record any graph break
as a **NO** row, investigate its control flow, and capture the confirmed cause in
`references/performance-notes.md`.

## Wire it into TorchFeatureExtractor

*Consulted during step 6 of "Instructions" — must happen before step 7 (the benchmarks need the
feature wired in; the compile/export-safety tests in steps 4-5 don't).*

`torch_feature_extractor.py`'s `_compile()`/`extract_features()` already target the end-state
described by this section — there is no adapter step, no intermediate `FeatureFactory` shape to
satisfy. `SUPPORTED_FEATURES: dict[str, type[Feature[Any]] | type[ParameterlessFeature]]` maps a
feature name straight to its class; `_compile()` constructs it via
`tstorch.feature_factory.construct_feature(feature_cls, params_or_none)` (the instance itself is
the closure composed into `torch.cat([f(y) for f in closures], dim=-1)`) and reads `.names`
directly — no `build`/`param_key`/`name` indirection anywhere in this file.

Wiring a feature in is therefore just:

```python
# torch_feature_extractor.py
from tstorch.calculators.torch_features_<domain> import NewFeatureClass

SUPPORTED_FEATURES: dict[str, type[Feature[Any]] | type[ParameterlessFeature]] = {
    ...,
    "feature_name": NewFeatureClass,
}
```

Add the complete domain's `{"name": Class}` mapping to `SUPPORTED_FEATURES` in one wiring pass.
This is a single shared dict in a single shared file, so serialize wiring changes across parallel
agents working on different domains (see "Scope for parallel agents").

**Non-negotiable**: `TorchFeatureExtractor` produces exactly one fused kernel per configuration —
every requested feature's output concatenated with `torch.cat` and the whole thing wrapped in a
single `torch.compile` call — on physical BST input. Every feature you migrate is checked, before
and after, against `report.md`'s existing numbers via the same `benchmark-feature.sh` workflow the
rest of this skill uses (see "Instructions" steps 1 and 8) — if a migrated feature's numbers move,
that's a regression to fix, not a shape change to explain away.

### `DEPRECATED_`-prefixed machinery in `feature_factory.py`

`feature_factory.py` still carries `DEPRECATED_FeatureFactory` (a `NamedTuple` of `build`/
`param_key`/`name` callables) and `DEPRECATED_parameter_free_factory`. **Neither is consumed by
`TorchFeatureExtractor`.** They exist only because some not-yet-migrated calculator files still
build their own module-level `dict[str, DEPRECATED_FeatureFactory]` (e.g.
`AUTOCORRELATION_FEATURE_FACTORIES` before this domain was migrated) — a dict nothing outside that
file imports or reads. After every calculator in the domain has a `Feature` class and a
`SUPPORTED_FEATURES` entry, delete the calculator file's
`dict[str, DEPRECATED_FeatureFactory]`, `_build` helpers, name adapters, and public function
facades. Construct and wire each class directly.

Once no calculator file builds a `dict[str, DEPRECATED_FeatureFactory]` anymore, delete
`DEPRECATED_FeatureFactory`/`DEPRECATED_parameter_free_factory` from `feature_factory.py` too — a
final, separate cleanup pass across the whole package, not something to do mid-migration for one
domain.
