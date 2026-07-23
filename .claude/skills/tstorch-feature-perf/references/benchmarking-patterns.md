# Benchmarking Patterns

Use this reference before creating, renaming, or interpreting benchmark functions in
`packages/tstorch/tests/benchmarks/calculators/test_<domain>.py`.

## Vocabulary

- `direct_call` bypasses the production extractor interface.
  - tstorch: construct/call the raw `Feature` class.
  - tsfresh: call the raw calculator function through `execute_tsfresh_direct`.
- `proxy_call` uses the production interface.
  - tstorch: `TorchFeatureExtractor`.
  - tsfresh: `extract_features`.
- `compiled` means the tstorch callable for that row is compiled and warmed before the measured
  run, except `compiled_cold_run`.
- `uncompiled` means the tstorch callable for that row is not compiled and is not warmed.
- `cold_run` appears exactly once, on row 4, and measures first-time compile/build cost.
- tsfresh has no compiled/uncompiled/cold-run axis. When a row only differs on that tstorch axis,
  render the tsfresh cell by reusing the measured tsfresh row for the same scale and mechanism.
  This is a report-rendering rule. `store.json` may carry metadata that points at the reused row,
  but do not duplicate benchmark timing records there.
- For batched-argument rows, research tsfresh's raw source before deciding what the tsfresh side
  means. `fctype` is only the dispatcher shape: in
  `feature_extraction.extraction._do_extraction_on_chunk`, `fctype == "combiner"` receives the
  whole `parameter_list`, while other calculators loop once per parameter dict with
  `func(x, **param)`. That is not enough by itself: each combiner calculator's body must also be
  read to see whether it shares work across the parameter list or simply loops internally. Row 7
  must record the researched mechanism as benchmark metadata via pytest-benchmark `extra_info`,
  and the renderer must display it. Row 8 reuses row 7's timing and batching metadata for the
  same count.

## Current Inventory

| # | Function | tstorch mechanism | tsfresh mechanism | Scale | Backend axis | Extractor build/compile state | Param-count axis | tsfresh argument batching | tsfresh cell source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `test_single_trajectory_direct_call_uncompiled_<domain>` | `direct_call`: raw `Feature` class construction+call, not compiled | `direct_call`: raw calculator call through `execute_tsfresh_direct` | single trajectory, canonical cases, each run once | `tstorch_cpu` / `tstorch_gpu` / `tsfresh` | no extractor, uncompiled | fixed, usually one params dict from `spec.kwargs` | n/a | measures its own tsfresh value |
| 2 | `test_single_trajectory_direct_call_compiled_<domain>` | `direct_call`: raw `Feature` class construction+call, compiled and warmed | none measured for this row | single trajectory, canonical cases, each run once | `tstorch_cpu` / `tstorch_gpu`, with tsfresh displayed by reuse | no extractor, compiled | fixed, usually one params dict from `spec.kwargs` | n/a | reuse row 1 at render time |
| 3 | `test_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_<domain>` | `direct_call`: raw `Feature` class construction+call, not compiled | `proxy_call`: `extract_features`, tsfresh's real batched entry point | batched trajectories, 10k grid | `tstorch_cpu` / `tstorch_gpu` / `tsfresh` | no tstorch extractor, uncompiled | fixed, usually one params dict from `spec.kwargs` | n/a | measures its own tsfresh value |
| 4 | `test_batched_trajectories_proxy_call_compiled_cold_run_<domain>` | `proxy_call`: fresh `TorchFeatureExtractor`, compiled during the measured round | none measured for this row | batched trajectories, 10k grid | `tstorch_cpu` / `tstorch_gpu`, with tsfresh displayed by reuse | cold, compiled | fixed, usually one params dict from `spec.kwargs` | n/a | reuse row 3 at render time |
| 5 | `test_batched_trajectories_proxy_call_compiled_<domain>` | `proxy_call`: `TorchFeatureExtractor` built once, warmed, compiled, reused | none measured for this row | batched trajectories, 10k grid | `tstorch_cpu` / `tstorch_gpu`, with tsfresh displayed by reuse | warm, compiled | fixed, usually one params dict from `spec.kwargs` | n/a | reuse row 3 at render time |
| 6 | `test_batched_trajectories_proxy_call_uncompiled_<domain>` | `proxy_call`: `TorchFeatureExtractor` built once, reused with `use_compile=False`, no warmup | none measured for this row | batched trajectories, 10k grid | `tstorch_cpu` / `tstorch_gpu`, with tsfresh displayed by reuse | uncompiled, no warmup | fixed, usually one params dict from `spec.kwargs` | n/a | reuse row 3 at render time |
| 7 | `test_batched_trajectories_proxy_call_uncompiled_batched_arguments_<domain>` | `proxy_call`: `TorchFeatureExtractor` built once, reused with `use_compile=False`, no warmup | `proxy_call`: `extract_features`, matched parameter count | batched trajectories, 10k grid | `tstorch_cpu` / `tstorch_gpu` / `tsfresh` | uncompiled, no warmup | varies: counts 1 / 5 / 10 | required researched signal: proxy loops per argument, calculator gets param list and shares work, or calculator gets param list but loops arguments | measures its own tsfresh value per count |
| 8 | `test_batched_trajectories_proxy_call_compiled_batched_arguments_<domain>` | `proxy_call`: warmed `TorchFeatureExtractor`, compiled and reused | none measured for this row | batched trajectories, 10k grid | `tstorch_cpu` / `tstorch_gpu`, with tsfresh displayed by reuse | warm, compiled | varies: counts 1 / 5 / 10 | reuse row 7's signal for the same count | reuse row 7 at render time for the same count |

The row 3 name is intentionally verbose because tstorch and tsfresh use different mechanisms in
that row. Do not shorten it to a generic `direct_call` or `proxy_call` name.

## Helper Mapping

Use helpers from `packages/tstorch/tests/utils/benchmark_utils.py` with names matching each
row's mechanism and state. If a helper does not exist yet, add or rename it to match the row
semantics instead of hiding a mechanism mismatch behind a shorter test name.

For rows 7 and 8, parametrize over `count in [1, 5, 10]` and build `params` with a private
`_feature_params_for_count(feature_name, count) -> list[dict[str, Any]]` in the domain benchmark
file. Do not put `tsfresh` in those test names; `backend` already covers tsfresh.

For row 7's tsfresh backend, record `tsfresh_argument_batching` in pytest-benchmark `extra_info`
from explicit `FeatureSpec` metadata. Fill that metadata only after reading the raw tsfresh
dispatcher and calculator body. The renderer must use that metadata; do not infer from the mere
presence of `tsfresh_param`, and do not treat `fctype="combiner"` as proof of shared computation.

For rows 7 and 8, record the measured speedup at a specific argument count in `FEATURE_NOTES`
(`scripts/benchmark/tstorch_bench_store.py`) when tstorch batches something tsfresh itself does
not.
