# tstorch feature benchmarks

These benchmarks compare one feature at a time across:

- `tstorch` CPU
- `tstorch` GPU when CUDA is available
- `tsfresh`

The benchmark suite stores generated trajectory data under `benchmarks/tstorch/cache/`.

## Reporting numbers

Never hand-type a benchmark number into a doc, and never run raw `pytest` and transcribe its
printed table — both are error-prone and drift silently. Use the wrapper script, which runs
the benchmarks, upserts the parsed stats into `packages/tstorch/reports/store.json` (keyed
by feature/variant/backend - other features already recorded are untouched), and regenerates
`packages/tstorch/reports/report.md`:

```bash
scripts/benchmark/benchmark-feature.sh --match <domain>/<feature>   # one feature
scripts/benchmark/benchmark-feature.sh --match <domain>             # every feature in a domain
```

`report.md` and `store.json` are git-tracked (the durable, deterministic source of truth);
raw per-run pytest-benchmark JSON dumps land in `packages/tstorch/reports/raw/` and are
gitignored. See `.github/skills/tstorch-feature-perf/SKILL.md`'s "Reading report.md" for the
reasoning.

## Ad-hoc / whole-suite runs

Run a first one-shot benchmark with no warmups:

```bash
uv run pytest packages/tstorch/tests/benchmarks \
  --benchmark-save=tstorch-feature-extraction \
  --benchmark-storage=file://benchmarks/tstorch/.benchmarks \
  --benchmark-json=benchmarks/tstorch/latest.json
```

The tests call `benchmark.pedantic(..., rounds=1, iterations=1, warmup_rounds=0)`.
The pytest-benchmark CLI options above persist results in both the plugin storage directory
and a single JSON report. This path is for one-off exploration - it does not update
`report.md`/`store.json`.

Run a single domain while iterating:

```bash
uv run pytest packages/tstorch/tests/benchmarks/calculators/test_statistical.py \
  --benchmark-save=tstorch-statistical \
  --benchmark-storage=file://benchmarks/tstorch/.benchmarks
```
