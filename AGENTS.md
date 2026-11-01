# Repository conventions

## Prime directive: API consistency is not negotiable

This project builds APIs meant for wide use (currently: the tstorch `Feature`/
`ParameterlessFeature` calculator API, used by every consumer of `TorchFeatureExtractor` and
every direct caller of a feature class). The primary job on any API surface is not "does this
task's code work" - it's "does the same member mean the same kind of thing for every variant of
this API, regardless of which internal class/branch/path produced it." An oversight where that
isn't true must never reach the user - catch it while designing or reviewing the change, not
after it ships.

A property or method that returns a structurally different *kind* of value depending on an
invisible implementation detail (which base class was inherited, which internal branch fired) is
a defect in the API, not a documentation gap and not a style nit. Concretely:

- Before shipping any shared interface (a protocol, a base class, a property every variant
  implements), check every variant against the identical contract: would a caller who does not
  know the internal class hierarchy be surprised by what they get back from any one of them? If
  yes, the contract itself is wrong.
- If explaining a rule - in a comment, docstring, or skill file - requires an "except when X, in
  which case Y behaves differently" carve-out for what should be one concept, that is a signal to
  fix X in the code, not to write clearer prose around it. Prose is never a substitute for a
  consistent contract.
- Never rationalize an inconsistency as intentional in order to finish the current task faster.
  Stop, name it explicitly, and either fix it or ask before proceeding - do not paper over it and
  move on.

## Comments

Never write comments or docstrings that narrate a change: what a piece of code used to
do, that something is deprecated, that a name is destined for deletion, or how an old
shape differed from the new one. When removing or replacing code, delete it outright and
say nothing about its history in the source. Never write a deprecation comment/notice
unless the user explicitly asks for one. A comment only earns its place by explaining a
non-obvious WHY a future reader needs to safely change the code going forward - a real
invariant, a subtle bug workaround, a constraint imposed from outside the file - never a
changelog of what was tried, removed, or is pending removal.

Never reference internal skills (e.g. `.claude/skills/...`, "see the X skill") from source
code comments or docstrings. Skills are a workflow tool for the assistant, not part of the
codebase's own documentation - a reader of the code has no access to them and shouldn't
need to.

A comment must describe the code's current behavior, never compare it to a prior version.
Phrases like "still works", "keeps working", "no longer accepts", "used to be", "now does
X instead of Y" all encode a diff against a state the reader cannot see - read 500 commits
from now, they're either meaningless or actively misleading, because the "before" they're
implicitly contrasted with is gone. Write what is true today, full stop:

- Bad: "`normalize` and `impute_method` keep working from the caller's perspective even
  though the wrapped extractor no longer accepts them."
- Still bad: "`normalize` and `impute_method` are handled by this adapter, not forwarded
  to the wrapped extractor." Naming what *doesn't* happen, or what a different component
  doesn't do, is still an implicit comparison - the reader has to know the wrapped
  extractor's contract to make sense of the negation. State only what this code does.
- Good: "`impute_method` selects the imputation strategy; when `normalize` is set, a
  `StandardScaler` is fit on the first batch and reused for subsequent calls."

This applies doubly to public APIs, where the comment is often the only contract a caller
sees. For internal implementation details, it's fine - and often necessary - to explain a
non-obvious choice with a "this looks odd, but here's why" comment (e.g. "using X instead
of the more obvious Y because Z"); that may reference a past constraint or incident as the
*reason* for the current behavior, but the behavior itself must still be described in the
present tense, not as a delta from what it replaced.

## Verifying changes

After any non-trivial change - a new module, a refactor touching multiple files, moving
code between files - run `scripts/ci.sh` before considering the change done. It syncs the
workspace, runs ruff (lint + format), and runs pyright across the main project, plotters,
and benchmarks. Fix everything it reports, including type errors introduced by moving code
between modules (e.g. a variable that's only bound inside a `try` block and used after it
unconditionally) - these are real bugs, not lint noise to silence.

`uv run pyright` on a `.ipynb` path (directly or via `include`) does not actually
type-check cell bodies in this pinned pyright version - even a trivial unannotated
function in a fresh notebook produces zero diagnostics. A clean `pyright` run on a
notebook is not evidence the code is clean. To really check a notebook, concatenate its
code cells' `source` (in order) into a `.py` file and run `pyright` on that instead.

## Test and benchmark taxonomy

There are exactly three categories of test/measurement code in this repo. Before adding
a new test or benchmark, decide which category it is - the category determines where the
file lives, not the other way around.

- **unit / functional** (`packages/<pkg>/tests/`): does one function/class behave
  correctly in isolation - shapes, edge cases, error handling, API contracts. No external
  reference implementation, no ground truth file, runs in milliseconds. This is the only
  category tstorch currently has; `packages/tstorch/tests/` mixes unit and accuracy
  concerns in one tree and is a known gap, not a pattern to copy into other packages.
- **numerical accuracy** (`tests/integration/` and `packages/<pkg>/tests/accuracy/`):
  validates that pybasin's *output* matches an independent reference - a published paper
  result, another tool's ground truth, or (once a result is trusted enough to stop
  re-deriving it externally) a frozen self-consistency snapshot. Always goes through
  `compute_classification_metrics`/MCC or F1, never a bare `np.isclose` on a single number
  unless there genuinely is only one number to check.
- **benchmarks / performance** (`benchmarks/` and `packages/<pkg>/benchmarks/`): timing
  only, never a correctness check on its own - each benchmark script still validates its
  own output against a known-good fraction/label before recording a time, but a benchmark
  test file is not where you add new correctness coverage.

### Root-level vs. package-level

The same three categories exist at two scopes, and the scope is what decides the
directory, not which package happens to own the case study's source files:

- **Root-level** (`tests/integration/<case>/`, `benchmarks/<suite>/`): the case study
  itself is the object under test - "does pybasin-the-tool reproduce this published
  physical result," compared against an external, independent tool (MATLAB bSTAB
  historically for `duffing`/`lorenz`/`pendulum`/`friction`) or the values published with
  the system (Menck et al. 2013 for `rossler_network`). These suites routinely span more
  than one workspace package (`packages/case_studies` for the system, `packages/pybasin`
  for the estimator, plus a third-party tool) and get their own `Project.toml`/`Manifest.toml` (Julia) or
  equivalent when they need one, plus a `compare_all.py` that merges every
  implementation's results into one table/plot published under
  `packages/pybasin/docs/assets/benchmarks/<suite>/`.
- **Package-level** (`packages/<pkg>/tests/accuracy/<case>/`,
  `packages/<pkg>/benchmarks/<grouper>/`): a *specific capability inside that package* is
  the object under test - e.g. `HistogramClusterer`/`PairwiseComparisonClusterer` against
  Attractors.jl's `GroupViaHistogram`/`GroupViaPairwiseComparison` in
  `packages/pybasin/tests/accuracy/{henon_map,magnetic_pendulum}/`. The case study used to
  exercise it is a vehicle, not the point - if the same capability also has edge cases
  that aren't naturally reachable through any real case study (e.g. an aggregation
  threshold's merge/split behavior), those get their own accuracy test file next to the
  case-study ones (`packages/pybasin/tests/accuracy/pairwise/test_algorithmic_cases.py`),
  not folded into a case study that doesn't naturally cover them. A package-level Julia
  project used by multiple accuracy/benchmark scripts within the same package can be
  shared (`packages/pybasin/tests/accuracy/_julia_project/`) instead of each script
  getting its own.

### CPU and GPU accuracy coverage share one parametrized test

For root integration coverage, an accuracy test that makes the same ground-truth claim
on CPU and GPU belongs in one test body with an explicit execution-target parameter.
Mark only the GPU parameter with
`gpu` and the relevant availability check, and give both parameters stable `cpu` and
`gpu` IDs. This keeps `pytest -m gpu` useful while ensuring that a machine without CUDA
still runs the CPU case instead of skipping the whole test. The shared runner owns device
selection, GPU tolerances, and any reference-coordinate sampling needed for cross-device
comparison.

Artifact-producing tests pass the execution target through to `ArtifactCollector`. CPU
outputs retain the publication filenames referenced by the docs; GPU outputs use the
`_gpu` suffix and cannot overwrite them. CPU-vs-CUDA kernel self-parity for a predictor
is a unit test beside that predictor because it validates the kernel rather than the full
case-study pipeline.

### Shared test helpers cross the root/package boundary through `packages/test_support`

Root `tests/integration/` and every `packages/<pkg>/tests/accuracy/` need the same
primitives (`compute_classification_metrics`, `julia_labels_to_pybasin`,
`requires_gpu_and_cuml`, ...). These live in the `test_support` workspace package, not in
`tests/integration/test_helpers.py` re-imported cross-tree: root `tests/` and each
package's `tests/` are governed by *different* pytest config files (root `pytest.ini` vs.
`packages/<pkg>/pyproject.toml`'s own `[tool.pytest.ini_options]`, picked up independently
of which directory you invoke `pytest` from), so a plain relative import between the two
trees resolves when run from the repo root and silently breaks when a package's tests are
run standalone (`cd packages/pybasin && uv run pytest`). A dependency on a shared
workspace package resolves identically either way. Case-study-specific driver functions
(`run_basin_stability_test`, artifact-generation glue) stay local to the tree that uses
them; only the primitives every tree needs move to `test_support`.

## Maintaining skills

Every skill has a "## Skill structure" section (or equivalently named) in its SKILL.md that
maps each file in the skill to the one kind of content it owns. Before editing any file in a
skill - adding a rule, fixing a rule, adding a note, anything - read that section first, every
time, even for a one-line change. This is not optional and not a one-time read.

1. Read "## Skill structure" (or the skill's equivalent index section) before touching any
   file.
2. From it, identify the single file that owns the category of information you're adding
   (a design rule, a process step, a performance finding, a per-feature history entry,
   whatever categories that skill defines).
3. Write the information there, once. Every other file that touches the same topic gets at
   most a bare pointer already established by that file's own structure (a name, a section
   title) - never a restatement of the rule, its rationale, or its exceptions, and never a
   second copy "just in case." If a file already has its own pointer to the owning section,
   leave it as a pointer; don't expand it into a summary.
4. If "Skill structure" doesn't make it clear which file owns the new information, stop and
   ask before writing anything. Do not guess, and do not write it in two places to be safe -
   duplicated information is the specific failure mode this process exists to prevent.
5. When a rule changes, edit it in its single owning location only. Touch a pointer elsewhere
   only if the pointer itself is now wrong (e.g. a renamed section).
