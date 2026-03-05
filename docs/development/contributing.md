# Contributing

## Development Setup

pybasin uses [uv](https://docs.astral.sh/uv/) for dependency management. After cloning the repository, a single command installs everything:

```bash
git clone https://github.com/adrianwix/pybasin.git
cd pybasinWorkspace
uv sync --all-groups
source .venv/bin/activate
```

This creates a virtual environment with all dependency groups -- dev tools, documentation, case studies, and experiments -- so every part of the codebase is immediately usable.

## Repository Structure

The workspace is a monorepo containing two publishable packages:

- **pybasin** (`src/pybasin/`) -- the main basin stability library
- **zigode** (`src/zigode/`) -- a Zig-compiled native ODE solver (separate `pyproject.toml`)

Both are managed under a single [uv workspace](https://docs.astral.sh/uv/concepts/workspaces/), sharing one lockfile while remaining independently publishable.

Alongside these packages live several directories that are not published:

- `thesis_utils/` -- thesis-specific plotting utilities (depend on `cpsmehelper`)
- `case_studies/` -- research case studies validating pybasin against MATLAB bSTAB
- `benchmarks/` -- performance benchmarks
- `tests/` -- unit and integration tests
- `docs/` -- MkDocs documentation source

## Dependency Management

Core runtime dependencies are declared in `[project.dependencies]` in the root `pyproject.toml`. Keep these minimal -- only packages needed for `import pybasin` to work without errors.

Optional features live under `[project.optional-dependencies]`:

| Extra         | Packages                              | Purpose                   |
| ------------- | ------------------------------------- | ------------------------- |
| `jax`         | jax, diffrax                          | `JaxSolver`               |
| `interactive` | plotly, dash, dash-mantine-components | `InteractivePlotter`      |
| `tsfresh`     | tsfresh                               | `TsfreshFeatureExtractor` |
| `nolds`       | nolds                                 | `NoldsFeatureExtractor`   |
| `torchode`    | torchode                              | `TorchOdeSolver`          |
| `all`         | all of the above                      | Everything                |

Development-only tools (ruff, pyright, pytest, type stubs, etc.) go in `[dependency-groups]` under `dev`, `docs`, `case-studies`, or `experiments`. These are never installed by end users.

To add a new runtime dependency:

```bash
uv add <package>            # core dependency
uv add --dev <package>      # dev-only dependency
```

Never edit `pyproject.toml` manually for adding dependencies.

### Adding an Optional Feature

When a new feature depends on a package that should not be required for all users:

1. Add the package to a new or existing extra in `[project.optional-dependencies]`
2. Guard the import with `try/except ImportError` in the module (see `nolds_feature_extractor.py` for the pattern)
3. Raise a clear `ImportError` with the install command if the user tries to use the feature without the package
4. Add the extra to the `all` group

## Code Quality

Run all checks (linter, formatter, type checker) in one command:

```bash
sh scripts/ci.sh
```

This executes ruff lint, ruff format, and pyright across the codebase. All three must pass before merging.

## Testing

```bash
uv run pytest                          # all tests
uv run pytest tests/unit/              # unit tests only
uv run pytest tests/integration/       # integration tests only
uv run pytest --cov=src/pybasin        # with coverage
```

Integration tests compare pybasin results against the original MATLAB bSTAB implementation.

## Adding a New Case Study

1. Create a new directory under `case_studies/`
2. Define the ODE system (subclass `ODESystem` or `JaxODESystem`)
3. Create a setup function returning sampler, solver, and ODE system configuration
4. Write a main script that runs `BasinStabilityEstimator.estimate_bs()`
5. Add a corresponding integration test under `tests/integration/`
6. Document the case study under `docs/case-studies/`
