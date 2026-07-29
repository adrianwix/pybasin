# Contributing

## Development Setup

pybasin uses [uv](https://docs.astral.sh/uv/) for dependency management. After
cloning the repository, sync the workspace from its root:

```bash
git clone https://github.com/adrianwix/pybasin.git
cd pybasin
uv sync --all-groups --all-extras
source .venv/bin/activate
```

This creates one virtual environment for the workspace packages, development
tools, documentation, case studies, and experiments.

## Repository Structure

The repository root is a uv workspace, not a Python distribution. Each importable
project owns its source and package metadata below `packages/`.

- **pybasin** (`packages/pybasin/`) -- the published basin stability library
- **zigode** (`packages/zigode/`) -- an experimental Zig ODE solver installed for
  local development, without release automation
- **tstorch** (`packages/tstorch/`) -- PyTorch-native time-series features
- **jax_features** (`packages/jax_features/`) -- JAX feature extraction
- **case_studies** (`packages/case_studies/`) -- research systems and validation
  workflows

Workspace-level directories contain shared material: `benchmarks/` and
`experiments/` hold research code, `tests/` owns integration and GPU validation,
and `docs/` contains the MkDocs site. Pybasin's unit tests live beside the package
in `packages/pybasin/tests/`.

## Dependency Management

Pybasin runtime dependencies are declared in
`packages/pybasin/pyproject.toml`. Keep the core list limited to packages needed
for `import pybasin`.

Optional features live under `[project.optional-dependencies]`:

| Extra         | Packages                              | Purpose                   |
| ------------- | ------------------------------------- | ------------------------- |
| `jax`         | jax, diffrax                          | `JaxSolver`               |
| `interactive` | plotly, dash, dash-mantine-components | `InteractivePlotter`      |
| `tsfresh`     | tsfresh                               | `TsfreshFeatureExtractor` |
| `nolds`       | nolds                                 | `NoldsFeatureExtractor`   |
| `torchode`    | torchode                              | `TorchOdeSolver`          |
| `all`         | all of the above                      | All optional backends     |

Development tools belong to the dependency groups in the root
`pyproject.toml`. Add pybasin dependencies through uv:

```bash
uv add --package pybasin <package>
uv add --package pybasin --optional <extra> <package>
```

Do not place development-only tools in the published package metadata.

### Adding an Optional Feature

When a feature depends on a package that should not be installed for every user:

1. Add the package to a new or existing optional dependency group.
2. Guard its import at the point where the optional feature is loaded.
3. Raise an `ImportError` that includes the appropriate installation command.
4. Add the extra to `all` when it is part of the supported complete installation.

## Code Quality

Run the repository checks with:

```bash
sh scripts/ci.sh
```

The script syncs the workspace, runs Ruff, and checks both the main and plotter
Pyright configurations.

## Testing

From the workspace root, use:

```bash
uv run pytest                                      # pybasin unit, integration, and GPU collection
uv run pytest packages/pybasin/tests/              # pybasin unit tests
uv run pytest tests/integration/                    # case-study integration tests
uv run pytest tests/gpu/                            # GPU-only tests
uv run pytest --cov=packages/pybasin/src/pybasin   # pybasin coverage
```

Integration tests compare pybasin results with the original MATLAB bSTAB
implementation. GPU tests require CUDA and cuML.

## Adding a New Case Study

1. Create the system under `packages/case_studies/src/case_studies/`.
2. Define the ODE system by subclassing the appropriate solver base.
3. Add a setup function that returns the sampler, solver, and system configuration.
4. Write the runnable study entry point.
5. Add its integration test under `tests/integration/`.
6. Document the study under `docs/case-studies/`.

## Building pybasin

Only pybasin participates in the release pipeline. Build its wheel and source
distribution explicitly:

```bash
uv build --package pybasin --out-dir dist
```

The root workspace and zigode are never selected by the publishing workflow. See
the [releasing guide](releasing.md) for versioning, TestPyPI, and PyPI steps.
