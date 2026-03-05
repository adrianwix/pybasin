# Releasing

PyBasin follows [Semantic Versioning](https://semver.org/) and publishes to [PyPI](https://pypi.org/project/pybasin/) via two GitHub Actions workflows. This page covers the branching strategy, pre-release testing, and the step-by-step process for shipping a new version.

## Versioning

Versions follow the `MAJOR.MINOR.PATCH` scheme with optional pre-release suffixes:

| Format    | Meaning                                            |
| --------- | -------------------------------------------------- |
| `0.2.0b1` | beta -- published to TestPyPI for consumer testing |
| `0.2.0`   | stable release -- published to PyPI                |

Alpha and release candidate suffixes exist in SemVer but are not used here. Feature branches serve the same purpose as an alpha stage -- code on `feat/xxx` is by definition not ready for release. Release candidates add a formal sign-off step that has no benefit for a solo-maintained library. The version on `main` is always the last stable release; the next version number only appears when a release branch is created.

The version lives in `pyproject.toml` and nowhere else. Bump it with `uv version`:

```bash
uv version --bump patch                # 0.1.2 --> 0.1.3
uv version --bump minor                # 0.1.2 --> 0.2.0
uv version --bump major                # 0.1.2 --> 1.0.0
uv version --bump minor --bump beta    # 0.1.2 --> 0.2.0b1
uv version --bump beta                 # 0.2.0b1 --> 0.2.0b2
uv version --bump stable               # 0.2.0b2 --> 0.2.0
```

## Branching Strategy

The repository uses trunk-based development with short-lived feature branches:

```
main            <- production-ready, protected
feat/xxx        <- feature work
fix/xxx         <- bug fixes
release/x.y.z   <- integration and version bump before a stable release
```

`main` is always releasable. All changes go through pull requests -- direct pushes are blocked by branch protection. Tags, however, are not subject to branch protection and can be pushed directly from the terminal at any time.

## CI Workflows

Two GitHub Actions workflows run automatically.

**`ci.yml`** triggers on every push and pull request to `main`. It runs ruff, pyright, and the unit test suite. Integration tests are excluded because they require a GPU.

**`publish.yml`** triggers on every tag matching `v*`. The pipeline has five jobs that run in sequence:

1. CI -- same checks as `ci.yml`, run again to guard the release
2. Build -- `uv build` produces the wheel and sdist, uploaded as a workflow artifact
3. Check -- detects whether the tag is a beta by matching against the pattern `vX.Y.Z(a|b|rc)N`
4. Publish to TestPyPI -- runs for every tag
5. Publish to PyPI -- skipped automatically for beta tags; requires manual approval in the GitHub Environment for stable releases
6. GitHub Release -- creates a release entry with auto-generated changelog; marked "Pre-release" on GitHub for beta tags

PyPI publishing uses OIDC Trusted Publishers -- no API tokens are stored anywhere.

## Releasing a Single Feature

For a small fix or isolated feature, work directly on a branch and merge it to `main`:

```bash
git checkout -b fix/solver-edge-case
# ... commit work ...
git push origin fix/solver-edge-case
# Open PR -> CI passes -> merge
```

When ready to ship after one or more such merges:

```bash
# Create a release branch
git checkout main && git pull
git checkout -b release/0.1.3
uv version --bump patch          # 0.1.2 --> 0.1.3
git add pyproject.toml
git commit -m "chore: release 0.1.3"
git push origin release/0.1.3
# Open PR -> merge
git checkout main && git pull
git tag v0.1.3
git push --tags
```

The tag push triggers `publish.yml`. After the TestPyPI step completes, approve the `pypi` environment gate in GitHub Actions to push to production PyPI.

## Releasing Multiple Features Together

When several branches are in flight simultaneously, use the release branch as an integration point:

```bash
git checkout main && git pull
git checkout -b release/0.2.0

git merge feat/new-solver
git merge feat/new-clustering
git merge feat/new-plot
```

At this point, cut a beta to test the integrated result:

```bash
uv version --bump minor --bump beta
git add pyproject.toml
git commit -m "chore: bump to 0.2.0b1"
git tag v0.2.0b1
git push origin release/0.2.0 --tags
```

The tag triggers `publish.yml`, which publishes `0.2.0b1` to TestPyPI. Install it in a clean environment to verify:

```bash
pip install --pre pybasin==0.2.0b1 \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/
```

If something is broken, fix it on the release branch and cut another beta:

```bash
# Fix the issue
git commit -m "fix: solver edge case in merged code"
uv version --bump beta
git tag v0.2.0b2
git push origin release/0.2.0 --tags
```

Once the release is stable, bump to the final version and merge to `main`:

```bash
uv version --bump stable
git add pyproject.toml
git commit -m "chore: release 0.2.0"
git push origin release/0.2.0
# Open PR -> merge
git checkout main && git pull
git tag v0.2.0
git push --tags
```

## Approving the PyPI Gate

For stable tags (`v0.2.0`, `v1.0.0`, etc.) the `publish-pypi` job pauses and waits for approval. To approve:

1. Go to the Actions tab on GitHub
2. Open the running `Publish` workflow
3. Click "Review deployments" on the `pypi` environment step
4. Approve

Beta tags skip this step entirely -- the job is automatically skipped by the workflow.

## Setup Prerequisites

The following must be configured in the GitHub repository before the first release:

**GitHub Environments** (`Settings -> Environments`):

- Create `testpypi` -- no required reviewers needed
- Create `pypi` -- add yourself as a required reviewer

**Trusted Publishers** (on both [pypi.org](https://pypi.org) and [test.pypi.org](https://test.pypi.org)):

- Publisher type: GitHub Actions
- Repository: `adrianwix/pybasin`
- Workflow: `publish.yml`
- Environment name: `pypi` (or `testpypi` for TestPyPI)
