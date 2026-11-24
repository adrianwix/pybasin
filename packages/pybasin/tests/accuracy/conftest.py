"""Artifact collection for package-level accuracy tests."""

from pathlib import Path

import pytest

from accuracy.artifacts import AccuracyArtifactCollector

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "results"

_artifact_collector: AccuracyArtifactCollector | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the documentation-artifact generation option."""
    parser.addoption(
        "--generate-accuracy-artifacts",
        action="store_true",
        default=False,
        help="Generate numerical-accuracy JSON artifacts for the case-study docs",
    )


@pytest.fixture(scope="session")
def accuracy_artifact_collector(
    request: pytest.FixtureRequest,
) -> AccuracyArtifactCollector | None:
    """Return a session collector when artifact generation is requested."""
    global _artifact_collector
    if request.config.getoption("--generate-accuracy-artifacts"):
        _artifact_collector = AccuracyArtifactCollector(RESULTS_DIR)
    return _artifact_collector


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write artifacts only after a successful accuracy-test session."""
    if exitstatus == 0 and _artifact_collector is not None:
        _artifact_collector.write()
