# pyright: basic
"""Shared fixtures for tstorch numerical-accuracy unit tests.

Reuses the same cached trajectories and builders as the benchmark suite
(``tests/utils/benchmark_utils.py`` and ``tests/utils/trajectory_builders.py``)
so unit tests exercise the exact same data the benchmarks measure against.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ..utils.benchmark_utils import BenchmarkSeriesData, load_or_build_cache
from ..utils.compile_recording import dump_compile_records
from ..utils.export_recording import dump_export_records
from ..utils.precision_recording import dump_precision_records


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--precision-json",
        action="store",
        default=None,
        help="Dump accuracy-vs-tsfresh stats recorded via precision_recording to this JSON path.",
    )
    parser.addoption(
        "--compile-json",
        action="store",
        default=None,
        help="Dump torch.compile(fullgraph=True) safety stats recorded via "
        "compile_recording to this JSON path.",
    )
    parser.addoption(
        "--export-json",
        action="store",
        default=None,
        help="Dump torch.export safety stats recorded via export_recording to this JSON path.",
    )


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Write every requested traceability record from this session."""
    precision_path_str = session.config.getoption("--precision-json")
    if precision_path_str:
        dump_precision_records(Path(precision_path_str))
    compile_path_str = session.config.getoption("--compile-json")
    if compile_path_str:
        dump_compile_records(Path(compile_path_str))
    export_path_str = session.config.getoption("--export-json")
    if export_path_str:
        dump_export_records(Path(export_path_str))


@pytest.fixture(scope="session")
def benchmark_cache_dir() -> Path:
    """Return the repo-local benchmark trajectory cache directory."""
    return Path("benchmarks/tstorch/cache")


@pytest.fixture(scope="session")
def canonical_trajectories(benchmark_cache_dir: Path) -> BenchmarkSeriesData:
    """Return cached canonical supervised template trajectories."""
    cache_path = benchmark_cache_dir / "canonical_supervised_bounded.pt"
    if cache_path.exists():

        def missing_builder() -> BenchmarkSeriesData:
            raise FileNotFoundError(cache_path)

        return load_or_build_cache(cache_path, missing_builder)
    trajectory_builders = importlib.import_module("..utils.trajectory_builders", __package__)

    return load_or_build_cache(
        cache_path,
        trajectory_builders.build_canonical_trajectories,
    )
