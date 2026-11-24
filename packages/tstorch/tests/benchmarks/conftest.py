# pyright: basic
"""Shared fixtures for tstorch pytest-benchmark suites."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from ..utils.benchmark_utils import BenchmarkSeriesData, load_or_build_cache


def _build_trajectories(builder_name: str) -> BenchmarkSeriesData:
    trajectory_builders = importlib.import_module("..utils.trajectory_builders", __package__)
    builder = getattr(trajectory_builders, builder_name)
    return builder()


def pytest_configure(config: pytest.Config) -> None:
    """Register benchmark markers used by this suite."""
    config.addinivalue_line("markers", "tstorch_benchmark: tstorch feature benchmark")


@pytest.fixture(scope="session")
def benchmark_cache_dir() -> Path:
    """Return the repo-local benchmark trajectory cache directory."""
    return Path("benchmarks/tstorch/cache")


@pytest.fixture(scope="session")
def canonical_trajectories(benchmark_cache_dir: Path) -> BenchmarkSeriesData:
    """Return cached canonical supervised template trajectories."""
    cache_path = benchmark_cache_dir / "canonical_supervised_bounded.pt"
    return load_or_build_cache(
        cache_path,
        lambda: _build_trajectories("build_canonical_trajectories"),
    )


@pytest.fixture(scope="session")
def pendulum_grid_trajectories(benchmark_cache_dir: Path) -> BenchmarkSeriesData:
    """Return cached 10,000-trajectory pendulum grid data."""
    cache_path = benchmark_cache_dir / "pendulum_grid_10000.pt"
    return load_or_build_cache(
        cache_path,
        lambda: _build_trajectories("build_pendulum_grid_trajectories"),
    )
