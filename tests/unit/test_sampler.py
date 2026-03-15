from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest
import torch

from pybasin import set_seed
from pybasin.sampler import CsvSampler, GaussianSampler, GridSampler, UniformRandomSampler


class SamplerParams(TypedDict):
    min_limits: list[float]
    max_limits: list[float]
    device: str


@pytest.fixture
def sampler_params() -> SamplerParams:
    return {
        "min_limits": [0.0, -1.0],
        "max_limits": [1.0, 1.0],
        "device": "cpu",
    }


def test_uniform_sampler_shape(sampler_params: SamplerParams) -> None:
    sampler = UniformRandomSampler(**sampler_params)
    n = 100
    samples = sampler.sample(n)

    # Verify samples has correct shape: 100 samples × 2 dimensions
    assert samples.shape == (n, 2)
    # Confirm samples are on CPU device as requested
    assert samples.device.type == "cpu"
    # Validate data type is float32 for efficiency
    assert samples.dtype == torch.float32


def test_uniform_sampler_bounds(sampler_params: SamplerParams) -> None:
    sampler = UniformRandomSampler(**sampler_params)
    samples = sampler.sample(1000)

    # First dimension samples are within [0.0, 1.0] bounds
    assert torch.all(samples[:, 0] >= 0.0) and torch.all(samples[:, 0] <= 1.0)
    # Second dimension samples are within [-1.0, 1.0] bounds
    assert torch.all(samples[:, 1] >= -1.0) and torch.all(samples[:, 1] <= 1.0)


def test_uniform_sampler_seed_reproducibility(sampler_params: SamplerParams) -> None:
    sampler = UniformRandomSampler(**sampler_params)

    set_seed(42)
    samples1 = sampler.sample(100)
    set_seed(42)
    samples2 = sampler.sample(100)

    # Same seed produces identical samples (reproducibility)
    assert torch.allclose(samples1, samples2)


def test_uniform_sampler_no_seed_non_deterministic(sampler_params: SamplerParams) -> None:
    sampler = UniformRandomSampler(**sampler_params)

    samples1 = sampler.sample(100)
    samples2 = sampler.sample(100)

    # Without a fixed seed, two consecutive calls should differ
    assert not torch.allclose(samples1, samples2)


def test_grid_sampler_coverage(sampler_params: SamplerParams) -> None:
    sampler = GridSampler(**sampler_params)
    samples = sampler.sample(100)

    # Grid has correct number of dimensions (2)
    assert samples.shape[1] == 2
    # Grid has at least 100 points (may have more due to ceiling operation)
    assert samples.shape[0] >= 100


def test_gaussian_sampler_distribution(sampler_params: SamplerParams) -> None:
    sampler = GaussianSampler(**sampler_params, std_factor=0.2)
    samples = sampler.sample(1000)

    mean = samples.mean(dim=0)
    expected_mean = torch.tensor([0.5, 0.0])
    # Sample mean ≈ [0.5, 0.0] (midpoint between bounds)
    assert torch.allclose(mean, expected_mean, atol=0.1)


class TestCsvSampler:
    """Tests for CsvSampler."""

    @pytest.fixture
    def csv_file(self, tmp_path: Path) -> Path:
        """Create a temporary CSV file for testing."""
        csv_content = """x1,x2,label
1.0,2.0,A
3.0,4.0,B
5.0,6.0,A
7.0,8.0,B
"""
        csv_path = tmp_path / "test_samples.csv"
        csv_path.write_text(csv_content)
        return csv_path

    def test_csv_sampler_loads_data(self, csv_file: Path) -> None:
        sampler = CsvSampler(
            csv_file, coordinate_columns=["x1", "x2"], label_column="label", device="cpu"
        )
        samples = sampler.sample()

        assert samples.shape == (4, 2)
        assert samples.dtype == torch.float32

    def test_csv_sampler_correct_values(self, csv_file: Path) -> None:
        sampler = CsvSampler(
            csv_file, coordinate_columns=["x1", "x2"], label_column="label", device="cpu"
        )
        samples = sampler.sample()

        expected = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32
        )
        assert torch.allclose(samples, expected)

    def test_csv_sampler_labels(self, csv_file: Path) -> None:
        sampler = CsvSampler(
            csv_file, coordinate_columns=["x1", "x2"], label_column="label", device="cpu"
        )

        assert sampler.labels is not None
        np.testing.assert_array_equal(sampler.labels, ["A", "B", "A", "B"])

    def test_csv_sampler_no_label_column(self, csv_file: Path) -> None:
        sampler = CsvSampler(csv_file, coordinate_columns=["x1", "x2"], device="cpu")

        assert sampler.labels is None

    def test_csv_sampler_n_samples(self, csv_file: Path) -> None:
        sampler = CsvSampler(
            csv_file, coordinate_columns=["x1", "x2"], label_column="label", device="cpu"
        )

        assert sampler.n_samples == 4

    def test_csv_sampler_partial_sample(self, csv_file: Path) -> None:
        sampler = CsvSampler(
            csv_file, coordinate_columns=["x1", "x2"], label_column="label", device="cpu"
        )
        samples = sampler.sample(n=2)

        assert samples.shape == (2, 2)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        assert torch.allclose(samples, expected)

    def test_csv_sampler_exceeds_available(self, csv_file: Path) -> None:
        sampler = CsvSampler(
            csv_file, coordinate_columns=["x1", "x2"], label_column="label", device="cpu"
        )

        with pytest.raises(ValueError, match="Requested 10 samples"):
            sampler.sample(n=10)

    def test_csv_sampler_custom_columns(self, tmp_path: Path) -> None:
        csv_content = """a,b,c,label
1.0,2.0,3.0,A
4.0,5.0,6.0,B
"""
        csv_path = tmp_path / "custom_cols.csv"
        csv_path.write_text(csv_content)

        sampler = CsvSampler(csv_path, coordinate_columns=["a", "c"], device="cpu")
        samples = sampler.sample()

        assert samples.shape == (2, 2)
        expected = torch.tensor([[1.0, 3.0], [4.0, 6.0]], dtype=torch.float32)
        assert torch.allclose(samples, expected)

    def test_csv_sampler_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            CsvSampler("/nonexistent/path.csv", coordinate_columns=["x1", "x2"])

    def test_csv_sampler_missing_columns(self, csv_file: Path) -> None:
        with pytest.raises(ValueError, match="Columns .* not found"):
            CsvSampler(csv_file, coordinate_columns=["x1", "missing"], device="cpu")

    def test_csv_sampler_limits_from_data(self, csv_file: Path) -> None:
        sampler = CsvSampler(csv_file, coordinate_columns=["x1", "x2"], device="cpu")

        min_limits_list: list[float] = sampler.min_limits.tolist()  # type: ignore[reportUnknownMemberType]
        max_limits_list: list[float] = sampler.max_limits.tolist()  # type: ignore[reportUnknownMemberType]
        assert min_limits_list == [1.0, 2.0]
        assert max_limits_list == [7.0, 8.0]
