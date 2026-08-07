# pyright: basic
from pathlib import Path

import numpy as np
import pytest
import torch

from pybasin.regions import BallRegion, BoxRegion, SphereSurfaceRegion
from pybasin.samplers import (
    CartesianProductSampler,
    CsvSampler,
    EvenlySpacedSampler,
    LatinHypercubeSampler,
    SobolSampler,
    TruncatedGaussianSampler,
    UniformRandomSampler,
    UnsupportedSamplerConfigurationError,
    UnsupportedSamplerRegionError,
)

# ---------------------------------------------------------------------------
# UniformRandomSampler
# ---------------------------------------------------------------------------


class TestSamplerDeviceDtypeOverride:
    """`device`/`dtype` passed once to the sampler, overriding the region's own placement."""

    def test_uniform_random_sampler_dtype_override(self) -> None:
        region = BoxRegion([0.0, -1.0], [1.0, 1.0])
        sampler = UniformRandomSampler(region, dtype=torch.float64)
        assert sampler.dtype == torch.float64
        assert sampler.region.dtype == torch.float64
        points = sampler.sample(10)
        assert points.dtype == torch.float64

    def test_evenly_spaced_sampler_dtype_override(self) -> None:
        region = BoxRegion([0.0, -1.0], [1.0, 1.0])
        sampler = EvenlySpacedSampler(region, dtype=torch.float64)
        assert sampler.sample(100).dtype == torch.float64

    def test_no_override_keeps_region_placement(self) -> None:
        region = BoxRegion(
            torch.tensor([0.0, -1.0], dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
        )
        sampler = UniformRandomSampler(region)
        assert sampler.dtype == torch.float64
        assert sampler.sample(10).dtype == torch.float64

    def test_region_itself_is_left_unmodified(self) -> None:
        # The sampler resolves its own (possibly-overridden) region; the region instance
        # passed in is untouched, since BoxRegion is immutable.
        region = BoxRegion([0.0, -1.0], [1.0, 1.0])
        UniformRandomSampler(region, dtype=torch.float64)
        assert region.dtype == torch.float32


class TestUniformRandomSampler:
    def test_box_shape_and_membership(self) -> None:
        region = BoxRegion(torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0]))
        sampler = UniformRandomSampler(region)
        points = sampler.sample(1000, seed=0)
        assert points.shape == (1000, 2)
        assert bool(region.contains(points).all())

    def test_box_fixed_dimension(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 10.0, 0.0]), torch.tensor([10.0, 10.0, 10.0]))
        sampler = UniformRandomSampler(region)
        points = sampler.sample(200, seed=0)
        assert bool((points[:, 1] == 10.0).all())

    def test_box_device_and_dtype(self) -> None:
        region = BoxRegion(
            torch.tensor([0.0, 0.0], dtype=torch.float64),
            torch.tensor([1.0, 1.0], dtype=torch.float64),
        )
        points = UniformRandomSampler(region).sample(10)
        assert points.dtype == torch.float64
        assert points.device == region.min.device

    def test_seed_reproducibility(self) -> None:
        region = BoxRegion(torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0]))
        sampler = UniformRandomSampler(region)
        points_a = sampler.sample(100, seed=42)
        points_b = sampler.sample(100, seed=42)
        assert torch.allclose(points_a, points_b)

    def test_no_seed_non_deterministic(self) -> None:
        region = BoxRegion(torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0]))
        sampler = UniformRandomSampler(region)
        assert not torch.allclose(sampler.sample(100), sampler.sample(100))

    def test_constructor_seed_reproducibility(self) -> None:
        region = BoxRegion(torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0]))
        sampler = UniformRandomSampler(region, seed=42)
        assert torch.equal(sampler.sample(100), sampler.sample(100))

    def test_call_seed_overrides_constructor_seed(self) -> None:
        region = BoxRegion(torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0]))
        sampler = UniformRandomSampler(region, seed=42)
        assert torch.equal(sampler.sample(100, seed=7), sampler.sample(100, seed=7))
        assert not torch.equal(sampler.sample(100), sampler.sample(100, seed=7))

    def test_ball_membership_and_approximately_uniform_by_volume(self) -> None:
        region = BallRegion(torch.zeros(3), radius=1.0)
        sampler = UniformRandomSampler(region)
        points = sampler.sample(20_000, seed=0)
        assert bool(region.contains(points).all())
        # Uniform by volume: P(||x|| <= r/2) should be (1/2)^3 = 0.125.
        radii = torch.linalg.vector_norm(points, dim=-1)
        fraction_inner = (radii <= 0.5).float().mean().item()
        assert pytest.approx(fraction_inner, abs=0.02) == 0.125

    def test_sphere_surface_membership_and_radius(self) -> None:
        region = SphereSurfaceRegion(torch.tensor([1.0, -2.0, 0.5]), radius=2.0)
        sampler = UniformRandomSampler(region)
        points = sampler.sample(5000, seed=0)
        assert bool(region.contains(points).all())
        radii = torch.linalg.vector_norm(points - region.center, dim=-1)
        assert radii.mean().item() == pytest.approx(2.0, abs=1e-4)

    def test_sphere_surface_approximately_uniform_directions(self) -> None:
        region = SphereSurfaceRegion(torch.zeros(3), radius=1.0)
        sampler = UniformRandomSampler(region)
        points = sampler.sample(20_000, seed=0)
        # Uniform-by-area directions average to ~0 in every coordinate.
        assert points.mean(dim=0).abs().max().item() < 0.03


# ---------------------------------------------------------------------------
# EvenlySpacedSampler
# ---------------------------------------------------------------------------


class TestEvenlySpacedSampler:
    def test_box_grid_count_and_fixed_dimension(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 10.0, 0.0]), torch.tensor([10.0, 10.0, 10.0]))
        sampler = EvenlySpacedSampler(region)
        n = 400
        points = sampler.sample(n)
        assert sampler.n_points(n) == points.shape[0]
        assert points.shape[0] >= n
        assert bool((points[:, 1] == 10.0).all())
        assert bool(region.contains(points).all())

    def test_ball_filtered_grid_membership(self) -> None:
        region = BallRegion(torch.zeros(2), radius=1.0)
        sampler = EvenlySpacedSampler(region)
        points = sampler.sample(2000)
        assert points.shape[1] == 2
        assert points.shape[0] <= sampler.n_points(2000)
        assert bool(region.contains(points).all())

    def test_sphere_surface_fibonacci_count_and_radius(self) -> None:
        region = SphereSurfaceRegion(torch.tensor([0.0, 0.0, 0.0]), radius=3.0)
        sampler = EvenlySpacedSampler(region)
        assert sampler.n_points(2000) == 2000
        points = sampler.sample(2000)
        assert points.shape == (2000, 3)
        radii = torch.linalg.vector_norm(points - region.center, dim=-1)
        assert torch.allclose(radii, torch.full_like(radii, 3.0), atol=1e-4)

    def test_sphere_surface_rejects_non_3d(self) -> None:
        region = SphereSurfaceRegion(torch.zeros(2), radius=1.0)
        with pytest.raises(UnsupportedSamplerConfigurationError):
            EvenlySpacedSampler(region)


# ---------------------------------------------------------------------------
# CartesianProductSampler
# ---------------------------------------------------------------------------


class TestCartesianProductSampler:
    def test_box_grid_count_and_fixed_dimension(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 10.0, 0.0]), torch.tensor([10.0, 10.0, 10.0]))
        coordinates = [
            torch.linspace(0.0, 10.0, 101),
            torch.tensor([10.0]),
            torch.linspace(0.0, 10.0, 101),
        ]
        sampler = CartesianProductSampler(region, coordinates=coordinates)
        assert sampler.n_points() == 101 * 1 * 101
        points = sampler.sample()
        assert points.shape == (101 * 1 * 101, 3)
        assert bool((points[:, 1] == 10.0).all())
        assert bool(region.contains(points).all())

    def test_asymmetric_per_dimension_ranges(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 100.0]), torch.tensor([10.0, 1000.0]))
        coordinates = [torch.arange(0.0, 10.01, 0.5), torch.arange(100.0, 1001.0, 100.0)]
        sampler = CartesianProductSampler(region, coordinates=coordinates)
        expected = coordinates[0].numel() * coordinates[1].numel()
        assert sampler.n_points() == expected
        points = sampler.sample()
        assert points.shape == (expected, 2)
        assert bool(region.contains(points).all())

    def test_filters_points_outside_region(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        coordinates = [torch.linspace(-1.0, 2.0, 7), torch.linspace(-1.0, 2.0, 7)]
        sampler = CartesianProductSampler(region, coordinates=coordinates)
        points = sampler.sample()
        assert bool(region.contains(points).all())
        assert points.shape[0] < 7 * 7

    def test_ball_filtered_grid_membership(self) -> None:
        region = BallRegion(torch.zeros(2), radius=1.0)
        coordinates = [torch.linspace(-1.0, 1.0, 51), torch.linspace(-1.0, 1.0, 51)]
        sampler = CartesianProductSampler(region, coordinates=coordinates)
        assert sampler.n_points() == 51 * 51
        points = sampler.sample()
        assert points.shape[1] == 2
        assert points.shape[0] < 51 * 51
        assert bool(region.contains(points).all())

    def test_sample_matches_n_points(self) -> None:
        region = BoxRegion(torch.zeros(2), torch.ones(2))
        coordinates = [torch.linspace(0.0, 1.0, 4), torch.linspace(0.0, 1.0, 4)]
        sampler = CartesianProductSampler(region, coordinates=coordinates)
        assert sampler.sample().shape[0] == sampler.n_points()

    def test_requires_matching_coordinates_length(self) -> None:
        region = BoxRegion(torch.zeros(2), torch.ones(2))
        with pytest.raises(ValueError):
            CartesianProductSampler(
                region, coordinates=[torch.zeros(3), torch.zeros(3), torch.zeros(3)]
            )

    def test_sphere_surface_not_supported(self) -> None:
        region = SphereSurfaceRegion(torch.zeros(3), radius=1.0)
        with pytest.raises(UnsupportedSamplerRegionError):
            CartesianProductSampler(
                region, coordinates=[torch.zeros(10), torch.zeros(10), torch.zeros(10)]
            )


# ---------------------------------------------------------------------------
# TruncatedGaussianSampler
# ---------------------------------------------------------------------------


class TestTruncatedGaussianSampler:
    def test_exact_count_and_membership(self) -> None:
        region = BoxRegion(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))
        sampler = TruncatedGaussianSampler(region, mean=torch.zeros(2), std=torch.ones(2))
        points = sampler.sample(500, seed=0)
        assert points.shape == (500, 2)
        assert bool(region.contains(points).all())

    def test_ball_region_supported(self) -> None:
        region = BallRegion(torch.zeros(2), radius=1.0)
        sampler = TruncatedGaussianSampler(region, mean=torch.zeros(2), std=0.5 * torch.ones(2))
        points = sampler.sample(300, seed=0)
        assert points.shape == (300, 2)
        assert bool(region.contains(points).all())

    def test_fixed_dimension_requires_matching_mean_and_zero_std(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 5.0]), torch.tensor([1.0, 5.0]))
        with pytest.raises(ValueError):
            TruncatedGaussianSampler(
                region, mean=torch.tensor([0.5, 5.0]), std=torch.tensor([0.1, 0.1])
            )
        with pytest.raises(ValueError):
            TruncatedGaussianSampler(
                region, mean=torch.tensor([0.5, 0.0]), std=torch.tensor([0.1, 0.0])
            )

    def test_sphere_surface_not_supported(self) -> None:
        region = SphereSurfaceRegion(torch.zeros(2), radius=1.0)
        with pytest.raises(UnsupportedSamplerRegionError):
            TruncatedGaussianSampler(region, mean=torch.zeros(2), std=torch.ones(2))

    def test_raises_after_max_candidate_batches(self) -> None:
        # Region is unreachable from the Gaussian's mean, so acceptance is ~0.
        region = BoxRegion(torch.tensor([100.0, 100.0]), torch.tensor([101.0, 101.0]))
        sampler = TruncatedGaussianSampler(
            region,
            mean=torch.zeros(2),
            std=torch.ones(2),
            candidate_batch_size=16,
            max_candidate_batches=3,
        )
        with pytest.raises(RuntimeError):
            sampler.sample(10, seed=0)

    def test_mean_centers_the_distribution(self) -> None:
        # Regression coverage for the old GaussianSampler's "centered at midpoint" case,
        # via rejection instead of clipping (no artificial mass on the boundary).
        region = BoxRegion(torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0]))
        mean = (region.min + region.max) / 2
        std = 0.2 * (region.max - region.min)
        sampler = TruncatedGaussianSampler(region, mean=mean, std=std)
        points = sampler.sample(1000, seed=0)
        assert torch.allclose(points.mean(dim=0), mean, atol=0.1)

    def test_constructor_seed_reproducibility(self) -> None:
        region = BoxRegion(torch.tensor([-1.0, -1.0]), torch.tensor([1.0, 1.0]))
        sampler = TruncatedGaussianSampler(region, mean=torch.zeros(2), std=torch.ones(2), seed=42)
        assert torch.equal(sampler.sample(500), sampler.sample(500))


# ---------------------------------------------------------------------------
# LatinHypercubeSampler
# ---------------------------------------------------------------------------


class TestLatinHypercubeSampler:
    def test_exact_count_and_membership(self) -> None:
        region = BoxRegion(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 10.0]))
        sampler = LatinHypercubeSampler(region)
        points = sampler.sample(4096, seed=0)
        assert points.shape == (4096, 2)
        assert bool(region.contains(points).all())

    def test_fixed_dimension_stays_fixed(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 10.0]), torch.tensor([10.0, 10.0]))
        sampler = LatinHypercubeSampler(region)
        points = sampler.sample(64, seed=0)
        assert bool((points[:, 1] == 10.0).all())

    def test_stratification_one_point_per_interval(self) -> None:
        region = BoxRegion(torch.tensor([0.0]), torch.tensor([1.0]))
        sampler = LatinHypercubeSampler(region)
        n = 32
        points = sampler.sample(n, seed=0)
        bin_index = torch.clamp((points[:, 0] * n).long(), max=n - 1)
        assert len(bin_index.unique()) == n

    def test_ball_region_not_supported(self) -> None:
        region = BallRegion(torch.zeros(2), radius=1.0)
        with pytest.raises(UnsupportedSamplerRegionError):
            LatinHypercubeSampler(region)

    def test_constructor_seed_reproducibility(self) -> None:
        region = BoxRegion(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 10.0]))
        sampler = LatinHypercubeSampler(region, seed=42)
        assert torch.equal(sampler.sample(4096), sampler.sample(4096))


# ---------------------------------------------------------------------------
# SobolSampler
# ---------------------------------------------------------------------------


class TestSobolSampler:
    def test_exact_count_and_membership(self) -> None:
        region = BoxRegion(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 10.0]))
        sampler = SobolSampler(region, seed=42)
        points = sampler.sample(256)
        assert points.shape == (256, 2)
        assert bool(region.contains(points).all())

    def test_fixed_dimension_stays_fixed(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 10.0]), torch.tensor([10.0, 10.0]))
        sampler = SobolSampler(region, seed=42)
        points = sampler.sample(64)
        assert bool((points[:, 1] == 10.0).all())

    def test_seed_reproducibility(self) -> None:
        region = BoxRegion(torch.zeros(2), torch.ones(2))
        points_a = SobolSampler(region, scramble=True, seed=7).sample(128)
        points_b = SobolSampler(region, scramble=True, seed=7).sample(128)
        assert torch.equal(points_a, points_b)

    def test_ball_region_not_supported(self) -> None:
        region = BallRegion(torch.zeros(2), radius=1.0)
        with pytest.raises(UnsupportedSamplerRegionError):
            SobolSampler(region)


# ---------------------------------------------------------------------------
# CsvSampler
# ---------------------------------------------------------------------------


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

    def test_csv_sampler_region_fitted_from_data(self, csv_file: Path) -> None:
        sampler = CsvSampler(csv_file, coordinate_columns=["x1", "x2"], device="cpu")

        assert isinstance(sampler.region, BoxRegion)
        assert sampler.region.min.tolist() == [1.0, 2.0]
        assert sampler.region.max.tolist() == [7.0, 8.0]
        assert bool(sampler.region.contains(sampler.sample()).all())

    def test_csv_sampler_is_deterministic(self, csv_file: Path) -> None:
        sampler = CsvSampler(csv_file, coordinate_columns=["x1", "x2"], device="cpu")
        assert sampler.is_deterministic is True

    def test_csv_sampler_repr(self, csv_file: Path) -> None:
        sampler = CsvSampler(csv_file, coordinate_columns=["x1", "x2"], device="cpu")
        assert repr(sampler) == f"CsvSampler({csv_file.name}, n=4)"
