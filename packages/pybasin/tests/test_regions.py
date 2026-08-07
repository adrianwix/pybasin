# pyright: basic
import pytest
import torch

from pybasin.regions import BallRegion, BoxRegion, SphereSurfaceRegion


class TestBoxRegion:
    def test_state_dim(self) -> None:
        region = BoxRegion(torch.tensor([0.0, -1.0]), torch.tensor([1.0, 1.0]))
        assert region.state_dim == 2

    def test_contains_membership(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        points = torch.tensor([[0.5, 0.5], [1.5, 0.5], [0.5, -0.1]])
        assert region.contains(points).tolist() == [True, False, False]

    def test_boundary_points_are_included(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        assert bool(region.contains(points).all())

    def test_broadcasted_batch_shape(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))
        points = torch.rand(5, 3, 2)
        mask = region.contains(points)
        assert mask.shape == (5, 3)

    def test_fixed_dimensions(self) -> None:
        region = BoxRegion(torch.tensor([0.0, 10.0, 0.0]), torch.tensor([10.0, 10.0, 10.0]))
        assert region.fixed_dimensions.tolist() == [False, True, False]
        assert region.active_dimensions.tolist() == [True, False, True]
        assert not bool(region.contains(torch.tensor([[5.0, 9.999, 5.0]])).any())
        assert bool(region.contains(torch.tensor([[5.0, 10.0, 5.0]])).all())

    def test_accepts_plain_python_lists(self) -> None:
        region = BoxRegion([0.0, -1.0], [1.0, 1.0])
        assert region.state_dim == 2
        assert region.min.tolist() == [0.0, -1.0]
        assert region.max.tolist() == [1.0, 1.0]
        assert region.dtype == torch.float32
        assert region.device.type == "cpu"
        assert isinstance(region.min, torch.Tensor)

    def test_accepts_mixed_list_and_tensor(self) -> None:
        region = BoxRegion([0.0, -1.0], torch.tensor([1.0, 1.0]))
        assert region.min.tolist() == [0.0, -1.0]
        assert region.max.tolist() == [1.0, 1.0]

    def test_rejects_non_one_dimensional_limits(self) -> None:
        with pytest.raises(ValueError):
            BoxRegion(torch.zeros(2, 2), torch.ones(2, 2))

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError):
            BoxRegion(torch.zeros(2), torch.ones(3))

    def test_rejects_min_greater_than_max(self) -> None:
        with pytest.raises(ValueError):
            BoxRegion(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))

    def test_rejects_dtype_mismatch(self) -> None:
        with pytest.raises(ValueError):
            BoxRegion(torch.zeros(2, dtype=torch.float32), torch.ones(2, dtype=torch.float64))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_gpu_device_preservation(self) -> None:
        region = BoxRegion(
            torch.tensor([0.0, 0.0], device="cuda"), torch.tensor([1.0, 1.0], device="cuda")
        )
        points = torch.rand(4, 2, device="cuda")
        assert region.contains(points).device.type == "cuda"


class TestBallRegion:
    def test_state_dim(self) -> None:
        region = BallRegion(torch.zeros(3), radius=1.0)
        assert region.state_dim == 3

    def test_accepts_plain_python_list(self) -> None:
        region = BallRegion([0.0, 0.0], radius=1.0)
        assert region.center.tolist() == [0.0, 0.0]
        assert isinstance(region.center, torch.Tensor)

    def test_contains_membership(self) -> None:
        region = BallRegion(torch.tensor([0.0, 0.0]), radius=1.0)
        points = torch.tensor([[0.0, 0.0], [0.9, 0.0], [1.1, 0.0]])
        assert region.contains(points).tolist() == [True, True, False]

    def test_boundary_point_is_included(self) -> None:
        region = BallRegion(torch.tensor([0.0, 0.0]), radius=1.0)
        assert bool(region.contains(torch.tensor([[1.0, 0.0]])).all())

    def test_broadcasted_batch_shape(self) -> None:
        region = BallRegion(torch.zeros(3), radius=1.0)
        points = torch.rand(4, 2, 3)
        assert region.contains(points).shape == (4, 2)

    def test_rejects_negative_radius(self) -> None:
        with pytest.raises(ValueError):
            BallRegion(torch.zeros(2), radius=-1.0)

    def test_rejects_non_one_dimensional_center(self) -> None:
        with pytest.raises(ValueError):
            BallRegion(torch.zeros(2, 2), radius=1.0)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_gpu_device_preservation(self) -> None:
        region = BallRegion(torch.zeros(2, device="cuda"), radius=1.0)
        points = torch.rand(4, 2, device="cuda")
        assert region.contains(points).device.type == "cuda"


class TestSphereSurfaceRegion:
    def test_state_dim(self) -> None:
        region = SphereSurfaceRegion(torch.zeros(3), radius=1.0)
        assert region.state_dim == 3

    def test_accepts_plain_python_list(self) -> None:
        region = SphereSurfaceRegion([0.0, 0.0, 0.0], radius=1.0)
        assert region.center.tolist() == [0.0, 0.0, 0.0]
        assert isinstance(region.center, torch.Tensor)

    def test_contains_only_the_surface(self) -> None:
        region = SphereSurfaceRegion(torch.tensor([0.0, 0.0]), radius=1.0)
        points = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.5, 0.0]])
        assert region.contains(points).tolist() == [True, False, False]

    def test_tolerance_widens_the_shell(self) -> None:
        region = SphereSurfaceRegion(torch.tensor([0.0, 0.0]), radius=1.0, tolerance=0.1)
        points = torch.tensor([[1.05, 0.0], [1.2, 0.0]])
        assert region.contains(points).tolist() == [True, False]

    def test_rejects_negative_radius(self) -> None:
        with pytest.raises(ValueError):
            SphereSurfaceRegion(torch.zeros(2), radius=-1.0)

    def test_rejects_negative_tolerance(self) -> None:
        with pytest.raises(ValueError):
            SphereSurfaceRegion(torch.zeros(2), radius=1.0, tolerance=-1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_gpu_device_preservation(self) -> None:
        region = SphereSurfaceRegion(torch.zeros(3, device="cuda"), radius=1.0)
        points = torch.rand(4, 3, device="cuda")
        assert region.contains(points).device.type == "cuda"
