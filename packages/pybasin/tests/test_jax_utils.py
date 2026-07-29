# pyright: basic
import jax
import jax.numpy as jnp
import pytest
import torch

from pybasin.jax_utils import get_jax_device, jax_to_torch, torch_to_jax


def _has_jax_gpu() -> bool:
    try:
        return len(jax.devices("gpu")) > 0
    except RuntimeError:
        return False


_no_gpu = pytest.mark.skipif(not _has_jax_gpu(), reason="No JAX GPU available")


class TestGetJaxDevice:
    """Tests for get_jax_device function."""

    def test_auto_detect(self):
        """Test auto-detection returns a valid device."""
        device = get_jax_device(None)
        assert device is not None

    def test_cpu_device(self):
        """Test CPU device selection."""
        device = get_jax_device("cpu")
        assert device.platform == "cpu"

    @_no_gpu
    def test_gpu_device(self):
        """Test GPU device selection."""
        device = get_jax_device("gpu")
        assert device.platform == "gpu"

    @_no_gpu
    def test_cuda_device(self):
        """Test CUDA device selection."""
        device = get_jax_device("cuda")
        assert device.platform == "gpu"

    @_no_gpu
    def test_cuda_indexed_device(self):
        """Test CUDA:0 device selection."""
        device = get_jax_device("cuda:0")
        assert device.platform == "gpu"

    def test_invalid_device(self):
        """Test invalid device raises error."""
        with pytest.raises(ValueError, match="Unknown device"):
            get_jax_device("invalid_device")


class TestTorchToJax:
    """Tests for torch_to_jax conversion."""

    def test_cpu_tensor_conversion(self):
        """Test CPU tensor converts correctly."""
        x_torch = torch.randn(10, 5)
        x_jax = torch_to_jax(x_torch)

        assert x_jax.shape == (10, 5)
        assert torch.allclose(x_torch, torch.tensor(x_jax.__array__()))

    def test_cpu_tensor_to_cpu_device(self):
        """Test CPU tensor to explicit CPU device."""
        x_torch = torch.randn(10, 5)
        device = get_jax_device("cpu")
        x_jax = torch_to_jax(x_torch, device)

        assert x_jax.devices().pop().platform == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
    def test_gpu_tensor_conversion(self):
        """Test GPU tensor converts correctly."""
        x_torch = torch.randn(10, 5, device="cuda")
        x_jax = torch_to_jax(x_torch)

        assert x_jax.shape == (10, 5)
        # Verify values match
        x_torch_cpu = x_torch.cpu()
        assert torch.allclose(x_torch_cpu, torch.tensor(x_jax.__array__()))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
    def test_gpu_tensor_stays_on_gpu(self):
        """Test GPU tensor stays on GPU by default."""
        x_torch = torch.randn(10, 5, device="cuda")
        x_jax = torch_to_jax(x_torch)

        assert x_jax.devices().pop().platform == "gpu"

    def test_preserves_dtype_float32(self):
        """Test float32 dtype is preserved."""
        x_torch = torch.randn(10, 5, dtype=torch.float32)
        x_jax = torch_to_jax(x_torch)

        assert x_jax.dtype.name == "float32"

    def test_3d_tensor_conversion(self):
        """Test 3D tensor (N, B, S) converts correctly."""
        x_torch = torch.randn(100, 5, 2)  # (N, B, S) shape
        x_jax = torch_to_jax(x_torch)

        assert x_jax.shape == (100, 5, 2)


class TestJaxToTorch:
    """Tests for jax_to_torch conversion."""

    def test_cpu_array_conversion(self):
        """Test CPU array converts correctly."""
        # Explicitly place on CPU
        device = get_jax_device("cpu")
        x_jax = jax.device_put(jnp.array([[1.0, 2.0], [3.0, 4.0]]), device)  # type: ignore[arg-type]
        x_torch = jax_to_torch(x_jax)

        assert x_torch.shape == (2, 2)
        assert x_torch.device.type == "cpu"

    def test_cpu_array_to_explicit_device(self):
        """Test CPU array to explicit CPU device."""
        device = get_jax_device("cpu")
        x_jax = jax.device_put(jnp.array([[1.0, 2.0], [3.0, 4.0]]), device)  # type: ignore[arg-type]
        x_torch = jax_to_torch(x_jax, device="cpu")

        assert x_torch.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
    def test_gpu_array_conversion(self):
        """Test GPU array converts correctly."""
        device = get_jax_device("gpu")
        x_jax = jax.device_put(jnp.array([[1.0, 2.0], [3.0, 4.0]]), device)  # type: ignore[arg-type]
        x_torch = jax_to_torch(x_jax)

        assert x_torch.shape == (2, 2)

    def test_preserves_values(self):
        """Test values are preserved after conversion."""
        device = get_jax_device("cpu")
        x_jax = jax.device_put(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]), device)  # type: ignore[arg-type]
        x_torch = jax_to_torch(x_jax)

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        assert torch.allclose(x_torch, expected)


class TestRoundTrip:
    """Tests for round-trip conversions."""

    def test_torch_jax_torch_cpu(self):
        """Test torch -> jax -> torch on CPU preserves values."""
        x_original = torch.randn(10, 5)
        x_jax = torch_to_jax(x_original)
        x_back = jax_to_torch(x_jax)

        assert torch.allclose(x_original, x_back)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
    def test_torch_jax_torch_gpu(self):
        """Test torch -> jax -> torch on GPU preserves values."""
        x_original = torch.randn(10, 5, device="cuda")
        x_jax = torch_to_jax(x_original)
        x_back = jax_to_torch(x_jax)

        assert torch.allclose(x_original.cpu(), x_back.cpu())

    def test_large_tensor_conversion(self):
        """Test large tensor conversion works correctly."""
        x_original = torch.randn(1000, 100, 10)
        x_jax = torch_to_jax(x_original)
        x_back = jax_to_torch(x_jax)

        assert x_back.shape == x_original.shape
        assert torch.allclose(x_original, x_back, atol=1e-6)
