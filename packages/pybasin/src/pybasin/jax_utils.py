"""JAX utility functions for device management and tensor conversion.

This module provides utilities for:
- Resolving JAX devices from string specifications
- Efficient tensor conversion between PyTorch and JAX (using DLPack when possible)
"""

from typing import Any

import jax
import jax.dlpack  # type: ignore[import-untyped]
import jax.numpy as jnp
import numpy as np
import torch
import torch.utils.dlpack  # type: ignore[import-untyped]
from jax import Array

# Use Any for JAX Device type since it's not properly exported in type stubs
JaxDevice = Any


def get_jax_device(device: str | None = None) -> JaxDevice:
    """Resolve a device string to a JAX device.

    ```python
    device = get_jax_device("cuda:0")
    device = get_jax_device("cpu")
    device = get_jax_device()  # auto-detect
    ```

    :param device: Device specification. Options:
        - None: Auto-detect (prefer GPU if available)
        - "cpu": CPU device
        - "gpu", "cuda", "cuda:0": GPU device
        - "cuda:N": Specific GPU device N
    :return: JAX Device object.
    """
    if device is None:
        # Auto-detect: use default JAX device (prefers GPU)
        return jax.devices()[0]  # type: ignore[no-any-return]

    device_lower = device.lower()

    if device_lower == "cpu":
        return jax.devices("cpu")[0]  # type: ignore[no-any-return]

    if device_lower in ("gpu", "cuda"):
        gpu_devices: list[JaxDevice] = jax.devices("gpu")  # type: ignore[assignment]
        if gpu_devices:
            return gpu_devices[0]
        raise RuntimeError("GPU requested but no GPU devices available")

    if device_lower.startswith("cuda:"):
        idx = int(device_lower.split(":")[1])
        gpu_devices_n: list[JaxDevice] = jax.devices("gpu")  # type: ignore[assignment]
        if idx < len(gpu_devices_n):
            return gpu_devices_n[idx]
        raise RuntimeError(
            f"GPU device {idx} requested but only {len(gpu_devices_n)} GPUs available"
        )

    raise ValueError(f"Unknown device: {device}. Use 'cpu', 'gpu', 'cuda', or 'cuda:N'")


def torch_to_jax(tensor: torch.Tensor, device: JaxDevice | None = None) -> Array:
    """Convert a PyTorch tensor to a JAX array.

    Uses DLPack for zero-copy conversion when both tensors are on GPU.
    Falls back to NumPy conversion for CPU tensors or cross-device transfers.

    ```python
    x_torch = torch.randn(100, 10, device="cuda")
    x_jax = torch_to_jax(x_torch)  # zero-copy on GPU
    ```

    :param tensor: PyTorch tensor to convert.
    :param device: Target JAX device. If None, uses the same device as the tensor.
    :return: JAX array on the specified device.
    """
    # Determine target device
    if device is None:
        if tensor.is_cuda:
            device = jax.devices("gpu")[tensor.device.index or 0]  # type: ignore[assignment]
        else:
            device = jax.devices("cpu")[0]  # type: ignore[assignment]

    # Try DLPack for zero-copy GPU transfer
    if tensor.is_cuda and device.platform == "gpu":  # type: ignore[union-attr]
        try:
            # Ensure tensor is contiguous for DLPack
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            dlpack_capsule = torch.utils.dlpack.to_dlpack(tensor)  # type: ignore[attr-defined]
            return jax.dlpack.from_dlpack(dlpack_capsule)  # type: ignore[return-value]
        except Exception:
            # Fall back to NumPy if DLPack fails
            pass

    # Fall back to NumPy conversion (requires CPU copy)
    np_array = tensor.detach().cpu().numpy()
    jax_array: Array = jnp.asarray(np_array)  # type: ignore[assignment]
    return jax.device_put(jax_array, device)  # type: ignore[return-value]


def jax_to_torch(array: Array, device: torch.device | str | None = None) -> torch.Tensor:
    """Convert a JAX array to a PyTorch tensor.

    Uses DLPack for zero-copy conversion when both arrays are on GPU.
    Falls back to NumPy conversion for CPU arrays or cross-device transfers.

    ```python
    x_jax = jnp.ones((100, 10))
    x_torch = jax_to_torch(x_jax, device="cuda:0")
    ```

    :param array: JAX array to convert.
    :param device: Target PyTorch device. If None, uses the same device as the array.
    :return: PyTorch tensor on the specified device.
    """
    # Determine source device
    jax_device: JaxDevice = array.devices().pop()
    is_gpu: bool = jax_device.platform == "gpu"

    # Determine target device
    if device is None:
        device = torch.device(f"cuda:{jax_device.id}") if is_gpu else torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    # Try DLPack for zero-copy GPU transfer
    if is_gpu and device.type == "cuda":
        try:
            dlpack_capsule = jax.dlpack.to_dlpack(array)  # type: ignore[attr-defined]
            return torch.utils.dlpack.from_dlpack(dlpack_capsule)  # type: ignore[attr-defined]
        except Exception:
            # Fall back to NumPy if DLPack fails
            pass

    # Fall back to NumPy conversion
    np_array: Any = np.asarray(array)
    return torch.tensor(np_array, device=device)
