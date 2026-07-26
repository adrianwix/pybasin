"""Caps JAX's GPU memory grab before its CUDA backend initializes.

JAX preallocates ~75% of GPU memory by default the first time it touches the
device. Running the CPU-forced and GPU-only integration tests in the same
pytest process, cuML/cupy/torch all need to allocate from what's left --
without this, a full GPU run can crash with CUDA_ERROR_OUT_OF_MEMORY partway
through a parameter sweep. Must be imported (for its side effect) before any
module that imports jax, directly or transitively -- see ``tests/conftest.py``.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
