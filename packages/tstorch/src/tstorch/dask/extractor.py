# pyright: basic
"""``DaskFeatureExtractor``: the lazy Dask Array graph for distributed feature extraction.

Builds a ``map_blocks`` graph over the batch (``B``) axis only. One CPU or CUDA execution
policy is set at construction (``device``); the actual work happens in
:meth:`DaskFeatureExtractor._extract_block`, which is the only thing embedded in each
task. As a ``staticmethod`` it closes over nothing but its own arguments - plain,
picklable constants (a :class:`tstorch.feature_plan.ResolvedFeaturePlan`, the
feature-name -> class mapping, the device string, and the compile flag) - never a
``Client``, a compiled extractor, or this class's own instance.
"""

import threading
from typing import Any, Literal

import dask.array as da
import numpy as np

from tstorch.dask._block_conversion import _block_to_tensor, _DeviceType, _tensor_to_block
from tstorch.dask._cache import _get_process_cache, _make_cache_key
from tstorch.feature_plan import extract_raw, normalize_features_config, resolve_feature_plan
from tstorch.settings import MINIMAL_FEATURE_NAMES, FeaturesConfig


class DaskFeatureExtractor:
    """Distributed counterpart to :class:`tstorch.torch_feature_extractor.TorchFeatureExtractor`.

    Accepts a three-dimensional Dask Array ``(B, S, T)`` where only ``B`` may have
    several chunks; ``S`` and ``T`` must each be a single complete chunk. Returns a lazy
    NumPy-backed Dask Array ``(B, F)`` - nothing is computed, submitted, or persisted
    until the caller does so through their own ``Client``.

    Uses the same ``features``/``features_per_state`` feature-selection contract as
    ``TorchFeatureExtractor``: one shared resolver produces identical output ordering,
    names, and columns regardless of which extractor or device resolves it.

    :param features: Default feature configuration to apply to all states. Same meaning
        as ``TorchFeatureExtractor``.
    :param features_per_state: Optional per-state override. Same meaning as
        ``TorchFeatureExtractor``.
    :param device: ``"cpu"`` or ``"cuda"``. Never an automatic fallback: a ``"cuda"``
        task fails if its worker has no usable CUDA device.
    :param use_compile: Whether each worker wraps its composed kernel with
        ``torch.compile``.
    """

    def __init__(
        self,
        *,
        features: Literal["minimal"] | FeaturesConfig | None = "minimal",
        features_per_state: dict[int, FeaturesConfig | None] | None = None,
        device: _DeviceType,
        use_compile: bool = True,
    ) -> None:
        if device not in ("cpu", "cuda"):
            raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")

        self.device: _DeviceType = device
        self.use_compile = use_compile

        self.features: dict[str, list[dict[str, Any]] | None] | None = (
            normalize_features_config(dict.fromkeys(MINIMAL_FEATURE_NAMES))
            if features == "minimal"
            else (normalize_features_config(features) if features is not None else None)
        )
        self.features_per_state: dict[int, dict[str, list[dict[str, Any]] | None] | None] = {
            state_idx: (
                normalize_features_config(state_features) if state_features is not None else None
            )
            for state_idx, state_features in (features_per_state or {}).items()
        }

        self._bind_lock = threading.Lock()
        self._bound_num_states: int | None = None
        self._extracted_feature_names: list[str] | None = None

    # staticmethod, not a bound method: dask pickles this callable to ship it to worker
    # processes, so it must not capture `self` - that would serialize this instance's
    # lock and bound-state flags into every task.
    @staticmethod
    def _extract_block(
        block: Any,
        *,
        state_config: dict[int, dict[str, list[dict[str, Any]] | None]],
        num_states: int,
        device: _DeviceType,
        use_compile: bool,
    ) -> Any:
        """The one function embedded in every worker task.

        Rebuilds a :class:`~tstorch.feature_plan.ResolvedFeaturePlan` from plain,
        picklable constants (cheap: no tensor or device work), then looks up or builds
        this process's compiled kernel from the shared worker-local cache before running
        the raw ``(B_chunk, S, T) -> (B_chunk, F)`` extraction path.
        """
        resolved = resolve_feature_plan(state_config, num_states)

        tensor = _block_to_tensor(block, device)
        key = _make_cache_key(resolved, tensor.shape[-1], tensor.dtype, device, use_compile)
        cache = _get_process_cache()
        compiled = cache.get_or_build(key, resolved, use_compile)

        features, _ = extract_raw(compiled, tensor)
        return _tensor_to_block(features)

    def _state_feature_config(
        self, num_states: int
    ) -> dict[int, dict[str, list[dict[str, Any]] | None]]:
        config: dict[int, dict[str, list[dict[str, Any]] | None]] = {}
        for state_idx in range(num_states):
            state_features = self.features_per_state.get(state_idx, self.features)
            if state_features is not None:
                config[state_idx] = state_features
        return config

    def extract_features(self, data: Any) -> Any:
        """Build the lazy ``(B, F)`` Dask Array graph for ``data``.

        :param data: A three-dimensional ``dask.array.Array`` ``(B, S, T)``. Only the
            ``B`` axis may have several chunks. NumPy-backed on ``device="cpu"``; NumPy
            or CuPy-backed on ``device="cuda"``.
        :return: A lazy NumPy-backed ``dask.array.Array`` ``(B, F)``, preserving the
            input's exact ``B`` chunk boundaries. Nothing is computed yet.
        :raises ValueError: If ``data``'s shape, chunking, or dtype is invalid, or if this
            extractor was already bound to a different state count.
        """
        if not isinstance(data, da.Array):
            raise TypeError(f"DaskFeatureExtractor requires a dask.array.Array, got {type(data)}")
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D (B, S, T) Dask Array, got ndim={data.ndim}")

        num_states, sequence_length = data.shape[1], data.shape[2]
        if (
            not isinstance(num_states, int)
            or not isinstance(sequence_length, int)
            or num_states <= 0
            or sequence_length <= 0
        ):
            raise ValueError(f"S and T must be known and nonzero; got shape={data.shape}")
        if len(data.chunks[1]) != 1 or len(data.chunks[2]) != 1:
            raise ValueError(
                "The state and time axes must each be a single chunk; got "
                f"chunks={data.chunks}. Only the batch axis (axis 0) may be split."
            )
        if not np.issubdtype(data.dtype, np.floating):
            raise ValueError(f"Expected a floating dtype, got {data.dtype}")

        with self._bind_lock:
            if self._bound_num_states is not None and self._bound_num_states != num_states:
                raise ValueError(
                    f"This extractor was bound to num_states={self._bound_num_states}; "
                    f"got num_states={num_states}. Construct a new extractor instead of "
                    "reusing one across state counts."
                )

            state_config = self._state_feature_config(num_states)
            resolved = resolve_feature_plan(state_config, num_states)
            feature_names = [spec.output_name for spec in resolved.specs]
            num_columns = len(resolved.specs)

            output = data.map_blocks(
                self._extract_block,
                state_config=state_config,
                num_states=num_states,
                device=self.device,
                use_compile=self.use_compile,
                drop_axis=[1, 2],
                new_axis=[1],
                chunks=(data.chunks[0], (num_columns,)),
                dtype=data.dtype,
                meta=np.empty((0, num_columns), dtype=data.dtype),
            )

            self._bound_num_states = num_states
            self._extracted_feature_names = feature_names

        return output

    @property
    def feature_names(self) -> list[str]:
        """Return feature names in the format ``state_X__feature_name``.

        :raises RuntimeError: If no lazy result has been successfully constructed yet.
        """
        if self._extracted_feature_names is None:
            raise RuntimeError("Feature names not available. Call extract_features first.")
        return self._extracted_feature_names
