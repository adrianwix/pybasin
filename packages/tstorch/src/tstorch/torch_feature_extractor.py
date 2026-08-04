# pyright: basic
"""Compiled PyTorch feature extractor for BST time series tensors.

Every requested feature - parameter-free or parametrized - goes through one uniform
mechanism: a feature class (see `tstorch.feature_factory`) is constructed directly - with
the distinct param dicts requested for it (deduped across every state) for a `Feature`
subclass, with no arguments for a `ParameterlessFeature` subclass - and the constructed
instance itself is the closure; `_compile()` composes every requested feature's closure
into a single `torch.cat([f(y) for f in closures], dim=-1)` kernel - no factory/adapter
layer in between.

This file does not build any features itself - each calculator file that contributes an
extractor-eligible feature exposes its own feature class, and
`tstorch.supported_features.SUPPORTED_FEATURES` maps feature name to class directly.
Requesting anything else raises instead of silently skipping it.
"""

from typing import Any, Literal, NamedTuple, Protocol

import torch
from torch import Tensor

from tstorch.feature_plan import (
    CompiledFeatureKernel,
    build_kernel,
    extract_raw,
    normalize_features_config,
    resolve_feature_plan,
)
from tstorch.settings import MINIMAL_FEATURE_NAMES, FeaturesConfig

DEFAULT_STEADY_FRACTION = 0.85


class TimeSeriesSolution(Protocol):
    """Protocol for objects accepted by :class:`TorchFeatureExtractor`.

    :ivar time: Time values of shape (T,).
    :ivar y: Time series values of shape (B, S, T).
    """

    time: Tensor
    y: Tensor


class _CompiledFeaturePlan(NamedTuple):
    """A resolved plan's kernel, specialized to a tensor device for reuse validation."""

    compiled: CompiledFeatureKernel
    device: torch.device


class TorchFeatureExtractor:
    """Configurable compiled feature extractor for ``(B, S, T)`` tensors.

    Compiles a state-major feature plan once per input shape/device: every requested
    (feature, params) combo is resolved to a closure by constructing its ``Feature``
    class directly (see ``SUPPORTED_FEATURES``) - the instance itself is the closure -
    and every closure is composed into a single ``kernel(y)`` call
    (``torch.cat([f(y) for f in closures], dim=-1)``), optionally wrapped with
    ``torch.compile`` (``use_compile``, default True) as one traced graph - no fallback:
    if ``torch.compile`` can't handle it, it raises, so the failure is visible instead of
    silently paying for both a compiled and an eager path. ``extract_features`` then does
    exactly one kernel call plus plain per-spec indexing (each spec selects one
    ``(state, feature, params)`` cell from the kernel's output with two Python ints - a
    view, not a copy - no gather needed). Reusing an instance for a different
    ``(num_states, device)`` than it was first compiled for is also an error - construct a
    new extractor instead of silently recompiling.

    Only the calculators in ``SUPPORTED_FEATURES`` are supported; requesting anything else
    raises at construction time.

    :param time_steady: Time threshold for filtering transients when ``extract_features`` receives
        a solution object. If ``None``, uses 85% of the integration time span. Ignored when passing
        a raw BST tensor directly.
    :param features: Default feature configuration to apply to all states. ``"minimal"`` uses the
        tsfresh Minimal-equivalent 10 features, an ``FeaturesConfig``-shaped mapping
        (``{name: None}`` for parameter-free, ``{name: [params, ...]}`` for parametrized) uses
        exactly that, and ``None`` skips states not configured in ``features_per_state``.
    :param features_per_state: Optional mapping from state index to feature configuration. Values
        override ``features`` for that state. Use ``None`` to skip a state.
    :param use_compile: Whether to wrap the composed kernel with ``torch.compile``.
    """

    def __init__(
        self,
        time_steady: float | None = None,
        features: Literal["minimal"] | FeaturesConfig | None = "minimal",
        features_per_state: dict[int, FeaturesConfig | None] | None = None,
        use_compile: bool = True,
    ):
        self.time_steady: float | None = time_steady
        self.use_compile: bool = use_compile

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

        self._compiled_plan: _CompiledFeaturePlan | None = None
        self._extracted_feature_names: list[str] | None = None

    def _resolve_time_steady(self, solution: TimeSeriesSolution) -> float:
        """Resolve the effective steady-state time threshold.

        :param solution: Object with ``time`` of shape (T,).
        :return: Time threshold used to filter transients.
        """
        if self.time_steady is not None:
            return self.time_steady
        t = solution.time
        t0 = float(t[0])
        t1 = float(t[-1])
        return t0 + DEFAULT_STEADY_FRACTION * (t1 - t0)

    def _coerce_input(self, data: Tensor | TimeSeriesSolution) -> Tensor:
        """Return a BST tensor from a raw tensor or solution object.

        :param data: Raw tensor of shape ``(B, S, T)`` or an object with ``time`` and ``y``.
        :return: Tensor of shape ``(B, S, T)``.
        :raises ValueError: If the tensor is not three-dimensional.
        """
        if isinstance(data, Tensor):
            y = data
        else:
            effective_steady = self._resolve_time_steady(data)
            idx_steady = torch.where(data.time >= effective_steady)[0]
            start_idx = idx_steady[0] if len(idx_steady) > 0 else 0
            y = data.y[:, :, start_idx:]

        if y.ndim != 3:
            raise ValueError(f"Expected a BST tensor with shape (B, S, T), got {tuple(y.shape)}")
        if y.stride(-1) != 1:
            raise ValueError("Expected the time dimension to be contiguous with stride 1")
        return y

    def _state_feature_config(
        self, num_states: int
    ) -> dict[int, dict[str, list[dict[str, Any]] | None]]:
        """Resolve the configured feature/params mapping for each state."""
        config: dict[int, dict[str, list[dict[str, Any]] | None]] = {}
        for state_idx in range(num_states):
            state_features = self.features_per_state.get(state_idx, self.features)
            if state_features is not None:
                config[state_idx] = state_features
        return config

    def _compile(self, num_states: int, device: torch.device) -> None:
        """Resolve the configured states/features and compile the kernel for this device."""
        state_config = self._state_feature_config(num_states)
        resolved = resolve_feature_plan(state_config, num_states)
        compiled = build_kernel(resolved, self.use_compile)

        self._compiled_plan = _CompiledFeaturePlan(compiled=compiled, device=device)

    def extract_features(self, data: Tensor | TimeSeriesSolution) -> Tensor:
        """Extract features from a raw BST tensor or a solution object.

        :param data: Tensor with shape ``(B, S, T)`` or an object whose ``y`` tensor has that shape.
        :return: Feature tensor of shape ``(B, F)`` on the same device as the input tensor.
        """
        y = self._coerce_input(data)
        num_states = y.shape[1]

        if self._compiled_plan is None:
            self._compile(num_states, y.device)
        elif (
            self._compiled_plan.compiled.plan.num_states != num_states
            or self._compiled_plan.device != y.device
        ):
            raise ValueError(
                "This extractor was compiled for "
                f"(num_states={self._compiled_plan.compiled.plan.num_states}, "
                f"device={self._compiled_plan.device}); "
                f"got (num_states={num_states}, device={y.device}). Construct a new extractor "
                "instead of silently recompiling for a different shape/device."
            )
        assert self._compiled_plan is not None

        features, feature_names = extract_raw(self._compiled_plan.compiled, y)
        self._extracted_feature_names = feature_names
        return features

    @property
    def feature_names(self) -> list[str]:
        """Return feature names in the format ``state_X__feature_name``."""
        if self._extracted_feature_names is None:
            raise RuntimeError("Feature names not available. Call extract_features first.")
        return self._extracted_feature_names
