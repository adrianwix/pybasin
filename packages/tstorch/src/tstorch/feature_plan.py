# pyright: basic
"""Device-independent feature-configuration resolution, shared by every extractor.

``normalize_features_config``/``resolve_feature_plan`` turn a caller's ``features``/
``features_per_state`` request into an immutable :class:`ResolvedFeaturePlan` - the same
ordered specs, feature names, output column count, deduplicated parameter groups, and
tensor-column mapping regardless of which extractor (local, Dask CPU, Dask CUDA) resolves
it. The plan contains no tensor, device, compiled callable, lock, or Dask object, so it is
safe to embed in a Dask task and to build a worker-cache key from.

``build_kernel`` then turns a resolved plan into a :class:`CompiledFeatureKernel`: the
composed ``torch.cat([f(y) for f in closures], dim=-1)`` call, optionally wrapped with
``torch.compile``. This is the only place either step happens; local extraction and every
Dask worker task call the same two functions.

Every one of these resolves feature names against ``tstorch.supported_features.SUPPORTED_FEATURES``
directly - it is the one ``feature_name -> Feature class`` mapping in the package, so there is
nothing for a caller to supply instead.
"""

from collections.abc import Callable, Hashable, Mapping
from typing import Any, NamedTuple

import torch
from torch import Tensor

from tstorch.feature_factory import (
    construct_feature,
    is_parameterless,
    param_key,
)
from tstorch.supported_features import SUPPORTED_FEATURES


class FeatureSpec(NamedTuple):
    """One output column requested by the resolved feature plan."""

    state_idx: int
    feature_name: str
    param_key: Hashable
    output_name: str


class ResolvedFeaturePlan(NamedTuple):
    """Feature configuration resolved for a fixed ``num_states``, independent of device.

    :ivar specs: One entry per requested output column, in stable order.
    :ivar num_states: The state count this plan was resolved for.
    :ivar column_of: ``(feature_name, param_key) -> absolute column index`` in the
        kernel's ``(B, S, F)`` output.
    :ivar unique_params_by_feature: Per feature name, the deduplicated param dicts
        requested for it anywhere (first-seen order) - what ``build_kernel`` constructs
        closures from.
    :ivar token: A deterministic, hashable value that changes whenever the resolved
        schema (specs, column mapping, or dedup groups) would change - suitable for a
        Dask task key or worker-cache key component.
    """

    specs: tuple[FeatureSpec, ...]
    num_states: int
    column_of: dict[tuple[str, Hashable], int]
    unique_params_by_feature: dict[str, list[dict[str, Any]]]
    token: Hashable


class CompiledFeatureKernel(NamedTuple):
    """A resolved plan's closures composed into one callable, optionally compiled.

    :ivar kernel: ``(B, S, T) -> (B, S, len(unique columns))``.
    :ivar plan: The :class:`ResolvedFeaturePlan` this kernel was built from.
    """

    kernel: Callable[[Tensor], Tensor]
    plan: ResolvedFeaturePlan


def normalize_features_config(
    features: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]] | None]:
    """Validate and freeze a per-state feature configuration.

    :raises ValueError: If a requested feature isn't in ``SUPPORTED_FEATURES``.
    """
    unsupported = [name for name in features if name not in SUPPORTED_FEATURES]
    if unsupported:
        raise ValueError(
            f"Unsupported feature(s) {unsupported}. Only these are wired in right now: "
            f"{sorted(SUPPORTED_FEATURES)}"
        )
    result: dict[str, list[dict[str, Any]] | None] = {}
    for name, params in features.items():
        if params is None:
            result[name] = None
        elif isinstance(params, list):
            result[name] = params
        else:
            result[name] = [dict(params)]
    return result


def resolve_feature_plan(
    state_config: dict[int, dict[str, list[dict[str, Any]] | None]],
    num_states: int,
) -> ResolvedFeaturePlan:
    """Resolve a per-state feature configuration into an immutable, device-independent plan.

    :param state_config: Per-state feature configuration, already normalized by
        :func:`normalize_features_config`, and already restricted to states that are
        actually configured (a skipped state has no entry).
    :param num_states: Total number of states in the target tensor's ``S`` axis.
    """
    unique_params_by_feature: dict[str, list[dict[str, Any]]] = {}
    seen_keys_by_feature: dict[str, set[Hashable]] = {}
    specs: list[FeatureSpec] = []

    for state_idx in range(num_states):
        state_features = state_config.get(state_idx)
        if state_features is None:
            continue
        for feature_name, param_list in state_features.items():
            feature_cls = SUPPORTED_FEATURES[feature_name]
            parameterless = is_parameterless(feature_cls)
            if parameterless:
                if param_list is not None:
                    raise ValueError(f"{feature_name} takes no parameters; got {param_list}")
                entries: list[dict[str, Any]] = [{}]
            else:
                entries = param_list if param_list is not None else [{}]
            for params in entries:
                key = param_key(params)
                seen_keys = seen_keys_by_feature.setdefault(feature_name, set())
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique_params_by_feature.setdefault(feature_name, []).append(params)
                ctor_params = None if parameterless else [params]
                name = construct_feature(feature_cls, ctor_params).names[0]
                output_name = f"state_{state_idx}__{name}"
                specs.append(FeatureSpec(state_idx, feature_name, key, output_name))

    column_of: dict[tuple[str, Hashable], int] = {}
    offset = 0
    for feature_name, params_list in unique_params_by_feature.items():
        for params in params_list:
            column_of[(feature_name, param_key(params))] = offset
            offset += 1

    token = (
        num_states,
        tuple(specs),
        tuple(
            (name, tuple(param_key(p) for p in params))
            for name, params in unique_params_by_feature.items()
        ),
    )

    return ResolvedFeaturePlan(
        specs=tuple(specs),
        num_states=num_states,
        column_of=column_of,
        unique_params_by_feature=unique_params_by_feature,
        token=token,
    )


def build_kernel(
    plan: ResolvedFeaturePlan,
    use_compile: bool,
) -> CompiledFeatureKernel:
    """Compose a resolved plan's closures into one callable, optionally ``torch.compile``'d.

    :param use_compile: Whether to wrap the composed kernel with ``torch.compile``. No
        eager fallback: if ``torch.compile`` can't handle it, it raises.
    """
    closures: list[Callable[[Tensor], Tensor]] = []
    for feature_name, params_list in plan.unique_params_by_feature.items():
        feature_cls = SUPPORTED_FEATURES[feature_name]
        ctor_params = None if is_parameterless(feature_cls) else params_list
        closures.append(construct_feature(feature_cls, ctor_params))

    def kernel_eager(
        y: Tensor, closures: tuple[Callable[[Tensor], Tensor], ...] = tuple(closures)
    ) -> Tensor:
        return torch.cat([f(y) for f in closures], dim=-1)

    kernel = torch.compile(kernel_eager, dynamic=True) if use_compile else kernel_eager
    return CompiledFeatureKernel(kernel=kernel, plan=plan)


def extract_raw(compiled: CompiledFeatureKernel, y: Tensor) -> tuple[Tensor, list[str]]:
    """Run one compiled kernel over one validated ``(B, S, T)`` tensor.

    The single raw extraction path shared by :class:`tstorch.torch_feature_extractor.TorchFeatureExtractor`
    and every Dask worker task - neither reimplements feature stacking or column selection.

    :param compiled: A kernel built by :func:`build_kernel` for ``y``'s state count.
    :param y: Tensor of shape ``(B, S, T)``, time dimension already contiguous.
    :return: ``(features, feature_names)`` where ``features`` has shape ``(B, F)`` on
        ``y``'s device and dtype, and ``feature_names`` has length ``F`` in the same order.
    """
    plan = compiled.plan
    if not plan.specs:
        return torch.zeros((y.shape[0], 0), dtype=y.dtype, device=y.device), []

    with torch.no_grad():
        # torch.compile failures raise straight through - no silent eager fallback.
        cube = compiled.kernel(y)  # (B, S, total distinct (feature, params) columns)

        all_features = [
            cube[:, spec.state_idx, plan.column_of[(spec.feature_name, spec.param_key)]]
            for spec in plan.specs
        ]
        all_feature_names = [spec.output_name for spec in plan.specs]

        features = torch.stack(all_features, dim=1)

    return features, all_feature_names
