"""Provider contracts for running shared feature assertions against array backends."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast, runtime_checkable

import pytest
import torch
from torch import Tensor, nn


@runtime_checkable
class TestFeature(Protocol):
    @property
    def names(self) -> Sequence[str]: ...

    def __call__(self, x: Tensor) -> Tensor: ...


class TorchCompile(Protocol):
    def __call__(
        self,
        model: TestFeature,
        *,
        fullgraph: bool,
        options: Mapping[str, str | int | bool],
    ) -> Callable[[Tensor], Tensor]: ...


@runtime_checkable
class FeatureTestProvider(Protocol):
    """Construct features through a test backend and return Torch comparison tensors."""

    name: str

    @property
    def compile_mode(self) -> str | None: ...

    @property
    def export_mode(self) -> str | None: ...

    def supports(self, feature_name: str) -> bool: ...

    def create(
        self,
        feature_name: str,
        params: Sequence[Mapping[str, object]] | None,
        native_factory: Callable[[], TestFeature],
    ) -> TestFeature: ...

    def compile(self, feature: TestFeature, x: Tensor) -> str | None: ...

    def export(self, feature: TestFeature, x: Tensor) -> str | None: ...


class NativeTorchProvider:
    name = "native-torch"

    @property
    def compile_mode(self) -> str:
        return "torch.compile(fullgraph=True)"

    @property
    def export_mode(self) -> str:
        return "torch.export"

    def supports(self, feature_name: str) -> bool:
        return True

    def create(
        self,
        feature_name: str,
        params: Sequence[Mapping[str, object]] | None,
        native_factory: Callable[[], TestFeature],
    ) -> TestFeature:
        del feature_name, params
        return native_factory()

    def compile(self, feature: TestFeature, x: Tensor) -> str | None:
        try:
            _compile(feature, fullgraph=True, options={"trace.enabled": True})(x)
        except Exception as exc:  # noqa: BLE001 - graph breaks have multiple exception types
            return f"{type(exc).__name__}: {exc}"
        finally:
            torch.compiler.reset()
        return None

    def export(self, feature: TestFeature, x: Tensor) -> str | None:
        class ExportShim(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature = feature

            def forward(self, value: Tensor) -> Tensor:
                return self.feature(value)

        try:
            torch.export.export(
                ExportShim(),
                (x,),
                dynamic_shapes={"value": {0: torch.export.Dim("batch")}},
            )
        except Exception as exc:  # noqa: BLE001 - exporter failures have multiple exception types
            error = f"{type(exc).__name__}: {exc}"
            try:
                draft = torch.export.draft_export(
                    ExportShim(),
                    (x,),
                    dynamic_shapes={"value": {0: torch.export.Dim("batch")}},
                )
                error = str(cast(object, vars(draft)["_report"]))
            except Exception as draft_exc:  # noqa: BLE001 - draft_export can fail independently
                error += (
                    f"\n(torch.export.draft_export also failed to produce a report: {draft_exc})"
                )
            return error
        return None


_provider: FeatureTestProvider = NativeTorchProvider()
_native_provider_selected = True
_compile: TorchCompile = cast(TorchCompile, vars(torch)["compile"])


def configure_provider(specification: str | None) -> None:
    global _provider
    global _native_provider_selected
    if specification is None:
        _provider = NativeTorchProvider()
        _native_provider_selected = True
        return
    module_name, separator, object_name = specification.partition(":")
    if not separator or not module_name or not object_name:
        raise pytest.UsageError("--feature-provider must have the form module:object")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise pytest.UsageError(f"cannot import provider module {module_name!r}: {exc}") from exc
    try:
        selected = getattr(module, object_name)
    except AttributeError as exc:
        raise pytest.UsageError(
            f"provider module {module_name!r} has no object {object_name!r}"
        ) from exc
    provider = selected() if isinstance(selected, type) else selected
    if not isinstance(provider, FeatureTestProvider):
        raise pytest.UsageError(f"{specification} does not implement FeatureTestProvider")
    _provider = provider
    _native_provider_selected = False


def feature_provider(feature_name: str) -> FeatureTestProvider:
    if not _provider.supports(feature_name):
        pytest.skip(f"provider {_provider.name!r} does not support feature {feature_name!r}")
    return _provider


def construct_test_feature(
    feature_name: str,
    params: Sequence[Mapping[str, object]] | None,
    native_factory: Callable[[], TestFeature],
) -> TestFeature:
    provider = feature_provider(feature_name)
    return provider.create(feature_name, params, native_factory)


def selected_provider_name() -> str:
    return _provider.name


def selected_provider_is_native() -> bool:
    return _native_provider_selected
