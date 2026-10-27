from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, Sequence

import pytest
from torch import Tensor

from . import feature_test_provider as providers


class UnsupportedProvider:
    name = "native-torch"
    compile_mode = None
    export_mode = None

    def supports(self, feature_name: str) -> bool:
        return False

    def create(
        self,
        feature_name: str,
        params: Sequence[Mapping[str, object]] | None,
        native_factory: Callable[[], providers.TestFeature],
    ) -> providers.TestFeature:
        raise AssertionError("unsupported feature reached a provider factory")

    def compile(self, feature: providers.TestFeature, x: Tensor) -> str | None:
        raise AssertionError("unsupported feature reached a compiler check")

    def export(self, feature: providers.TestFeature, x: Tensor) -> str | None:
        raise AssertionError("unsupported feature reached an export check")


class InvalidProvider:
    pass


class FakeFeature:
    names = ["feature"]

    def __call__(self, x: Tensor) -> Tensor:
        return x


class RecordingProvider:
    name = "recording"
    compile_mode = None
    export_mode = None

    def __init__(self) -> None:
        self.created: tuple[str, Sequence[Mapping[str, object]] | None] | None = None

    def supports(self, feature_name: str) -> bool:
        return feature_name == "feature"

    def create(
        self,
        feature_name: str,
        params: Sequence[Mapping[str, object]] | None,
        native_factory: Callable[[], providers.TestFeature],
    ) -> providers.TestFeature:
        del native_factory
        self.created = (feature_name, params)
        return FakeFeature()

    def compile(self, feature: providers.TestFeature, x: Tensor) -> str | None:
        raise AssertionError("compile mode is disabled")

    def export(self, feature: providers.TestFeature, x: Tensor) -> str | None:
        raise AssertionError("export mode is disabled")


def test_default_provider_is_native_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "_provider", UnsupportedProvider())
    monkeypatch.setattr(providers, "_native_provider_selected", False)

    providers.configure_provider(None)

    assert providers.selected_provider_name() == "native-torch"
    assert providers.selected_provider_is_native()


def test_unsupported_feature_does_not_fall_back_to_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(providers, "_provider", UnsupportedProvider())
    monkeypatch.setattr(providers, "_native_provider_selected", False)

    with pytest.raises(pytest.skip.Exception, match="does not support feature"):
        providers.construct_test_feature("unported_feature", None, lambda: FakeFeature())


def test_external_provider_named_native_is_not_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "_tstorch_test_provider", sys.modules[__name__])
    monkeypatch.setattr(providers, "_provider", providers.NativeTorchProvider())
    monkeypatch.setattr(providers, "_native_provider_selected", True)

    providers.configure_provider("_tstorch_test_provider:UnsupportedProvider")

    assert providers.selected_provider_name() == "native-torch"
    assert not providers.selected_provider_is_native()


def test_provider_factory_receives_name_and_params_without_native_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = RecordingProvider()
    monkeypatch.setattr(providers, "_provider", selected)
    monkeypatch.setattr(providers, "_native_provider_selected", False)
    params = [{"lag": 3}]

    result = providers.construct_test_feature(
        "feature", params, lambda: pytest.fail("external provider used native factory")
    )

    assert result.names == ["feature"]
    assert selected.created == ("feature", params)


def test_invalid_provider_specification_is_actionable() -> None:
    with pytest.raises(pytest.UsageError, match="cannot import provider module"):
        providers.configure_provider("_missing_codex_provider:Provider")


def test_provider_without_protocol_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "_tstorch_test_provider", sys.modules[__name__])

    with pytest.raises(pytest.UsageError, match="does not implement FeatureTestProvider"):
        providers.configure_provider("_tstorch_test_provider:InvalidProvider")
