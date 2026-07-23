"""Shared unit-test-only assertions: numerical-accuracy-vs-tsfresh checks and
torch.compile(fullgraph=True)/torch.export safety checks for feature classes.

Split out from `tests/utils/benchmark_utils.py`: these are unit-test concerns (is the
result correct? does it trace cleanly?), not performance-benchmark concerns, even though a
couple of them reuse tsfresh-calling infrastructure (`execute_tsfresh_direct`) that also
serves the benchmark suite.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import torch
from torch import Tensor, nn
from tstorch.feature_factory import (
    Feature,
    ParameterlessFeature,
    construct_feature,
    is_parameterless,
)

from .benchmark_utils import (
    BenchmarkSeriesData,
    FeatureSpec,
    execute_tsfresh_direct,
    tsfresh_dispatcher_fctype,
)
from .compile_recording import record_compile
from .export_recording import record_export
from .precision_recording import record_precision

_compile = cast(
    Callable[..., Callable[[Tensor], Tensor]],
    vars(torch)["compile"],
)


def _draft_export_report(program: object) -> object:
    """Return the diagnostic report attached dynamically by ``draft_export``."""
    return cast(object, vars(program)["_report"])


def assert_feature_matches_tsfresh(
    feature_cls: type[Feature[Any]] | type[ParameterlessFeature],
    spec: FeatureSpec,
    canonical_trajectories: BenchmarkSeriesData,
    params: Sequence[Mapping[str, Any]] | None = None,
    *,
    tsfresh_params: Sequence[Mapping[str, Any]] | None = None,
    input_dtype: torch.dtype | None = None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """Assert one feature class matches tsfresh across every canonical trajectory.

    The same call covers ``ParameterlessFeature`` and parametrized ``Feature`` classes.
    Parametrized features pass all requested params in one constructor call. The utility
    detects tsfresh's simple or combiner calling convention, checks every output column for
    every canonical case, and records one aggregate precision row per feature parameter for
    the generated report.

    Supply ``tsfresh_params`` when tstorch and tsfresh use different parameter names. Its
    entries correspond positionally to ``params``; report rows retain the tstorch params.
    """
    # Normalize both feature APIs into one output-column contract. A parameterless feature has
    # one implicit output; a parametrized feature has one output per entry in `params`.
    parameterless = is_parameterless(feature_cls)
    if parameterless:
        assert params is None, f"{spec.name}: parameter-free feature received params={params}"
        assert tsfresh_params is None, (
            f"{spec.name}: parameter-free feature received tsfresh_params={tsfresh_params}"
        )
        feature_params: list[dict[str, Any]] | None = None
        report_params = [spec.kwargs or {}]
        resolved_tsfresh_params: list[dict[str, Any]] | None = None
    else:
        assert params is not None, f"{spec.name}: parametrized feature requires params"
        feature_params = [dict(param) for param in params]
        assert feature_params, f"{spec.name}: numerical-accuracy params must be non-empty"
        report_params = feature_params
        resolved_tsfresh_params = [
            dict(param) for param in (tsfresh_params if tsfresh_params is not None else params)
        ]
        assert len(resolved_tsfresh_params) == len(feature_params), (
            f"{spec.name}: expected {len(feature_params)} tsfresh params, "
            f"got {len(resolved_tsfresh_params)}"
        )

    feature = construct_feature(feature_cls, feature_params)
    n_outputs = len(report_params)

    # Statistics stay separate per output parameter but accumulate across every canonical
    # dynamical system. The generated documentation therefore reports domain-wide accuracy,
    # rather than whichever trajectory happened to run last.
    diffs_per_output: list[list[float]] = [[] for _ in range(n_outputs)]
    nan_mismatches = [0] * n_outputs
    passed = [True] * n_outputs
    failures: list[str] = []
    recorded_dtype = input_dtype
    for case_name, y_bst in canonical_trajectories.case_tensors.items():
        if input_dtype is not None:
            y_bst = y_bst.to(input_dtype)
        elif recorded_dtype is None:
            recorded_dtype = y_bst.dtype
        result = feature(y_bst)
        expected_shape = (*y_bst.shape[:2], n_outputs)
        assert result.shape == expected_shape, (
            f"{spec.name}/{case_name}: expected {expected_shape}, got {result.shape}"
        )
        assert result.dtype == y_bst.dtype, (
            f"{spec.name}/{case_name}: expected dtype {y_bst.dtype}, got {result.dtype}"
        )
        batch_size, n_states = y_bst.shape[:2]

        # Convert every tsfresh calling convention to `expected_by_output[output][flat_row]`.
        # Simple calculators are called once per parameter; combiners return all parameter
        # results in each flattened (batch, state) row.
        if parameterless:
            expected_by_output = [execute_tsfresh_direct(y_bst, spec)]
        elif tsfresh_dispatcher_fctype(spec) == "combiner":
            combiner_spec = replace(spec, tsfresh_param=resolved_tsfresh_params)
            combined_rows = execute_tsfresh_direct(y_bst, combiner_spec)
            normalized_rows = [list(row) for row in combined_rows]
            expected_by_output = [
                [float(row[output_idx][1]) for row in normalized_rows]
                for output_idx in range(n_outputs)
            ]
        else:
            assert resolved_tsfresh_params is not None
            expected_by_output = [
                execute_tsfresh_direct(
                    y_bst,
                    replace(
                        spec,
                        kwargs=tsfresh_param,
                        tsfresh_kwargs=tsfresh_param,
                        tsfresh_param=None,
                    ),
                )
                for tsfresh_param in resolved_tsfresh_params
            ]

        for output_idx, output_params in enumerate(report_params):
            expected_rows = expected_by_output[output_idx]
            assert len(expected_rows) == batch_size * n_states, (
                f"{spec.name}/{case_name}/{output_params}: expected "
                f"{batch_size * n_states} tsfresh rows, got {len(expected_rows)}"
            )
            for row_idx, expected in enumerate(expected_rows):
                # `execute_tsfresh_direct` flattens in batch-major, then state-major order.
                # Reconstructing those indices is required to compare the same trajectory/state.
                batch_idx, state_idx = divmod(row_idx, n_states)
                got = float(result[batch_idx, state_idx, output_idx].item())
                expected_float = float(expected)
                # NaN parity is recorded explicitly because subtraction cannot distinguish
                # a matching NaN pair from a one-sided semantic mismatch.
                if math.isnan(expected_float) or math.isnan(got):
                    if not (math.isnan(expected_float) and math.isnan(got)):
                        nan_mismatches[output_idx] += 1
                        passed[output_idx] = False
                        failures.append(
                            f"{spec.name}/{output_params} case={case_name} b={batch_idx} "
                            f"s={state_idx}: got {got}, expected {expected_float}"
                        )
                    continue
                diff = 0.0 if got == expected_float else abs(got - expected_float)
                diffs_per_output[output_idx].append(diff)
                if diff > atol + rtol * abs(expected_float):
                    passed[output_idx] = False
                    failures.append(
                        f"{spec.name}/{output_params} case={case_name} b={batch_idx} "
                        f"s={state_idx}: got {got}, expected {expected_float}"
                    )

    # Write one precision row per feature parameter only after all canonical cases contribute.
    # `report.md` and the website documentation consume these aggregate rows directly.
    for output_idx, output_params in enumerate(report_params):
        record_precision(
            spec.name,
            output_params,
            dtype=str(recorded_dtype).removeprefix("torch."),
            rows_compared=len(diffs_per_output[output_idx]),
            total_abs_diff=sum(diffs_per_output[output_idx]),
            max_abs_diff=max(diffs_per_output[output_idx], default=0.0),
            nan_mismatches=nan_mismatches[output_idx],
            passed=passed[output_idx],
            atol=atol,
            rtol=rtol,
        )
    if failures:
        raise AssertionError(
            f"{spec.name}: {len(failures)} mismatch(es):\n" + "\n".join(failures[:20])
        )


def assert_feature_names_are_valid(
    feature_name: str,
    feature_cls: type[Feature[Any]] | type[ParameterlessFeature],
    distinct_params: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Assert every feature produces a non-empty `.names` entry per requested param, and
    that distinct params produce distinct names.

    Omit `distinct_params` (or pass `None`) for a `ParameterlessFeature` - there are no
    params to vary, so this only checks that its one name is non-empty.

    For a `Feature[ParamsT]`, `distinct_params` is required. Constructs
    `feature_cls(distinct_params)` (the batched call) and `feature_cls([p])` per entry (the
    single-param call `TorchFeatureExtractor._compile()` uses to name one column), and
    checks both give the same names - a param must never be silently absorbed into an
    unchanged name regardless of which door constructed the instance. Accepts a `Sequence`
    (not just `list`) so a caller's own `list[SomeFeatureParams]` - a `TypedDict` subtype,
    and `list` is invariant - can be passed without a cast.
    """
    if is_parameterless(feature_cls):
        names = construct_feature(feature_cls, None).names
        assert len(names) == 1, f"{feature_name}: expected exactly 1 name, got {names}"
        assert names[0], f"{feature_name}: every feature must have a non-empty name, got {names}"
        return

    assert distinct_params is not None, (
        f"{feature_name}: distinct_params is required for a parametrized Feature"
    )
    params_list = list(distinct_params)
    batched_names = construct_feature(feature_cls, params_list).names
    assert len(batched_names) == len(params_list), (
        f"{feature_name}: expected {len(params_list)} names, got {len(batched_names)}"
    )
    assert all(batched_names), (
        f"{feature_name}: every feature must have a non-empty name, got {batched_names}"
    )

    singleton_names = [construct_feature(feature_cls, [p]).names[0] for p in params_list]
    assert singleton_names == batched_names, (
        f"{feature_name}: singleton-call names {singleton_names} disagree with "
        f"batched-call names {batched_names} for the same params"
    )

    if len(params_list) > 1:
        assert len(set(batched_names)) == len(batched_names), (
            f"{feature_name}: distinct params must produce distinct names, got {batched_names}"
        )


class _ExportShim(nn.Module):
    """Only exists to satisfy `torch.export.export()`'s `isinstance(mod, nn.Module)` check
    - no parameters, no buffers, no hooks, adds nothing to what's actually traced. A feature
    class itself never becomes an `nn.Module` - this shim is built fresh per call, not part
    of the class design."""

    def __init__(self, fn: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self._fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self._fn(x)


def assert_feature_compiles_fullgraph(
    feature_name: str,
    feature_cls: type[Feature[Any]] | type[ParameterlessFeature],
    params: Sequence[Mapping[str, Any]] | None,
    y_bst: Tensor,
) -> None:
    """Assert the constructed feature is `torch.compile(fullgraph=True)`-safe.

    Constructs the instance directly and compiles it - this already *is* the closure
    `TorchFeatureExtractor._compile()` compiles once the feature is wired into
    `SUPPORTED_FEATURES`, so there is no registry lookup or adapter involved. Works whether
    or not the feature is wired in yet - nothing to skip. Pass `None` for `params` when
    `feature_cls` is a `ParameterlessFeature` - constructed with no arguments.

    Uses `fullgraph=True` (raises instead of silently falling back to eager on a graph break)
    and `options={"trace.enabled": True}` (dumps the Dynamo/Inductor trace under
    `pybasin/torch_compile_debug/` - gitignored - so a break can be diagnosed from the
    artifacts instead of just the exception message - see
    https://docs.pytorch.org/docs/2.12/generated/torch.compile.html), then runs the compiled
    instance once on `y_bst`.

    Records pass/fail (`compile_recording.record_compile`) before asserting, so a failing
    feature still shows up in the report's "torch.compile(fullgraph=True) safety" table
    instead of only failing the test silently. Resets Dynamo's cache afterwards so this
    compile attempt can't leak guards/state into an unrelated test.
    """
    instance = construct_feature(feature_cls, params)
    compiled = _compile(instance, fullgraph=True, options={"trace.enabled": True})
    error: str | None = None
    try:
        compiled(y_bst)
    except Exception as exc:  # noqa: BLE001 - Dynamo raises many distinct exception types for graph breaks
        error = f"{type(exc).__name__}: {exc}"
    finally:
        torch.compiler.reset()

    recorded_params = dict(params[0]) if params else None
    record_compile(feature_name, recorded_params, compiled=error is None, error=error)
    assert error is None, f"{feature_name} is not torch.compile(fullgraph=True)-safe: {error}"


def assert_feature_exports_cleanly(
    feature_name: str,
    feature_cls: type[Feature[Any]] | type[ParameterlessFeature],
    params: Sequence[Mapping[str, Any]] | None,
    y_bst: Tensor,
) -> None:
    """Assert the constructed feature survives `torch.export.export()`, alongside (not
    instead of) `assert_feature_compiles_fullgraph`. A genuinely independent check even
    though both require the same fullgraph-equivalent strictness (zero tolerance for
    untraceable code): `torch.export` uses its own tracer, not Dynamo -> Inductor, so a
    feature can pass one and fail the other. Also the actual prerequisite for ever pursuing
    AOTInductor/C++ deployment. Pass `None` for `params` when `feature_cls` is a
    `ParameterlessFeature` - constructed with no arguments.

    Constructs the instance directly, wrapped in `_ExportShim` since `torch.export.export()`
    requires an `nn.Module` at the top level (verified directly against the installed
    PyTorch: a bare callable raises `ValueError` immediately) - the shim doesn't change what's
    traced, only satisfies that check. No registry lookup involved, so this works whether or
    not the feature is wired into `SUPPORTED_FEATURES` yet.

    On failure, re-runs via `torch.export.draft_export()` instead of just re-raising the
    first exception - draft-export traces through untraceable spots and can produce one
    report listing every graph break encountered, not just the first, which goes straight
    into the raised `AssertionError`. `draft_export` itself is not always able to produce a
    report (confirmed: it can raise its own, different exception on the same input that
    broke `export()` proper, e.g. a data-dependent branch combined with the dynamic batch
    dimension below) - when that happens the original `export()` exception is reported
    instead, not silently swallowed.
    """
    instance = construct_feature(feature_cls, params)
    shim = _ExportShim(instance)
    dynamic_shapes = {"x": {0: torch.export.Dim("batch")}}

    error: str | None = None
    try:
        torch.export.export(shim, (y_bst,), dynamic_shapes=dynamic_shapes)
    except Exception as exc:  # noqa: BLE001 - torch.export raises many distinct exception types
        error = f"{type(exc).__name__}: {exc}"
        try:
            draft = torch.export.draft_export(shim, (y_bst,), dynamic_shapes=dynamic_shapes)
            error = str(_draft_export_report(draft))
        except Exception as draft_exc:  # noqa: BLE001 - draft_export can itself fail to produce a report
            error += f"\n(torch.export.draft_export also failed to produce a report: {draft_exc})"

    recorded_params = dict(params[0]) if params else None
    record_export(feature_name, recorded_params, exported=error is None, error=error)
    assert error is None, f"{feature_name} does not torch.export cleanly:\n{error}"
