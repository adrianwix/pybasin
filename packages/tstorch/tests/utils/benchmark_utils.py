# pyright: basic
"""Shared pytest-benchmark helpers for tstorch feature calculators."""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import os
import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor
from tstorch.feature_factory import Feature, construct_feature, is_parameterless
from tstorch.supported_features import SUPPORTED_FEATURES
from tstorch.torch_feature_extractor import TorchFeatureExtractor

try:
    import tsfresh.feature_extraction.feature_calculators as tsfresh_calculators
    from tsfresh import extract_features as tsfresh_extract_features
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent.
    tsfresh_calculators = None
    tsfresh_extract_features = None

# Disk cache for tsfresh results (gitignored - see .gitignore). tsfresh's own reference
# calculators are slow enough (extract_features on 10,000 trajectories: multi-second per
# feature; approximate_entropy/sample_entropy: O(T^2)) that re-running the same call on
# every iteration of a tstorch rewrite is real wasted time, not just test overhead. Keyed
# by the *resolved* tsfresh call (function/feature name + kwargs actually passed + the
# exact input data), not by FeatureSpec identity, so it's correct even across differently
# constructed specs that happen to resolve to the same call, and there is no staleness
# risk from reusing an entry - the key already captures everything that determines the
# result. Invalidate by deleting the directory; nothing here does that automatically.
TSFRESH_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cached_tsfresh"


def _tsfresh_cache_key(kind: str, name: str, payload: dict[str, Any], y_bst: Tensor) -> str:
    hasher = hashlib.sha256()
    hasher.update(kind.encode())
    hasher.update(name.encode())
    hasher.update(repr(sorted(payload.items(), key=lambda kv: kv[0])).encode())
    _update_tensor_digest(hasher, y_bst)
    return hasher.hexdigest()


def _update_tensor_digest(hasher: Any, y_bst: Tensor) -> None:
    y_np = y_bst.detach().cpu().numpy()
    hasher.update(str(y_np.dtype).encode())
    hasher.update(str(y_np.shape).encode())
    hasher.update(y_np.tobytes())


def _tsfresh_version() -> str:
    try:
        return version("tsfresh")
    except PackageNotFoundError:
        return "unknown"


def _tsfresh_extract_features_n_jobs() -> int:
    return os.cpu_count() or 1


def _load_tsfresh_cache(key: str) -> Any:
    path = TSFRESH_CACHE_DIR / f"{key}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)


def _save_tsfresh_cache(key: str, value: Any) -> None:
    TSFRESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = TSFRESH_CACHE_DIR / f"{key}.pkl"
    with path.open("wb") as file:
        pickle.dump(value, file)


Backend = Literal["tstorch_cpu", "tstorch_gpu", "tsfresh"]
TsfreshArgumentBatching = Literal[
    "proxy_loops_per_argument",
    "calculator_gets_param_list_and_shares_work",
    "calculator_gets_param_list_but_loops_arguments",
    "unsupported",
]


@dataclass(frozen=True)
class FeatureSpec:
    """One benchmarkable feature and its cross-library parameter mapping."""

    name: str
    domain: str
    kwargs: dict[str, Any] | None = None
    tsfresh_name: str | None = None
    tsfresh_kwargs: dict[str, Any] | None = None
    tsfresh_param: list[dict[str, Any]] | None = None
    supports_tstorch_direct: bool = True
    supports_tsfresh: bool = True
    supports_batched_extractor: bool = True
    # Full params list passed to a migrated `Feature` class's constructor
    # (`SUPPORTED_FEATURES[name](params)(x)`) - set only once that feature has moved
    # off the old `DEPRECATED_FeatureFactory` machinery. Falls back to `[kwargs or {}]`
    # when unset, so an already-`kwargs`-configured spec keeps working unchanged the
    # moment its function entry is swapped for a `Feature` class, no spec edit required
    # at migration time.
    params: list[dict[str, Any]] | None = None
    # Required before adding a row-7 batched-argument benchmark for a tsfresh-backed feature.
    # Do not infer this from the presence of `tsfresh_param`: tsfresh's proxy API accepts
    # `list[dict]` for both `fctype="combiner"` calculators and `fctype="simple"` calculators,
    # but the simple path loops over `func(x, **param)` once per argument. Even combiner functions
    # need their raw calculator body checked to determine whether they share work or merely loop.
    tsfresh_argument_batching: TsfreshArgumentBatching | None = None
    tsfresh_argument_batching_source: str | None = None


@dataclass(frozen=True)
class BenchmarkSeriesData:
    """Cached trajectories used by feature benchmarks."""

    time: Tensor
    y_bst: Tensor
    case_tensors: dict[str, Tensor]
    case_times: dict[str, Tensor]
    case_slices: dict[str, tuple[int, int]]
    metadata: dict[str, Any]


def run_once(
    benchmark: Any,
    func: Callable[[], Any],
    *,
    cleanup_cuda: bool = False,
    rounds: int = 1,
) -> Any:
    """Run `rounds` independent invocations with no artificial warmup.

    Still zero explicit warmup (`warmup_rounds=0`) - round 1 is a genuine cold call - but
    `rounds > 1` lets pytest-benchmark report mean/stddev/min/max across independent calls
    instead of a single anecdotal number, without hiding real per-call cost behind a warmup.
    """
    if cleanup_cuda:
        clean_cuda_memory()
    result = benchmark.pedantic(func, rounds=rounds, iterations=1, warmup_rounds=0)
    if cleanup_cuda:
        synchronize_if_cuda(result)
        del result
        clean_cuda_memory()
        return None
    return result


def clean_cuda_memory() -> None:
    """Synchronize and release cached CUDA memory between benchmarked runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def synchronize_if_cuda(value: Any) -> Any:
    """Synchronize CUDA work if the benchmarked output lives on CUDA."""
    if isinstance(value, Tensor) and value.is_cuda:
        torch.cuda.synchronize(value.device)
    elif isinstance(value, (list, tuple)):
        for item in value:
            synchronize_if_cuda(item)
    elif isinstance(value, dict):
        for item in value.values():
            synchronize_if_cuda(item)
    return value


def _params_for_spec(spec: FeatureSpec) -> list[dict[str, Any]]:
    return spec.params if spec.params is not None else [spec.kwargs or {}]


def _extractor_params_for_spec(spec: FeatureSpec) -> list[dict[str, Any]] | None:
    params = _params_for_spec(spec)
    return None if params == [{}] and spec.kwargs is None and spec.params is None else params


def tsfresh_argument_batching(spec: FeatureSpec) -> TsfreshArgumentBatching:
    """Return the researched tsfresh argument-batching mechanism for ``spec``.

    This must be populated from the raw tsfresh calculator source before row-7
    batched-argument benchmarks are added. ``fctype`` is a useful dispatcher check, but not
    sufficient: a combiner receives the full param list, yet its own body may still loop without
    sharing meaningful computation.
    """
    if not spec.supports_tsfresh:
        return "unsupported"
    if spec.tsfresh_argument_batching is None:
        pytest.fail(
            f"{spec.name} is missing researched tsfresh_argument_batching metadata; "
            "inspect tsfresh.feature_extraction.feature_calculators before running "
            "batched-argument benchmarks"
        )
    if spec.tsfresh_argument_batching_source is None:
        pytest.fail(f"{spec.name} is missing tsfresh_argument_batching_source")
    return spec.tsfresh_argument_batching


def tsfresh_dispatcher_fctype(spec: FeatureSpec) -> str | None:
    """Return tsfresh's dispatcher fctype for diagnostics; not the final batching verdict."""
    if tsfresh_calculators is None or not spec.supports_tsfresh:
        return None
    name = spec.tsfresh_name or spec.name
    if not hasattr(tsfresh_calculators, name):
        return None
    func = getattr(tsfresh_calculators, name)
    return getattr(func, "fctype", None)


def _record_tsfresh_argument_batching(
    benchmark: Any, spec: FeatureSpec, params: list[dict[str, Any]]
) -> None:
    batching = tsfresh_argument_batching(spec)
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info["tsfresh_argument_batching"] = batching
        benchmark.extra_info["tsfresh_argument_count"] = len(params)
        benchmark.extra_info["tsfresh_dispatcher_fctype"] = tsfresh_dispatcher_fctype(spec)
        benchmark.extra_info["tsfresh_argument_batching_source"] = (
            spec.tsfresh_argument_batching_source
        )


def _resolved_tsfresh_payload(spec: FeatureSpec) -> tuple[str, dict[str, Any]]:
    name = spec.tsfresh_name or spec.name
    if spec.tsfresh_param is not None:
        return name, {"param": spec.tsfresh_param}
    kwargs = spec.tsfresh_kwargs if spec.tsfresh_kwargs is not None else spec.kwargs or {}
    return name, kwargs


def _resolved_tsfresh_extract_features_payload(spec: FeatureSpec) -> tuple[str, dict[str, Any]]:
    feature_name = spec.tsfresh_name or spec.name
    if spec.tsfresh_param is not None:
        return feature_name, {feature_name: spec.tsfresh_param}
    if spec.tsfresh_kwargs is not None:
        return feature_name, {feature_name: [spec.tsfresh_kwargs]}
    if spec.kwargs is not None:
        return feature_name, {feature_name: [spec.kwargs]}
    return feature_name, {feature_name: None}


def _tsfresh_measurement_key(
    *,
    variant: str,
    kind: str,
    feature_name: str,
    payload: dict[str, Any],
    inputs: Tensor | dict[str, Tensor],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(_tsfresh_version().encode())
    hasher.update(variant.encode())
    hasher.update(kind.encode())
    if kind == "extract_features":
        hasher.update(str(_tsfresh_extract_features_n_jobs()).encode())
    hasher.update(feature_name.encode())
    hasher.update(repr(sorted(payload.items(), key=lambda kv: kv[0])).encode())
    if isinstance(inputs, dict):
        for name, tensor in sorted(inputs.items()):
            hasher.update(name.encode())
            _update_tensor_digest(hasher, tensor)
    else:
        _update_tensor_digest(hasher, inputs)
    return hasher.hexdigest()


def _record_tsfresh_measurement(
    benchmark: Any,
    *,
    variant: str,
    kind: str,
    feature_name: str,
    payload: dict[str, Any],
    inputs: Tensor | dict[str, Tensor],
) -> None:
    if hasattr(benchmark, "extra_info"):
        benchmark.extra_info["tsfresh_version"] = _tsfresh_version()
        benchmark.extra_info["tsfresh_measurement_key"] = _tsfresh_measurement_key(
            variant=variant,
            kind=kind,
            feature_name=feature_name,
            payload=payload,
            inputs=inputs,
        )
        benchmark.extra_info["tsfresh_measurement_kind"] = kind
        if kind == "extract_features":
            benchmark.extra_info["tsfresh_n_jobs"] = _tsfresh_extract_features_n_jobs()


def _skip_cached_tsfresh_measurement(
    *,
    variant: str,
    spec: FeatureSpec,
    feature_name: str,
    payload: dict[str, Any],
    inputs: Tensor | dict[str, Tensor],
    kind: str,
) -> None:
    if os.environ.get("TSTORCH_BENCHMARK_REMEASURE_TSFRESH") == "1":
        return
    store_path_str = os.environ.get("TSTORCH_BENCHMARK_STORE")
    if not store_path_str:
        return
    store_path = Path(store_path_str)
    if not store_path.exists():
        return
    expected_key = _tsfresh_measurement_key(
        variant=variant,
        kind=kind,
        feature_name=feature_name,
        payload=payload,
        inputs=inputs,
    )
    store = json.loads(store_path.read_text())
    existing = store.get(spec.name, {}).get("performance", {}).get(variant, {}).get("tsfresh")
    if existing is None:
        return
    if existing.get("tsfresh_measurement_key") == expected_key:
        pytest.skip(
            "cached tsfresh benchmark measurement is current for "
            f"{spec.name}/{variant} on tsfresh {_tsfresh_version()}"
        )


def execute_tstorch_direct(y_bst: Tensor, spec: FeatureSpec, device: str) -> Tensor:
    """Construct ``spec``'s ``Feature`` class directly and call it on ``(B, S, T)`` data,
    bypassing `TorchFeatureExtractor` entirely (no impute/normalize/kernel fusion) - isolates
    the raw calculator's own cost.

    Looks the class up in `SUPPORTED_FEATURES` - the same registry `TorchFeatureExtractor`
    itself uses - so a feature not yet wired in there skips rather than failing; there is no
    separate direct-call registry.
    """
    if spec.name not in SUPPORTED_FEATURES:
        pytest.skip(f"{spec.name} is not wired into TorchFeatureExtractor yet")
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        y_bst = y_bst.to("cuda")

    feature_cls = SUPPORTED_FEATURES[spec.name]
    ctor_params = None if is_parameterless(feature_cls) else _params_for_spec(spec)
    result = construct_feature(feature_cls, ctor_params)(y_bst)
    synchronize_if_cuda(result)
    return result


def execute_tstorch_extractor(y_bst: Tensor, spec: FeatureSpec, device: str) -> Tensor:
    """Execute one feature through the public tstorch extractor."""
    if not spec.supports_batched_extractor or spec.name not in SUPPORTED_FEATURES:
        pytest.skip(f"{spec.name} is not wired into TorchFeatureExtractor yet")
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        y_bst = y_bst.to("cuda")

    extractor = TorchFeatureExtractor(
        features={spec.name: _extractor_params_for_spec(spec)},
    )
    result = extractor.extract_features(y_bst)
    synchronize_if_cuda(result)
    return result


def execute_tsfresh_direct(
    y_bst: Tensor, spec: FeatureSpec, *, use_cache: bool = True
) -> list[Any]:
    """Execute one tsfresh calculator directly for all trajectories and states.

    Disk-cached under ``TSFRESH_CACHE_DIR`` - see that constant's docstring.
    """
    if tsfresh_calculators is None:
        pytest.skip("tsfresh is not installed")
    if not spec.supports_tsfresh:
        pytest.skip(f"{spec.name} has no tsfresh calculator mapping")

    name, kwargs = _resolved_tsfresh_payload(spec)
    if not hasattr(tsfresh_calculators, name):
        pytest.skip(f"tsfresh has no calculator named {name}")

    cache_key = _tsfresh_cache_key("direct", name, kwargs, y_bst)
    if use_cache:
        cached = _load_tsfresh_cache(cache_key)
        if cached is not None:
            return cached

    func = getattr(tsfresh_calculators, name)
    y_np = y_bst.detach().cpu().numpy()
    time_index = pd.to_timedelta(np.arange(y_np.shape[2]), unit="s")
    results: list[Any] = []
    for batch_idx in range(y_np.shape[0]):
        for state_idx in range(y_np.shape[1]):
            series = pd.Series(y_np[batch_idx, state_idx], index=time_index)
            results.append(func(series, **kwargs))
    if use_cache:
        _save_tsfresh_cache(cache_key, results)
    return results


def execute_tsfresh_extract_features(
    y_bst: Tensor, spec: FeatureSpec, *, use_cache: bool = True
) -> pd.DataFrame:
    """Execute one feature through tsfresh.extract_features over all B*S trajectories at once.

    Uses all visible CPU cores for tsfresh's worker pool. Forcing ``n_jobs=1`` here would
    single-thread tsfresh while tstorch keeps its own vectorized batching, an unfair comparison
    for a scale benchmark.

    Disk-cached under ``TSFRESH_CACHE_DIR`` (see that constant's docstring) - by far the
    most expensive tsfresh call in this benchmark suite (multi-second per feature on
    10,000 trajectories), so caching this one gives the largest iteration-speed win.
    """
    if tsfresh_extract_features is None:
        pytest.skip("tsfresh is not installed")
    if not spec.supports_tsfresh:
        pytest.skip(f"{spec.name} has no tsfresh feature mapping")

    feature_name, fc_parameters = _resolved_tsfresh_extract_features_payload(spec)

    cache_key = _tsfresh_cache_key("extract_features", feature_name, fc_parameters, y_bst)
    if use_cache:
        cached = _load_tsfresh_cache(cache_key)
        if cached is not None:
            return cached

    df = to_tsfresh_frame(y_bst)
    result = tsfresh_extract_features(
        df,
        column_id="id",
        column_sort="time",
        default_fc_parameters=fc_parameters,
        disable_progressbar=True,
        n_jobs=_tsfresh_extract_features_n_jobs(),
    )
    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"Expected tsfresh.extract_features to return a DataFrame, got {type(result)}"
        )
    if use_cache:
        _save_tsfresh_cache(cache_key, result)
    return result


def to_tsfresh_frame(y_bst: Tensor) -> pd.DataFrame:
    """Convert ``(B, S, T)`` data into tsfresh wide dataframe format."""
    y_np = y_bst.detach().cpu().numpy()
    n_batch, n_states, n_time = y_np.shape
    y_flat = y_np.transpose(0, 2, 1).reshape(-1, n_states)
    data: dict[str, Any] = {
        "id": np.repeat(np.arange(n_batch), n_time),
        "time": np.tile(np.arange(n_time), n_batch),
    }
    for state_idx in range(n_states):
        data[f"state_{state_idx}"] = y_flat[:, state_idx]
    return pd.DataFrame(data)


# Number of independent (still zero-warmup) rounds for the single-feature benchmark - was 1
# round, noisy enough (see the tstorch-feature-perf skill) that agents had to manually re-run
# the whole pytest invocation 2-3 times to trust a result. 5 rounds bakes that into one run.
SINGLE_FEATURE_BENCHMARK_ROUNDS = 5


def run_single_trajectory_direct_call_uncompiled_benchmark(
    benchmark: Any,
    canonical_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> Any:
    case_tensors = canonical_trajectories.case_tensors
    if backend == "tstorch_cpu":
        if not spec.supports_tstorch_direct:
            pytest.skip(f"{spec.name} is not implemented for the tstorch direct benchmark")
        return run_once(
            benchmark,
            lambda: [execute_tstorch_direct(y_bst, spec, "cpu") for y_bst in case_tensors.values()],
            rounds=SINGLE_FEATURE_BENCHMARK_ROUNDS,
        )
    if backend == "tstorch_gpu":
        if not spec.supports_tstorch_direct:
            pytest.skip(f"{spec.name} is not implemented for the tstorch direct benchmark")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        return run_once(
            benchmark,
            lambda: [
                execute_tstorch_direct(y_bst, spec, "cuda") for y_bst in case_tensors.values()
            ],
            cleanup_cuda=True,
            rounds=SINGLE_FEATURE_BENCHMARK_ROUNDS,
        )
    if tsfresh_calculators is None:
        pytest.skip("tsfresh is not installed")
    if not spec.supports_tsfresh:
        pytest.skip(f"{spec.name} has no tsfresh calculator mapping")
    name = spec.tsfresh_name or spec.name
    if not hasattr(tsfresh_calculators, name):
        pytest.skip(f"tsfresh has no calculator named {name}")
    name, payload = _resolved_tsfresh_payload(spec)
    variant = "single_trajectory_direct_call_uncompiled"
    _skip_cached_tsfresh_measurement(
        variant=variant,
        spec=spec,
        feature_name=name,
        payload=payload,
        inputs=case_tensors,
        kind="direct",
    )
    _record_tsfresh_measurement(
        benchmark,
        variant=variant,
        kind="direct",
        feature_name=name,
        payload=payload,
        inputs=case_tensors,
    )
    return run_once(
        benchmark,
        lambda: [
            execute_tsfresh_direct(y_bst, spec, use_cache=False) for y_bst in case_tensors.values()
        ],
        rounds=SINGLE_FEATURE_BENCHMARK_ROUNDS,
    )


def run_single_trajectory_direct_call_compiled_benchmark(
    benchmark: Any,
    canonical_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> Any:
    if backend == "tsfresh":
        pytest.skip("tsfresh direct-call baseline is reused from the uncompiled variant")
    if not spec.supports_tstorch_direct or spec.name not in SUPPORTED_FEATURES:
        pytest.skip(f"{spec.name} is not wired into TorchFeatureExtractor yet")
    device = "cuda" if backend == "tstorch_gpu" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    case_tensors = {
        case_name: y_bst.to(device)
        for case_name, y_bst in canonical_trajectories.case_tensors.items()
    }
    feature_cls = SUPPORTED_FEATURES[spec.name]
    params = _params_for_spec(spec)
    compiled_features: dict[str, Callable[[Tensor], Tensor]] = {}
    for case_name, y_bst in case_tensors.items():
        ctor_params = None if is_parameterless(feature_cls) else params
        feature = construct_feature(feature_cls, ctor_params)
        compiled_feature = torch.compile(feature, dynamic=True)
        synchronize_if_cuda(compiled_feature(y_bst))
        compiled_features[case_name] = compiled_feature

    def call() -> list[Tensor]:
        results = [compiled_features[case_name](y_bst) for case_name, y_bst in case_tensors.items()]
        for result in results:
            synchronize_if_cuda(result)
        return results

    return run_once(
        benchmark,
        call,
        cleanup_cuda=(device == "cuda"),
        rounds=SINGLE_FEATURE_BENCHMARK_ROUNDS,
    )


def run_batched_trajectories_proxy_call_compiled_cold_run_benchmark(
    benchmark: Any,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> Any:
    y_bst = pendulum_grid_trajectories.y_bst
    if backend == "tstorch_cpu":
        if not spec.supports_batched_extractor:
            pytest.skip(f"{spec.name} is not enabled for the batched extractor benchmark")
        return run_once(
            benchmark,
            lambda: execute_tstorch_extractor(y_bst, spec, "cpu"),
        )
    if backend == "tstorch_gpu":
        if not spec.supports_batched_extractor:
            pytest.skip(f"{spec.name} is not enabled for the batched extractor benchmark")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        return run_once(
            benchmark,
            lambda: execute_tstorch_extractor(y_bst, spec, "cuda"),
            cleanup_cuda=True,
        )
    pytest.skip(
        "tsfresh's 10,000-trajectory proxy-call number lives on the asymmetric direct/proxy "
        "benchmark - see "
        "test_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_<domain>, "
        "not duplicated here"
    )


def _run_batched_trajectories_proxy_call_benchmark(
    benchmark: Any,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
    *,
    use_compile: bool,
) -> Any:
    if not spec.supports_batched_extractor or spec.name not in SUPPORTED_FEATURES:
        pytest.skip(f"{spec.name} is not wired into TorchFeatureExtractor yet")
    if backend == "tsfresh":
        pytest.skip("tsfresh has no extractor/compile step to warm up")
    device = "cuda" if backend == "tstorch_gpu" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    y_bst = pendulum_grid_trajectories.y_bst.to(device)
    extractor = TorchFeatureExtractor(
        features={spec.name: _extractor_params_for_spec(spec)},
        use_compile=use_compile,
    )
    if use_compile:
        synchronize_if_cuda(extractor.extract_features(y_bst))

    def call() -> Tensor:
        result = extractor.extract_features(y_bst)
        synchronize_if_cuda(result)
        return result

    return run_once(
        benchmark,
        call,
        cleanup_cuda=(device == "cuda"),
        rounds=BATCHED_TRAJECTORIES_BENCHMARK_ROUNDS,
    )


def run_batched_trajectories_proxy_call_compiled_benchmark(
    benchmark: Any,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> Any:
    return _run_batched_trajectories_proxy_call_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend, use_compile=True
    )


def run_batched_trajectories_proxy_call_uncompiled_benchmark(
    benchmark: Any,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> Any:
    return _run_batched_trajectories_proxy_call_benchmark(
        benchmark, pendulum_grid_trajectories, spec, backend, use_compile=False
    )


def run_single_feature_extractor_proxy_call_benchmark(
    benchmark: Any,
    y_bst: Tensor,
    feature_name: str,
    feature_params: list[dict[str, Any]] | None,
    device: str,
    *,
    use_compile: bool,
) -> Any:
    """Small-scale `TorchFeatureExtractor` proxy-call benchmark shape, generalized past a
    single feature/domain (takes `feature_name`/`feature_params` directly) - used by the
    generic synthetic-`Scale`-feature validation test, not by any per-domain benchmark file.

    Compiled runs perform one untimed call to pay compile cost before timed rounds.
    Uncompiled runs do not warm up.
    """
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        y_bst = y_bst.to("cuda")

    extractor = TorchFeatureExtractor(
        features={feature_name: feature_params},
        use_compile=use_compile,
    )
    if use_compile:
        synchronize_if_cuda(extractor.extract_features(y_bst))  # untimed: triggers compilation

    def call() -> Tensor:
        result = extractor.extract_features(y_bst)
        synchronize_if_cuda(result)
        return result

    return run_once(
        benchmark,
        call,
        cleanup_cuda=(device == "cuda"),
        rounds=SINGLE_FEATURE_BENCHMARK_ROUNDS,
    )


# 10,000-trajectory calls are expensive enough (tsfresh: ~1.3s/call for agg_autocorrelation)
# that 5 rounds is still fast (~7s), but a single round's stddev=0 is not a real measurement -
# same reasoning as SINGLE_FEATURE_BENCHMARK_ROUNDS above.
BATCHED_TRAJECTORIES_BENCHMARK_ROUNDS = 5


def run_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_benchmark(
    benchmark: Any,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
) -> Any:
    """Benchmark one feature's direct-call implementation on the 10,000-trajectory
    pendulum grid dataset, bypassing TorchFeatureExtractor. One vectorized call over the
    whole tensor, not a per-trajectory loop.
    """
    y_bst = pendulum_grid_trajectories.y_bst
    if backend == "tstorch_cpu":
        return run_once(
            benchmark,
            lambda: execute_tstorch_direct(y_bst, spec, "cpu"),
            rounds=BATCHED_TRAJECTORIES_BENCHMARK_ROUNDS,
        )
    if backend == "tstorch_gpu":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        return run_once(
            benchmark,
            lambda: execute_tstorch_direct(y_bst, spec, "cuda"),
            cleanup_cuda=True,
            rounds=BATCHED_TRAJECTORIES_BENCHMARK_ROUNDS,
        )
    if tsfresh_extract_features is None:
        pytest.skip("tsfresh is not installed")
    if not spec.supports_tsfresh:
        pytest.skip(f"{spec.name} has no tsfresh feature mapping")
    name, payload = _resolved_tsfresh_extract_features_payload(spec)
    variant = "batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call"
    _skip_cached_tsfresh_measurement(
        variant=variant,
        spec=spec,
        feature_name=name,
        payload=payload,
        inputs=y_bst,
        kind="extract_features",
    )
    _record_tsfresh_measurement(
        benchmark,
        variant=variant,
        kind="extract_features",
        feature_name=name,
        payload=payload,
        inputs=y_bst,
    )
    return run_once(
        benchmark,
        lambda: execute_tsfresh_extract_features(y_bst, spec, use_cache=False),
        rounds=BATCHED_TRAJECTORIES_BENCHMARK_ROUNDS,
    )


ARGUMENT_COUNT_BENCHMARK_ROUNDS = 5


def run_feature_class_benchmark(
    benchmark: Any,
    y_bst: Tensor,
    feature_cls: type[Feature[Any]],
    params: list[dict[str, Any]],
    device: str,
) -> Any:
    """Benchmark one `Feature` class constructed with a specific `params` list, called
    once on `y_bst`. Hold `y_bst` fixed at many trajectories and vary `len(params)`
    across separate calls to measure how a shared-computation win grows with argument
    count - a different axis from `run_single_trajectory_direct_call_uncompiled_benchmark`/
    `run_batched_trajectories_tstorch_direct_call_uncompiled_tsfresh_proxy_call_benchmark`,
    which vary trajectory count (`B*S`) at a fixed, small param count instead.

    For the tsfresh-side comparison at the same `params`, reuse
    `execute_tsfresh_extract_features` directly with `dataclasses.replace(spec,
    tsfresh_param=params)` - no separate helper needed, tsfresh's own `extract_features`
    already accepts an arbitrary param list and shares work internally exactly as much as
    tsfresh itself does for that feature (fully for `fctype=combiner`, not at all for
    `fctype=simple` - see `references/api-design.md`'s "Decision tree").
    """
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        y_bst = y_bst.to("cuda")
    feature = feature_cls(params)
    return run_once(
        benchmark,
        lambda: synchronize_if_cuda(feature(y_bst)),
        cleanup_cuda=(device == "cuda"),
        rounds=ARGUMENT_COUNT_BENCHMARK_ROUNDS,
    )


def _run_batched_trajectories_proxy_call_batched_arguments_benchmark(
    benchmark: Any,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
    params: list[dict[str, Any]],
    *,
    use_compile: bool,
    tsfresh_params: list[dict[str, Any]] | None = None,
) -> Any:
    y_bst = pendulum_grid_trajectories.y_bst
    if backend == "tsfresh":
        if use_compile:
            pytest.skip("tsfresh has no compile step - already measured in the uncompiled variant")
        if not spec.supports_tsfresh:
            pytest.skip(f"{spec.name} has no tsfresh feature mapping")
        resolved_tsfresh_params = tsfresh_params if tsfresh_params is not None else params
        _record_tsfresh_argument_batching(benchmark, spec, resolved_tsfresh_params)
        tsfresh_spec = dataclasses.replace(spec, tsfresh_param=resolved_tsfresh_params)
        name, payload = _resolved_tsfresh_extract_features_payload(tsfresh_spec)
        variant = f"batched_trajectories_proxy_call_uncompiled_batched_arguments_count{len(params)}"
        _skip_cached_tsfresh_measurement(
            variant=variant,
            spec=spec,
            feature_name=name,
            payload=payload,
            inputs=y_bst,
            kind="extract_features",
        )
        _record_tsfresh_measurement(
            benchmark,
            variant=variant,
            kind="extract_features",
            feature_name=name,
            payload=payload,
            inputs=y_bst,
        )
        return run_once(
            benchmark,
            lambda: execute_tsfresh_extract_features(y_bst, tsfresh_spec, use_cache=False),
            rounds=BATCHED_TRAJECTORIES_BENCHMARK_ROUNDS,
        )

    if not spec.supports_batched_extractor or spec.name not in SUPPORTED_FEATURES:
        pytest.skip(f"{spec.name} is not wired into TorchFeatureExtractor yet")
    device = "cuda" if backend == "tstorch_gpu" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    y_bst = y_bst.to(device)
    extractor = TorchFeatureExtractor(
        features={spec.name: params},
        use_compile=use_compile,
    )
    if use_compile:
        synchronize_if_cuda(extractor.extract_features(y_bst))

    def call() -> Tensor:
        result = extractor.extract_features(y_bst)
        synchronize_if_cuda(result)
        return result

    return run_once(
        benchmark,
        call,
        cleanup_cuda=(device == "cuda"),
        rounds=BATCHED_TRAJECTORIES_BENCHMARK_ROUNDS,
    )


def run_batched_trajectories_proxy_call_uncompiled_batched_arguments_benchmark(
    benchmark: Any,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
    params: list[dict[str, Any]],
    *,
    tsfresh_params: list[dict[str, Any]] | None = None,
) -> Any:
    return _run_batched_trajectories_proxy_call_batched_arguments_benchmark(
        benchmark,
        pendulum_grid_trajectories,
        spec,
        backend,
        params,
        use_compile=False,
        tsfresh_params=tsfresh_params,
    )


def run_batched_trajectories_proxy_call_compiled_batched_arguments_benchmark(
    benchmark: Any,
    pendulum_grid_trajectories: BenchmarkSeriesData,
    spec: FeatureSpec,
    backend: Backend,
    params: list[dict[str, Any]],
    *,
    tsfresh_params: list[dict[str, Any]] | None = None,
) -> Any:
    return _run_batched_trajectories_proxy_call_batched_arguments_benchmark(
        benchmark,
        pendulum_grid_trajectories,
        spec,
        backend,
        params,
        use_compile=True,
        tsfresh_params=tsfresh_params,
    )


def load_or_build_cache(
    path: Path, builder: Callable[[], BenchmarkSeriesData]
) -> BenchmarkSeriesData:
    """Load cached benchmark trajectories or build and persist them."""
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return BenchmarkSeriesData(
            time=payload["time"],
            y_bst=payload["y_bst"],
            case_tensors=payload.get("case_tensors", {}),
            case_times=payload.get("case_times", {}),
            case_slices=payload["case_slices"],
            metadata=payload["metadata"],
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    data = builder()
    torch.save(
        {
            "time": data.time.cpu(),
            "y_bst": data.y_bst.cpu(),
            "case_tensors": {name: value.cpu() for name, value in data.case_tensors.items()},
            "case_times": {name: value.cpu() for name, value in data.case_times.items()},
            "case_slices": data.case_slices,
            "metadata": data.metadata,
        },
        path,
    )
    return data


def filter_steady(time: Tensor, y_tsb: Tensor, time_steady: float) -> tuple[Tensor, Tensor]:
    """Filter a ``(T, B, S)`` solution to the steady-state time window."""
    idx = torch.where(time >= time_steady)[0]
    start = int(idx[0].item()) if idx.numel() else 0
    return time[start:].detach().cpu(), y_tsb[start:].detach().cpu()


def combine_case_data(items: Iterable[tuple[str, Tensor, Tensor]]) -> BenchmarkSeriesData:
    """Combine per-case steady trajectories into a canonical benchmark collection."""
    case_slices: dict[str, tuple[int, int]] = {}
    case_tensors: dict[str, Tensor] = {}
    case_times: dict[str, Tensor] = {}
    start = 0

    materialized = list(items)
    for name, time, y_bst in materialized:
        end = start + y_bst.shape[0]
        case_slices[name] = (start, end)
        start = end
        case_tensors[name] = y_bst.contiguous()
        case_times[name] = time

    return BenchmarkSeriesData(
        time=torch.empty(0, dtype=torch.float32),
        y_bst=torch.empty(0, dtype=torch.float32),
        case_tensors=case_tensors,
        case_times=case_times,
        case_slices=case_slices,
        metadata={
            "note": "Canonical trajectories preserve per-case state/time shapes.",
            "num_trajectories": sum(y.shape[0] for y in case_tensors.values()),
            "case_shapes": {name: tuple(y.shape) for name, y in case_tensors.items()},
        },
    )


def finite_result_count(value: Any) -> int:
    """Return a coarse materialization count used by smoke checks."""
    if isinstance(value, Tensor):
        return int(torch.isfinite(value.detach().cpu()).sum().item())
    if isinstance(value, pd.DataFrame):
        return int(np.isfinite(value.to_numpy(dtype=float, na_value=np.nan)).sum())
    if isinstance(value, list):
        return len(value)
    return 1 if value is not None else 0


def grid_sample_count(n: int, dimensions: int) -> int:
    """Return the exact grid sample count produced for all-varying dimensions."""
    return int(math.ceil(n ** (1 / dimensions)) ** dimensions)
