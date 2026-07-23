# pyright: basic
from __future__ import annotations

import argparse
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal

import scipy
import torch
import tsfresh
from cwt_peaks_kernel import NumberCwtPeaksKernel
from torch import Tensor
from tsfresh.feature_extraction.feature_calculators import number_cwt_peaks


@dataclass(frozen=True)
class VariantRuntime:
    """
    Prepared runtime paths for one CWT implementation.

    :ivar name: Convolution implementation name.
    :ivar model: Eager tensor kernel.
    :ivar exported: Exported program with a dynamic batch dimension.
    :ivar compiled: Full-graph Inductor callable.
    :ivar aoti: Loaded AOTInductor package.
    :ivar compile_seconds: First-call Inductor compilation time.
    :ivar package_seconds: AOTInductor packaging time.
    :ivar artifact_path: Packaged native artifact.
    """

    name: str
    model: NumberCwtPeaksKernel
    exported: Any
    compiled: Callable[[Tensor], Tensor]
    aoti: Callable[[Tensor], Tensor]
    compile_seconds: float
    package_seconds: float
    artifact_path: Path


def benchmark_ms(function: Callable[[], object], *, rounds: int, warmup_rounds: int) -> float:
    """
    Measure median wall-clock latency.

    :param function: Zero-argument operation to measure.
    :param rounds: Number of measured calls.
    :param warmup_rounds: Number of unmeasured calls.
    :return: Median latency in milliseconds.
    """
    for _ in range(warmup_rounds):
        function()
    samples: list[float] = []
    for _ in range(rounds):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return median(samples)


def scipy_reference(x: Tensor, width_counts: tuple[int, ...]) -> Tensor:
    """
    Evaluate tsfresh's SciPy-backed reference implementation.

    :param x: Tensor of shape ``(batch, states, time)``.
    :param width_counts: Requested inclusive width counts.
    :return: Counts with shape ``(batch, states, len(width_counts))``.
    """
    values = x.detach().cpu().numpy()
    result = [
        [
            [number_cwt_peaks(values[batch, state], width) for width in width_counts]
            for state in range(values.shape[1])
        ]
        for batch in range(values.shape[0])
    ]
    return torch.tensor(result, dtype=x.dtype)


def package_aoti(
    exported: Any,
    output_path: Path,
) -> Path:
    """
    Export and package the kernel with AOTInductor.

    :param exported: Exported shape-specialized feature kernel.
    :param output_path: Destination ``.pt2`` archive.
    :return: Path to the packaged native artifact.
    """
    packaged = torch._inductor.aoti_compile_and_package(  # pyright: ignore[reportAttributeAccessIssue]
        exported, package_path=str(output_path)
    )
    return Path(packaged)


def prepare_variant(
    name: str,
    model: NumberCwtPeaksKernel,
    example: Tensor,
    output_dir: Path,
    *,
    max_batch: int,
) -> VariantRuntime:
    """
    Export, compile, package, and validate one implementation.

    :param name: Convolution implementation name.
    :param model: Shape-specialized feature kernel.
    :param example: Input used to trace and validate compiled paths.
    :param output_dir: Directory for the AOTInductor package.
    :param max_batch: Maximum exported batch size.
    :return: Prepared runtime paths and setup measurements.
    """
    dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=max_batch)}}
    exported = torch.export.export(model, (example,), dynamic_shapes=dynamic_shapes)
    eager = model(example)
    if not torch.equal(exported.module()(example), eager):
        raise AssertionError(f"torch.export changed the {name} output")

    compiled = torch.compile(model, fullgraph=True)
    started = time.perf_counter()
    compiled_result = compiled(example)
    compile_seconds = time.perf_counter() - started
    if not torch.equal(compiled_result, eager):
        raise AssertionError(f"Inductor changed the {name} output")

    artifact_path = output_dir / f"number_cwt_peaks_{name}.pt2"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    package_aoti(exported, artifact_path)
    package_seconds = time.perf_counter() - started
    aoti = torch._inductor.aoti_load_package(  # pyright: ignore[reportAttributeAccessIssue]
        str(artifact_path)
    )
    if not torch.equal(aoti(example), eager):
        raise AssertionError(f"AOTInductor changed the {name} output")

    return VariantRuntime(
        name=name,
        model=model,
        exported=exported,
        compiled=compiled,
        aoti=aoti,
        compile_seconds=compile_seconds,
        package_seconds=package_seconds,
        artifact_path=artifact_path,
    )


def run(args: argparse.Namespace) -> None:
    """
    Run correctness, compilation, packaging, and performance checks.

    :param args: Parsed command-line options.
    """
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    width_counts = tuple(args.width_counts)
    convolution_names: tuple[Literal["conv1d", "direct"], ...] = ("conv1d", "direct")
    models = {
        name: NumberCwtPeaksKernel(
            length=args.timesteps,
            width_counts=width_counts,
            convolution=name,
        )
        for name in convolution_names
    }

    correctness_input = torch.randn(
        (args.correctness_batch, args.states, args.timesteps), dtype=torch.float64
    )
    reference = scipy_reference(correctness_input, width_counts)
    for name, model in models.items():
        eager = model(correctness_input)
        if not torch.equal(eager, reference):
            mismatch_count = int((eager != reference).sum())
            raise AssertionError(f"{name} SciPy parity failed for {mismatch_count} outputs")

    export_batch = max(2, args.correctness_batch)
    export_input = torch.randn((export_batch, args.states, args.timesteps), dtype=torch.float32)
    export_reference = scipy_reference(export_input, width_counts)
    for name, model in models.items():
        if not torch.equal(model(export_input), export_reference):
            raise AssertionError(f"{name} float32 SciPy parity failed")

    torch._inductor.config.cpp.simdlen = 1  # pyright: ignore[reportAttributeAccessIssue]
    variants = [
        prepare_variant(
            name,
            model,
            export_input,
            args.output_dir,
            max_batch=max(args.large_batch, export_batch),
        )
        for name, model in models.items()
    ]

    degenerate_input = torch.ones((1, 1, 31), dtype=torch.float64)
    degenerate_reference = scipy_reference(degenerate_input, width_counts)

    print("Seeded random correctness: exact counts for all SciPy comparisons")
    for name in convolution_names:
        degenerate_model = NumberCwtPeaksKernel(
            length=31,
            width_counts=width_counts,
            convolution=name,
        )
        degenerate_tensor = degenerate_model(degenerate_input)
        print(
            f"Length-31 all-ones {name} counts "
            f"(tensor={degenerate_tensor.flatten().tolist()}, "
            f"SciPy={degenerate_reference.flatten().tolist()})"
        )

    print()
    print("| CWT | export nodes | compile s | package s | artifact bytes |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for variant in variants:
        print(
            f"| {variant.name} | {len(list(variant.exported.graph.nodes))} | "
            f"{variant.compile_seconds:.3f} | {variant.package_seconds:.3f} | "
            f"{variant.artifact_path.stat().st_size} |"
        )

    print()
    print("| batch | path | SciPy/tsfresh ms | tensor eager ms | Inductor ms | AOTI ms |")
    print("| ---: | --- | ---: | ---: | ---: | ---: |")
    for batch in (1, args.large_batch):
        x = torch.randn((batch, args.states, args.timesteps), dtype=torch.float32)
        expected = scipy_reference(x, width_counts)
        scipy_ms = benchmark_ms(
            lambda x=x: scipy_reference(x, width_counts),
            rounds=args.rounds,
            warmup_rounds=args.warmup_rounds,
        )
        for variant in variants:
            if not torch.equal(variant.model(x), expected):
                raise AssertionError(
                    f"{variant.name} SciPy parity failed for benchmark batch {batch}"
                )
            variant.compiled(x)
            variant.aoti(x)
            eager_ms = benchmark_ms(
                lambda variant=variant, x=x: variant.model(x),
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
            )
            compiled_ms = benchmark_ms(
                lambda variant=variant, x=x: variant.compiled(x),
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
            )
            aoti_ms = benchmark_ms(
                lambda variant=variant, x=x: variant.aoti(x),
                rounds=args.rounds,
                warmup_rounds=args.warmup_rounds,
            )
            print(
                f"| {batch} | {variant.name} | {scipy_ms:.6f} | {eager_ms:.6f} | "
                f"{compiled_ms:.6f} | {aoti_ms:.6f} |"
            )

    print()
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"SciPy: {scipy.__version__}")
    print(f"tsfresh: {tsfresh.__version__}")
    print(f"Threads: {args.threads}")
    print("Inductor SIMD length: 1 (workaround for current CPU mask-vector code generation)")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line options.

    :return: Experiment configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--width-counts", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--correctness-batch", type=int, default=8)
    parser.add_argument("--large-batch", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("artifacts"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
