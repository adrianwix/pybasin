from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch
from torch import Tensor, nn
from torch.utils.cpp_extension import include_paths, library_paths
from tstorch.calculators.torch_features_statistical import Maximum, Mean, Minimum
from tstorch.torch_feature_extractor import TorchFeatureExtractor

BASIC_STATS_FEATURES = {"mean": None, "maximum": None, "minimum": None}


@dataclass(frozen=True)
class TimingRow:
    group: str
    path: str
    shape: tuple[int, int, int]
    median_ms: float
    max_abs_diff: float | None = None


class FeatureShim(nn.Module):
    def __init__(self, feature: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.feature = feature

    def forward(self, x: Tensor) -> Tensor:
        return self.feature(x)


class ExtractorGeneratedKernelBasicStats(nn.Module):
    def __init__(self, *, states: int) -> None:
        super().__init__()
        extractor = TorchFeatureExtractor(
            features=BASIC_STATS_FEATURES,
            normalize=False,
            impute_method="extreme",
            use_compile=False,
        )
        extractor._compile(states, torch.device("cpu"))
        assert extractor._compiled_plan is not None
        self.kernel = extractor._compiled_plan.kernel

    def forward(self, x: Tensor) -> Tensor:
        return self.kernel(x)


class TorchFeatureExtractorBasicStats(nn.Module):
    def __init__(self, *, use_compile: bool) -> None:
        super().__init__()
        self.extractor = TorchFeatureExtractor(
            features=BASIC_STATS_FEATURES,
            normalize=False,
            impute_method="extreme",
            use_compile=use_compile,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.extractor.extract_features(x)


def benchmark_ms(
    fn: Callable[[Tensor], Tensor],
    x: Tensor,
    *,
    rounds: int,
    warmup_rounds: int,
) -> float:
    with torch.inference_mode():
        for _ in range(warmup_rounds):
            fn(x)

        values: list[float] = []
        for _ in range(rounds):
            started = time.perf_counter_ns()
            fn(x)
            values.append((time.perf_counter_ns() - started) / 1e6)
    return median(values)


def export_and_package(
    module: nn.Module,
    package_path: Path,
    *,
    export_batch: int,
    max_batch: int,
    states: int,
    timesteps: int,
) -> Path:
    x_export = torch.randn((export_batch, states, timesteps), dtype=torch.float32)
    dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=max_batch)}}
    with torch.no_grad():
        exported = torch.export.export(module, (x_export,), dynamic_shapes=dynamic_shapes)
        packaged = torch._inductor.aoti_compile_and_package(  # pyright: ignore[reportAttributeAccessIssue]
            exported,
            package_path=str(package_path),
        )
    return Path(packaged)


def cxx_compile_command(output_path: Path) -> list[str]:
    source_path = Path(__file__).with_name("aoti_basic_stats_runner.cpp")
    command = [
        "c++",
        "-std=c++17",
        "-O3",
        "-march=native",
        str(source_path),
        "-o",
        str(output_path),
    ]
    for include_path in include_paths():
        command.append(f"-I{include_path}")
    for library_path in library_paths():
        command.append(f"-L{library_path}")
        command.append(f"-Wl,-rpath,{library_path}")
    command.extend(["-ltorch", "-ltorch_cpu", "-lc10", "-pthread"])
    return command


def cxx_run_command(
    runner_path: Path,
    extractor_package: Path,
    mean_package: Path,
    maximum_package: Path,
    minimum_package: Path,
    *,
    large_batch: int,
    states: int,
    timesteps: int,
    rounds: int,
    threads: int,
) -> list[str]:
    return [
        str(runner_path),
        str(extractor_package),
        str(mean_package),
        str(maximum_package),
        str(minimum_package),
        str(large_batch),
        str(states),
        str(timesteps),
        str(rounds),
        str(threads),
    ]


def run_python_benchmark(
    *,
    large_batch: int,
    states: int,
    timesteps: int,
    rounds: int,
    warmup_rounds: int,
) -> list[TimingRow]:
    extractor_eager = TorchFeatureExtractorBasicStats(use_compile=False)
    extractor_compiled = TorchFeatureExtractorBasicStats(use_compile=True)
    inputs = [
        torch.randn((1, states, timesteps), dtype=torch.float32),
        torch.randn((large_batch, states, timesteps), dtype=torch.float32),
    ]

    rows: list[TimingRow] = []
    for x in inputs:
        baseline = extractor_eager(x)
        compiled_result = extractor_compiled(x)
        rows.append(
            TimingRow(
                group="combined",
                path="TorchFeatureExtractor(use_compile=False), Python",
                shape=tuple(x.shape),
                median_ms=benchmark_ms(
                    extractor_eager,
                    x,
                    rounds=rounds,
                    warmup_rounds=warmup_rounds,
                ),
                max_abs_diff=0.0,
            )
        )
        rows.append(
            TimingRow(
                group="combined",
                path="TorchFeatureExtractor(use_compile=True), Python",
                shape=tuple(x.shape),
                median_ms=benchmark_ms(
                    extractor_compiled,
                    x,
                    rounds=rounds,
                    warmup_rounds=warmup_rounds,
                ),
                max_abs_diff=(baseline - compiled_result).abs().max().item(),
            )
        )
    return rows


def parse_cxx_rows(output: str) -> list[TimingRow]:
    rows: list[TimingRow] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) != 7 or parts[0] != "TIMING":
            continue
        _, group, path, batch, states, timesteps, median_ms = parts
        rows.append(
            TimingRow(
                group=group,
                path=path,
                shape=(int(batch), int(states), int(timesteps)),
                median_ms=float(median_ms),
            )
        )
    return rows


def parse_cxx_diffs(output: str) -> dict[str, float]:
    diffs: dict[str, float] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0] == "DIFF":
            diffs[parts[1]] = float(parts[2])
    return diffs


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def markdown_table(rows: list[TimingRow]) -> str:
    lines = [
        "| Group | Path | Shape | Median ms | Max abs diff |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        diff = "" if row.max_abs_diff is None else f"{row.max_abs_diff:.6g}"
        lines.append(f"| {row.group} | {row.path} | `{row.shape}` | {row.median_ms:.6g} | {diff} |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    *,
    python_rows: list[TimingRow],
    cxx_rows: list[TimingRow],
    cxx_diffs: dict[str, float],
    packages: dict[str, Path],
    compile_command: list[str],
    run_command: list[str],
    cxx_output: str | None,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# AOTInductor Basic Statistics Report",
        "",
        "## Configuration",
        "",
        f"- PyTorch: `{torch.__version__}`",
        f"- Python: `{platform.python_version()}`",
        f"- CPU threads for Python: `{torch.get_num_threads()}`",
        f"- C++ large-batch threads: `{args.cxx_threads}`",
        f"- Small shape: `(1, {args.states}, {args.timesteps})`",
        f"- Large shape: `({args.large_batch}, {args.states}, {args.timesteps})`",
        f"- Python rounds: `{args.rounds}`",
        f"- C++ rounds: `{args.cxx_rounds}`",
        "",
        "## Packages",
        "",
    ]
    for name, path in packages.items():
        lines.append(f"- {name}: `{path}`")
    lines.extend([
        "",
        "## Python Timings",
        "",
        markdown_table(python_rows),
        "",
        "## C++ Timings",
        "",
        markdown_table(cxx_rows),
        "",
        "## C++ Correctness Checks",
        "",
    ])
    if cxx_diffs:
        for name, diff in cxx_diffs.items():
            lines.append(f"- {name}: `{diff:.6g}`")
    else:
        lines.append("- C++ runner was not executed.")
    lines.extend([
        "",
        "## Commands",
        "",
        "Compile C++ runner:",
        "",
        "```bash",
        shell_join(compile_command),
        "```",
        "",
        "Run C++ benchmark:",
        "",
        "```bash",
        shell_join(run_command),
        "```",
    ])
    if cxx_output is not None:
        lines.extend([
            "",
            "Raw C++ output:",
            "",
            "```text",
            cxx_output.strip(),
            "```",
        ])
    report_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    default_artifact_dir = Path(__file__).resolve().parent / "artifacts"
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_artifact_dir)
    parser.add_argument("--report-name", default="report.md")
    parser.add_argument("--export-batch", type=int, default=8)
    parser.add_argument("--large-batch", type=int, default=10000)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--cxx-rounds", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cxx-threads", type=int, default=24)
    parser.add_argument("--skip-cxx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.export_batch <= 1:
        raise ValueError("Use --export-batch > 1 when declaring batch dynamic.")

    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    max_batch = max(args.export_batch, args.large_batch)
    packages = {
        "combined TorchFeatureExtractor generated kernel": args.output_dir
        / "extractor_generated_kernel.pt2",
        "separated Mean FeatureClass": args.output_dir / "mean.pt2",
        "separated Maximum FeatureClass": args.output_dir / "maximum.pt2",
        "separated Minimum FeatureClass": args.output_dir / "minimum.pt2",
    }

    exported_packages = {
        "combined TorchFeatureExtractor generated kernel": export_and_package(
            ExtractorGeneratedKernelBasicStats(states=args.states),
            packages["combined TorchFeatureExtractor generated kernel"],
            export_batch=args.export_batch,
            max_batch=max_batch,
            states=args.states,
            timesteps=args.timesteps,
        ),
        "separated Mean FeatureClass": export_and_package(
            FeatureShim(Mean()),
            packages["separated Mean FeatureClass"],
            export_batch=args.export_batch,
            max_batch=max_batch,
            states=args.states,
            timesteps=args.timesteps,
        ),
        "separated Maximum FeatureClass": export_and_package(
            FeatureShim(Maximum()),
            packages["separated Maximum FeatureClass"],
            export_batch=args.export_batch,
            max_batch=max_batch,
            states=args.states,
            timesteps=args.timesteps,
        ),
        "separated Minimum FeatureClass": export_and_package(
            FeatureShim(Minimum()),
            packages["separated Minimum FeatureClass"],
            export_batch=args.export_batch,
            max_batch=max_batch,
            states=args.states,
            timesteps=args.timesteps,
        ),
    }

    python_rows = run_python_benchmark(
        large_batch=args.large_batch,
        states=args.states,
        timesteps=args.timesteps,
        rounds=args.rounds,
        warmup_rounds=args.warmup_rounds,
    )

    runner_path = args.output_dir / "aoti_basic_stats_runner"
    compile_command = cxx_compile_command(runner_path)
    run_command = cxx_run_command(
        runner_path,
        exported_packages["combined TorchFeatureExtractor generated kernel"],
        exported_packages["separated Mean FeatureClass"],
        exported_packages["separated Maximum FeatureClass"],
        exported_packages["separated Minimum FeatureClass"],
        large_batch=args.large_batch,
        states=args.states,
        timesteps=args.timesteps,
        rounds=args.cxx_rounds,
        threads=args.cxx_threads,
    )

    cxx_output: str | None = None
    cxx_rows: list[TimingRow] = []
    cxx_diffs: dict[str, float] = {}
    if not args.skip_cxx:
        subprocess.run(compile_command, check=True)
        completed = subprocess.run(run_command, check=True, capture_output=True, text=True)
        cxx_output = completed.stdout
        cxx_rows = parse_cxx_rows(cxx_output)
        cxx_diffs = parse_cxx_diffs(cxx_output)

    report_path = args.output_dir / args.report_name
    write_report(
        report_path,
        python_rows=python_rows,
        cxx_rows=cxx_rows,
        cxx_diffs=cxx_diffs,
        packages=exported_packages,
        compile_command=compile_command,
        run_command=run_command,
        cxx_output=cxx_output,
        args=args,
    )

    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
