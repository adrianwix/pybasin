from __future__ import annotations

import argparse
import importlib.util
import platform
import shlex
import subprocess
import sysconfig
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch
from executorch import exir
from torch import Tensor, nn
from tstorch.torch_feature_extractor import TorchFeatureExtractor

BASIC_STATS_FEATURES = {"mean": None, "maximum": None, "minimum": None}


@dataclass(frozen=True)
class TimingRow:
    group: str
    path: str
    shape: tuple[int, int, int]
    median_ms: float
    max_abs_diff: float | None


class ExtractorGeneratedKernelBasicStats(nn.Module):
    def __init__(self, *, states: int) -> None:
        super().__init__()
        extractor = TorchFeatureExtractor(
            features=BASIC_STATS_FEATURES,
            use_compile=False,
        )
        extractor._compile(states, torch.device("cpu"))
        assert extractor._compiled_plan is not None
        self.kernel = extractor._compiled_plan.kernel

    def forward(self, x: Tensor) -> Tensor:
        return self.kernel(x)


def benchmark_ms(fn, x: Tensor, *, rounds: int, warmup_rounds: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup_rounds):
            fn(x)

        values: list[float] = []
        for _ in range(rounds):
            started = time.perf_counter_ns()
            fn(x)
            values.append((time.perf_counter_ns() - started) / 1e6)
    return median(values)


def export_to_executorch(
    module: nn.Module,
    output_path: Path,
    *,
    export_batch: int,
    max_batch: int,
    states: int,
    timesteps: int,
) -> bytes:
    x_export = torch.randn((export_batch, states, timesteps), dtype=torch.float32)
    dynamic_shapes = {"x": {0: torch.export.Dim("batch", min=1, max=max_batch)}}
    with torch.no_grad():
        exported = torch.export.export(module, (x_export,), dynamic_shapes=dynamic_shapes)
        edge_program = exir.to_edge(exported)
        executorch_program = edge_program.to_executorch()
    output_path.write_bytes(executorch_program.buffer)
    return executorch_program.buffer


def package_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def ldd_output(path: Path) -> str | None:
    if not path.exists():
        return None
    result = subprocess.run(["ldd", str(path)], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        return result.stderr.strip() or result.stdout.strip()
    return result.stdout.strip()


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
                max_abs_diff=None,
            )
        )
    return rows


def site_packages_root() -> Path:
    return Path(sysconfig.get_path("purelib"))


def portable_runtime_so_path() -> Path | None:
    pybindings_dir = site_packages_root() / "executorch" / "extension" / "pybindings"
    matches = sorted(pybindings_dir.glob("_portable_lib*.so"))
    return matches[0] if matches else None


def executorch_version() -> str:
    try:
        import executorch.version as version  # noqa: PLC0415
    except ImportError:
        return "unknown"
    return str(getattr(version, "__version__", "unknown"))


def cxx_compile_command(output_path: Path) -> list[str]:
    source_path = Path(__file__).with_name("executorch_basic_stats_runner.cpp")
    site_packages = site_packages_root()
    python_include = Path(sysconfig.get_path("include"))
    python_libdir = Path(sysconfig.get_config_var("LIBDIR"))
    python_library = (
        str(sysconfig.get_config_var("LDLIBRARY")).removeprefix("lib").removesuffix(".so")
    )
    executorch_root = site_packages / "executorch"
    torch_root = site_packages / "torch"
    pybindings_dir = executorch_root / "extension" / "pybindings"
    qualcomm_dir = executorch_root / "backends" / "qualcomm"
    torch_lib_dir = torch_root / "lib"
    portable_so = portable_runtime_so_path()
    if portable_so is None:
        raise FileNotFoundError("ExecuTorch _portable_lib shared object was not found")

    return [
        "c++",
        "-std=c++20",
        "-O3",
        "-march=native",
        str(source_path),
        f"-I{executorch_root / 'include'}",
        f"-I{torch_root / 'include'}",
        f"-I{torch_root / 'include' / 'torch' / 'csrc' / 'api' / 'include'}",
        f"-I{python_include}",
        f"-L{pybindings_dir}",
        f"-L{qualcomm_dir}",
        f"-L{torch_lib_dir}",
        f"-L{python_libdir}",
        "-Wl,--allow-shlib-undefined",
        f"-Wl,-rpath,{pybindings_dir}",
        f"-Wl,-rpath,{qualcomm_dir}",
        f"-Wl,-rpath,{torch_lib_dir}",
        f"-Wl,-rpath,{python_libdir}",
        f"-l:{portable_so.name}",
        "-l:libqnn_executorch_backend.so",
        "-ltorch_python",
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
        f"-l{python_library}",
        "-lpthread",
        "-ldl",
        "-lutil",
        "-lm",
        "-o",
        str(output_path),
    ]


def cxx_run_command(
    runner_path: Path,
    pte_path: Path,
    *,
    large_batch: int,
    states: int,
    timesteps: int,
    rounds: int,
    warmup_rounds: int,
) -> list[str]:
    return [
        str(runner_path),
        str(pte_path),
        str(large_batch),
        str(states),
        str(timesteps),
        str(rounds),
        str(warmup_rounds),
    ]


def run_python_runtime_rows(
    module: nn.Module,
    buffer: bytes,
    *,
    large_batch: int,
    states: int,
    timesteps: int,
    rounds: int,
    warmup_rounds: int,
) -> tuple[list[TimingRow], str | None]:
    try:
        from executorch.runtime import Runtime, Verification  # noqa: PLC0415
    except ImportError as error:
        return [], str(error)

    runtime = Runtime.get()
    program = runtime.load_program(buffer, verification=Verification.Minimal)
    method = program.load_method("forward")
    assert method is not None, "executorch program has no 'forward' method"

    rows: list[TimingRow] = []
    for batch in (1, large_batch):
        x = torch.randn((batch, states, timesteps), dtype=torch.float32)

        def run_executorch(input_tensor: Tensor) -> Tensor:
            return method.execute((input_tensor,))[0]

        with torch.inference_mode():
            expected = module(x)
            actual = run_executorch(x)
        rows.append(
            TimingRow(
                group="combined",
                path="TorchFeatureExtractor generated kernel, ExecuTorch Python runtime",
                shape=tuple(x.shape),
                median_ms=benchmark_ms(
                    run_executorch,
                    x,
                    rounds=rounds,
                    warmup_rounds=warmup_rounds,
                ),
                max_abs_diff=(expected - actual).abs().max().item(),
            )
        )
    return rows, None


def write_report(
    report_path: Path,
    *,
    rows: list[TimingRow],
    pte_path: Path,
    graph_path: Path,
    runtime_so_path: Path | None,
    python_runtime_error: str | None,
    compile_command: list[str] | None,
    run_command: list[str] | None,
    cxx_error: str | None,
    cxx_output: str | None,
    command: list[str],
    args: argparse.Namespace,
) -> None:
    executorch_spec = importlib.util.find_spec("executorch")
    executorch_root = (
        None if executorch_spec is None else next(iter(executorch_spec.submodule_search_locations))
    )
    runtime_deps = ldd_output(runtime_so_path) if runtime_so_path is not None else None
    lines = [
        "# ExecuTorch Basic Statistics Report",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- PyTorch: `{torch.__version__}`",
        f"- ExecuTorch: `{executorch_version()}`",
        f"- ExecuTorch artifact: `{pte_path}` ({pte_path.stat().st_size} bytes)",
        f"- Export graph: `{graph_path}`",
        f"- Dynamic batch: `1..{args.large_batch}`",
        "",
        "## Timings",
        "",
        markdown_table(rows) if rows else "No runtime timings were collected.",
    ]
    if python_runtime_error is not None:
        lines.extend([
            "",
            "## Python Runtime",
            "",
            "Import failed:",
            "",
            "```text",
            python_runtime_error,
            "```",
        ])
    if cxx_error is not None:
        lines.extend([
            "",
            "## C++ Runtime",
            "",
            "Build or run failed:",
            "",
            "```text",
            cxx_error,
            "```",
        ])
    elif cxx_output is not None:
        lines.extend(["", "## C++ Runtime Output", "", "```text", cxx_output.strip(), "```"])

    lines.extend([
        "",
        "## Artifacts",
        "",
        f"- combined TorchFeatureExtractor generated kernel: `{pte_path}` ({pte_path.stat().st_size} bytes)",
        f"- exported graph: `{graph_path}` ({graph_path.stat().st_size} bytes)",
    ])
    if executorch_root is not None:
        lines.append(
            f"- ExecuTorch Python package: `{executorch_root}` ({package_size(Path(executorch_root))} bytes)"
        )
    if runtime_so_path is not None:
        lines.append(
            f"- ExecuTorch Python runtime binding: `{runtime_so_path}` ({runtime_so_path.stat().st_size} bytes)"
        )
    if runtime_deps:
        lines.extend(["", "## Runtime Dependencies", "", "```text", runtime_deps, "```"])
    lines.extend(["", "## Commands", ""])
    for item in [command, compile_command, run_command]:
        if item is not None:
            lines.extend(["```bash", shell_join(item), "```", ""])
    report_path.write_text("\n".join(lines).rstrip() + "\n")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts")
    parser.add_argument("--large-batch", type=int, default=10000)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    module = ExtractorGeneratedKernelBasicStats(states=args.states).eval()
    pte_path = args.output_dir / "extractor_generated_kernel.pte"
    graph_path = args.output_dir / "extractor_generated_kernel_graph.py"
    buffer = export_to_executorch(
        module,
        pte_path,
        export_batch=2,
        max_batch=args.large_batch,
        states=args.states,
        timesteps=args.timesteps,
    )
    graph_path.write_text(
        torch.export.export(
            module, (torch.randn((2, args.states, args.timesteps)),)
        ).graph_module.code
    )

    rows, python_runtime_error = run_python_runtime_rows(
        module,
        buffer,
        large_batch=args.large_batch,
        states=args.states,
        timesteps=args.timesteps,
        rounds=args.rounds,
        warmup_rounds=args.warmup_rounds,
    )

    runner_path = args.output_dir / "executorch_basic_stats_runner"
    compile_command: list[str] | None = None
    run_command: list[str] | None = None
    cxx_error: str | None = None
    cxx_output: str | None = None
    try:
        compile_command = cxx_compile_command(runner_path)
        subprocess.run(compile_command, check=True, capture_output=True, text=True)
        run_command = cxx_run_command(
            runner_path,
            pte_path,
            large_batch=args.large_batch,
            states=args.states,
            timesteps=args.timesteps,
            rounds=args.rounds,
            warmup_rounds=args.warmup_rounds,
        )
        result = subprocess.run(run_command, check=True, capture_output=True, text=True)
        cxx_output = result.stdout
        rows.extend(parse_cxx_rows(result.stdout))
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            cxx_error = (error.stderr or error.stdout or str(error)).strip()
        else:
            cxx_error = str(error)

    command = [
        "uv",
        "run",
        "python",
        str(Path(__file__).resolve()),
        "--large-batch",
        str(args.large_batch),
        "--states",
        str(args.states),
        "--timesteps",
        str(args.timesteps),
        "--rounds",
        str(args.rounds),
        "--warmup-rounds",
        str(args.warmup_rounds),
        "--threads",
        str(args.threads),
    ]
    runtime_so = portable_runtime_so_path()
    write_report(
        args.output_dir / "report.md",
        rows=rows,
        pte_path=pte_path,
        graph_path=graph_path,
        runtime_so_path=runtime_so if runtime_so is not None and runtime_so.exists() else None,
        python_runtime_error=python_runtime_error,
        compile_command=compile_command,
        run_command=run_command,
        cxx_error=cxx_error,
        cxx_output=cxx_output,
        command=command,
        args=args,
    )
    print(f"report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
