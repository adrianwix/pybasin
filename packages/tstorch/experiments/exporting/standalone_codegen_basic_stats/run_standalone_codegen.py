from __future__ import annotations

import argparse
import platform
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

HEADER_SOURCE = """\
#ifndef TSTORCH_BASIC_STATS_H
#define TSTORCH_BASIC_STATS_H

#ifdef __cplusplus
extern "C" {
#endif

void tstorch_basic_stats_f32(
    const float* x,
    float* out,
    int batch,
    int states,
    int timesteps
);

#ifdef __cplusplus
}
#endif

#endif
"""


C_SOURCE = """\
#include "tstorch_basic_stats.h"

#include <float.h>

void tstorch_basic_stats_f32(
    const float* x,
    float* out,
    int batch,
    int states,
    int timesteps
) {
    const int total_series = batch * states;
    #pragma omp parallel for if(total_series > 1024)
    for (int series = 0; series < total_series; ++series) {
        const int series_offset = series * timesteps;
        float sum = 0.0F;
        float maximum = -FLT_MAX;
        float minimum = FLT_MAX;

        #pragma omp simd reduction(+:sum) reduction(max:maximum) reduction(min:minimum)
        for (int t = 0; t < timesteps; ++t) {
            const float value = x[series_offset + t];
            sum += value;
            maximum = value > maximum ? value : maximum;
            minimum = value < minimum ? value : minimum;
        }

        const int out_offset = series * 3;
        out[out_offset] = sum / (float)timesteps;
        out[out_offset + 1] = maximum;
        out[out_offset + 2] = minimum;
    }
}
"""


RUNNER_SOURCE = """\
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "tstorch_basic_stats.h"

namespace {

void fill_input(std::vector<float>& x) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (float& value : x) {
        value = dist(rng);
    }
}

void reference_basic_stats(
    const std::vector<float>& x,
    std::vector<float>& out,
    int batch,
    int states,
    int timesteps
) {
    const int total_series = batch * states;
    for (int series = 0; series < total_series; ++series) {
        const int series_offset = series * timesteps;
        float sum = 0.0F;
        float maximum = -INFINITY;
        float minimum = INFINITY;
        for (int t = 0; t < timesteps; ++t) {
            const float value = x[series_offset + t];
            sum += value;
            maximum = value > maximum ? value : maximum;
            minimum = value < minimum ? value : minimum;
        }
        const int out_offset = series * 3;
        out[out_offset] = sum / (float)timesteps;
        out[out_offset + 1] = maximum;
        out[out_offset + 2] = minimum;
    }
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::abs((double)a[i] - (double)b[i]);
        max_diff = diff > max_diff ? diff : max_diff;
    }
    return max_diff;
}

double benchmark_ms(
    const std::vector<float>& x,
    std::vector<float>& out,
    int batch,
    int states,
    int timesteps,
    int rounds,
    int warmup_rounds
) {
    for (int i = 0; i < warmup_rounds; ++i) {
        tstorch_basic_stats_f32(x.data(), out.data(), batch, states, timesteps);
    }

    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i) {
        tstorch_basic_stats_f32(x.data(), out.data(), batch, states, timesteps);
    }
    const auto ended = std::chrono::steady_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = ended - started;
    return elapsed.count() / rounds;
}

int parse_int_arg(char** argv, int index, int fallback) {
    if (argv[index] == nullptr) {
        return fallback;
    }
    return std::atoi(argv[index]);
}

void run_case(int batch, int states, int timesteps, int rounds, int warmup_rounds) {
    std::vector<float> x((size_t)batch * (size_t)states * (size_t)timesteps);
    std::vector<float> out((size_t)batch * (size_t)states * 3U);
    std::vector<float> reference(out.size());
    fill_input(x);
    reference_basic_stats(x, reference, batch, states, timesteps);
    tstorch_basic_stats_f32(x.data(), out.data(), batch, states, timesteps);

    std::cout << std::setprecision(10);
    std::cout << "DIFF\\tstandalone_vs_reference_batch_" << batch << "\\t"
              << max_abs_diff(out, reference) << "\\n";
    std::cout << "TIMING\\tstandalone_codegen\\tbasic_stats_f32\\t" << batch << "\\t"
              << states << "\\t" << timesteps << "\\t"
              << benchmark_ms(x, out, batch, states, timesteps, rounds, warmup_rounds)
              << "\\n";
}

}  // namespace

int main(int argc, char** argv) {
    const int large_batch = argc > 1 ? parse_int_arg(argv, 1, 10000) : 10000;
    const int states = argc > 2 ? parse_int_arg(argv, 2, 2) : 2;
    const int timesteps = argc > 3 ? parse_int_arg(argv, 3, 1000) : 1000;
    const int rounds = argc > 4 ? parse_int_arg(argv, 4, 100) : 100;
    const int warmup_rounds = argc > 5 ? parse_int_arg(argv, 5, 10) : 10;

    run_case(1, states, timesteps, rounds, warmup_rounds);
    run_case(large_batch, states, timesteps, rounds, warmup_rounds);
    return 0;
}
"""


@dataclass(frozen=True)
class TimingRow:
    group: str
    path: str
    shape: tuple[int, int, int]
    median_ms: float


def parse_timing_rows(output: str) -> list[TimingRow]:
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


def parse_diffs(output: str) -> dict[str, float]:
    diffs: dict[str, float] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0] == "DIFF":
            diffs[parts[1]] = float(parts[2])
    return diffs


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_sources(generated_dir: Path) -> tuple[Path, Path, Path]:
    generated_dir.mkdir(parents=True, exist_ok=True)
    header_path = generated_dir / "tstorch_basic_stats.h"
    source_path = generated_dir / "tstorch_basic_stats.c"
    runner_path = generated_dir / "benchmark_basic_stats.cpp"
    header_path.write_text(HEADER_SOURCE)
    source_path.write_text(C_SOURCE)
    runner_path.write_text(RUNNER_SOURCE)
    return header_path, source_path, runner_path


def compile_command(source_path: Path, runner_source_path: Path, output_path: Path) -> list[str]:
    return [
        "c++",
        "-std=c++17",
        "-O3",
        "-march=native",
        "-fopenmp",
        str(runner_source_path),
        str(source_path),
        "-o",
        str(output_path),
    ]


def run_command(
    executable_path: Path,
    *,
    large_batch: int,
    states: int,
    timesteps: int,
    rounds: int,
    warmup_rounds: int,
    threads: int,
) -> list[str]:
    return [
        "env",
        f"OMP_NUM_THREADS={threads}",
        str(executable_path),
        str(large_batch),
        str(states),
        str(timesteps),
        str(rounds),
        str(warmup_rounds),
    ]


def file_size(path: Path) -> int:
    return path.stat().st_size


def ldd_output(path: Path) -> str:
    completed = subprocess.run(["ldd", str(path)], check=True, capture_output=True, text=True)
    return completed.stdout


def markdown_table(rows: list[TimingRow]) -> str:
    lines = [
        "| Group | Path | Shape | Median ms |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row.group} | {row.path} | `{row.shape}` | {row.median_ms:.6g} |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    *,
    header_path: Path,
    source_path: Path,
    runner_source_path: Path,
    executable_path: Path,
    compile_cmd: list[str],
    run_cmd: list[str],
    raw_output: str,
    ldd: str,
    rows: list[TimingRow],
    diffs: dict[str, float],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Standalone Codegen Basic Statistics Report",
        "",
        "## Configuration",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- Small shape: `(1, {args.states}, {args.timesteps})`",
        f"- Large shape: `({args.large_batch}, {args.states}, {args.timesteps})`",
        f"- C++ rounds: `{args.rounds}`",
        f"- Warmup rounds: `{args.warmup_rounds}`",
        f"- OpenMP threads: `{args.threads}`",
        "",
        "## Artifacts",
        "",
        f"- Header: `{header_path}` ({file_size(header_path)} bytes)",
        f"- Generated C source: `{source_path}` ({file_size(source_path)} bytes)",
        f"- Benchmark source: `{runner_source_path}` ({file_size(runner_source_path)} bytes)",
        f"- Executable: `{executable_path}` ({file_size(executable_path)} bytes)",
        "",
        "## Timings",
        "",
        markdown_table(rows),
        "",
        "## Correctness Checks",
        "",
    ]
    for name, diff in diffs.items():
        lines.append(f"- {name}: `{diff:.6g}`")
    lines.extend([
        "",
        "## Runtime Dependencies",
        "",
        "```text",
        ldd.strip(),
        "```",
        "",
        "## Commands",
        "",
        "Compile:",
        "",
        "```bash",
        shell_join(compile_cmd),
        "```",
        "",
        "Run:",
        "",
        "```bash",
        shell_join(run_cmd),
        "```",
        "",
        "Raw output:",
        "",
        "```text",
        raw_output.strip(),
        "```",
    ])
    report_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-dir", type=Path, default=root / "generated")
    parser.add_argument("--artifacts-dir", type=Path, default=root / "artifacts")
    parser.add_argument("--large-batch", type=int, default=10000)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup-rounds", type=int, default=10)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    header_path, source_path, runner_source_path = write_sources(args.generated_dir)
    executable_path = args.artifacts_dir / "benchmark_basic_stats"

    compile_cmd = compile_command(source_path, runner_source_path, executable_path)
    subprocess.run(compile_cmd, check=True)
    run_cmd = run_command(
        executable_path,
        large_batch=args.large_batch,
        states=args.states,
        timesteps=args.timesteps,
        rounds=args.rounds,
        warmup_rounds=args.warmup_rounds,
        threads=args.threads,
    )
    completed = subprocess.run(run_cmd, check=True, capture_output=True, text=True)
    raw_output = completed.stdout
    rows = parse_timing_rows(raw_output)
    diffs = parse_diffs(raw_output)
    ldd = ldd_output(executable_path)

    report_path = args.artifacts_dir / "report.md"
    write_report(
        report_path,
        header_path=header_path,
        source_path=source_path,
        runner_source_path=runner_source_path,
        executable_path=executable_path,
        compile_cmd=compile_cmd,
        run_cmd=run_cmd,
        raw_output=raw_output,
        ldd=ldd,
        rows=rows,
        diffs=diffs,
        args=args,
    )
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
