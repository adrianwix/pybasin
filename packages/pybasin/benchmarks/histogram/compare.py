# pyright: basic
"""Compare the histogram grouper benchmark results (pybasin CPU/CUDA vs Attractors.jl).

Run with:
    uv run python -m packages.pybasin.benchmarks.histogram.compare
"""

from packages.pybasin.benchmarks._compare_grouper import compare_grouper

if __name__ == "__main__":
    compare_grouper("histogram")
