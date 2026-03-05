# Benchmarks Overview

This section documents pybasin's performance characteristics and compares it against the original MATLAB implementation ([bSTAB-M](https://github.com/TUHH-DYN/bSTAB)).

## Test Hardware

- **CPU**: Intel Core Ultra 9 275HX
- **GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM)

## Key Findings

- **GPU delivers massive speedups at scale**: At N=100k samples, pybasin on GPU (Diffrax) is **~25× faster** than MATLAB for end-to-end basin stability estimation (12s vs 309s)
- **Near-constant GPU time**: JAX/Diffrax on CUDA maintains ~11-12s integration time regardless of sample size, enabling large-scale studies without linear time scaling
- **CPU competitive for smaller workloads**: pybasin CPU (Diffrax) is **~1.2× faster** than MATLAB at N=5k and scales to **3-5× faster** at N>5k
- **GPU overhead at small N**: For sample sizes below ~10k, CPU solvers outperform GPU due to data transfer and kernel launch overhead
- **JAX/Diffrax is the recommended solver for GPU workloads**: Best performance on both CPU and GPU, plus unique support for per-trajectory event-based termination (critical for unbounded systems). Install with `pip install pybasin[jax]`
- **Integration dominates runtime**: ODE integration accounts for ~70% of total estimation time, classification ~26%, and feature extraction ~2.5%. Solver selection has the most impact on performance.

## Benchmark Pages

### [Basin Stability Estimator](basin-stability-estimator.md)

Detailed breakdown of the full estimation pipeline showing how time is distributed across each step (sampling, integration, feature extraction, classification). Includes an interactive flame graph for profiling analysis.

### [End-to-End Performance](end-to-end.md)

Compares the complete basin stability estimation workflow between pybasin and MATLAB bSTAB-M across different sample sizes. Demonstrates pybasin's scalability advantage, especially on GPU.

### [Solver Comparison](solvers.md)

Evaluates different ODE solver backends (JAX/Diffrax, PyTorch/torchdiffeq, SciPy) across CPU and GPU. Shows how JAX achieves near-constant integration time on GPU regardless of sample size.
