# pyright: basic
# pyright: reportAttributeAccessIssue=false
"""Benchmark and correctness validation for the pendulum system using diffeqpy.

Uses diffeqpy (Julia's DifferentialEquations.jl via Python) with the DP5 solver
(Dormand-Prince 5th order) and EnsembleProblem for parallel trajectory integration.

All ODE/prob_func/output_func code runs entirely in Julia to eliminate Python↔Julia
bridge overhead.  The CPU path mirrors benchmark_julia_ensemble.jl exactly:
  - Out-of-place ODE returning ``SVector{2,Float64}`` via ``ODEProblem{false}``
  - ``prob_func`` builds each IC as a ``SVector{2,Float64}`` (stack-allocated, no alloc)
  - ``output_func`` classifies each trajectory inline in Julia using the same
    steady-state heuristic (|max(ω)−mean(ω)| < 0.01 over t≥950)
  - saveat = 10 000 points over [0, 1000], reltol=1e-8, abstol=1e-6
  - Per-round seeds (seed = round index) to match Julia benchmark

Scaling benchmark times only the ``solve`` call (including output_func work),
so the measurement is directly comparable to the Julia numbers.

Correctness is validated by computing basin stability and comparing to the JAX reference
pipeline defined in case_studies/pendulum/setup_pendulum_system.py.

Pendulum ODE:
    dθ/dt  = ω
    dω/dt  = −α·ω + T − K·sin(θ)

Parameters: alpha=0.1, T=0.5, K=1.0

Usage:
    uv run python -m experiments.solver.diffeqpy
    uv run python -m experiments.solver.diffeqpy --gpu  # GPU mode
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
from diffeqpy import de  # type: ignore[import-untyped]
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------------------------------------------------------
# Constants matching case_studies/pendulum/setup_pendulum_system.py
# ---------------------------------------------------------------------------
ALPHA: float = 0.1
T_PARAM: float = 0.5
K_PARAM: float = 1.0

T_SPAN: tuple[float, float] = (0.0, 1000.0)
T_STEADY: float = 950.0
N_STEADY_STEPS: int = 100

# Match benchmark_julia_ensemble.jl exactly
N_VALUES: list[int] = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
NUM_ROUNDS: int = 5

# Solver settings matching the Julia benchmark
RTOL: float = 1e-8
ATOL: float = 1e-6
N_SAVE: int = 10_000
SAVE_TIMES: list[float] = list(np.linspace(T_SPAN[0], T_SPAN[1], N_SAVE))

STEADY_STATE_T: float = 950.0
FP_THRESHOLD: float = 0.01

TEMPLATE_Y0: list[list[float]] = [[0.4, 0.0], [2.7, 0.0]]
LABELS: list[str] = ["FP", "LC"]

REFERENCE_BS: dict[str, float] = {"FP": 0.152, "LC": 0.848}
BS_TOLERANCE: float = 0.02

RESULTS_DIR: Path = Path(__file__).parent / "results"

N_THREADS: int = os.cpu_count() or 1

USE_GPU: bool = False

_fast_prob: Any = None
_cuda_module: Any = None
_jl_prob_func: Any = None
_jl_output_func: Any = None


# ---------------------------------------------------------------------------
# ODE (defined in Julia via de.seval for maximum speed)
# ---------------------------------------------------------------------------

# CPU: out-of-place ODE returning SVector{2,Float64} — mirrors benchmark_julia_ensemble.jl.
# Out-of-place + SVector is faster for small 2-D systems because the compiler can
# keep the state in registers and avoids heap allocation entirely.
_jul_f: Any = de.seval("""
using StaticArrays: SVector
function pendulum_rule(u, p, t)
    theta = u[1]
    omega = u[2]
    alpha = p[1]
    T_ext = p[2]
    k     = p[3]
    return SVector{2,Float64}(omega, -alpha * omega + T_ext - k * sin(theta))
end
""")

_U0_SV: Any = de.seval("SVector{2,Float64}(0.0, 0.0)")
_P: list[float] = [ALPHA, T_PARAM, K_PARAM]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _setup_cpu() -> None:
    """Build the base ODE problem for CPU ensemble solves."""
    global _fast_prob, _jl_prob_func, _jl_output_func
    from juliacall import Main as jl  # type: ignore[import-untyped]

    os.environ.setdefault("JULIA_NUM_THREADS", str(N_THREADS))

    # ODEProblem{false} = out-of-place; SVector u0 lets Julia infer the output type.
    _fast_prob = jl.seval("""
    using OrdinaryDiffEq: ODEProblem
    using StaticArrays: SVector
    ODEProblem{false}(pendulum_rule,
                      SVector{2,Float64}(0.0, 0.0),
                      (0.0, 1000.0),
                      [0.1, 0.5, 1.0])
    """)

    # prob_func: build each IC as a stack-allocated SVector (no heap alloc per trajectory)
    _jl_prob_func = jl.seval("""
    using OrdinaryDiffEq: remake
    (prob, i, repeat) -> remake(prob, u0 = SVector{2,Float64}(jl_ics[i, 1], jl_ics[i, 2]))
    """)

    # output_func: classify each trajectory inline in Julia — identical logic to
    # classify_solution() in benchmark_julia_ensemble.jl.  Returns (label::Int, false)
    # so the EnsembleSolution contains only labels, not full trajectories.
    _jl_output_func = jl.seval(f"""
    function jl_output_func(sol, i)
        steady_idx = findfirst(t -> t >= {STEADY_STATE_T}, sol.t)
        steady_idx === nothing && (steady_idx = 1)
        omega_max = -Inf
        omega_sum = 0.0
        n_pts = length(sol.u) - steady_idx + 1
        for k in steady_idx:length(sol.u)
            omega = sol.u[k][2]
            omega_sum += omega
            omega > omega_max && (omega_max = omega)
        end
        delta = abs(omega_max - omega_sum / n_pts)
        label = delta < {FP_THRESHOLD} ? 1 : 2
        return (label, false)
    end
    """)


def _setup_gpu() -> None:
    """Build the GPU ODE problem with StaticArrays and Float32.

    Requires Julia packages (install once)::

        julia -e 'import Pkg; Pkg.add(["DifferentialEquations", "StaticArrays", "DiffEqGPU", "CUDA"])'
    """
    global _fast_prob, _cuda_module, _jl_prob_func
    from diffeqpy import cuda  # type: ignore[import-untyped]
    from juliacall import Main as jl  # type: ignore[import-untyped]

    _cuda_module = cuda

    jl.seval("using StaticArrays")
    jl.seval("""
    function f_gpu(u, p, t)
        theta = u[1]
        omega = u[2]
        alpha = p[1]
        T_ext = p[2]
        K = p[3]
        du1 = omega
        du2 = -alpha * omega + T_ext - K * sin(theta)
        return SVector{2}(du1, du2)
    end
    """)
    _fast_prob = jl.seval("""
    u0_gpu = @SVector Float32[0.0f0, 0.0f0]
    p_gpu = @SVector Float32[0.1f0, 0.5f0, 1.0f0]
    tspan_gpu = (0.0f0, 1000.0f0)
    ODEProblem{false}(f_gpu, u0_gpu, tspan_gpu, p_gpu)
    """)

    # GPU prob_func: convert ICs to SVector{2, Float32}
    _jl_prob_func = jl.seval("""
    (prob, i, repeat) -> begin
        u0 = @SVector Float32[Float32(jl_ics[i, 1]), Float32(jl_ics[i, 2])]
        remake(prob, u0 = u0)
    end
    """)


# ---------------------------------------------------------------------------
# Ensemble solving
# ---------------------------------------------------------------------------


def _set_ics(y0_all: np.ndarray) -> None:
    """Upload initial conditions to Julia's Main as a Matrix{Float64}."""
    from juliacall import Main as jl  # type: ignore[import-untyped]

    jl.jl_ics = jl.convert(jl.seval("Matrix{Float64}"), y0_all)


def _solve_ensemble_benchmark(y0_all: np.ndarray) -> list[int]:
    """Solve + classify entirely in Julia; return per-trajectory int labels.

    Mirrors benchmark_julia_ensemble.jl exactly:
    - saveat = SAVE_TIMES (10 000 points)
    - reltol = RTOL, abstol = ATOL
    - output_func classifies each trajectory inline (no full solution kept)

    :param y0_all: Initial conditions, shape (n, 2).
    :return: List of int labels (1=FP, 2=LC), length n.
    """
    from juliacall import Main as jl  # type: ignore[import-untyped]

    n: int = len(y0_all)
    _set_ics(y0_all)

    if USE_GPU:
        # GPU: EnsembleGPUKernel does not support output_func; classify after solve.
        # Timing matches benchmark_julia_ensemble.jl GPU path (solve-only, classify outside).
        cuda = _cuda_module
        gpu_ensemble_prob = de.EnsembleProblem(
            _fast_prob,
            prob_func=_jl_prob_func,
            safetycopy=False,
        )
        sim = de.solve(
            gpu_ensemble_prob,
            cuda.GPUTsit5(),
            cuda.EnsembleGPUKernel(cuda.CUDABackend()),
            trajectories=n,
            saveat=1.0,  # save every 1 s (GPU memory budget)
            adaptive=True,
            dt=0.1,
            reltol=1e-6,
            abstol=1e-6,
        )
        # Classify after CUDA sync; included in timed region (cheap N int bridge calls).
        return [int(sim[i + 1]) for i in range(n)]
    else:
        # CPU: output_func runs inside solve (classification timing included),
        # matching benchmark_julia_ensemble.jl CPU path exactly.
        cpu_ensemble_prob = de.EnsembleProblem(
            _fast_prob,
            prob_func=_jl_prob_func,
            output_func=_jl_output_func,
            safetycopy=False,
        )
        sim = de.solve(
            cpu_ensemble_prob,
            de.DP5(),
            de.EnsembleThreads(),
            trajectories=n,
            saveat=jl.collect(jl.seval(f"range(0.0, 1000.0, length={N_SAVE})")),
            reltol=RTOL,
            abstol=ATOL,
        )
        return [int(sim[i + 1]) for i in range(n)]


def _solve_ensemble_features(y0_all: np.ndarray, saveat: list[float]) -> Any:
    """Solve and return full solutions for Python-side feature extraction.

    Used only by the basin-stability validation path.

    :param y0_all: Initial conditions, shape (n, 2).
    :param saveat: Time points to save at.
    :return: EnsembleSolution with full trajectories.
    """
    n: int = len(y0_all)
    _set_ics(y0_all)

    ensemble_prob = de.EnsembleProblem(_fast_prob, prob_func=_jl_prob_func, safetycopy=False)

    if USE_GPU:
        cuda = _cuda_module
        return de.solve(
            ensemble_prob,
            cuda.GPUTsit5(),
            cuda.EnsembleGPUKernel(cuda.CUDABackend()),
            trajectories=n,
            saveat=saveat,
            adaptive=True,
            dt=0.1,
            reltol=1e-6,
            abstol=1e-6,
        )
    return de.solve(
        ensemble_prob,
        de.DP5(),
        de.EnsembleThreads(),
        trajectories=n,
        saveat=saveat,
        reltol=RTOL,
        abstol=ATOL,
    )


# ---------------------------------------------------------------------------
# IC generation
# ---------------------------------------------------------------------------


def _generate_ics(n: int, seed: int = 42) -> np.ndarray:
    """Generate *n* random ICs inside the pendulum sampling region.

    Uses per-round seeds when called from the benchmark (seed=round_idx), matching
    the approach in benchmark_julia_ensemble.jl.

    :param n: Number of initial conditions.
    :param seed: RNG seed.
    """
    theta_eq: float = float(np.arcsin(T_PARAM / K_PARAM))
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(-np.pi + theta_eq, np.pi + theta_eq, n),
        rng.uniform(-10.0, 10.0, n),
    ])


# ---------------------------------------------------------------------------
# Scaling benchmark  (times only the ODE solve)
# ---------------------------------------------------------------------------


def _benchmark_scaling() -> list[dict[str, object]]:
    """Run the scaling benchmark.

    Times only the ODE solve (+ inline classification via output_func).
    Uses per-round seeds and solver settings matching benchmark_julia_ensemble.jl.
    """
    print(f"\n{'=' * 60}")
    if USE_GPU:
        print("  diffeqpy EnsembleProblem (GPUTsit5, CUDA)")
    else:
        print(f"  diffeqpy EnsembleProblem (DP5, EnsembleThreads, {N_THREADS} threads)")
        print(f"  saveat={N_SAVE} pts, reltol={RTOL}, abstol={ATOL}")
    print(f"{'=' * 60}")

    _gc = de.seval("GC.gc")

    # Warmup: trigger JIT compilation before timing (seed=0 matches Julia's warmup)
    for _ in range(2):
        warmup_y0: np.ndarray = _generate_ics(100, seed=0)
        _solve_ensemble_benchmark(warmup_y0)
    _gc()

    results: list[dict[str, object]] = []
    for n in N_VALUES:
        round_times: list[float] = []

        for round_idx in range(1, NUM_ROUNDS + 1):
            # Per-round seed matches Julia: sample_ics(n, round_idx)
            y0: np.ndarray = _generate_ics(n, seed=round_idx)
            _gc()
            start: float = time.perf_counter()
            labels = _solve_ensemble_benchmark(y0)
            elapsed: float = time.perf_counter() - start
            round_times.append(elapsed)
            print(f"  N={n:>7,}  round {round_idx}/{NUM_ROUNDS}: {elapsed:.3f}s")

        # Validate at large N (matches Julia's validation logic)
        if n >= 10_000:
            fp_frac = labels.count(1) / n
            lc_frac = labels.count(2) / n
            fp_ok = abs(fp_frac - REFERENCE_BS["FP"]) <= BS_TOLERANCE
            lc_ok = abs(lc_frac - REFERENCE_BS["LC"]) <= BS_TOLERANCE
            status = "PASS" if fp_ok and lc_ok else "FAIL"
            print(f"    validation [{status}]: FP={fp_frac:.4f}, LC={lc_frac:.4f}")

        min_t: float = min(round_times)
        mean_t: float = mean(round_times)
        std_t: float = stdev(round_times) if len(round_times) > 1 else 0.0
        per_ic_us: float = min_t / n * 1e6
        print(
            f"  N={n:>7,}  min={min_t:.3f}s  mean={mean_t:.3f}s "
            f"±{std_t:.3f}s  {per_ic_us:.1f} µs/IC\n"
        )
        results.append({
            "N": n,
            "round_times": round_times,
            "mean_time": mean_t,
            "std_time": std_t,
            "min_time": min_t,
            "max_time": max(round_times),
        })
    return results


# ---------------------------------------------------------------------------
# Feature extraction  (Python-side on data returned by diffeqpy)
# ---------------------------------------------------------------------------


def _log_delta(series: np.ndarray) -> float:
    """log(|max - mean| + 1e-10) over a 1-D time series."""
    return float(np.log(np.abs(series.max() - series.mean()) + 1e-10))


def _extract_features_batch(y0_all: np.ndarray) -> np.ndarray:
    """Return feature vector (log_delta of ω) for each IC in *y0_all*.

    ODE solve via diffeqpy with saveat over the steady-state window;
    feature computation in numpy.
    """
    n: int = y0_all.shape[0]
    saveat: list[float] = list(np.linspace(T_STEADY, T_SPAN[1], N_STEADY_STEPS))
    n_t: int = len(saveat)

    sol = _solve_ensemble_features(y0_all, saveat=saveat)

    features: np.ndarray = np.empty(n)
    for i in range(n):
        traj = sol[i + 1]
        omega_series: np.ndarray = np.array([float(traj.u[j][1]) for j in range(n_t)])
        features[i] = _log_delta(omega_series)

    return features


# ---------------------------------------------------------------------------
# Basin stability
# ---------------------------------------------------------------------------

N_BASIN: int = 10000


def _compute_basin_stability() -> dict[str, object]:
    """Compute basin stability for the pendulum using diffeqpy."""
    print(f"\n{'=' * 60}")
    print(f"  Basin stability validation  (N={N_BASIN})")
    print(f"{'=' * 60}")

    template_y0: np.ndarray = np.array(TEMPLATE_Y0, dtype=float)
    template_features: np.ndarray = _extract_features_batch(template_y0).reshape(-1, 1)

    knn: KNeighborsClassifier = KNeighborsClassifier(n_neighbors=1)
    knn.fit(template_features, LABELS)

    y0_all: np.ndarray = _generate_ics(N_BASIN, seed=42)
    sample_features: np.ndarray = _extract_features_batch(y0_all).reshape(-1, 1)
    predictions: np.ndarray = np.array(knn.predict(sample_features))

    basin_stability: dict[str, float] = {
        label: float(np.mean(predictions == label)) for label in LABELS
    }

    print("  Basin stability estimates:")
    for label, bs in basin_stability.items():
        print(f"    {label}: {bs:.4f}")

    return {
        "n": N_BASIN,
        "basin_stability": basin_stability,
        "template_features": {
            LABELS[i]: float(template_features[i, 0]) for i in range(len(LABELS))
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    global USE_GPU

    parser = argparse.ArgumentParser(description="Benchmark diffeqpy ODE solver")
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU (CUDA) for parallel integration"
    )
    args = parser.parse_args()

    USE_GPU = args.gpu

    if USE_GPU:
        print("Initializing GPU support...")
        _setup_gpu()
        solver_name = "GPUTsit5"
        backend_info = "CUDA (EnsembleGPUKernel)"
    else:
        print("Initializing CPU support...")
        _setup_cpu()
        solver_name = "DP5"
        backend_info = f"CPU ({N_THREADS} threads, EnsembleThreads)"

    print(f"Backend: {backend_info}")
    print(f"Solver: {solver_name}")
    print(f"N values: {N_VALUES}")
    print(f"Rounds: {NUM_ROUNDS}  |  t_span: {T_SPAN}")

    benchmark_results: list[dict[str, object]] = _benchmark_scaling()
    basin_result: dict[str, object] = _compute_basin_stability()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    suffix: str = "_gpu" if USE_GPU else "_cpu"
    bench_path: Path = RESULTS_DIR / f"diffeqpy_scaling{suffix}.json"
    with open(bench_path, "w") as f:
        json.dump(
            {
                "solver": solver_name,
                "backend": "GPU" if USE_GPU else "CPU",
                "n_threads": N_THREADS if not USE_GPU else None,
                "num_rounds": NUM_ROUNDS,
                "t_span": list(T_SPAN),
                "benchmarks": benchmark_results,
            },
            f,
            indent=2,
        )
    print(f"\nSaved: {bench_path}")

    basin_path: Path = RESULTS_DIR / f"diffeqpy_basin_stability{suffix}.json"
    with open(basin_path, "w") as f:
        json.dump(basin_result, f, indent=2)
    print(f"Saved: {basin_path}")


if __name__ == "__main__":
    main()
