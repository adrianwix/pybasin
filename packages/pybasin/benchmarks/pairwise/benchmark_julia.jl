"""
Basin stability scaling benchmark using Attractors.jl's `GroupViaPairwiseComparison`.

Benchmarks the full `AttractorsViaFeaturizing` workflow (ODE integration, featurizing,
and sequential leader-clustering grouping) for the magnetic pendulum -- the same system
and parameters as `packages/pybasin/tests/accuracy/magnetic_pendulum/
generate_ground_truth.jl` -- across different N values. Supports incremental runs:
existing results are loaded and only missing N values are benchmarked. Delete the output
JSON to force a full re-run.

Run with:
    julia --project=packages/pybasin/tests/accuracy/_julia_project \
        packages/pybasin/benchmarks/pairwise/benchmark_julia.jl
"""

using OrdinaryDiffEq: Vern9
using DynamicalSystems: AttractorsViaFeaturizing, GroupViaPairwiseComparison,
    basins_fractions, statespace_sampler, HRectangle
using StaticArrays: SVector
using Statistics: mean, std
using JSON

include(joinpath(@__DIR__, "..", "..", "..", "case_studies", "src", "case_studies",
    "magnetic_pendulum", "magnetic_pendulum.jl"))

# --- Configuration ---

const N_VALUES = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000]
const NUM_ROUNDS = 5

const RESULTS_PATH = joinpath(@__DIR__, "results", "julia_pairwise.json")

function make_system()
    diffeq = (alg = Vern9(), reltol = 1.0e-9, abstol = 1.0e-9)
    return magnetic_pendulum_projected(; γ = 1, d = 0.2, α = 0.2, ω = 0.8, N = 3, diffeq)
end

featurizer(A, t) = SVector(A[end][1], A[end][2])

function make_sampler(seed::Int)
    sampler, = statespace_sampler(HRectangle([-2.0, -2.0], [2.0, 2.0]), seed)
    return sampler
end

# --- Incremental merge ---

function load_existing_results()::Dict
    if isfile(RESULTS_PATH)
        return JSON.parsefile(RESULTS_PATH)
    end
    return Dict("num_rounds" => NUM_ROUNDS, "benchmarks" => [])
end

function append_result(existing::Dict, result::Dict)
    push!(existing["benchmarks"], result)
    sort!(existing["benchmarks"], by=b -> b["N"])
    existing["num_rounds"] = NUM_ROUNDS

    mkpath(dirname(RESULTS_PATH))
    open(RESULTS_PATH, "w") do f
        JSON.print(f, existing, 2)
    end
    println("  Saved N=$(result["N"]) to: $RESULTS_PATH")
end

# --- Expected basin fractions (from generate_ground_truth.jl's magnetic pendulum run) ---

# generate_ground_truth.jl's magnetic pendulum run (N=1000, seed 1234) measures
# {0.377, 0.314, 0.309} -- but that triple is itself an N=1000 sampling statistic, not the
# converged basin fractions: at N=5000+ with the same seed, the three (symmetric, 3-magnet)
# basins measure close to an even 1/3 each instead, since the earlier 0.377/0.314/0.309
# split hadn't yet converged at N=1000. So this scaling benchmark validates structurally --
# exactly 3 attractors, none negligible, fractions summing to 1 -- rather than pinning to
# the N=1000-specific triple, which `packages/pybasin/tests/accuracy/magnetic_pendulum/
# test_magnetic_pendulum.py`'s Julia ground-truth parity test already checks exactly.
const MIN_PLAUSIBLE_FRACTION = 0.05

function validate_fractions(fractions::Dict, n::Int)::Bool
    vals = sort(collect(values(fractions)))
    if length(vals) != 3
        println("  VALIDATION FAILED: Expected 3 attractors, got $(length(vals)): $fractions")
        return false
    end
    if any(vals .< MIN_PLAUSIBLE_FRACTION)
        println("  VALIDATION FAILED for N=$n: a group is implausibly small: $(round.(vals, digits=4))")
        return false
    end
    total = sum(vals)
    if abs(total - 1.0) > 1e-9
        println("  VALIDATION FAILED for N=$n: fractions sum to $total, not 1.0")
        return false
    end

    println("  Validation passed: $(round.(vals, digits=4)) ✓")
    return true
end

# --- Benchmark ---

function run_benchmark(n::Int)::Union{Dict, Nothing}
    grouping = GroupViaPairwiseComparison(threshold = 1, rescale_features = false)

    round_times = Float64[]

    for round_idx in 1:NUM_ROUNDS
        ds = make_system()
        mapper = AttractorsViaFeaturizing(ds, featurizer, grouping; Ttr = 500.0)

        # Fixed seed 1234 (matches attractor_mapping.jl's own test and
        # pairwise_ground_truth.jl's magnetic_pendulum fixture): the magnetic pendulum's
        # basin boundaries are fractal, so a per-round seed produces fractions that vary
        # well outside a tight tolerance from one draw to the next even at N=5000.
        sampler = make_sampler(1234)
        t_start = time_ns()
        fractions = basins_fractions(mapper, sampler; N = n, show_progress = false)
        elapsed = (time_ns() - t_start) / 1e9
        push!(round_times, elapsed)
        println("  Round $round_idx/$NUM_ROUNDS: $(round(elapsed, digits=3))s")

        if round_idx == 1 && !validate_fractions(fractions, n)
            return nothing
        end
    end

    return Dict(
        "N" => n,
        "round_times" => round_times,
        "mean_time" => mean(round_times),
        "std_time" => std(round_times),
        "min_time" => minimum(round_times),
        "max_time" => maximum(round_times),
    )
end

function main()
    existing = load_existing_results()
    completed = Set{Int}(bench["N"] for bench in existing["benchmarks"])
    pending = sort([n for n in N_VALUES if !(n in completed)])

    if isempty(pending)
        println("All N values already benchmarked. Delete $RESULTS_PATH to re-run.")
        return
    end

    println("Already completed: $(sort(collect(completed)))")
    println("Pending N values: $pending")

    warmup_n = max(100, first(pending))
    println("\nWarmup run (N=$warmup_n, excluded from timing)...")
    grouping = GroupViaPairwiseComparison(threshold = 1, rescale_features = false)
    ds = make_system()
    mapper = AttractorsViaFeaturizing(ds, featurizer, grouping; Ttr = 500.0)
    basins_fractions(mapper, make_sampler(0); N = warmup_n, show_progress = false)
    println("Warmup complete.\n")

    for n in pending
        println("Benchmarking N=$n ($NUM_ROUNDS rounds)...")
        result = run_benchmark(n)
        if result === nothing
            println("  Stopping benchmark due to validation failure at N=$n.")
            return
        end
        println("  Mean: $(round(result["mean_time"], digits=3))s ± $(round(result["std_time"], digits=3))s")
        append_result(existing, result)
        println()
    end
end

main()
