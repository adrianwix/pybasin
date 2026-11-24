"""
Histogram-grouping scaling benchmark using Attractors.jl's `GroupViaHistogram`.

Benchmarks the full `AttractorsViaFeaturizing` workflow (Hénon map iteration,
featurizing, and histogram-binning) across different N values. Supports incremental
runs: existing results are loaded and only missing N values are benchmarked. Delete the
output JSON to force a full re-run.

Run with:
    julia --project=packages/pybasin/tests/accuracy/_julia_project \
        packages/pybasin/benchmarks/histogram/benchmark_julia.jl
"""

using DynamicalSystems: AttractorsViaFeaturizing, GroupViaHistogram, FixedRectangularBinning,
    basins_fractions, statespace_sampler, HRectangle
using Statistics: mean, std
using JSON

include(joinpath(@__DIR__, "..", "..", "..", "case_studies", "src", "case_studies",
    "henon_map", "henon_map.jl"))

const N_VALUES = [100, 200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000]
const NUM_ROUNDS = 5

const RESULTS_PATH = joinpath(@__DIR__, "results", "julia_histogram.json")

const BIN_RANGE = range(-10, 240; step = 50)

make_gconfig() = GroupViaHistogram(FixedRectangularBinning(BIN_RANGE, 2))

function make_sampler(seed::Int)
    sampler, = statespace_sampler(HRectangle([-2.0, -2.0], [2.0, 2.0]), seed)
    return sampler
end

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

const EXPECTED_FRAC_BOUNDED = 0.45
const EXPECTED_FRAC_DIVERGENT = 0.55
const FRAC_RTOL = 0.15

function validate_fractions(fractions::Dict, n::Int)::Bool
    n < 10_000 && return true

    vals = sort(collect(values(fractions)))
    if length(vals) != 2
        println("  VALIDATION FAILED: expected 2 occupied bins, got $(length(vals)): $fractions")
        return false
    end

    lo_ok = abs(vals[1] - EXPECTED_FRAC_BOUNDED) / EXPECTED_FRAC_BOUNDED <= FRAC_RTOL
    hi_ok = abs(vals[2] - EXPECTED_FRAC_DIVERGENT) / EXPECTED_FRAC_DIVERGENT <= FRAC_RTOL
    if !lo_ok || !hi_ok
        println("  VALIDATION FAILED for N=$n: $(round.(vals, digits=4))")
        return false
    end

    println("  Validation passed: $(round.(vals, digits=4)) ✓")
    return true
end

function run_benchmark(n::Int)::Union{Dict, Nothing}
    gconfig = make_gconfig()
    round_times = Float64[]

    for round_idx in 1:NUM_ROUNDS
        ds = henon_system()
        mapper = AttractorsViaFeaturizing(ds, henon_mean_state_featurizer, gconfig; threaded = false)
        sampler = make_sampler(round_idx)

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
    ds = henon_system()
    mapper = AttractorsViaFeaturizing(ds, henon_mean_state_featurizer, make_gconfig(); threaded = false)
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
