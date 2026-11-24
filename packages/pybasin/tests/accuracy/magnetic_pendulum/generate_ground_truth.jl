"""
Full-pipeline ground truth for the magnetic pendulum accuracy suite.

Samples initial conditions, integrates the real ODE (Vern9, matching
`Attractors.jl`'s own tolerances), extracts the final `(x, y)` position, and
groups the result with both `GroupViaHistogram` and
`GroupViaPairwiseComparison` -- the same two groupers
`packages/pybasin/tests/accuracy/magnetic_pendulum/test_magnetic_pendulum.py`
runs against these same initial conditions through pybasin's own pipeline.

ODE, parameters, grid, and pairwise threshold reproduce `Attractors.jl` @
`df7c420b4c1d69e1d662bb679a46f8ef8b53bcaa` `test/mapping/attractor_mapping.jl`'s
"Magnetic pendulum: projected system" test exactly
(`GroupViaPairwiseComparison(threshold=1.0, rescale_features=false)`), matching
`case_studies/magnetic_pendulum/main_magnetic_pendulum_pairwise_experiment.py`.
The histogram column uses per-axis bin edges chosen to keep each of the three
magnets' tightly-converged equilibria (empirically `(-0.50, 0.86)`, `(-0.50,
-0.86)`, `(0.99, 0.00)`) comfortably inside a single bin with a wide margin
(>=0.36) from every edge -- a uniform fine 21-edge grid over `[-1.5, 1.5]`
places an edge within ~0.04 of one equilibrium, letting ordinary cross-solver
floating-point differences (`Vern9` vs pybasin's own solver) flip individual
points across it. Matches
`case_studies/magnetic_pendulum/main_magnetic_pendulum_histogram_experiment.py`.

Run with:
    julia --project=packages/pybasin/tests/accuracy/_julia_project \
        packages/pybasin/tests/accuracy/magnetic_pendulum/generate_ground_truth.jl
"""

using DynamicalSystems: AttractorsViaFeaturizing, GroupViaHistogram, FixedRectangularBinning,
    GroupViaPairwiseComparison, extract_features, group_features, statespace_sampler
using OrdinaryDiffEq: Vern9
using StaticArrays: SVector

include(joinpath(@__DIR__, "..", "..", "..", "..", "case_studies", "src", "case_studies",
    "magnetic_pendulum", "magnetic_pendulum.jl"))

const OUTPUT_PATH = joinpath(@__DIR__, "ground_truths", "magnetic_pendulum.csv")

const BIN_RANGE_X = range(-1.5, 1.5; length = 3)  # edges -1.5, 0.0, 1.5
const BIN_RANGE_Y = range(-1.5, 1.5; length = 4)  # edges -1.5, -0.5, 0.5, 1.5
const N = 1000
const SEED = 1234
const TTR = 500.0

function main()
    diffeq = (alg = Vern9(), reltol = 1.0e-9, abstol = 1.0e-9)
    ds = magnetic_pendulum_projected(; γ = 1, d = 0.2, α = 0.2, ω = 0.8, N = 3, diffeq)

    histogram_config = GroupViaHistogram(FixedRectangularBinning((BIN_RANGE_X, BIN_RANGE_Y)))
    pairwise_config = GroupViaPairwiseComparison(threshold = 1.0, rescale_features = false)

    featurizer(A, t) = SVector(A[end][1], A[end][2])
    mapper = AttractorsViaFeaturizing(ds, featurizer, histogram_config; Ttr = TTR)

    xg = range(-2, 2; length = 201)
    yg = range(-2, 2; length = 201)
    sampler, _ = statespace_sampler((xg, yg), SEED)
    ics = [copy(sampler()) for _ in 1:N]

    println("Integrating $N magnetic pendulum trajectories (Vern9, this takes a while)...")
    features = extract_features(mapper, ics; show_progress = false)

    labels_histogram = group_features(features, histogram_config)
    labels_pairwise = group_features(features, pairwise_config)

    mkpath(dirname(OUTPUT_PATH))
    open(OUTPUT_PATH, "w") do f
        println(f, "x,y,vx,vy,label_histogram,label_pairwise")
        for i in 1:N
            ic = ics[i]
            println(f, "$(ic[1]),$(ic[2]),0.0,0.0,$(labels_histogram[i]),$(labels_pairwise[i])")
        end
    end
    println("Saved ground truth to: $OUTPUT_PATH")

    n_hist = length(unique(labels_histogram))
    n_pair = length(unique(labels_pairwise))
    println("Histogram grouper found $n_hist groups; pairwise grouper found $n_pair groups.")
end

main()
