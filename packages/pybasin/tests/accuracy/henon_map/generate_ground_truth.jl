"""
Full-pipeline ground truth for the Hénon map accuracy suite.

Samples initial conditions, iterates the map, extracts the mean-state feature
(`Attractors.jl`'s own Hénon featurizer, `(200, 200)` sentinel for divergence),
and groups the result with both `GroupViaHistogram` and
`GroupViaPairwiseComparison` -- the same two groupers
`packages/pybasin/tests/accuracy/henon_map/test_henon_map.py` runs against
these same initial conditions through pybasin's own pipeline.

Reproduces `Attractors.jl` @ `df7c420b4c1d69e1d662bb679a46f8ef8b53bcaa`
`test/mapping/histogram_grouping.jl` for the histogram column
(`FixedRectangularBinning(range(-10, 240; step = 50), 2)`), matching the bin
edges in `case_studies/henon_map/main_henon_histogram_experiment.py`. The
pairwise column uses `GroupViaPairwiseComparison(threshold=1.0,
rescale_features=false)`, matching
`case_studies/henon_map/main_henon_pairwise_experiment.py`.

Run with:
    julia --project=packages/pybasin/tests/accuracy/_julia_project \
        packages/pybasin/tests/accuracy/henon_map/generate_ground_truth.jl
"""

using DynamicalSystems: AttractorsViaFeaturizing, GroupViaHistogram, FixedRectangularBinning,
    GroupViaPairwiseComparison, extract_features, group_features, statespace_sampler, StateSpaceSet

include(joinpath(@__DIR__, "..", "..", "..", "..", "case_studies", "src", "case_studies",
    "henon_map", "henon_map.jl"))

const OUTPUT_PATH = joinpath(@__DIR__, "ground_truths", "henon_map.csv")

const BIN_RANGE = range(-10, 240; step = 50)
const N = 1000
const SEED = 12444

function main()
    ds = henon_system(HENON_DEFAULT_P)

    histogram_config = GroupViaHistogram(FixedRectangularBinning(BIN_RANGE, 2))
    pairwise_config = GroupViaPairwiseComparison(threshold = 1.0, rescale_features = false)

    mapper = AttractorsViaFeaturizing(ds, henon_mean_state_featurizer, histogram_config; threaded = false)

    xg = yg = range(-2.0, 2.0; length = 100)
    sampler, _ = statespace_sampler((xg, yg), SEED)
    ics = StateSpaceSet([copy(sampler()) for _ in 1:N])

    println("Extracting features for $N Hénon initial conditions (p=$HENON_DEFAULT_P, seed=$SEED)...")
    features = extract_features(mapper, ics; show_progress = false)

    labels_histogram = group_features(features, histogram_config)
    labels_pairwise = group_features(features, pairwise_config)

    mkpath(dirname(OUTPUT_PATH))
    open(OUTPUT_PATH, "w") do f
        println(f, "x,y,label_histogram,label_pairwise")
        for i in 1:N
            ic = ics[i]
            println(f, "$(ic[1]),$(ic[2]),$(labels_histogram[i]),$(labels_pairwise[i])")
        end
    end
    println("Saved ground truth to: $OUTPUT_PATH")

    n_hist = length(unique(labels_histogram))
    n_pair = length(unique(labels_pairwise))
    println("Histogram grouper found $n_hist groups; pairwise grouper found $n_pair groups.")
end

main()
