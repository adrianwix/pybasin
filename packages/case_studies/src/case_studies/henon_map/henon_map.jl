"""
Hénon map, a small, fast, genuinely chaotic-or-divergent discrete map:

    x' = 1 - p1*x^2 + y
    y' = p2*x

With the classic parameters `p = [1.4, 0.3]` most initial conditions in a
generously sized box around the origin diverge to infinity; only a small
fraction land on the bounded chaotic attractor. That mix of "many divergent
trajectories, one bounded attractor" makes this map a useful ground-truth case
for `GroupViaHistogram`'s out-of-range (`-1`)/sentinel-bin handling.

Source: `Attractors.jl` @ `df7c420b4c1d69e1d662bb679a46f8ef8b53bcaa`
`test/mapping/histogram_grouping.jl:6-20` (rule and mean-state-with-sentinel featurizer,
verbatim; the `HENON_DEFAULT_P`/`henon_system` wrapper names are local to this file).
Line numbers and content are only guaranteed to match at that commit -- re-verify against
current Attractors.jl source before trusting them at a later one.
"""

using DynamicalSystems: DeterministicIteratedMap
using StaticArrays: SVector

const HENON_DEFAULT_P = [1.4, 0.3]

henon_rule(x, p, n) = SVector{2}(1.0 - p[1] * x[1]^2 + x[2], p[2] * x[1])

"""
    henon_system(p = HENON_DEFAULT_P)

Construct a `DeterministicIteratedMap` for the Hénon map with parameters `p`,
starting from the origin.
"""
henon_system(p::AbstractVector = HENON_DEFAULT_P) = DeterministicIteratedMap(henon_rule, zeros(2), p)

"""
    henon_mean_state_featurizer(A, t)

Mean-state featurizer: the feature is the per-axis mean of the trajectory,
unless any coordinate is `Inf`/`NaN`/`> 10`, in which case the sentinel
feature `(200.0, 200.0)` is returned instead. Unsupervised grouping configs
like `GroupViaHistogram` have no notion of "diverged to infinity" -- without
this clamp, `Inf`/`NaN` features would either error out of the histogram
encoding or (if in range) get silently mixed into a real bin.
"""
function henon_mean_state_featurizer(A, t)
    x = SVector(sum(A[:, 1]) / size(A, 1), sum(A[:, 2]) / size(A, 1))
    return if any(c -> (isinf(c) || isnan(c) || c > 10), x)
        SVector(200.0, 200.0)
    else
        x
    end
end
