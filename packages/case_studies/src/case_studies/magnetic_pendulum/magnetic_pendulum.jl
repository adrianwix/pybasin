"""
Magnetic pendulum: a damped pendulum bob suspended above `N` point magnets,
projected onto position space (velocity fixed at `[0, 0]`). ODE, parameters,
and grid match `Attractors.jl`'s own "Magnetic pendulum: projected system"
test -- fixture accuracy depends on this exact system.

Source: `Attractors.jl` @ `df7c420b4c1d69e1d662bb679a46f8ef8b53bcaa`
`test/mapping/attractor_mapping.jl:196-247` (`@testset "Magnetic pendulum: projected
system"`) -- `MagneticPendulumParams` (197-203), `magnetic_pendulum_rule` (204-216), and
`magnetic_pendulum` (217-224) are verbatim; `magnetic_pendulum_projected` factors out
that testset's own `ProjectedDynamicalSystem(ds, 1:2, [0.0, 0.0])` call (line 231) into a
reusable function. Line numbers and content are only guaranteed to match at that commit
-- re-verify against current Attractors.jl source before trusting them at a later one.
"""

using DynamicalSystems: CoupledODEs, ProjectedDynamicalSystem
using StaticArrays: SVector

mutable struct MagneticPendulumParams
    γs::Vector{Float64}
    d::Float64
    α::Float64
    ω::Float64
    magnets::Vector{SVector{2, Float64}}
end

function magnetic_pendulum_rule(u, p, t)
    x, y, vx, vy = u
    γs::Vector{Float64}, d::Float64, α::Float64, ω::Float64 = p.γs, p.d, p.α, p.ω
    dx, dy = vx, vy
    dvx, dvy = @. -ω^2 * (x, y) - α * (vx, vy)
    for (i, ma) in enumerate(p.magnets)
        δx, δy = (x - ma[1]), (y - ma[2])
        D = sqrt(δx^2 + δy^2 + d^2)
        dvx -= γs[i] * (x - ma[1]) / D^3
        dvy -= γs[i] * (y - ma[2]) / D^3
    end
    return SVector(dx, dy, dvx, dvy)
end

"""
    magnetic_pendulum(u = ...; γ, d, α, ω, N, γs, diffeq)

Construct a `CoupledODEs` for the `N`-magnet pendulum, magnets placed evenly
on the unit circle.
"""
function magnetic_pendulum(
        u = [sincos(0.12553 * 2π)..., 0, 0];
        γ = 1.0, d = 0.3, α = 0.2, ω = 0.5, N = 3, γs = fill(γ, N), diffeq
    )
    m = [SVector(cos(2π * i / N), sin(2π * i / N)) for i in 1:N]
    p = MagneticPendulumParams(γs, d, α, ω, m)
    return CoupledODEs(magnetic_pendulum_rule, u, p; diffeq)
end

"""
    magnetic_pendulum_projected(; γ, d, α, ω, N, diffeq)

`magnetic_pendulum` projected onto position space (`x, y`), velocity fixed at
`[0, 0]` -- the system every ground-truth/benchmark fixture in this repo
actually samples initial conditions over.
"""
function magnetic_pendulum_projected(; γ = 1, d = 0.2, α = 0.2, ω = 0.8, N = 3, diffeq)
    ds_full = magnetic_pendulum(; γ, d, α, ω, N, diffeq)
    return ProjectedDynamicalSystem(ds_full, 1:2, [0.0, 0.0])
end
