# Labelled steady-state trajectories for systems from BasinsCollection
# (https://github.com/awage/BasinsCollection, A. Wagemakers, MIT licence).
#
# Each system's equations, parameters, solver settings and AttractorsViaRecurrences
# configuration are taken from the corresponding BasinsCollection script. Labels come from
# the recurrence mapper, an algorithm independent of every representation evaluated in
# Python. Initial conditions are drawn uniformly from the region the collection plots.
#
# A sample is kept only if the mapper assigns the same attractor to the initial condition and
# to the final state of the exported trajectory, so a trajectory still on a transient at the
# end of the window is discarded rather than mislabelled.
#
# Forced systems are integrated from t = 0 with a transient of a whole number of forcing
# periods and SAMPLES_PER_PERIOD samples per period, so every SAMPLES_PER_PERIOD-th sample
# lies on the stroboscopic section at forcing phase 0.
#
# Usage: julia --project=. generate.jl [system ...]    (default: all systems)
# Output: data/<system>.npz with y (N, S, T), t (T,), y0 (N, S), labels (N,), forcing_period.

using Attractors, OrdinaryDiffEqVerner, StaticArrays, NPZ, Random, LinearAlgebra

const N_IC = parse(Int, get(ENV, "N_IC", "1500"))
const SAMPLES_PER_PERIOD = 64
const N_PERIODS = 12
const N_SAMPLES_AUTONOMOUS = 1000
const OUT = joinpath(@__DIR__, "data")

struct Spec
    ds::CoupledODEs
    mapper::Any
    sample::Function        # rng -> full initial state
    to_mapper::Function     # full state -> state the mapper expects
    forcing_period::Float64 # NaN when autonomous
    dt::Float64             # autonomous sampling step
    ttr::Float64            # transient discarded before the window
    check_end::Bool         # whether the exported end state can be re-labelled by the mapper
end

Spec(ds, mapper, sample, to_mapper, forcing_period, dt, ttr) =
    Spec(ds, mapper, sample, to_mapper, forcing_period, dt, ttr, true)

uniform(rng, lo, hi) = lo + (hi - lo) * rand(rng)

function wrap_angle_callback()
    condition(u, t, integrator) = (integrator.u[1] < -π || integrator.u[1] > π)
    function affect!(integrator)
        uu = integrator.u
        integrator.u = SVector(uu[1] < 0 ? uu[1] + 2π : uu[1] - 2π, uu[2])
        u_modified!(integrator, true)
    end
    DiscreteCallback(condition, affect!; save_positions = (false, false))
end

function forced(ds, T, grid, sample; ttr_periods = 300, kwargs...)
    smap = StroboscopicMap(ds, T)
    mapper = AttractorsViaRecurrences(smap, grid; kwargs...)
    Spec(ds, mapper, sample, identity, T, NaN, ttr_periods * T)
end

# ---------------------------------------------------------------- forced systems

function duffing_ω115()
    f(u, p, t) = SVector(u[2], -p[1] * u[2] + u[1] - u[1]^3 + p[2] * sin(p[3] * t))
    d, F, ω = 0.05, 0.098, 1.15
    ds = CoupledODEs(f, rand(2), [d, F, ω]; diffeq = (; reltol = 1e-9, alg = Vern9(), maxiters = 1e6))
    g = range(-2.2, 2.2, length = 1200)
    forced(ds, 2π / ω, (g, g), rng -> [uniform(rng, -2.2, 2.2), uniform(rng, -2.2, 2.2)])
end

function forced_pendulum()
    f(u, p, t) = SVector(u[2], -p[1] * u[2] - sin(u[1]) + p[2] * cos(p[3] * t))
    d, F, ω = 0.2, 1.66, 1.0
    diffeq = (; reltol = 1e-9, alg = Vern9(), callback = wrap_angle_callback())
    ds = CoupledODEs(f, rand(2), [d, F, ω]; diffeq)
    grid = (range(-π, π, length = 1200), range(-4.0, 4.0, length = 1200))
    forced(ds, 2π / ω, grid, rng -> [uniform(rng, -π, π), uniform(rng, -4.0, 4.0)])
end

function parametric_pendulum()
    function f(u, p, t)
        A1, A2, ω = p
        x, y = u
        SVector(y, -0.2y - x - (9.81 - A2 * ω^2 * cos(ω * t)) * sin(x) - A1 * ω^2 * sin(ω * t) * cos(x))
    end
    A1, A2, ω = 0.0, 4.1, 1.5
    ds = CoupledODEs(f, rand(2), [A1, A2, ω]; diffeq = (; reltol = 1e-9, alg = Vern9(), maxiters = 1e6))
    g = range(-10, 10, length = 1200)
    forced(ds, 2π / ω, (g, g), rng -> [uniform(rng, -10, 10), uniform(rng, -10, 10)])
end

function split_ring_resonator()
    f(u, p, t) = SVector(u[2], -0.38u[2] - u[1] + 0.4u[1]^2 - 0.08u[1]^3 + p[2] * cos(p[1] * t))
    ω, μ = 1.0285, 35.0
    ds = CoupledODEs(f, rand(2), [ω, μ]; diffeq = (; reltol = 1e-9, alg = Vern9(), maxiters = 1e6))
    g = range(-50, 50, length = 10000)
    forced(ds, 2π / ω, (g, g), rng -> [uniform(rng, -10, 10), uniform(rng, -15, 15)])
end

function megastable_sprott()
    f(u, p, t) = SVector(u[2], -0.33^2 * u[1] + u[2] * cos(u[1]) + sin(0.73t))
    ds = CoupledODEs(f, rand(2), Float64[]; diffeq = (; reltol = 1e-6, alg = Vern9()))
    g = range(-5000, 5000, length = 50000)
    forced(ds, 2π / 0.73, (g, g), rng -> [uniform(rng, -20, 20), uniform(rng, -20, 20)];
        consecutive_recurrences = 1000, attractor_locate_steps = 1000)
end

# ---------------------------------------------------------------- autonomous systems

function thomas()
    f(u, p, t) = SVector(sin(u[2]) - p[1] * u[1], sin(u[3]) - p[1] * u[2], sin(u[1]) - p[1] * u[3])
    ds = CoupledODEs(f, rand(3), [0.1665])
    g = range(-7, 7, length = 10000)
    mapper = AttractorsViaRecurrences(ds, (g, g, g))
    Spec(ds, mapper, rng -> [uniform(rng, -5, 5), uniform(rng, -5, 5), 0.0], identity, NaN, 0.5, 2000.0)
end

function lorenz84()
    function f(u, p, t)
        F, G, a, b = p
        x, y, z = u
        SVector(-y^2 - z^2 - a * x + a * F, x * y - y - b * x * z + G, b * x * y + x * z - z)
    end
    ds = CoupledODEs(f, rand(3), [6.846, 1.287, 0.25, 4.0]; diffeq = (; alg = Vern9(), reltol = 1e-7, maxiters = 1e8))
    g = range(-4, 4, length = 250)
    mapper = AttractorsViaRecurrences(ds, (g, g, g);
        consecutive_recurrences = 2000, attractor_locate_steps = 2000, consecutive_attractor_steps = 100)
    Spec(ds, mapper, rng -> [uniform(rng, 0.5, 1.5), uniform(rng, -1.5, -0.5), 0.0], identity, NaN, 0.05, 1000.0)
end

function li_sprott()
    function f!(du, u, p, t)
        a, b = p
        x, y, z = u
        du[1] = y + y * z
        du[2] = y * z - a * x * z
        du[3] = b * z^2 - y^2
        nothing
    end
    ds = CoupledODEs(f!, rand(3), [13.0, 0.55]; diffeq = (; alg = Vern9(), reltol = 1e-9, maxiters = 1e8))
    g = range(-5, 5, length = 10000)
    mapper = AttractorsViaRecurrences(ds, (g, g, g); sparse = true,
        consecutive_recurrences = 1000, attractor_locate_steps = 1000, maximum_iterations = Int(1e8))
    Spec(ds, mapper, rng -> [uniform(rng, -1, 1), uniform(rng, -5, 5), -1.0], identity, NaN, 0.05, 1000.0)
end

function rikitake()
    function f(u, p, t)
        μ, α = p
        x, y, z = u
        SVector(-μ * x + y * z, -μ * y + x * (z - α), 1 - x * y)
    end
    ds = CoupledODEs(f, rand(3), [0.5, 1.0]; diffeq = (; alg = Vern9(), reltol = 1e-6, maxiters = 1e8))
    psys = ProjectedDynamicalSystem(ds, [1, 2], [0.0])
    g = range(-5, 5, length = 1001)
    mapper = AttractorsViaRecurrences(psys, (g, g); Δt = 0.1,
        consecutive_recurrences = 50000, attractor_locate_steps = 10000)
    # The mapper runs on (x, y) with z completed as 0, so it cannot re-label an end state
    # whose z is not 0; labels come from the initial condition only.
    Spec(ds, mapper, rng -> [uniform(rng, -2.5, 2.5), uniform(rng, -2.5, 2.5), 0.0], u -> u[1:2], NaN, 0.05, 1000.0, false)
end

function magnetic_pendulum()
    magnets = [SVector(cos(2π * i / 3), sin(2π * i / 3)) for i in 1:3]
    function f(u, p, t)
        x, y, vx, vy = u
        γ, d, α, ω = p
        ax, ay = -ω^2 * x - α * vx, -ω^2 * y - α * vy
        for m in magnets
            D = sqrt((x - m[1])^2 + (y - m[2])^2 + d^2)
            ax -= γ * (x - m[1]) / D^3
            ay -= γ * (y - m[2]) / D^3
        end
        SVector(vx, vy, ax, ay)
    end
    ds = CoupledODEs(f, [0.7, 0.7, 0, 0], [1.0, 0.3, 0.2, 0.5]; diffeq = (; alg = Vern9(), reltol = 1e-6, maxiters = 1e8))
    psys = ProjectedDynamicalSystem(ds, [1, 2], [0.0, 0.0])
    g = range(-10.0, 10.0, length = 5001)
    mapper = AttractorsViaRecurrences(psys, (g, g); Δt = 0.1)
    Spec(ds, mapper, rng -> [uniform(rng, -5, 5), uniform(rng, -5, 5), 0.0, 0.0], u -> u[1:2], NaN, 0.1, 500.0)
end

function gyrostat()
    function f(u, p, t)
        x1, x2, x3 = u
        SVector(-2x1 - 0.7933x2 + 0.1914x3 + x2 * x3 / 3,
            1.19x1 + 3.48x2 - x1 * x3,
            -0.5742x1 - 5.8x3 + x1 * x2 + 22.8)
    end
    ds = CoupledODEs(f, rand(3); diffeq = (; alg = Vern9(), reltol = 1e-9, maxiters = 1e8))
    g = range(-30, 30, length = 1501)
    mapper = AttractorsViaRecurrences(ds, (g, g, g); Ttr = 2000, consecutive_recurrences = 1000)
    Spec(ds, mapper, rng -> [uniform(rng, -50, 50), uniform(rng, -50, 50), 0.0], identity, NaN, 0.02, 2000.0)
end

const SYSTEMS = Dict(
    "bc_duffing" => duffing_ω115,
    "bc_forced_pendulum" => forced_pendulum,
    "bc_parametric_pendulum" => parametric_pendulum,
    "bc_split_ring" => split_ring_resonator,
    "bc_megastable" => megastable_sprott,
    "bc_thomas" => thomas,
    "bc_lorenz84" => lorenz84,
    "bc_li_sprott" => li_sprott,
    "bc_rikitake" => rikitake,
    "bc_magnetic_pendulum" => magnetic_pendulum,
    "bc_gyrostat" => gyrostat,
)

function window(spec::Spec)
    if isnan(spec.forcing_period)
        return spec.dt, spec.dt * (N_SAMPLES_AUTONOMOUS - 1)
    end
    dt = spec.forcing_period / SAMPLES_PER_PERIOD
    return dt, dt * (SAMPLES_PER_PERIOD * N_PERIODS - 1)
end

function generate(name::String)
    spec = SYSTEMS[name]()
    rng = MersenneTwister(0)
    dt, T = window(spec)
    ys, y0s, labels = Array{Float32, 2}[], Vector{Float64}[], Int[]
    n_rejected = 0
    t0 = time()
    while length(labels) < N_IC && length(labels) + n_rejected < 3N_IC
        u0 = spec.sample(rng)
        label = spec.mapper(spec.to_mapper(u0))
        if label < 1
            n_rejected += 1
            continue
        end
        X, _ = trajectory(spec.ds, T, u0; Ttr = spec.ttr, Δt = dt)
        Y = Matrix(X)  # (T, S)
        # The mapper restarts at t = 0, so a forced system's end state must lie on the
        # phase-0 section: the first sample of the last forcing period.
        i_end = isnan(spec.forcing_period) ? length(X) : length(X) - SAMPLES_PER_PERIOD + 1
        consistent = !spec.check_end || spec.mapper(spec.to_mapper(Vector(X[i_end]))) == label
        if !all(isfinite, Y) || !consistent
            n_rejected += 1
            continue
        end
        push!(ys, Float32.(Y)); push!(y0s, u0); push!(labels, label)
    end
    n, (nt, s) = length(ys), size(ys[1])
    y = Array{Float32}(undef, n, s, nt)
    for i in 1:n
        y[i, :, :] = permutedims(ys[i])
    end
    t = collect(spec.ttr .+ (0:nt-1) .* dt)
    mkpath(OUT)
    npzwrite(joinpath(OUT, "$name.npz"), Dict(
        "y" => y, "t" => t, "y0" => permutedims(reduce(hcat, y0s)), "labels" => labels,
        "forcing_period" => [spec.forcing_period]))
    counts = sort([(l, count(==(l), labels)) for l in unique(labels)])
    println("$name: kept $n, rejected $n_rejected, $(round(time() - t0; digits = 1))s, shape=$(size(y)), attractors=$counts")
    flush(stdout)
end

for name in (isempty(ARGS) ? sort(collect(keys(SYSTEMS))) : ARGS)
    try
        generate(name)
    catch err
        println("$name: FAILED $(sprint(showerror, err))")
        flush(stdout)
    end
end
