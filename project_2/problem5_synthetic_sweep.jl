include("algorithms.jl")
include("save_outputs.jl")

using Random
using Distributions
using LinearAlgebra
using Statistics
using Printf
using CSV
using DataFrames

###################################
### P5 extension: Synthetic Beta copula sweep WITH demand heterogeneity
###
### Like P3, we sweep correlation ρ using a bivariate Gaussian copula with
### Beta(2,5) marginals. But now we also attach demand profiles (industrial
### for PV, residential for wind) as in P5, to see whether demand-adjusted
### synthetic predictions better align with real monthly savings.
###################################

function generate_correlated_pair(ρ::Float64, T::Int; seed::Int = 238)
    Random.seed!(seed)
    z1 = randn(T)
    ε  = randn(T)
    ρ_safe = clamp(ρ, -0.999, 0.999)
    z2 = ρ_safe .* z1 .+ sqrt(1 - ρ_safe^2) .* ε

    Φ = Normal(0.0, 1.0)
    marginal = Beta(2.0, 5.0)
    g1 = quantile.(marginal, cdf.(Φ, z1))
    g2 = quantile.(marginal, cdf.(Φ, z2))
    return g1, g2
end

function make_demand_profile(T::Int, profile::Symbol)::Vector{Float64}
    d = zeros(T)
    for t in 1:T
        hour = mod(t - 1, 24)
        if profile == :industrial
            d[t] = hour in 8:17 ? 0.75 : 0.10
        elseif profile == :residential
            if hour in 7:8
                d[t] = 0.55
            elseif hour in 18:21
                d[t] = 0.85
            elseif hour in 9:17
                d[t] = 0.30
            else
                d[t] = 0.12
            end
        elseif profile == :flat
            d[t] = 0.35
        elseif profile == :none
            d[t] = 0.0
        else
            error("Unknown demand profile: $profile")
        end
    end
    return d
end

###################################
### Sweep: same network as P3/P4/P5
###################################

function p5_synthetic_costs_at_rho(ρ::Float64;
        T::Int = 744,
        seed::Int = 238,
        demand_1::Symbol = :industrial,
        demand_2::Symbol = :residential,
        demand_scale::Float64 = 0.5)

    g1, g2 = generate_correlated_pair(ρ, T; seed=seed)

    n    = 2
    N    = [1, 2]
    Tset = collect(1:T)
    L    = [(1, 0), (2, 0), (1, 2)]
    INV  = Dict((1, 0) => 90.0, (2, 0) => 100.0, (1, 2) => 50.0)
    P    = 0.0

    g = zeros(Float64, n, T)
    g[1, :] = g1
    g[2, :] = g2

    d1_raw = make_demand_profile(T, demand_1) .* demand_scale
    d2_raw = make_demand_profile(T, demand_2) .* demand_scale
    d = zeros(Float64, n, T)
    d[1, :] = d1_raw
    d[2, :] = d2_raw

    C = compute_all_costs(n, N, Tset, g, L, INV; P=P, d=d)

    C1   = C[[1]]
    C2   = C[[2]]
    C12  = C[[1, 2]]
    savings = C1 + C2 - C12
    ρ_hat = cor(g1, g2)

    x_star  = nucleolus_sequential_lp(2, C)
    share_1 = C12 > 0 ? x_star[1] / C12 : 0.5
    share_2 = C12 > 0 ? x_star[2] / C12 : 0.5

    return (
        rho_target   = ρ,
        rho_realized = ρ_hat,
        C_1          = C1,
        C_2          = C2,
        C_12         = C12,
        savings      = savings,
        share_1      = share_1,
        share_2      = share_2,
    )
end

###################################
### Main driver
###################################

function run_p5_synthetic_sweep(;
        ρ_grid = -1.0:0.1:1.0,
        T::Int = 744,
        seed::Int = 238,
        demand_1::Symbol = :industrial,
        demand_2::Symbol = :residential,
        demand_scale::Float64 = 0.5,
        save::Bool = true)

    println("\n" * "="^60)
    println("P5 Synthetic Sweep: Beta copula + demand heterogeneity")
    println("  T=$T, demand_1=$demand_1, demand_2=$demand_2, scale=$demand_scale")
    println("="^60)

    rhos = collect(ρ_grid)
    rows = []

    for ρ in rhos
        @printf("\n  ρ_target = %+0.2f ...", ρ)
        res = p5_synthetic_costs_at_rho(ρ; T=T, seed=seed,
                demand_1=demand_1, demand_2=demand_2, demand_scale=demand_scale)
        push!(rows, res)
        @printf(" savings = %.2f", res.savings)
    end

    println("\n")

    if save
        dir = joinpath(@__DIR__, "results", "problem5")
        mkpath(dir)
        df = DataFrame(rows)
        outpath = joinpath(dir, "synthetic_sweep_with_demand.csv")
        CSV.write(outpath, df)
        println("  saved: $outpath")
    end

    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_p5_synthetic_sweep()
end
