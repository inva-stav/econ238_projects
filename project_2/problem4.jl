include("algorithms.jl")
include("save_outputs.jl")

using LinearAlgebra
using Statistics
using Printf
using Plots
using CSV
using DataFrames

###################################
### P4 data: real renewables.ninja capacity factors
### n = 2 (PV at site 1, wind at site 2), one chosen month, hourly resolution.
###################################

const P4_DATA_DIR = joinpath(@__DIR__, "p4_data")
const P4_PV_PATH   = joinpath(P4_DATA_DIR, "ninja_pv_35.1151_-118.8173_uncorrected.csv")
const P4_WIND_PATH = joinpath(P4_DATA_DIR, "ninja_wind_35.0745_-118.3761_uncorrected.csv")

# Hours-per-month for 2019 (non-leap). Cumulative offsets are 1-indexed start hours.
const P4_MONTH_HOURS = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
const P4_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

function load_ninja_capacity_factors(path::String)::Vector{Float64}
    # Skip the 3 leading "#"-prefixed comment lines, then read header + data.
    df = CSV.read(path, DataFrame; comment="#")
    cf = Float64.(df.electricity)
    @assert length(cf) == 8760 "Expected 8760 hourly rows, got $(length(cf)) from $path"
    return cf
end

function month_window(month::Int, T_hours::Int)::UnitRange{Int}
    @assert 1 <= month <= 12 "month must be in 1..12"
    start = sum(P4_MONTH_HOURS[1:month-1]) + 1
    avail = P4_MONTH_HOURS[month]
    stop  = start + min(T_hours, avail) - 1
    return start:stop
end

###################################
### Driver
###################################
function run_p4(; month::Int = 1, T_hours::Int = 720, save::Bool = true)
    println("\n" * "="^60)
    println("P4: real renewables.ninja data,  month = $(P4_MONTH_NAMES[month]),  T = $T_hours h")
    println("="^60)

    pv_full   = load_ninja_capacity_factors(P4_PV_PATH)
    wind_full = load_ninja_capacity_factors(P4_WIND_PATH)
    win = month_window(month, T_hours)
    g_pv   = pv_full[win]
    g_wind = wind_full[win]
    T = length(win)

    n   = 2
    N   = [1, 2]
    Tset = collect(1:T)
    L   = [(1, 0), (2, 0), (1, 2)]
    INV = Dict((1, 0) => 90.0, (2, 0) => 100.0, (1, 2) => 50.0)
    P   = 0.0

    g = zeros(Float64, n, T)
    g[1, :] = g_pv
    g[2, :] = g_wind

    C   = compute_all_costs(n, N, Tset, g, L, INV; P=P)
    C1  = C[[1]]
    C2  = C[[2]]
    C12 = C[[1, 2]]
    save_amt = C1 + C2 - C12
    ρ_hat    = cor(g[1, :], g[2, :])

    x_star = nucleolus_sequential_lp(2, C)
    share_1 = x_star[1] / C12
    share_2 = x_star[2] / C12
    ε       = C1 - x_star[1]

    @printf("  realized ρ̂ (PV vs wind, month sample) = %+0.4f\n", ρ_hat)
    @printf("  mean CF: PV = %.4f,  wind = %.4f\n", mean(g_pv), mean(g_wind))
    @printf("  C({1}) = %.4f,  C({2}) = %.4f,  C({1,2}) = %.4f\n", C1, C2, C12)
    @printf("  savings = C({1}) + C({2}) - C({1,2}) = %.4f\n", save_amt)
    @printf("  x* = (%.4f, %.4f),  shares = (%.4f, %.4f)\n",
        x_star[1], x_star[2], share_1, share_2)
    @printf("  excess (single-player) ε = %.6f\n", ε)

    if save
        dir = joinpath(@__DIR__, "results", "problem4")
        mkpath(dir)

        save_metadata(; n=n, T=T, seed=nothing, C_N=C12, problem_dir=dir)
        save_coalition_costs(C, C12, dir)
        save_nucleolus(x_star, C12, dir)

        open(joinpath(dir, "summary.csv"), "w") do io
            println(io, "month,month_name,T,rho_hat,mean_cf_pv,mean_cf_wind,C_1,C_2,C_12,savings,share_1,share_2,epsilon")
            @printf(io, "%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                month, P4_MONTH_NAMES[month], T, ρ_hat,
                mean(g_pv), mean(g_wind),
                C1, C2, C12, save_amt, share_1, share_2, ε)
        end
        println("  saved: $(joinpath(dir, "summary.csv"))")

        ts = plot(1:T, g_pv,
            lw = 1.2, label = "PV (player 1)",
            xlabel = "hour of month", ylabel = "capacity factor",
            title  = "$(P4_MONTH_NAMES[month]) 2019 — renewables.ninja (MERRA-2)",
            legend = :topright)
        plot!(ts, 1:T, g_wind, lw = 1.2, label = "wind (player 2)")
        savefig(ts, joinpath(dir, "generation_timeseries.png"))
        println("  saved: $(joinpath(dir, "generation_timeseries.png"))")
    end

    return (
        month   = month,
        T       = T,
        rho_hat = ρ_hat,
        C1      = C1,
        C2      = C2,
        C12     = C12,
        savings = save_amt,
        share_1 = share_1,
        share_2 = share_2,
        eps     = ε,
    )
end

# to run (guard prevents auto-execution when this file is include'd by another script)
if abspath(PROGRAM_FILE) == @__FILE__
    run_p4()
end
