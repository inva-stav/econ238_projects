include("algorithms.jl")
include("save_outputs.jl")

using LinearAlgebra
using Statistics
using Printf
using CSV
using DataFrames

###################################
### P5: Demand-side heterogeneity
### n=2 (PV + wind), real renewables.ninja data, synthetic demand profiles.
###
### Each player now has a local demand d[i,t] in addition to generation g[i,t].
### Only the net injection g[i,t] - d[i,t] must use the transmission network.
### Players who self-consume well need less grid capacity.
###################################

const P5_DATA_DIR  = joinpath(@__DIR__, "p4_data")
const P5_PV_PATH   = joinpath(P5_DATA_DIR, "ninja_pv_35.1151_-118.8173_uncorrected.csv")
const P5_WIND_PATH = joinpath(P5_DATA_DIR, "ninja_wind_35.0745_-118.3761_uncorrected.csv")

const P5_MONTH_HOURS = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
const P5_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

function load_cf(path::String)::Vector{Float64}
    df = CSV.read(path, DataFrame; comment="#")
    cf = Float64.(df.electricity)
    @assert length(cf) == 8760
    return cf
end

function p5_month_window(month::Int, T_hours::Int)::UnitRange{Int}
    start = sum(P5_MONTH_HOURS[1:month-1]) + 1
    stop  = start + min(T_hours, P5_MONTH_HOURS[month]) - 1
    return start:stop
end

###################################
### Synthetic demand profiles (repeating daily pattern, normalized to [0,1])
###################################

function make_demand_profile(T::Int, profile::Symbol)::Vector{Float64}
    d = zeros(T)
    for t in 1:T
        hour = mod(t - 1, 24)  # 0-23 within each day
        if profile == :industrial
            # Flat block 8am-6pm (hours 8-17), low overnight
            d[t] = hour in 8:17 ? 0.75 : 0.10
        elseif profile == :residential
            # Double-peaked: morning 7-9, evening 18-22
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

function self_consumption_rate(g::Vector{Float64}, d::Vector{Float64})
    total_gen = sum(g)
    total_gen == 0.0 && return 0.0
    return sum(min.(g, d)) / total_gen
end

###################################
### Run one scenario
###################################

function run_p5_scenario(;
        month::Int,
        T_hours::Int,
        demand_1::Symbol,
        demand_2::Symbol,
        demand_scale::Float64 = 1.0)

    pv_full   = load_cf(P5_PV_PATH)
    wind_full = load_cf(P5_WIND_PATH)
    win = p5_month_window(month, T_hours)
    g_pv   = pv_full[win]
    g_wind = wind_full[win]
    T = length(win)

    n    = 2
    N    = [1, 2]
    Tset = collect(1:T)
    L    = [(1, 0), (2, 0), (1, 2)]
    INV  = Dict((1, 0) => 90.0, (2, 0) => 100.0, (1, 2) => 50.0)
    P    = 0.0

    g = zeros(Float64, n, T)
    g[1, :] = g_pv
    g[2, :] = g_wind

    d1_raw = make_demand_profile(T, demand_1) .* demand_scale
    d2_raw = make_demand_profile(T, demand_2) .* demand_scale
    d = zeros(Float64, n, T)
    d[1, :] = d1_raw
    d[2, :] = d2_raw

    C   = compute_all_costs(n, N, Tset, g, L, INV; P=P, d=d)
    C1  = C[[1]]
    C2  = C[[2]]
    C12 = C[[1, 2]]
    savings = C1 + C2 - C12

    x_star  = nucleolus_sequential_lp(2, C)
    share_1 = C12 > 0 ? x_star[1] / C12 : 0.5
    share_2 = C12 > 0 ? x_star[2] / C12 : 0.5
    ε       = C1 - x_star[1]

    sc_1 = self_consumption_rate(g_pv,   d1_raw)
    sc_2 = self_consumption_rate(g_wind,  d2_raw)

    return (
        month       = month,
        T           = T,
        demand_1    = string(demand_1),
        demand_2    = string(demand_2),
        demand_scale = demand_scale,
        sc_1        = sc_1,
        sc_2        = sc_2,
        C1          = C1,
        C2          = C2,
        C12         = C12,
        savings     = savings,
        share_1     = share_1,
        share_2     = share_2,
        eps         = ε,
    )
end

###################################
### Experiment 1: Demand-profile pairing comparison (fixed month, fixed scale)
###################################

function run_p5_pairings(; month::Int = 7, T_hours::Int = 720, demand_scale::Float64 = 0.5)
    scenarios = [
        (:none,        :none,        "Baseline (no demand)"),
        (:industrial,  :residential, "PV+Industrial / Wind+Residential"),
        (:residential, :industrial,  "PV+Residential / Wind+Industrial"),
        (:industrial,  :industrial,  "Both Industrial"),
        (:residential, :residential, "Both Residential"),
        (:flat,        :flat,        "Both Flat"),
    ]

    rows = []
    for (d1, d2, label) in scenarios
        @printf("\n  scenario: %s\n", label)
        res = run_p5_scenario(; month=month, T_hours=T_hours,
                                demand_1=d1, demand_2=d2, demand_scale=demand_scale)
        push!(rows, merge(res, (label = label,)))
        @printf("    SC₁=%.3f  SC₂=%.3f  C1=%.2f  C2=%.2f  C(N)=%.2f  save=%.2f  shares=(%.3f,%.3f)\n",
            res.sc_1, res.sc_2, res.C1, res.C2, res.C12, res.savings, res.share_1, res.share_2)
    end
    return rows
end

###################################
### Experiment 2: Demand-scale sweep (fixed pairing, fixed month)
###################################

function run_p5_scale_sweep(;
        month::Int = 7, T_hours::Int = 720,
        demand_1::Symbol = :industrial, demand_2::Symbol = :residential,
        scales = 0.0:0.1:1.0)

    rows = []
    for α in scales
        res = run_p5_scenario(; month=month, T_hours=T_hours,
                                demand_1=demand_1, demand_2=demand_2, demand_scale=α)
        push!(rows, res)
        @printf("    α=%.2f  SC₁=%.3f  SC₂=%.3f  C(N)=%.2f  save=%.2f\n",
            α, res.sc_1, res.sc_2, res.C12, res.savings)
    end
    return rows
end

###################################
### Experiment 3: Monthly sweep with demand (same as P4 sweep but with load)
###################################

function run_p5_monthly(; demand_1::Symbol = :industrial, demand_2::Symbol = :residential,
                          demand_scale::Float64 = 0.5)
    rows = []
    for m in 1:12
        T_hours = P5_MONTH_HOURS[m]
        res = run_p5_scenario(; month=m, T_hours=T_hours,
                                demand_1=demand_1, demand_2=demand_2, demand_scale=demand_scale)
        push!(rows, merge(res, (month_name = P5_MONTH_NAMES[m],)))
    end
    return rows
end

###################################
### Main driver
###################################

function run_p5(; save::Bool = true)
    println("\n" * "="^60)
    println("P5: Demand-side heterogeneity — PV + Wind + load profiles")
    println("="^60)

    dir = joinpath(@__DIR__, "results", "problem5")
    mkpath(dir)

    # --- Experiment 1: pairing comparison ---
    println("\n--- Experiment 1: Demand-profile pairings (Jul, scale=0.5) ---")
    pairings = run_p5_pairings(; month=7, demand_scale=0.5)

    if save
        df = DataFrame(pairings)
        CSV.write(joinpath(dir, "pairings.csv"), df)
        println("\n  saved: $(joinpath(dir, "pairings.csv"))")
    end

    # --- Experiment 2: scale sweep ---
    println("\n--- Experiment 2: Demand-scale sweep (Jul, PV+Ind / Wind+Res) ---")
    scale_rows = run_p5_scale_sweep(; month=7, scales=0.0:0.05:1.0)

    if save
        df = DataFrame(scale_rows)
        CSV.write(joinpath(dir, "scale_sweep.csv"), df)
        println("  saved: $(joinpath(dir, "scale_sweep.csv"))")
    end

    # --- Experiment 3: monthly sweep with demand ---
    println("\n--- Experiment 3: Monthly sweep (PV+Ind / Wind+Res, scale=0.5) ---")
    monthly = run_p5_monthly(; demand_1=:industrial, demand_2=:residential, demand_scale=0.5)

    if save
        df = DataFrame(monthly)
        CSV.write(joinpath(dir, "monthly_with_demand.csv"), df)
        println("  saved: $(joinpath(dir, "monthly_with_demand.csv"))")
    end

    # --- Experiment 3b: monthly sweep without demand (for comparison) ---
    println("\n--- Experiment 3b: Monthly sweep (no demand, for comparison) ---")
    monthly_no_d = run_p5_monthly(; demand_1=:none, demand_2=:none, demand_scale=0.0)

    if save
        df = DataFrame(monthly_no_d)
        CSV.write(joinpath(dir, "monthly_no_demand.csv"), df)
        println("  saved: $(joinpath(dir, "monthly_no_demand.csv"))")
    end

    # --- Save demand profile shapes for plotting ---
    if save
        T_example = 168  # one week
        profiles = DataFrame(
            hour        = 1:T_example,
            industrial  = make_demand_profile(T_example, :industrial),
            residential = make_demand_profile(T_example, :residential),
            flat        = make_demand_profile(T_example, :flat),
        )
        CSV.write(joinpath(dir, "demand_profiles.csv"), profiles)
        println("  saved: $(joinpath(dir, "demand_profiles.csv"))")
    end

    println("\nP5 complete.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_p5()
end
