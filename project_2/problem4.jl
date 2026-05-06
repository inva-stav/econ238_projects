include("algorithms.jl")
include("save_outputs.jl")

using LinearAlgebra
using Statistics
using Printf
using CSV
using DataFrames

###################################
### P4 data: real renewables.ninja capacity factors
### n = 2 (PV at site 1, wind at site 2), one chosen month, hourly resolution.
###################################

const P4_DATA_DIR = joinpath(@__DIR__, "p4_data")
const P4_PV_PATH   = joinpath(P4_DATA_DIR, "ninja_pv_35.1151_-118.8173_uncorrected.csv")
const P4_WIND_PATH = joinpath(P4_DATA_DIR, "ninja_wind_35.0745_-118.3761_uncorrected.csv")
const P4_PV_LAT,   P4_PV_LON   = 35.1151, -118.8173
const P4_WIND_LAT, P4_WIND_LON = 35.0745, -118.3761

# Hours-per-month for 2019 (non-leap). Cumulative offsets are 1-indexed start hours.
const P4_MONTH_HOURS = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
const P4_MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Plotting helpers (kept in a separate file to keep this module focused on the
# coalition-cost / nucleolus pipeline). Included AFTER the constants above so
# the plot helpers can reference them via global lookup.
include("problem4_plots.jl")

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

# Average ranks for ties, so Spearman handles PV's many zeros correctly.
function tiedrank(x::AbstractVector)
    n = length(x)
    p = sortperm(x)
    r = zeros(Float64, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[p[j+1]] == x[p[i]]
            j += 1
        end
        avg = (i + j) / 2
        for k in i:j
            r[p[k]] = avg
        end
        i = j + 1
    end
    return r
end

function correlation_diagnostics(g_pv::Vector{Float64}, g_wind::Vector{Float64})
    pearson_all = cor(g_pv, g_wind)
    spearman    = cor(tiedrank(g_pv), tiedrank(g_wind))

    # Daytime-only Pearson: drop hours where PV is exactly zero (night).
    day = findall(>(0.0), g_pv)
    pearson_day = length(day) > 1 ? cor(g_pv[day], g_wind[day]) : NaN

    # Daily-mean Pearson: aggregate to days, correlate the 24-h means.
    T = length(g_pv)
    n_full_days = T ÷ 24
    if n_full_days >= 2
        pv_d   = [mean(g_pv[(d-1)*24+1 : d*24])   for d in 1:n_full_days]
        wind_d = [mean(g_wind[(d-1)*24+1 : d*24]) for d in 1:n_full_days]
        pearson_daily = cor(pv_d, wind_d)
    else
        pearson_daily = NaN
    end

    return (pearson_all = pearson_all, pearson_daytime = pearson_day,
            spearman = spearman, pearson_daily = pearson_daily,
            n_daytime_hours = length(day))
end

###################################
### Driver
###################################
function run_p4(; month::Union{Int,Nothing} = nothing, T_hours::Int = 8760, save::Bool = true)
    is_full_year = month === nothing && T_hours == 8760
    period_label = is_full_year ? "Full year 2019" :
                   month === nothing ? @sprintf("first %d hours of 2019", T_hours) :
                                       @sprintf("%s 2019", P4_MONTH_NAMES[month])

    println("\n" * "="^60)
    println("P4: real renewables.ninja data,  period = $period_label,  T = $T_hours h")
    println("="^60)

    pv_full   = load_ninja_capacity_factors(P4_PV_PATH)
    wind_full = load_ninja_capacity_factors(P4_WIND_PATH)
    if month === nothing
        T = min(T_hours, 8760)
        g_pv   = pv_full[1:T]
        g_wind = wind_full[1:T]
    else
        win = month_window(month, T_hours)
        g_pv   = pv_full[win]
        g_wind = wind_full[win]
        T = length(win)
    end

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
    corr     = correlation_diagnostics(g_pv, g_wind)
    ρ_hat    = corr.pearson_all

    x_star = nucleolus_sequential_lp(2, C)
    share_1 = x_star[1] / C12
    share_2 = x_star[2] / C12
    ε       = C1 - x_star[1]

    @printf("  mean CF: PV = %.4f,  wind = %.4f\n", mean(g_pv), mean(g_wind))
    @printf("  std  CF: PV = %.4f,  wind = %.4f\n", std(g_pv),  std(g_wind))
    @printf("  correlations (PV vs wind):\n")
    @printf("    Pearson  (all hours)         = %+0.4f\n", corr.pearson_all)
    @printf("    Pearson  (daytime, n=%4d h)  = %+0.4f\n", corr.n_daytime_hours, corr.pearson_daytime)
    @printf("    Spearman (rank, all hours)   = %+0.4f\n", corr.spearman)
    @printf("    Pearson  (daily-mean CFs)    = %+0.4f\n", corr.pearson_daily)
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
            println(io, "period,T,rho_hat,mean_cf_pv,mean_cf_wind,C_1,C_2,C_12,savings,share_1,share_2,epsilon")
            @printf(io, "%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                period_label, T, ρ_hat,
                mean(g_pv), mean(g_wind),
                C1, C2, C12, save_amt, share_1, share_2, ε)
        end
        println("  saved: $(joinpath(dir, "summary.csv"))")

        open(joinpath(dir, "correlations.csv"), "w") do io
            println(io, "measure,value,n_obs")
            @printf(io, "pearson_all_hours,%.6f,%d\n",      corr.pearson_all,     T)
            @printf(io, "pearson_daytime,%.6f,%d\n",        corr.pearson_daytime, corr.n_daytime_hours)
            @printf(io, "spearman_all_hours,%.6f,%d\n",     corr.spearman,        T)
            @printf(io, "pearson_daily_mean,%.6f,%d\n",     corr.pearson_daily,   T ÷ 24)
        end
        println("  saved: $(joinpath(dir, "correlations.csv"))")

        plot_all_p4(g_pv, g_wind, period_label, is_full_year, month, corr, dir)
    end

    return (
        period  = period_label,
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

###################################
### Curtailment extension
### Sweep λ_curtail and observe how the cooperative game responds.
### λ_curtail = Inf  → no curtailment (current model)
### λ_curtail = 0    → degenerate (curtail everything → zero cost)
###################################
function run_p4_curtailment_sweep(;
        λ_grid::Vector{Float64} = [Inf, 1.0, 0.5, 0.2, 0.1, 0.05, 0.025, 0.01, 0.0],
        T_hours::Int = 8760, save::Bool = true)
    println("\n" * "="^60)
    println("P4 curtailment sweep:  T = $T_hours h,  |λ_grid| = $(length(λ_grid))")
    println("="^60)

    pv_full   = load_ninja_capacity_factors(P4_PV_PATH)
    wind_full = load_ninja_capacity_factors(P4_WIND_PATH)
    T = min(T_hours, 8760)
    g_pv   = pv_full[1:T]
    g_wind = wind_full[1:T]

    n   = 2
    N   = [1, 2]
    Tset = collect(1:T)
    L   = [(1, 0), (2, 0), (1, 2)]
    INV = Dict((1, 0) => 90.0, (2, 0) => 100.0, (1, 2) => 50.0)
    P   = 0.0
    g   = zeros(Float64, n, T)
    g[1, :] = g_pv
    g[2, :] = g_wind

    available_pv   = sum(g_pv)
    available_wind = sum(g_wind)

    rows = NamedTuple[]
    for λ in λ_grid
        sols = compute_all_solutions(n, N, Tset, g, L, INV; P=P, λ_curtail=λ)
        C    = Dict(k => v.cost for (k, v) in sols)
        C1   = C[[1]]; C2 = C[[2]]; C12 = C[[1, 2]]
        sav  = C1 + C2 - C12

        x_star  = nucleolus_sequential_lp(2, C)
        share_1 = x_star[1] / max(C12, 1e-12)
        share_2 = x_star[2] / max(C12, 1e-12)

        # Investment-only cost components (strip the λ·Σc penalty for fair comparison).
        inv_1   = sols[[1]].invest_cost
        inv_2   = sols[[2]].invest_cost
        inv_12  = sols[[1, 2]].invest_cost

        # Curtailment fractions: standalone vs grand-coalition decisions.
        curt_pv_alone   = sum(get(sols[[1]].c,    (1, t), 0.0) for t in Tset) / max(available_pv,   1e-12)
        curt_wind_alone = sum(get(sols[[2]].c,    (2, t), 0.0) for t in Tset) / max(available_wind, 1e-12)
        curt_pv_grand   = sum(get(sols[[1, 2]].c, (1, t), 0.0) for t in Tset) / max(available_pv,   1e-12)
        curt_wind_grand = sum(get(sols[[1, 2]].c, (2, t), 0.0) for t in Tset) / max(available_wind, 1e-12)

        @printf("λ = %-7s | C1=%7.3f C2=%7.3f C12=%7.3f sav=%6.3f shares=(%.3f,%.3f) | curt%%(grand) PV=%5.2f Wind=%5.2f\n",
            isfinite(λ) ? @sprintf("%.4f", λ) : "Inf",
            C1, C2, C12, sav, share_1, share_2,
            100*curt_pv_grand, 100*curt_wind_grand)

        push!(rows, (
            λ_curtail        = λ,
            C_1              = C1,
            C_2              = C2,
            C_12             = C12,
            invest_1         = inv_1,
            invest_2         = inv_2,
            invest_12        = inv_12,
            savings          = sav,
            invest_savings   = inv_1 + inv_2 - inv_12,
            share_1          = share_1,
            share_2          = share_2,
            x_star_1         = x_star[1],
            x_star_2         = x_star[2],
            curt_pv_alone    = curt_pv_alone,
            curt_wind_alone  = curt_wind_alone,
            curt_pv_grand    = curt_pv_grand,
            curt_wind_grand  = curt_wind_grand,
            F_pv_grand       = sols[[1, 2]].F[(1, 0)],
            F_wind_grand     = sols[[1, 2]].F[(2, 0)],
            F_link_grand     = sols[[1, 2]].F[(1, 2)],
        ))
    end

    if save
        dir = joinpath(@__DIR__, "results", "problem4", "curtailment")
        mkpath(dir)
        open(joinpath(dir, "curtailment_sweep.csv"), "w") do io
            ks = keys(rows[1])
            println(io, join(ks, ","))
            for r in rows
                vals = [isfinite(r[k]) ? @sprintf("%.6f", r[k]) :
                        (r[k] === Inf ? "Inf" : "NaN") for k in ks]
                println(io, join(vals, ","))
            end
        end
        println("  saved: $(joinpath(dir, "curtailment_sweep.csv"))")

        plot_curtailment_sweep(rows, dir)
    end

    return rows
end

###################################
### Dispatch-pattern plots: see what curtailment actually does to the data.
### Solves the grand coalition at one focus λ (for time-domain plots) plus a
### handful of overlay λs (for duration-curve peak-shaving overlay).
###################################
function run_p4_curtailment_dispatch(;
        λ_focus::Float64 = 0.1,
        λ_overlay::Vector{Float64} = [0.5, 0.1, 0.05],
        T_hours::Int = 8760, save::Bool = true)
    println("\n" * "="^60)
    println("P4 dispatch-pattern plots:  λ_focus = $λ_focus,  overlay = $λ_overlay")
    println("="^60)

    pv_full   = load_ninja_capacity_factors(P4_PV_PATH)
    wind_full = load_ninja_capacity_factors(P4_WIND_PATH)
    T = min(T_hours, 8760)
    g_pv   = pv_full[1:T]
    g_wind = wind_full[1:T]

    n   = 2
    N   = [1, 2]
    Tset = collect(1:T)
    L   = [(1, 0), (2, 0), (1, 2)]
    INV = Dict((1, 0) => 90.0, (2, 0) => 100.0, (1, 2) => 50.0)
    g   = zeros(Float64, n, T)
    g[1, :] = g_pv
    g[2, :] = g_wind
    s_grand = [1, 1]

    function curt_arrays(λ::Float64)
        sol = solve_coalition_lp(s_grand, N, Tset, g, L, INV; λ_curtail=λ)
        c_pv   = [get(sol.c, (1, t), 0.0) for t in 1:T]
        c_wind = [get(sol.c, (2, t), 0.0) for t in 1:T]
        return c_pv, c_wind
    end

    c_pv_focus,   c_wind_focus   = curt_arrays(λ_focus)
    overlay_curts = Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}}()
    for λ in λ_overlay
        overlay_curts[λ] = curt_arrays(λ)
    end

    @printf("  λ_focus = %.4f  →  PV curt %.2f%%,  wind curt %.2f%%\n",
        λ_focus,
        100 * sum(c_pv_focus) / sum(g_pv),
        100 * sum(c_wind_focus) / sum(g_wind))

    if save
        dir = joinpath(@__DIR__, "results", "problem4", "curtailment")
        mkpath(dir)
        plot_curtailment_dispatch_panels(g_pv, g_wind,
            c_pv_focus, c_wind_focus, λ_focus,
            overlay_curts, dir)
    end

    return (g_pv = g_pv, g_wind = g_wind,
            c_pv_focus = c_pv_focus, c_wind_focus = c_wind_focus,
            overlay_curts = overlay_curts)
end

# to run
run_p4()
