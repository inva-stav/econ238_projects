using JuMP
using HiGHS
using Statistics
using Plots
using CSV
using DataFrames
using Printf
using Dates

# =============================================================================
# Task 4 — ERCOT demand response optimization
#
# Tolling + DR model using ERCOT summer data (2015–2024, Jun–Sep).
# Each calendar month = one scenario (N=40). T=720 hours per scenario (30 days).
# Generators: Wind + Solar (ERCOT CFs). Single price series (HB_BUSAVG RTM).
# =============================================================================

const ERCOT_DATA_DIR = joinpath(@__DIR__, "task4", "task4_data")
const RESULTS_DIR    = joinpath(@__DIR__, "results", "task4")
mkpath(RESULTS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load ERCOT summer data → (T × N) matrices
# ─────────────────────────────────────────────────────────────────────────────

function load_ercot_summer_data()
    prices_df = CSV.read(joinpath(ERCOT_DATA_DIR, "prices_summer.csv"), DataFrame)
    wind_df   = CSV.read(joinpath(ERCOT_DATA_DIR, "wind_cf_summer.csv"), DataFrame)
    solar_df  = CSV.read(joinpath(ERCOT_DATA_DIR, "solar_cf_summer.csv"), DataFrame)

    parse_dt(s) = DateTime(string(s)[1:19], "yyyy-mm-dd HH:MM:SS")
    for df in (prices_df, wind_df, solar_df)
        df[!, :dt]    = parse_dt.(df[!, :datetime_utc])
        df[!, :year]  = year.(df[!, :dt])
        df[!, :month] = month.(df[!, :dt])
    end

    # Left join on prices. June has 6 UTC-midnight hours missing from the
    # CST-based fuel mix; fill with 0 (solar exact; wind conservative).
    df = leftjoin(select(prices_df, :dt, :lmp_usd_mwh => :price),
                  select(wind_df,   :dt, :wind_cf),  on=:dt)
    df = leftjoin(df, select(solar_df, :dt, :solar_cf), on=:dt)
    df[!, :wind_cf]  = coalesce.(df[!, :wind_cf],  0.0)
    df[!, :solar_cf] = coalesce.(df[!, :solar_cf], 0.0)
    df[!, :year]  = year.(df[!, :dt])
    df[!, :month] = month.(df[!, :dt])
    sort!(df, :dt)

    scenario_keys = sort(unique(collect(zip(df[!, :year], df[!, :month]))))
    N_scen = length(scenario_keys)
    T_scen = 720

    π_mat     = Matrix{Float64}(undef, T_scen, N_scen)
    wind_mat  = Matrix{Float64}(undef, T_scen, N_scen)
    solar_mat = Matrix{Float64}(undef, T_scen, N_scen)

    for (ω, (yr, mo)) in enumerate(scenario_keys)
        rows = sort(df[(df[!, :year] .== yr) .& (df[!, :month] .== mo), :], :dt)
        π_mat[:, ω]     = rows[1:T_scen, :price]
        wind_mat[:, ω]  = rows[1:T_scen, :wind_cf]
        solar_mat[:, ω] = rows[1:T_scen, :solar_cf]
    end

    return π_mat, wind_mat, solar_mat, scenario_keys
end

const π_ERCOT, wind_cf_mat, solar_cf_mat, SCENARIO_LABELS = load_ercot_summer_data()

const T             = 720
const N             = size(π_ERCOT, 2)   # 40
const HOURS_PER_DAY = 24
const NUM_DAYS      = T ÷ HOURS_PER_DAY  # 30
const HOURS_IN_DAY  = Dict(d => ((HOURS_PER_DAY*(d-1)+1):(HOURS_PER_DAY*d)) for d in 1:NUM_DAYS)

const EC_Wind  = 10.0
const EC_Solar = 10.0

println("Loaded $(N) scenarios, $(T) hours each.")
println("Scenario range: $(SCENARIO_LABELS[1]) → $(SCENARIO_LABELS[end])")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CVaR (reused from Task 1)
# ─────────────────────────────────────────────────────────────────────────────

function cvar(profit::AbstractVector{<:Real}, α::Real)
    n = length(profit)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, z)
    @variable(m, δ[1:n] >= 0)
    @constraint(m, [ω=1:n], δ[ω] >= z - profit[ω])
    @objective(m, Max, z - (1 / (α * n)) * sum(δ))
    optimize!(m)
    return objective_value(m)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Profit vector (post-solve, for reporting)
# ─────────────────────────────────────────────────────────────────────────────

function compute_profit_vec(Psell, Qsell, QWind, QSolar, Qdr_contracted, Qdr_disp;
                             PWind=0.0, PSolar=0.0, Pdr=0.0, scenario_indices=1:N)
    n = length(scenario_indices)
    Π = zeros(n)
    for (i, ω) in enumerate(scenario_indices)
        Π[i] = sum(
            (Psell - π_ERCOT[t,ω]) * (Qsell - Qdr_disp[t,i])
            + (π_ERCOT[t,ω] * wind_cf_mat[t,ω]  - PWind)  * QWind
            + (π_ERCOT[t,ω] * solar_cf_mat[t,ω] - PSolar) * QSolar
        for t in 1:T) - Qdr_contracted * Pdr
    end
    return Π
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Contracting MILP
# ─────────────────────────────────────────────────────────────────────────────

"""
Solve the contracting MILP.

- `active`: NamedTuple (wind=Bool, solar=Bool) to toggle generators.
- `scenario_indices`: which columns of data matrices to use (pass `1:1` for
  a single-scenario test).

Returns scalars plus `y_vals` (T×n DR event indicators), `s_vals` (T×n event
starts), and `profit_vec` (length-n scenario profits).
"""
function solve_contracting_lp(Psell::Float64;
                               λ::Float64=0.5, α::Float64=0.05,
                               Qsell_max::Float64=10.0,
                               Qdr_contracted_max::Float64=5.0,
                               PWind=0.0, PSolar=0.0, Pdr=0.0,
                               active=(wind=true, solar=true),
                               scenario_indices=1:N)
    n = length(scenario_indices)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, 0 <= Qsell          <= Qsell_max)
    @variable(m, 0 <= QWind          <= (active.wind  ? EC_Wind  : 0.0))
    @variable(m, 0 <= QSolar         <= (active.solar ? EC_Solar : 0.0))
    @variable(m, 0 <= Qdr_contracted <= Qdr_contracted_max)
    @variable(m, Qdr_disp[1:T, 1:n] >= 0)
    @variable(m, z)
    @variable(m, δ[1:n] >= 0)
    @variable(m, y[1:T, 1:n], Bin)
    @variable(m, s[1:T, 1:n], Bin)

    day_starts = [(d-1)*HOURS_PER_DAY + 1 for d in 1:NUM_DAYS]
    @constraint(m, [t in day_starts, i in 1:n], s[t,i] >= y[t,i])
    @constraint(m, [t in 2:T, i in 1:n], s[t,i] >= y[t,i] - y[t-1,i])
    @constraint(m, [d in 1:NUM_DAYS, i in 1:n], sum(s[t,i] for t in HOURS_IN_DAY[d]) <= 1)
    @constraint(m, [d in 1:NUM_DAYS, i in 1:n], sum(y[t,i] for t in HOURS_IN_DAY[d]) <= 6)
    @constraint(m, [i in 1:n], sum(s[t,i] for t in 1:T) <= 10)
    @constraint(m, [t in 1:T, i in 1:n], Qdr_disp[t,i] <= Qdr_contracted)
    @constraint(m, [t in 1:T, i in 1:n], Qdr_disp[t,i] <= Qdr_contracted_max * y[t,i])

    @expression(m, Π[i=1:n],
        sum(
            (Psell - π_ERCOT[t, scenario_indices[i]]) * (Qsell - Qdr_disp[t,i])
            + (π_ERCOT[t, scenario_indices[i]] * wind_cf_mat[t, scenario_indices[i]]  - PWind)  * QWind
            + (π_ERCOT[t, scenario_indices[i]] * solar_cf_mat[t, scenario_indices[i]] - PSolar) * QSolar
        for t in 1:T) - Qdr_contracted * Pdr
    )

    @constraint(m, [i=1:n], δ[i] >= z - Π[i])
    @objective(m, Max,
        λ * (z - (1/(α*n)) * sum(δ)) + (1-λ) * (1/n) * sum(Π))

    optimize!(m)

    qs    = value(Qsell)
    qw    = value(QWind)
    qsol  = value(QSolar)
    qdr_c = value(Qdr_contracted)
    qdr_d  = [value(Qdr_disp[t,i]) for t in 1:T, i in 1:n]
    y_vals = [round(Int, value(y[t,i])) for t in 1:T, i in 1:n]
    s_vals = [round(Int, value(s[t,i])) for t in 1:T, i in 1:n]

    profit_vec = compute_profit_vec(Psell, qs, qw, qsol, qdr_c, qdr_d;
                                    PWind=PWind, PSolar=PSolar, Pdr=Pdr,
                                    scenario_indices=scenario_indices)
    EΠ       = mean(profit_vec)
    CVaR_val = cvar(profit_vec, α)
    ρval     = λ * CVaR_val + (1-λ) * EΠ

    return (Qsell=qs, QWind=qw, QSolar=qsol, Qdr_contracted=qdr_c,
            EΠ=EΠ, CVaR=CVaR_val, ρ=ρval,
            y_vals=y_vals, s_vals=s_vals, profit_vec=profit_vec)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Sweep helpers
# ─────────────────────────────────────────────────────────────────────────────

function run_psell_sweep(; Psell_grid=20.0:10.0:120.0, λ=0.5, α=0.05,
                           Qsell_max=10.0, Qdr_contracted_max=5.0,
                           PWind=0.0, PSolar=0.0, Pdr=0.0)
    configs = [
        ("Wind+Solar",    (wind=true, solar=true), 0.0),
        ("Wind+Solar+DR", (wind=true, solar=true), Qdr_contracted_max),
    ]
    results = Dict(name => NamedTuple[] for (name, _, _) in configs)
    total   = length(Psell_grid) * length(configs)
    count   = 0
    for (name, act, qdr_max) in configs
        for Ps in Psell_grid
            r = solve_contracting_lp(Float64(Ps); λ=λ, α=α,
                                      Qsell_max=Qsell_max,
                                      Qdr_contracted_max=qdr_max,
                                      PWind=PWind, PSolar=PSolar, Pdr=Pdr,
                                      active=act)
            push!(results[name], r)
            count += 1
            println("  Psell sweep $count/$total  ($(name), Ps=\$$(Int(Ps)))")
        end
    end
    Ps_vals = collect(Psell_grid)
    # Save to CSV
    rows = [(config=name, Psell=Ps, Qsell=r.Qsell, QWind=r.QWind, QSolar=r.QSolar,
             Qdr_contracted=r.Qdr_contracted, EΠ=r.EΠ, CVaR=r.CVaR, ρ=r.ρ)
            for (name, rs) in results for (Ps, r) in zip(Ps_vals, rs)]
    CSV.write(joinpath(RESULTS_DIR, "sweep_bilateral_price.csv"), DataFrame(rows))
    println("  Saved sweep_bilateral_price.csv")
    return Ps_vals, results
end

function load_psell_sweep()
    path = joinpath(RESULTS_DIR, "sweep_bilateral_price.csv")
    df = CSV.read(path, DataFrame)
    Ps_vals = sort(unique(df[!, :Psell]))
    results = Dict{String, Vector{NamedTuple}}()
    for name in unique(df[!, :config])
        sub = sort(df[df[!, :config] .== name, :], :Psell)
        results[name] = [(Qsell=row.Qsell, QWind=row.QWind, QSolar=row.QSolar,
                          Qdr_contracted=row.Qdr_contracted,
                          EΠ=row.EΠ, CVaR=row.CVaR, ρ=row.ρ)
                         for row in eachrow(sub)]
    end
    return Ps_vals, results
end

function run_lambda_sweep(Psell; λ_grid=0.1:0.1:1.0, α=0.05,
                           Qsell_max=10.0, Qdr_contracted_max=5.0,
                           PWind=0.0, PSolar=0.0, Pdr=0.0)
    rr_results = NamedTuple[]
    for (k, λv) in enumerate(λ_grid)
        r = solve_contracting_lp(Float64(Psell); λ=Float64(λv), α=α,
                                  Qsell_max=Qsell_max,
                                  Qdr_contracted_max=Qdr_contracted_max,
                                  PWind=PWind, PSolar=PSolar, Pdr=Pdr)
        push!(rr_results, r)
        println("  λ sweep $(k)/$(length(λ_grid))  (λ=$(λv))")
    end
    λ_vals = collect(λ_grid)
    # Save to CSV
    rows = [(λ=λv, Qsell=r.Qsell, QWind=r.QWind, QSolar=r.QSolar,
             Qdr_contracted=r.Qdr_contracted, EΠ=r.EΠ, CVaR=r.CVaR, ρ=r.ρ)
            for (λv, r) in zip(λ_vals, rr_results)]
    CSV.write(joinpath(RESULTS_DIR, "sweep_risk_weight.csv"), DataFrame(rows))
    println("  Saved sweep_risk_weight.csv")
    return λ_vals, rr_results
end

function load_lambda_sweep()
    df = CSV.read(joinpath(RESULTS_DIR, "sweep_risk_weight.csv"), DataFrame)
    sort!(df, :λ)
    rr_results = [(Qsell=row.Qsell, QWind=row.QWind, QSolar=row.QSolar,
                   Qdr_contracted=row.Qdr_contracted,
                   EΠ=row.EΠ, CVaR=row.CVaR, ρ=row.ρ)
                  for row in eachrow(df)]
    return df[!, :λ], rr_results
end

function run_qdr_sweep(Psell; qdr_grid=0.0:0.5:5.0, λ=0.5, α=0.05, Qsell_max=10.0,
                       PWind=0.0, PSolar=0.0, Pdr=0.0)
    results = NamedTuple[]
    total = length(qdr_grid)
    for (k, qdr_max) in enumerate(qdr_grid)
        r = solve_contracting_lp(Float64(Psell); λ=λ, α=α,
                                  Qsell_max=Qsell_max,
                                  Qdr_contracted_max=Float64(qdr_max),
                                  PWind=PWind, PSolar=PSolar, Pdr=Pdr)
        push!(results, r)
        println("  Qdr sweep $(k)/$(total)  (Qdr_max=$(qdr_max) MW)")
    end
    qdr_vals = collect(qdr_grid)
    rows = [(Qdr_max=q, Qsell=r.Qsell, QWind=r.QWind, QSolar=r.QSolar,
             Qdr_contracted=r.Qdr_contracted, EΠ=r.EΠ, CVaR=r.CVaR, ρ=r.ρ)
            for (q, r) in zip(qdr_vals, results)]
    CSV.write(joinpath(RESULTS_DIR, "sweep_qdr_capacity.csv"), DataFrame(rows))
    println("  Saved sweep_qdr_capacity.csv")
    return qdr_vals, results
end

function load_qdr_sweep()
    df = CSV.read(joinpath(RESULTS_DIR, "sweep_qdr_capacity.csv"), DataFrame)
    sort!(df, :Qdr_max)
    results = [(Qsell=row.Qsell, QWind=row.QWind, QSolar=row.QSolar,
                Qdr_contracted=row.Qdr_contracted,
                EΠ=row.EΠ, CVaR=row.CVaR, ρ=row.ρ)
               for row in eachrow(df)]
    return df[!, :Qdr_max], results
end

function run_tail_scenario_analysis(Psell; α=0.05, λ=0.5, Qsell_max=10.0,
                                     PWind=0.0, PSolar=0.0, Pdr=0.0)
    # Solve with Qdr=0 to get per-scenario profit vector
    println("  Solving with Qdr_max=0 MW...")
    r0 = solve_contracting_lp(Float64(Psell); λ=λ, α=α,
                               Qsell_max=Qsell_max, Qdr_contracted_max=0.0,
                               PWind=PWind, PSolar=PSolar, Pdr=Pdr)

    # Load Qdr=5 profits from baseline cache
    pv5 = CSV.read(joinpath(RESULTS_DIR, "baseline_scenario_profits.csv"), DataFrame)[!, :profit]

    profit0 = r0.profit_vec
    profit5 = Vector{Float64}(pv5)

    n_tail = max(1, floor(Int, α * N))
    tail0  = sort(sortperm(profit0)[1:n_tail])
    tail5  = sort(sortperm(profit5)[1:n_tail])

    sc_labels = [string(yr)*"-"*lpad(mo,2,'0') for (yr,mo) in SCENARIO_LABELS]
    df = DataFrame(
        scenario     = sc_labels,
        profit_qdr0  = profit0,
        profit_qdr5  = profit5,
        in_tail_qdr0 = [i in tail0 for i in 1:N],
        in_tail_qdr5 = [i in tail5 for i in 1:N],
    )
    CSV.write(joinpath(RESULTS_DIR, "tail_scenario_analysis.csv"), df)
    println("  Saved tail_scenario_analysis.csv")
    return (profit0=profit0, profit5=profit5, tail0=tail0, tail5=tail5, sc_labels=sc_labels)
end

function load_tail_scenario_analysis()
    df = CSV.read(joinpath(RESULTS_DIR, "tail_scenario_analysis.csv"), DataFrame;
                  types=Dict(:scenario => String))
    tail0 = findall(df[!, :in_tail_qdr0])
    tail5 = findall(df[!, :in_tail_qdr5])
    return (profit0=df[!, :profit_qdr0], profit5=df[!, :profit_qdr5],
            tail0=tail0, tail5=tail5, sc_labels=df[!, :scenario])
end

function plot_tail_scenarios(td)
    cap = 250.0
    px  = clamp.(π_ERCOT, -Inf, cap)
    hrs = 1:T

    lbl0 = "tail Qdr=0 MW  (" * join(td.sc_labels[td.tail0], ", ") * ")"
    lbl5 = "tail Qdr=5 MW  (" * join(td.sc_labels[td.tail5], ", ") * ")"

    plt = plot(xlabel="Hour within month",
               ylabel="Spot price (\$/MWh, capped \$250)",
               title="Price scenario paths — CVaR₀.₀₅ tails for Qdr ∈ {0, 5} MW  (Psell=\$70, λ=0.5)",
               legend=:topright, size=(1100,520),
               left_margin=8Plots.mm, bottom_margin=5Plots.mm)

    for ω in 1:N
        plot!(plt, hrs, px[:, ω]; color=:gray, alpha=0.15, linewidth=0.5, label="")
    end
    for (k, ω) in enumerate(td.tail0)
        plot!(plt, hrs, px[:, ω]; color=:red,      alpha=0.9, linewidth=1.8, label=(k==1 ? lbl0 : ""))
    end
    for (k, ω) in enumerate(td.tail5)
        plot!(plt, hrs, px[:, ω]; color=:royalblue, alpha=0.9, linewidth=1.8, label=(k==1 ? lbl5 : ""))
    end

    savefig(plt, joinpath(RESULTS_DIR, "tail_price_scenarios.png"))
    println("Saved tail_price_scenarios.png")
    return plt
end

function plot_ranked_profits(td)
    # Sort scenarios by Qdr=0 profit (worst → best)
    order     = sortperm(td.profit0)
    p0_sorted = td.profit0[order]
    p5_sorted = td.profit5[order]
    labels_sorted = td.sc_labels[order]
    ranks = 1:N

    n_tail = length(td.tail0)
    tail_ranks = 1:n_tail   # tail is always the leftmost n_tail after sorting

    plt = plot(xlabel="Scenario rank (1 = worst profit without DR)",
               ylabel="Scenario profit (\$)",
               title="Ranked scenario profits — Qdr=0 vs Qdr=5 MW  (Psell=\$70, λ=0.5)",
               legend=:topleft, size=(950, 520),
               left_margin=8Plots.mm, bottom_margin=5Plots.mm)

    # All scenarios: lines
    plot!(plt, ranks, p0_sorted; color=:red,      linewidth=1.5, label="Profit — Qdr=0 MW", alpha=0.7)
    plot!(plt, ranks, p5_sorted; color=:steelblue, linewidth=1.5, label="Profit — Qdr=5 MW", alpha=0.7)

    # Tail markers
    scatter!(plt, tail_ranks, p0_sorted[tail_ranks]; color=:red,      markersize=7, markerstrokewidth=0, label="")
    scatter!(plt, tail_ranks, p5_sorted[tail_ranks]; color=:steelblue, markersize=7, markerstrokewidth=0, label="")

    # Vertical line at tail cutoff
    vline!(plt, [n_tail + 0.5]; color=:black, linestyle=:dash, linewidth=1.2,
           label=@sprintf("CVaR₀.₀₅ cutoff (rank ≤ %d)", n_tail))

    # Annotate tail scenario labels
    for r in tail_ranks
        annotate!(plt, r + 0.3, p0_sorted[r],
                  text(labels_sorted[r], 7, :left, :red))
    end

    # Shaded region between Qdr=0 and Qdr=5 in the tail
    plot!(plt, tail_ranks, p0_sorted[tail_ranks];
          fillrange=p5_sorted[tail_ranks], fillalpha=0.15, color=:steelblue,
          linewidth=0, label="DR benefit in tail")

    hline!(plt, [0.0]; color=:gray, linestyle=:dot, linewidth=1, label="")

    savefig(plt, joinpath(RESULTS_DIR, "ranked_scenario_profits.png"))
    println("Saved ranked_scenario_profits.png")
    return plt
end

function plot_profit_vs_qdr(qdr_vals, qdr_results)
    EΠ_vals   = [r.EΠ   for r in qdr_results]
    CVaR_vals = [r.CVaR for r in qdr_results]
    ρ_vals    = [r.ρ    for r in qdr_results]

    plt = plot(xlabel="DR contracted capacity (MW)",
               ylabel="Value (\$)",
               title="Profit vs DR contracted capacity  (Psell=\$70, λ=0.5, Wind+Solar)",
               legend=:bottomright, size=(800,500),
               left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    plot!(plt, qdr_vals, EΠ_vals;   label="E[Π]",                      color=:steelblue, linewidth=2)
    plot!(plt, qdr_vals, CVaR_vals; label="CVaR₀.₀₅",                  color=:red,       linewidth=2)
    plot!(plt, qdr_vals, ρ_vals;    label="ρ = λ·CVaR + (1−λ)·E[Π]",  color=:purple,    linewidth=2, linestyle=:dash)

    savefig(plt, joinpath(RESULTS_DIR, "profit_vs_qdr.png"))
    println("Saved profit_vs_qdr.png")
    return plt
end

# Save/load the full optimization result (scalars + dispatch matrices)
function save_result_full(r; Psell=70.0)
    sc_labels = [string(yr)*"-"*lpad(mo,2,'0') for (yr,mo) in SCENARIO_LABELS]
    # Scalars: optimal contract decisions for the baseline run
    CSV.write(joinpath(RESULTS_DIR, "baseline_optimal_contract.csv"),
              DataFrame(Psell=Psell, Qsell=r.Qsell, QWind=r.QWind, QSolar=r.QSolar,
                        Qdr_contracted=r.Qdr_contracted,
                        EΠ=r.EΠ, CVaR=r.CVaR, ρ=r.ρ))
    # Per-scenario profits under the optimal contract
    CSV.write(joinpath(RESULTS_DIR, "baseline_scenario_profits.csv"),
              DataFrame(scenario=sc_labels, profit=r.profit_vec))
    # Binary matrix: y[t,ω]=1 means DR is active in hour t, scenario ω
    y_df = DataFrame(r.y_vals, sc_labels)
    insertcols!(y_df, 1, :t => 1:T)
    CSV.write(joinpath(RESULTS_DIR, "baseline_dr_active.csv"), y_df)
    # Binary matrix: s[t,ω]=1 means a new DR event starts in hour t, scenario ω
    s_df = DataFrame(r.s_vals, sc_labels)
    insertcols!(s_df, 1, :t => 1:T)
    CSV.write(joinpath(RESULTS_DIR, "baseline_dr_starts.csv"), s_df)
    println("  Saved baseline_optimal_contract.csv, baseline_scenario_profits.csv, baseline_dr_active.csv, baseline_dr_starts.csv")
end

function load_result_full()
    sc   = CSV.read(joinpath(RESULTS_DIR, "baseline_optimal_contract.csv"),  DataFrame)
    pv   = CSV.read(joinpath(RESULTS_DIR, "baseline_scenario_profits.csv"),  DataFrame)
    y_df = CSV.read(joinpath(RESULTS_DIR, "baseline_dr_active.csv"),         DataFrame)
    s_df = CSV.read(joinpath(RESULTS_DIR, "baseline_dr_starts.csv"),         DataFrame)
    y_vals = Matrix{Int}(y_df[:, 2:end])
    s_vals = Matrix{Int}(s_df[:, 2:end])
    return (Qsell=sc[1,:Qsell], QWind=sc[1,:QWind], QSolar=sc[1,:QSolar],
            Qdr_contracted=sc[1,:Qdr_contracted],
            EΠ=sc[1,:EΠ], CVaR=sc[1,:CVaR], ρ=sc[1,:ρ],
            y_vals=y_vals, s_vals=s_vals, profit_vec=pv[!,:profit])
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot functions
# ─────────────────────────────────────────────────────────────────────────────

# ── Plot 0: Spot price histogram ─────────────────────────────────────────────

function plot_price_histogram(; cap=300.0)
    prices = vec(π_ERCOT)
    n_total  = length(prices)
    n_capped = sum(prices .> cap)
    prices_shown = clamp.(prices, -Inf, cap)

    pct_above = 100*n_capped/n_total
    plt = histogram(prices_shown; bins=80, normalize=:pdf,
                    xlabel="Spot price (USD/MWh)",
                    ylabel="Density",
                    title=@sprintf("ERCOT RTM spot prices — Jun–Sep 2015–2024  (N=%d hours, %.1f%% above %.0f USD cap)",
                                   n_total, pct_above, cap),
                    label="", color=:steelblue, alpha=0.7,
                    size=(800, 450),
                    left_margin=8Plots.mm, bottom_margin=8Plots.mm)
    vline!(plt, [mean(prices)];
           color=:black, linestyle=:dash, linewidth=2,
           label=@sprintf("Mean  %.0f USD/MWh", mean(prices)))
    vline!(plt, [median(prices)];
           color=:gray, linestyle=:dot, linewidth=2,
           label=@sprintf("Median %.0f USD/MWh", median(prices)))
    vline!(plt, [Psell];
           color=:red, linestyle=:dash, linewidth=2,
           label=@sprintf("Psell = %.0f USD/MWh", Psell))
    savefig(plt, joinpath(RESULTS_DIR, "price_histogram.png"))
    println("Saved price_histogram.png")
    println(@sprintf("  Mean=\$%.1f  Median=\$%.1f  Std=\$%.1f  Min=\$%.1f  Max=\$%.1f",
                     mean(prices), median(prices), std(prices), minimum(prices), maximum(prices)))
    return plt
end

# ── Plot 4: Seasonality ───────────────────────────────────────────────────────

function plot_seasonality()
    month_nums   = [6, 7, 8, 9]
    month_labels = ["Jun", "Jul", "Aug", "Sep"]
    xs = 1:4

    mo_idx = Dict(mo => [i for (i, (_, m)) in enumerate(SCENARIO_LABELS) if m == mo]
                  for mo in month_nums)

    # Cap prices at 300 to keep the axis readable (spikes ≤ $9000 exist)
    cap = 300.0
    p_mean = [mean(clamp.(vec(π_ERCOT[:, mo_idx[mo]]), -Inf, cap)) for mo in month_nums]
    p_q25  = [quantile(clamp.(vec(π_ERCOT[:, mo_idx[mo]]), -Inf, cap), 0.25) for mo in month_nums]
    p_q75  = [quantile(clamp.(vec(π_ERCOT[:, mo_idx[mo]]), -Inf, cap), 0.75) for mo in month_nums]

    w_mean = [mean(vec(wind_cf_mat[:, mo_idx[mo]]))              for mo in month_nums]
    w_q25  = [quantile(vec(wind_cf_mat[:, mo_idx[mo]]), 0.25)    for mo in month_nums]
    w_q75  = [quantile(vec(wind_cf_mat[:, mo_idx[mo]]), 0.75)    for mo in month_nums]

    s_mean = [mean(vec(solar_cf_mat[:, mo_idx[mo]]))             for mo in month_nums]
    s_q25  = [quantile(vec(solar_cf_mat[:, mo_idx[mo]]), 0.25)   for mo in month_nums]
    s_q75  = [quantile(vec(solar_cf_mat[:, mo_idx[mo]]), 0.75)   for mo in month_nums]

    plt_p = plot(xs, p_mean;
                 ribbon=(p_mean .- p_q25, p_q75 .- p_mean),
                 label="Mean (IQR)", color=:darkred, linewidth=2, fillalpha=0.25,
                 xticks=(xs, month_labels), xlabel="Month",
                 ylabel="Price (\$/MWh)", title="ERCOT Prices (capped at \$300/MWh)")

    plt_w = plot(xs, w_mean;
                 ribbon=(w_mean .- w_q25, w_q75 .- w_mean),
                 label="Mean (IQR)", color=:steelblue, linewidth=2, fillalpha=0.25,
                 xticks=(xs, month_labels), xlabel="Month",
                 ylabel="Capacity Factor", title="Wind CF", ylim=(0,1))

    plt_s = plot(xs, s_mean;
                 ribbon=(s_mean .- s_q25, s_q75 .- s_mean),
                 label="Mean (IQR)", color=:darkorange, linewidth=2, fillalpha=0.25,
                 xticks=(xs, month_labels), xlabel="Month",
                 ylabel="Capacity Factor", title="Solar CF", ylim=(0,1))

    plt = plot(plt_p, plt_w, plt_s; layout=(1,3), size=(1200,420),
               left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    savefig(plt, joinpath(RESULTS_DIR, "seasonality.png"))
    println("Saved seasonality.png")
    return plt
end

# ── Plot 1: Willingness-to-supply ─────────────────────────────────────────────

function plot_willingness_to_supply(Ps_vals, sweep_results)
    rs_ws  = sweep_results["Wind+Solar"]
    rs_dr  = sweep_results["Wind+Solar+DR"]
    mean_price = mean(vec(π_ERCOT))

    plt = plot(xlabel="P_sell (\$/MWh)", ylabel="Quantity (MW)",
               title="Optimal quantities vs bilateral contract price  (λ=0.5, N=40 scenarios)",
               legend=:topleft, size=(900,500),
               left_margin=8Plots.mm, bottom_margin=5Plots.mm)

    # Wind+Solar (no DR): Qsell, QWind, QSolar
    plot!(plt, Ps_vals, [r.Qsell  for r in rs_ws]; label="Qsell — W+S",      color=:steelblue,  linewidth=2)
    plot!(plt, Ps_vals, [r.QWind  for r in rs_ws]; label="QWind — W+S",      color=:steelblue,  linewidth=1.5, linestyle=:dash)
    plot!(plt, Ps_vals, [r.QSolar for r in rs_ws]; label="QSolar — W+S",     color=:steelblue,  linewidth=1.5, linestyle=:dot)

    # Wind+Solar+DR: Qsell, QWind, QSolar, Qdr_contracted
    plot!(plt, Ps_vals, [r.Qsell          for r in rs_dr]; label="Qsell — W+S+DR",         color=:red,    linewidth=2)
    plot!(plt, Ps_vals, [r.QWind          for r in rs_dr]; label="QWind — W+S+DR",         color=:red,    linewidth=1.5, linestyle=:dash)
    plot!(plt, Ps_vals, [r.QSolar         for r in rs_dr]; label="QSolar — W+S+DR",        color=:red,    linewidth=1.5, linestyle=:dot)
    plot!(plt, Ps_vals, [r.Qdr_contracted for r in rs_dr]; label="Qdr_contracted — W+S+DR",color=:purple, linewidth=2,   linestyle=:dash)

    vline!(plt, [mean_price];
           label=@sprintf("Mean spot \$%.0f/MWh", mean_price),
           color=:black, linestyle=:dot, linewidth=1.5)

    savefig(plt, joinpath(RESULTS_DIR, "willingness_to_supply.png"))
    println("Saved willingness_to_supply.png")
    return plt
end

# ── Plot 3: Certainty equivalent & diversification benefit ────────────────────

function plot_certainty_equivalent(Ps_vals, sweep_results)
    ρ_dr = [r.ρ for r in sweep_results["Wind+Solar+DR"]]
    ρ_ws = [r.ρ for r in sweep_results["Wind+Solar"]]
    dr_benefit = ρ_dr .- ρ_ws

    plt_ce = plot(xlabel="P_sell (\$/MWh)", ylabel="Certainty equivalent ρ (\$)",
                  title="Certainty equivalent  (λ=0.5)",
                  legend=:topleft, size=(800,450),
                  left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    plot!(plt_ce, Ps_vals, ρ_dr; label="ρ  Wind+Solar+DR", color=:red,      linewidth=2)
    plot!(plt_ce, Ps_vals, ρ_ws; label="ρ  Wind+Solar",    color=:steelblue, linewidth=2)

    plt_dr = plot(Ps_vals, dr_benefit;
                  xlabel="P_sell (\$/MWh)", ylabel="DR benefit  Δρ (\$)",
                  title="ρ(W+S+DR) − ρ(W+S)",
                  label="DR benefit", color=:purple, linewidth=2,
                  fill=0, fillalpha=0.2, legend=:topleft,
                  left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    hline!(plt_dr, [0.0]; color=:black, linestyle=:dot, label="")

    plt = plot(plt_ce, plt_dr; layout=(1,2), size=(1400,450),
               left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    savefig(plt, joinpath(RESULTS_DIR, "certainty_equivalent.png"))
    println("Saved certainty_equivalent.png")
    return plt
end

# ── Plot 2: Risk-return frontier ──────────────────────────────────────────────

function plot_risk_return_frontier(λ_vals, rr_results)
    EΠ_vals  = [r.EΠ  for r in rr_results]
    CVaR_vals = [r.CVaR for r in rr_results]
    RP_vals  = EΠ_vals .- CVaR_vals   # risk premium = how far mean exceeds tail

    plt = scatter(RP_vals, EΠ_vals;
                  xlabel="Risk  E[Π] − CVaR₀.₀₅ (\$)",
                  ylabel="Return  E[Π] (\$)",
                  title="Risk-return frontier  (Psell=\$70, Wind+Solar)",
                  marker_z=λ_vals, color=:viridis, markerstrokewidth=0,
                  markersize=7, colorbar_title="λ",
                  label="", legend=false,
                  size=(700,500),
                  left_margin=8Plots.mm, bottom_margin=5Plots.mm)

    # Annotate first and last λ
    annotate!(plt, RP_vals[1]  + 500, EΠ_vals[1],
              text(@sprintf("λ=%.1f", λ_vals[1]),   8, :left))
    annotate!(plt, RP_vals[end]+ 500, EΠ_vals[end],
              text(@sprintf("λ=%.1f", λ_vals[end]), 8, :left))

    savefig(plt, joinpath(RESULTS_DIR, "risk_return_frontier.png"))
    println("Saved risk_return_frontier.png")
    return plt
end

# ── Plot 5: DR dispatch analysis ──────────────────────────────────────────────

function plot_dr_dispatch(result_full)
    y = result_full.y_vals   # T × N
    s = result_full.s_vals   # T × N

    # Hour-of-day frequency (UTC; ERCOT local = UTC-6)
    hourly_rate = zeros(HOURS_PER_DAY)
    for h in 1:HOURS_PER_DAY
        t_slots = h:HOURS_PER_DAY:T
        hourly_rate[h] = mean(y[t_slots, :])
    end
    plt1 = bar(0:23, hourly_rate;
               xlabel="Hour of day (UTC;  ERCOT local = UTC−6)",
               ylabel="Fraction of scenario-days with DR active",
               title="DR dispatch frequency by hour-of-day",
               legend=false, color=:steelblue,
               left_margin=8Plots.mm, bottom_margin=8Plots.mm,
               size=(800,400))

    # Events per scenario
    n_events = [sum(s[:,i]) for i in 1:N]
    sc_labels = [string(yr)*"-"*lpad(mo,2,'0') for (yr,mo) in SCENARIO_LABELS]

    plt2 = bar(1:N, n_events;
               xlabel="Scenario (year-month)", ylabel="DR events called",
               title="DR events per scenario  (max = 10/month)",
               legend=false, color=:coral,
               xticks=(1:N, sc_labels), xrotation=90,
               yticks=0:2:10, ylim=(0,11),
               left_margin=8Plots.mm, bottom_margin=35Plots.mm,
               size=(1200,450))
    hline!(plt2, [10.0]; color=:black, linestyle=:dash, linewidth=1.5, label="")

    # Price distribution: DR-active hours vs DR-inactive hours
    dr_mask = y .== 1
    prices_dr   = clamp.(vec(π_ERCOT)[vec(dr_mask)],   -100, 400)
    prices_nodr = clamp.(vec(π_ERCOT)[.!vec(dr_mask)], -100, 400)

    plt3 = histogram(prices_nodr; normalize=:pdf, bins=60,
                     label="DR inactive", alpha=0.5, color=:blue,
                     xlabel="Price (\$/MWh, clamped at ±400)",
                     ylabel="Density",
                     title="Price when DR active vs inactive",
                     left_margin=8Plots.mm, bottom_margin=5Plots.mm,
                     size=(700,400))
    histogram!(plt3, prices_dr; normalize=:pdf, bins=60,
               label="DR active", alpha=0.6, color=:red)
    vline!(plt3, [mean(prices_dr)];   color=:red,  linestyle=:dash, linewidth=2, label=@sprintf("DR mean \$%.0f", mean(prices_dr)))
    vline!(plt3, [mean(prices_nodr)]; color=:blue, linestyle=:dash, linewidth=2, label=@sprintf("No-DR mean \$%.0f", mean(prices_nodr)))

    # Save individually so each is readable
    savefig(plt1, joinpath(RESULTS_DIR, "dr_hourly_freq.png"))
    savefig(plt2, joinpath(RESULTS_DIR, "dr_events_per_scenario.png"))
    savefig(plt3, joinpath(RESULTS_DIR, "dr_price_distribution.png"))
    println("Saved dr_hourly_freq.png, dr_events_per_scenario.png, dr_price_distribution.png")
    return plt1, plt2, plt3
end


#################################################################################
##### RUNNING THE SCRIPT ########################################################
#################################################################################

Psell              = 55.0
λ                  = 0.5
α                  = 0.05
Qsell_max          = 10.0
Qdr_contracted_max = 2.0
PWind              = 16.0
PSolar             = 13.0
Pdr                = 100.0

# ── Spot price histogram (fast — no optimization) ────────────────────────────
println("\nGenerating spot price histogram...")
plot_price_histogram()

# ── Single-scenario smoke test ───────────────────────────────────────────────
println("\nRunning single-scenario test (scenario 1: $(SCENARIO_LABELS[1]))...")
t0 = time()
result_test = solve_contracting_lp(Psell; λ=λ, α=α,
    Qsell_max=Qsell_max, Qdr_contracted_max=Qdr_contracted_max,
    PWind=PWind, PSolar=PSolar, Pdr=Pdr,
    scenario_indices=1:1)
elapsed = time() - t0
println("Done in $(round(elapsed; digits=1))s")
@printf("  Qsell=%.2f MW  QWind=%.2f MW  QSolar=%.2f MW  Qdr_contracted=%.2f MW\n",
        result_test.Qsell, result_test.QWind, result_test.QSolar, result_test.Qdr_contracted)
@printf("  E[Π]=\$%.2f  CVaR=\$%.2f  ρ=\$%.2f\n",
        result_test.EΠ, result_test.CVaR, result_test.ρ)

# ── Full 40-scenario run (cache-or-compute) ───────────────────────────────────
if isfile(joinpath(RESULTS_DIR, "baseline_optimal_contract.csv"))
    println("\nLoading full result from cache...")
    result_full = load_result_full()
else
    println("\nRunning full 40-scenario optimization...")
    t1 = time()
    result_full = solve_contracting_lp(Psell; λ=λ, α=α,
        Qsell_max=Qsell_max, Qdr_contracted_max=Qdr_contracted_max,
        PWind=PWind, PSolar=PSolar, Pdr=Pdr)
    println("Done in $(round(time()-t1; digits=1))s")
    save_result_full(result_full; Psell=Psell)
end
@printf("  Qsell=%.2f MW  QWind=%.2f MW  QSolar=%.2f MW  Qdr_contracted=%.2f MW\n",
        result_full.Qsell, result_full.QWind, result_full.QSolar, result_full.Qdr_contracted)
@printf("  E[Π]=\$%.2f  CVaR=\$%.2f  ρ=\$%.2f\n",
        result_full.EΠ, result_full.CVaR, result_full.ρ)

# ── DR dispatch plots (fast — uses result_full) ───────────────────────────────
println("\nGenerating DR dispatch plots...")
plot_dr_dispatch(result_full)

# ── Tail scenario analysis (cache-or-compute, ~20s) ──────────────────────────
if isfile(joinpath(RESULTS_DIR, "tail_scenario_analysis.csv"))
    println("\nLoading tail scenario analysis from cache...")
    tail_data = load_tail_scenario_analysis()
else
    println("\nRunning tail scenario analysis (≈20s)...")
    tail_data = run_tail_scenario_analysis(Psell; α=α, λ=λ, Qsell_max=Qsell_max,
                                           PWind=PWind, PSolar=PSolar, Pdr=Pdr)
end
println("\nGenerating tail price scenarios plot...")
plot_tail_scenarios(tail_data)
println("Generating ranked scenario profits plot...")
plot_ranked_profits(tail_data)

# ── Qdr sweep → profit vs DR capacity (disabled) ─────────────────────────────
# if isfile(joinpath(RESULTS_DIR, "sweep_qdr_capacity.csv"))
#     println("\nLoading Qdr sweep from cache...")
#     qdr_vals, qdr_results = load_qdr_sweep()
# else
#     println("\nRunning Qdr sweep (≈4 min)...")
#     t_q = time()
#     qdr_vals, qdr_results = run_qdr_sweep(Psell; qdr_grid=0.0:0.5:2.0, λ=λ, α=α,
#                                            Qsell_max=Qsell_max,
#                                            PWind=PWind, PSolar=PSolar, Pdr=Pdr)
#     println("Qdr sweep done in $(round(time()-t_q; digits=0))s")
# end
# println("\nGenerating profit vs DR capacity plot...")
# plot_profit_vs_qdr(qdr_vals, qdr_results)

# ── Psell sweep → willingness-to-supply & certainty equivalent (cache only) ──
if isfile(joinpath(RESULTS_DIR, "sweep_bilateral_price.csv"))
    println("\nLoading Psell sweep from cache...")
    Ps_vals, sweep_results = load_psell_sweep()
    println("Generating willingness-to-supply plot...")
    plot_willingness_to_supply(Ps_vals, sweep_results)
    println("Generating certainty equivalent plot...")
    plot_certainty_equivalent(Ps_vals, sweep_results)
else
    println("\nNo Psell sweep cache found — skipping willingness-to-supply & certainty equivalent.")
end

# ── λ sweep → risk-return frontier (cache only) ───────────────────────────────
if isfile(joinpath(RESULTS_DIR, "sweep_risk_weight.csv"))
    println("\nLoading λ sweep from cache...")
    λ_vals, rr_results = load_lambda_sweep()
    println("Generating risk-return frontier plot...")
    plot_risk_return_frontier(λ_vals, rr_results)
else
    println("\nNo λ sweep cache found — skipping risk-return frontier.")
end

println("\n═══ Task 4 complete — plots saved to $(RESULTS_DIR) ═══")
