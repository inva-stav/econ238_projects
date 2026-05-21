using JuMP
using HiGHS
using Statistics
using Plots
using CSV
using DataFrames
using Printf
using Dates

# =============================================================================
# Task 4 — ERCOT DR optimization (energy-budget LP formulation)
#
# Replaces the MILP event-scheduling constraints with a single per-scenario
# MWh budget: sum_t Qdr_disp[t,ω] ≤ DR_BUDGET_HOURS × Qdr_contracted_max.
# Pure LP — no binary variables — solves in seconds instead of ~7 minutes.
# =============================================================================

const ERCOT_DATA_DIR = joinpath(@__DIR__, "task4", "task4_data")
const RESULTS_DIR    = joinpath(@__DIR__, "results", "task4_lp")
mkpath(RESULTS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load ERCOT summer data → (T × N) matrices  (identical to task4.jl)
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

const EC_Wind  = 10.0
const EC_Solar = 10.0

# Energy budget equivalent to MILP maximum: 10 events × 6 h/event = 60 h
const DR_BUDGET_HOURS = 60.0

println("Loaded $(N) scenarios, $(T) hours each.")
println("Scenario range: $(SCENARIO_LABELS[1]) → $(SCENARIO_LABELS[end])")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Bootstrap data loader
# ─────────────────────────────────────────────────────────────────────────────

function load_bootstrap_data(n::Int=1000)
    tag = "bootstrap_$(n)"
    read_wide(name) = begin
        df = CSV.read(joinpath(ERCOT_DATA_DIR, "$(name)_$(tag).csv"), DataFrame)
        Matrix{Float64}(df[:, 2:end])   # drop the :t column
    end
    π_bs    = read_wide("prices")
    wind_bs = read_wide("wind_cf")
    sol_bs  = read_wide("solar_cf")
    println("Loaded bootstrap data: $(size(π_bs,2)) scenarios, $(size(π_bs,1)) hours each.")
    return π_bs, wind_bs, sol_bs
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. CVaR
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
                             PWind=0.0, PSolar=0.0, Pdr=0.0,
                             price_mat=π_ERCOT, wind_mat=wind_cf_mat, solar_mat=solar_cf_mat)
    n = size(price_mat, 2)
    Π = zeros(n)
    for i in 1:n
        Π[i] = sum(
            (Psell - price_mat[t,i]) * (Qsell - Qdr_disp[t,i])
            + (price_mat[t,i] * wind_mat[t,i]  - PWind)  * QWind
            + (price_mat[t,i] * solar_mat[t,i] - PSolar) * QSolar
        for t in 1:T) - Qdr_contracted * Pdr
    end
    return Π
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Contracting LP (energy-budget DR formulation)
# ─────────────────────────────────────────────────────────────────────────────

"""
Solve the contracting LP with an energy-budget DR constraint.

DR dispatch is bounded by:
  - per-hour capacity:  Qdr_disp[t,ω] ≤ Qdr_contracted
  - monthly MWh budget: Σ_t Qdr_disp[t,ω] ≤ DR_BUDGET_HOURS × Qdr_contracted_max

No binary variables — pure LP, solves in seconds.
"""
function solve_contracting_lp(Psell::Float64;
                               λ::Float64=0.5, α::Float64=0.05,
                               Qsell_max::Float64=10.0,
                               Qdr_contracted_max::Float64=5.0,
                               PWind=0.0, PSolar=0.0, Pdr=0.0,
                               active=(wind=true, solar=true),
                               price_mat=π_ERCOT,
                               wind_mat=wind_cf_mat,
                               solar_mat=solar_cf_mat)
    n = size(price_mat, 2)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, 0 <= Qsell          <= Qsell_max)
    @variable(m, 0 <= QWind          <= (active.wind  ? EC_Wind  : 0.0))
    @variable(m, 0 <= QSolar         <= (active.solar ? EC_Solar : 0.0))
    @variable(m, 0 <= Qdr_contracted <= Qdr_contracted_max)
    @variable(m, Qdr_disp[1:T, 1:n] >= 0)
    @variable(m, z)
    @variable(m, δ[1:n] >= 0)

    dr_energy_cap = DR_BUDGET_HOURS * Qdr_contracted_max
    @constraint(m, [i in 1:n], sum(Qdr_disp[t,i] for t in 1:T) <= dr_energy_cap)
    @constraint(m, [t in 1:T, i in 1:n], Qdr_disp[t,i] <= Qdr_contracted)

    @expression(m, Π[i=1:n],
        sum(
            (Psell - price_mat[t,i]) * (Qsell - Qdr_disp[t,i])
            + (price_mat[t,i] * wind_mat[t,i]  - PWind)  * QWind
            + (price_mat[t,i] * solar_mat[t,i] - PSolar) * QSolar
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
    qdr_d = [value(Qdr_disp[t,i]) for t in 1:T, i in 1:n]

    profit_vec = compute_profit_vec(Psell, qs, qw, qsol, qdr_c, qdr_d;
                                    PWind=PWind, PSolar=PSolar, Pdr=Pdr,
                                    price_mat=price_mat, wind_mat=wind_mat, solar_mat=solar_mat)
    EΠ       = mean(profit_vec)
    CVaR_val = cvar(profit_vec, α)
    ρval     = λ * CVaR_val + (1-λ) * EΠ

    return (Qsell=qs, QWind=qw, QSolar=qsol, Qdr_contracted=qdr_c,
            EΠ=EΠ, CVaR=CVaR_val, ρ=ρval,
            qdr_disp=qdr_d, profit_vec=profit_vec)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Sweep helpers
# ─────────────────────────────────────────────────────────────────────────────

function run_qdr_sweep(Psell; qdr_grid=0.0:0.5:2.0, λ=0.5, α=0.05, Qsell_max=10.0,
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

function run_psell_sweep(; Psell_grid=20.0:5.0:120.0, λ=0.5, α=0.05,
                          Qsell_max=10.0, Qdr_contracted_max=2.0,
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
    rows = [(config=name, Psell=Ps, Qsell=r.Qsell, QWind=r.QWind, QSolar=r.QSolar,
             Qdr_contracted=r.Qdr_contracted, EΠ=r.EΠ, CVaR=r.CVaR, ρ=r.ρ)
            for (name, rs) in results for (Ps, r) in zip(Ps_vals, rs)]
    CSV.write(joinpath(RESULTS_DIR, "sweep_bilateral_price.csv"), DataFrame(rows))
    println("  Saved sweep_bilateral_price.csv")
    return Ps_vals, results
end

function run_lambda_sweep(Psell; λ_grid=0.0:0.1:1.0, α=0.05,
                           Qsell_max=10.0, Qdr_contracted_max=2.0,
                           PWind=0.0, PSolar=0.0, Pdr=0.0)
    results = NamedTuple[]
    total = length(λ_grid)
    for (k, λv) in enumerate(λ_grid)
        r = solve_contracting_lp(Float64(Psell); λ=Float64(λv), α=α,
                                  Qsell_max=Qsell_max,
                                  Qdr_contracted_max=Qdr_contracted_max,
                                  PWind=PWind, PSolar=PSolar, Pdr=Pdr)
        push!(results, r)
        println("  λ sweep $(k)/$(total)  (λ=$(λv))")
    end
    λ_vals = collect(λ_grid)
    rows = [(λ=λv, Qsell=r.Qsell, QWind=r.QWind, QSolar=r.QSolar,
             Qdr_contracted=r.Qdr_contracted, EΠ=r.EΠ, CVaR=r.CVaR, ρ=r.ρ)
            for (λv, r) in zip(λ_vals, results)]
    CSV.write(joinpath(RESULTS_DIR, "sweep_risk_weight.csv"), DataFrame(rows))
    println("  Saved sweep_risk_weight.csv")
    return λ_vals, results
end

function run_tail_scenario_analysis(Psell; α=0.05, λ=0.5, Qsell_max=10.0,
                                     PWind=0.0, PSolar=0.0, Pdr=0.0,
                                     Qdr_contracted_max=2.0)
    println("  Solving with Qdr_max=0 MW...")
    r0 = solve_contracting_lp(Float64(Psell); λ=λ, α=α,
                               Qsell_max=Qsell_max, Qdr_contracted_max=0.0,
                               PWind=PWind, PSolar=PSolar, Pdr=Pdr)
    println("  Solving with Qdr_max=$(Qdr_contracted_max) MW...")
    r_max = solve_contracting_lp(Float64(Psell); λ=λ, α=α,
                                  Qsell_max=Qsell_max, Qdr_contracted_max=Qdr_contracted_max,
                                  PWind=PWind, PSolar=PSolar, Pdr=Pdr)

    n_tail = max(1, floor(Int, α * N))
    tail0  = sort(sortperm(r0.profit_vec)[1:n_tail])
    tail_max = sort(sortperm(r_max.profit_vec)[1:n_tail])

    sc_labels = [string(yr)*"-"*lpad(mo,2,'0') for (yr,mo) in SCENARIO_LABELS]
    qdr_tag = Int(Qdr_contracted_max)
    df = DataFrame(
        :scenario                          => sc_labels,
        :profit_qdr0                       => r0.profit_vec,
        Symbol("profit_qdr$(qdr_tag)")     => r_max.profit_vec,
        :in_tail_qdr0                      => [i in tail0    for i in 1:N],
        Symbol("in_tail_qdr$(qdr_tag)")    => [i in tail_max for i in 1:N],
    )
    CSV.write(joinpath(RESULTS_DIR, "tail_scenario_analysis.csv"), df)
    println("  Saved tail_scenario_analysis.csv")
    return (profit0=r0.profit_vec, profit_max=r_max.profit_vec,
            tail0=tail0, tail_max=tail_max, sc_labels=sc_labels)
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Save / load baseline
# ─────────────────────────────────────────────────────────────────────────────

function save_result_full(r; Psell=55.0)
    sc_labels = [string(yr)*"-"*lpad(mo,2,'0') for (yr,mo) in SCENARIO_LABELS]
    CSV.write(joinpath(RESULTS_DIR, "baseline_optimal_contract.csv"),
              DataFrame(Psell=Psell, Qsell=r.Qsell, QWind=r.QWind, QSolar=r.QSolar,
                        Qdr_contracted=r.Qdr_contracted,
                        EΠ=r.EΠ, CVaR=r.CVaR, ρ=r.ρ))
    CSV.write(joinpath(RESULTS_DIR, "baseline_scenario_profits.csv"),
              DataFrame(scenario=sc_labels, profit=r.profit_vec))
    # Continuous dispatch matrix (MWh per hour per scenario)
    disp_df = DataFrame(r.qdr_disp, sc_labels)
    insertcols!(disp_df, 1, :t => 1:T)
    CSV.write(joinpath(RESULTS_DIR, "baseline_dr_dispatch.csv"), disp_df)
    println("  Saved baseline_optimal_contract.csv, baseline_scenario_profits.csv, baseline_dr_dispatch.csv")
end

function load_result_full()
    sc      = CSV.read(joinpath(RESULTS_DIR, "baseline_optimal_contract.csv"),  DataFrame)
    pv      = CSV.read(joinpath(RESULTS_DIR, "baseline_scenario_profits.csv"),  DataFrame)
    disp_df = CSV.read(joinpath(RESULTS_DIR, "baseline_dr_dispatch.csv"),       DataFrame)
    qdr_disp = Matrix{Float64}(disp_df[:, 2:end])
    return (Qsell=sc[1,:Qsell], QWind=sc[1,:QWind], QSolar=sc[1,:QSolar],
            Qdr_contracted=sc[1,:Qdr_contracted],
            EΠ=sc[1,:EΠ], CVaR=sc[1,:CVaR], ρ=sc[1,:ρ],
            qdr_disp=qdr_disp, profit_vec=pv[!,:profit])
end

function save_bs_result(r, n_bs; Psell=55.0)
    n = size(r.qdr_disp, 2)
    scen_cols = ["s$(lpad(i,4,'0'))" for i in 1:n]
    prefix = "bs$(n_bs)"
    CSV.write(joinpath(RESULTS_DIR, "$(prefix)_baseline_optimal_contract.csv"),
              DataFrame(Psell=Psell, Qsell=r.Qsell, QWind=r.QWind, QSolar=r.QSolar,
                        Qdr_contracted=r.Qdr_contracted,
                        EΠ=r.EΠ, CVaR=r.CVaR, ρ=r.ρ))
    CSV.write(joinpath(RESULTS_DIR, "$(prefix)_baseline_scenario_profits.csv"),
              DataFrame(scenario=scen_cols, profit=r.profit_vec))
    disp_df = DataFrame(r.qdr_disp, scen_cols)
    insertcols!(disp_df, 1, :t => 1:T)
    CSV.write(joinpath(RESULTS_DIR, "$(prefix)_baseline_dr_dispatch.csv"), disp_df)
    println("  Saved $(prefix)_baseline_optimal_contract.csv, $(prefix)_baseline_scenario_profits.csv, $(prefix)_baseline_dr_dispatch.csv")
end

function load_bs_result(n_bs)
    prefix  = "bs$(n_bs)"
    sc      = CSV.read(joinpath(RESULTS_DIR, "$(prefix)_baseline_optimal_contract.csv"),  DataFrame)
    pv      = CSV.read(joinpath(RESULTS_DIR, "$(prefix)_baseline_scenario_profits.csv"),  DataFrame)
    disp_df = CSV.read(joinpath(RESULTS_DIR, "$(prefix)_baseline_dr_dispatch.csv"),       DataFrame)
    qdr_disp = Matrix{Float64}(disp_df[:, 2:end])
    return (Qsell=sc[1,:Qsell], QWind=sc[1,:QWind], QSolar=sc[1,:QSolar],
            Qdr_contracted=sc[1,:Qdr_contracted],
            EΠ=sc[1,:EΠ], CVaR=sc[1,:CVaR], ρ=sc[1,:ρ],
            qdr_disp=qdr_disp, profit_vec=pv[!,:profit])
end

# ─────────────────────────────────────────────────────────────────────────────
# 7. Run
# ─────────────────────────────────────────────────────────────────────────────

Psell              = 55.0
λ                  = 0.5
α                  = 0.05
Qsell_max          = 10.0
Qdr_contracted_max = 2.0
PWind              = 16.0
PSolar             = 13.0
Pdr                = 100.0

# ── Single-scenario smoke test ───────────────────────────────────────────────
println("\nRunning single-scenario test (scenario 1: $(SCENARIO_LABELS[1]))...")
t0 = time()
result_test = solve_contracting_lp(Psell; λ=λ, α=α,
    Qsell_max=Qsell_max, Qdr_contracted_max=Qdr_contracted_max,
    PWind=PWind, PSolar=PSolar, Pdr=Pdr,
    price_mat=π_ERCOT[:, 1:1], wind_mat=wind_cf_mat[:, 1:1], solar_mat=solar_cf_mat[:, 1:1])
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
    println("\nRunning full 40-scenario LP optimization...")
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

# ── Tail scenario analysis (cache-or-compute) ─────────────────────────────────
if isfile(joinpath(RESULTS_DIR, "tail_scenario_analysis.csv"))
    println("\nLoading tail scenario analysis from cache...")
else
    println("\nRunning tail scenario analysis...")
    run_tail_scenario_analysis(Psell; α=α, λ=λ, Qsell_max=Qsell_max,
                               PWind=PWind, PSolar=PSolar, Pdr=Pdr,
                               Qdr_contracted_max=Qdr_contracted_max)
end

# ── Qdr sweep (cache-or-compute) ──────────────────────────────────────────────
if isfile(joinpath(RESULTS_DIR, "sweep_qdr_capacity.csv"))
    println("\nLoading Qdr sweep from cache...")
else
    println("\nRunning Qdr sweep...")
    t_q = time()
    run_qdr_sweep(Psell; qdr_grid=0.0:0.25:Qdr_contracted_max, λ=λ, α=α,
                  Qsell_max=Qsell_max, PWind=PWind, PSolar=PSolar, Pdr=Pdr)
    println("Qdr sweep done in $(round(time()-t_q; digits=1))s")
end

# ── Psell sweep (cache-or-compute) ───────────────────────────────────────────
if isfile(joinpath(RESULTS_DIR, "sweep_bilateral_price.csv"))
    println("\nLoading Psell sweep from cache...")
else
    println("\nRunning Psell sweep...")
    t_p = time()
    run_psell_sweep(; Psell_grid=20.0:5.0:120.0, λ=λ, α=α,
                     Qsell_max=Qsell_max, Qdr_contracted_max=Qdr_contracted_max,
                     PWind=PWind, PSolar=PSolar, Pdr=Pdr)
    println("Psell sweep done in $(round(time()-t_p; digits=1))s")
end

# ── λ sweep (cache-or-compute) ────────────────────────────────────────────────
if isfile(joinpath(RESULTS_DIR, "sweep_risk_weight.csv"))
    println("\nLoading λ sweep from cache...")
else
    println("\nRunning λ sweep...")
    t_l = time()
    run_lambda_sweep(Psell; λ_grid=0.0:0.1:1.0, α=α,
                     Qsell_max=Qsell_max, Qdr_contracted_max=Qdr_contracted_max,
                     PWind=PWind, PSolar=PSolar, Pdr=Pdr)
    println("λ sweep done in $(round(time()-t_l; digits=1))s")
end

# ── Bootstrap 1000-scenario run (cache-or-compute) ────────────────────────────
N_BS = 1000
bs_cache = joinpath(RESULTS_DIR, "bs$(N_BS)_baseline_optimal_contract.csv")
if isfile(bs_cache)
    println("\nLoading bootstrap $(N_BS)-scenario result from cache...")
    result_bs = load_bs_result(N_BS)
else
    println("\nLoading bootstrap data...")
    π_bs, wind_bs, solar_bs = load_bootstrap_data(N_BS)
    println("Running bootstrap $(N_BS)-scenario LP optimization...")
    t_bs = time()
    result_bs = solve_contracting_lp(Psell; λ=λ, α=α,
        Qsell_max=Qsell_max, Qdr_contracted_max=Qdr_contracted_max,
        PWind=PWind, PSolar=PSolar, Pdr=Pdr,
        price_mat=π_bs, wind_mat=wind_bs, solar_mat=solar_bs)
    println("Done in $(round(time()-t_bs; digits=1))s")
    save_bs_result(result_bs, N_BS; Psell=Psell)
end
@printf("  Bootstrap  Qsell=%.2f MW  QWind=%.2f MW  QSolar=%.2f MW  Qdr_contracted=%.2f MW\n",
        result_bs.Qsell, result_bs.QWind, result_bs.QSolar, result_bs.Qdr_contracted)
@printf("  Bootstrap  E[Π]=\$%.2f  CVaR=\$%.2f  ρ=\$%.2f\n",
        result_bs.EΠ, result_bs.CVaR, result_bs.ρ)

println("\n═══ task4_dr_budget_lp complete — results in $(RESULTS_DIR) ═══\n")
