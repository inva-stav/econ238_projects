using JuMP
using HiGHS
using Statistics
using Plots
using CSV
using DataFrames
using Printf

# =============================================================================
# Task 3 — Willingness-to-supply curve and the value of the portfolio
#
# Contract format: Tolling (fixed retainer P_i * Q_i per period; TC receives
#   GSF_i * Q_i of generation and sells at spot).
#
# Rationale: Tolling transfers the full price-and-quantity risk to the TC,
#   making the portfolio diversification effect most visible.
# =============================================================================

const DATA_DIR = joinpath(@__DIR__, "data")
const RESULTS_DIR = joinpath(@__DIR__, "results", "task4_high_red_cost")
mkpath(RESULTS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load scenario data from CSVs (exported from the workbook)
# ─────────────────────────────────────────────────────────────────────────────

function load_matrix(filename)
    path = joinpath(DATA_DIR, filename)
    df = CSV.read(path, DataFrame; header=false)
    return Matrix{Float64}(df)          # 12 × 2000
end

const GSF_Wind = load_matrix("wind_gsf.csv")   # 12 × 2000
const GSF_SH   = load_matrix("sh_gsf.csv")
const GSF_Bio  = load_matrix("bio_gsf.csv")
const π_SE     = load_matrix("spot_se.csv")     # 12 × 2000
const π_NE     = load_matrix("spot_ne.csv")

const T  = 12
const N  = size(GSF_Wind, 2)     # 2000
const Ht = Float64[744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]

const EC_Wind = 11.41
const EC_SH   = 17.50
const EC_Bio  = 17.59

println("Data loaded: T=$T, N=$N, total hours=$(sum(Ht))")

# ─────────────────────────────────────────────────────────────────────────────
# 2. CVaR function (reused from Task 1)
# ─────────────────────────────────────────────────────────────────────────────

function cvar(profit::AbstractVector{<:Real}, α::Real)
    N = length(profit)
    m = Model(HiGHS.Optimizer)
    set_silent(m)
    @variable(m, z)
    @variable(m, δ[1:N] >= 0)
    @constraint(m, [ω=1:N], δ[ω] >= z - profit[ω])
    @objective(m, Max, z - (1 / (α * N)) * sum(δ))
    optimize!(m)
    return objective_value(m)
end

function rho(profit::AbstractVector{<:Real}, α::Real, λ::Real)
    return λ * cvar(profit, α) + (1 - λ) * mean(profit)
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Annual profit vector under TOLLING for a given decision q
# ─────────────────────────────────────────────────────────────────────────────

function annual_profit_tolling(Qsell, Q_Wind, Q_SH, Q_Bio)
    Π = zeros(N)
    for ω in 1:N
        for t in 1:T
            profit_t = (Qsell * (140.0 - π_NE[t,ω])       # placeholder Psell; replaced in LP
                        + Q_Wind * (π_NE[t,ω] * GSF_Wind[t,ω] - 100.0)
                        + Q_SH   * (π_SE[t,ω] * GSF_SH[t,ω]   - 100.0)
                        + Q_Bio  * (π_SE[t,ω] * GSF_Bio[t,ω]   - 100.0))
            Π[ω] += Ht[t] * profit_t
        end
    end
    return Π
end

"""
Build the annual profit vector for arbitrary P_sell (prices P_i fixed at 100).
Under tolling:  Π_{t,ω} = (P_sell - π^NE_{t,ω}) Q_sell
                         + Σ_i [ π^{n(i)}_{t,ω} GSF^i_{t,ω} Q_i  -  P_i Q_i ]
"""
function annual_profit_vec(Psell, Qsell, QWind, QSH, QBio;
                           PWind=100.0, PSH=100.0, PBio=100.0)
    Π = zeros(N)
    for ω in 1:N
        for t in 1:T
            π_t = (Psell - π_NE[t,ω]) * Qsell +
                  (π_NE[t,ω] * GSF_Wind[t,ω] - PWind) * QWind +
                  (π_SE[t,ω] * GSF_SH[t,ω]   - PSH)   * QSH +
                  (π_SE[t,ω] * GSF_Bio[t,ω]   - PBio)  * QBio
            Π[ω] += Ht[t] * π_t
        end
    end
    return Π
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Contracting LP  (model 27, tolling format)
# ─────────────────────────────────────────────────────────────────────────────

"""
Precompute per-scenario annual coefficients so the LP stays linear in q.

Under tolling the annual profit for scenario ω is:
  Π_ω(q) = a_sell[ω] * Q_sell + a_wind[ω] * Q_wind
          + a_sh[ω] * Q_SH   + a_bio[ω] * Q_bio

where the a-coefficients absorb the hours and spot-price / GSF data.
"""

function optimize_demand_response_dispatch(Psell, Qsell_max, Q_reduced_max, cost_per_reduce,call_max_per_year, Q_duration_max)


function precompute_coefficients(Psell; PWind=100.0, PSH=100.0, PBio=100.0)
    a_sell = zeros(N)
    a_wind = zeros(N)
    a_sh   = zeros(N)
    a_bio  = zeros(N)
    for ω in 1:N
        GSF_DR = 
        # sweep depth, duration (number of events), 

        for t in 1:T
            a_sell[ω] += Ht[t] * (Psell - π_NE[t,ω])
            a_wind[ω] += Ht[t] * (π_NE[t,ω] * GSF_Wind[t,ω] - PWind)
            a_sh[ω]   += Ht[t] * (π_SE[t,ω] * GSF_SH[t,ω]   - PSH)
            a_bio[ω]  += Ht[t] * (π_SE[t,ω] * GSF_Bio[t,ω]   - PBio)
            a_dr 
        end
    end
    return a_sell, a_wind, a_sh, a_bio
end

"""
Solve the contracting LP (model 27, tolling) for a given P_sell and risk
parameters.  `active` is a NamedTuple indicating which generators are available:
  active = (wind=true, sh=true, bio=true)   for the joint portfolio
  active = (wind=true, sh=false, bio=false)  for wind-only, etc.

Returns (Qsell*, QWind*, QSH*, QBio*, E[Π*], CVaR*, ρ*).
"""
function solve_contracting_lp(Psell::Float64;
                               λ::Float64=0.5, α::Float64=0.05,
                               Qsell_max::Float64=100.0,
                               active=(wind=true, sh=true, bio=true),
                               PWind=100.0, PSH=100.0, PBio=100.0)

    a_sell, a_wind, a_sh, a_bio = precompute_coefficients(Psell;
                                        PWind=PWind, PSH=PSH, PBio=PBio)

    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, 0 <= Qsell <= Qsell_max)
    @variable(m, 0 <= QWind <= (active.wind ? EC_Wind : 0.0))
    @variable(m, 0 <= QSH   <= (active.sh   ? EC_SH   : 0.0))
    @variable(m, 0 <= QBio  <= (active.bio  ? EC_Bio  : 0.0))
    @variable(m, z)
    @variable(m, δ[1:N] >= 0)

    # Π_ω as an affine expression of the decision variables
    @expression(m, Π[ω=1:N],
        a_sell[ω] * Qsell + a_wind[ω] * QWind + a_sh[ω] * QSH + a_bio[ω] * QBio)

    @constraint(m, [ω=1:N], δ[ω] >= z - Π[ω])

    @objective(m, Max,
        λ * (z - (1 / (α * N)) * sum(δ)) +
        (1 - λ) * (1 / N) * sum(Π))

    optimize!(m)

    qs  = value(Qsell)
    qw  = value(QWind)
    qsh = value(QSH)
    qb  = value(QBio)

    profit_vec = annual_profit_vec(Psell, qs, qw, qsh, qb;
                                   PWind=PWind, PSH=PSH, PBio=PBio)
    EΠ   = mean(profit_vec)
    CVaR = cvar(profit_vec, α)
    ρval = λ * CVaR + (1 - λ) * EΠ

    return (Qsell=qs, QWind=qw, QSH=qsh, QBio=qb, EΠ=EΠ, CVaR=CVaR, ρ=ρval)
end

# Add demand reduction option variables and constraints
function solve_contracting_lp(Psell::Float64;
                               λ::Float64=0.5, α::Float64=0.05,
                               Qsell_max::Float64=100.0,
                               Q_reduced_max::Float64=20.0,  # Maximum reduction allowed
                               cost_per_reduce::Float64=5.0, # Cost per unit reduction
                               active=(wind=true, sh=true, bio=true),
                               PWind=100.0, PSH=100.0, PBio=100.0)

    a_sell, a_wind, a_sh, a_bio = precompute_coefficients(Psell;
                                        PWind=PWind, PSH=PSH, PBio=PBio)

    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, 0 <= Qsell <= Qsell_max)
    @variable(m, 0 <= QWind <= (active.wind ? EC_Wind : 0.0))
    @variable(m, 0 <= QSH   <= (active.sh   ? EC_SH   : 0.0))
    @variable(m, 0 <= QBio  <= (active.bio  ? EC_Bio  : 0.0))
    @variable(m, 0 <= Q_reduce <= Q_reduced_max)  # New variable for demand reduction
    @variable(m, z)
    @variable(m, δ[1:N] >= 0)

    # Adjust profit calculation to include demand reduction cost
    @expression(m, Π[ω=1:N],
        a_sell[ω] * (Qsell - Q_reduce) + a_wind[ω] * QWind + a_sh[ω] * QSH + a_bio[ω] * QBio)

    @constraint(m, [ω=1:N], δ[ω] >= z - Π[ω])

    @objective(m, Max,
        λ * (z - (1 / (α * N)) * sum(δ)) +
        (1 - λ) * (1 / N) * sum(Π) - cost_per_reduce * Q_reduce)  # Include reduction cost

    optimize!(m)

    qs  = value(Qsell)
    qw  = value(QWind)
    qsh = value(QSH)
    qb  = value(QBio)
    qred = value(Q_reduce)  # Retrieve the reduction value

    profit_vec = annual_profit_vec(Psell, qs - qred, qw, qsh, qb;
                                   PWind=PWind, PSH=PSH, PBio=PBio)
    EΠ   = mean(profit_vec)
    CVaR = cvar(profit_vec, α)
    ρval = λ * CVaR + (1 - λ) * EΠ

    return (Qsell=qs, QWind=qw, QSH=qsh, QBio=qb, Qreduce=qred, EΠ=EΠ, CVaR=CVaR, ρ=ρval)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Task 3(a): Willingness-to-supply plot
# ─────────────────────────────────────────────────────────────────────────────

function run_task3a(; Psell_grid=0.0:5.0:250.0, λ=0.5, α=0.05, Qsell_max=100.0)
    configs = [
        ("Wind only", (wind=true,  sh=false, bio=false)),
        ("SH only",   (wind=false, sh=true,  bio=false)),
        ("Bio only",  (wind=false, sh=false, bio=true)),
        ("Portfolio",  (wind=true,  sh=true,  bio=true)),
    ]

    results = Dict{String, Vector{NamedTuple}}()
    for (name, act) in configs
        results[name] = NamedTuple[]
    end

    total = length(Psell_grid) * length(configs)
    count = 0
    for Ps in Psell_grid
        for (name, act) in configs
            r = solve_contracting_lp(Float64(Ps); λ=λ, α=α,
                                     Qsell_max=Qsell_max, active=act)
            push!(results[name], r)
            count += 1
            if count % 40 == 0
                println("  Progress: $count / $total")
            end
        end
    end

    return Psell_grid, results
end

println("\n─── Running Task 3(a): Willingness-to-supply sweep ───")
Psell_grid, results = run_task3a()

# --- Plot (a): Qsell* and total EC purchased vs P_sell ---
colors = Dict("Wind only"=>:blue, "SH only"=>:green, "Bio only"=>:orange, "Portfolio"=>:red)
Ps = collect(Psell_grid)

plt_a = plot(xlabel="P_sell (\$/MWh)", ylabel="Quantity (avgMW)",
             title="Willingness-to-supply (tolling, λ=0.5, α=0.05)",
             legend=:outertopright, size=(900,500),
             left_margin=8Plots.mm, bottom_margin=5Plots.mm)

for name in ["Wind only", "SH only", "Bio only", "Portfolio"]
    Qsell_vals = [r.Qsell for r in results[name]]
    Qtotal_vals = [r.QWind + r.QSH + r.QBio for r in results[name]]
    plot!(plt_a, Ps, Qsell_vals; label="Qsell — $name",
          color=colors[name], linewidth=2)
    plot!(plt_a, Ps, Qtotal_vals; label="ΣQi — $name",
          color=colors[name], linewidth=2, linestyle=:dash)
end

savefig(plt_a, joinpath(RESULTS_DIR, "willingness_to_supply.png"))
println("Saved willingness_to_supply.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Task 3(b): Certainty equivalent — portfolio vs sum of individuals
# ─────────────────────────────────────────────────────────────────────────────

println("\n─── Running Task 3(b): Certainty equivalent comparison ───")

plt_b = plot(xlabel="P_sell (\$/MWh)", ylabel="Value (\$)",
             title="Certainty equivalent & risk premium: portfolio vs. individuals",
             legend=:outertopright, size=(1000,550),
             left_margin=10Plots.mm, bottom_margin=5Plots.mm)

ρ_portfolio = [r.ρ for r in results["Portfolio"]]
ρ_sum_indiv = [results["Wind only"][i].ρ + results["SH only"][i].ρ + results["Bio only"][i].ρ
               for i in eachindex(Ps)]
RP_portfolio = [r.EΠ - r.CVaR for r in results["Portfolio"]]
RP_sum_indiv = [ρ_sum_indiv[i] - results["Portfolio"][i].CVaR for i in eachindex(Ps)]

plot!(plt_b, Ps, ρ_portfolio; label="ρ — Portfolio", color=:red, linewidth=2)
plot!(plt_b, Ps, ρ_sum_indiv; label="ρ — Sum of individuals", color=:blue, linewidth=2)
plot!(plt_b, Ps, RP_portfolio; label="RP — Portfolio", color=:red, linestyle=:dash, linewidth=2)
plot!(plt_b, Ps, RP_sum_indiv; label="RP — Sum of individuals", color=:blue, linestyle=:dash, linewidth=2)

savefig(plt_b, joinpath(RESULTS_DIR, "certainty_equivalent_comparison.png"))
println("Saved certainty_equivalent_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Solve LP with reduction option and generate plots
# ─────────────────────────────────────────────────────────────────────────────

#function run_task4(; Psell_grid=0.0:5.0:250.0, λ=0.5, α=0.05, Qsell_max=100.0, Q_reduced_max=20.0, cost_per_reduce=5.0)
function run_task4(; Psell_grid=0.0:5.0:250.0, λ=0.5, α=0.05, Qsell_max=100.0, Q_reduced_max=20.0, cost_per_reduce=100.0)
    configs = [
        ("Wind only", (wind=true,  sh=false, bio=false)),
        ("SH only",   (wind=false, sh=true,  bio=false)),
        ("Bio only",  (wind=false, sh=false, bio=true)),
        ("Portfolio",  (wind=true,  sh=true,  bio=true)),
    ]

    results = Dict{String, Vector{NamedTuple}}()
    for (name, act) in configs
        results[name] = NamedTuple[]
    end

    total = length(Psell_grid) * length(configs)
    count = 0
    for Ps in Psell_grid
        for (name, act) in configs
            r = solve_contracting_lp(Float64(Ps); λ=λ, α=α, Qsell_max=Qsell_max, Q_reduced_max=Q_reduced_max, cost_per_reduce=cost_per_reduce, active=act)
            push!(results[name], r)
            count += 1
            if count % 40 == 0
                println("  Progress: $count / $total")
            end
        end
    end

    # Plot 1: Willingness-to-supply
    println("\n─── Generating Willingness-to-Supply Plot ───")
    colors = Dict("Wind only"=>:blue, "SH only"=>:green, "Bio only"=>:orange, "Portfolio"=>:red)
    Ps = collect(Psell_grid)

    plt_a = plot(xlabel="P_sell (\$/MWh)", ylabel="Quantity (avgMW)",
                 title="Willingness-to-supply with reduction option",
                 legend=:outertopright, size=(900,500),
                 left_margin=8Plots.mm, bottom_margin=5Plots.mm)

    for name in ["Wind only", "SH only", "Bio only", "Portfolio"]
        Qsell_vals = [r.Qsell for r in results[name]]
        Qtotal_vals = [r.QWind + r.QSH + r.QBio for r in results[name]]
        Qreduce_vals = [r.Qreduce for r in results[name]]
        plot!(plt_a, Ps, Qsell_vals; label="Qsell — $name",
              color=colors[name], linewidth=2)
        plot!(plt_a, Ps, Qtotal_vals; label="ΣQi — $name",
              color=colors[name], linewidth=2, linestyle=:dash)
        plot!(plt_a, Ps, Qreduce_vals; label="Qreduce — $name",
              color=colors[name], linewidth=2, linestyle=:dot)
    end

    savefig(plt_a, joinpath(RESULTS_DIR, "willingness_to_supply_with_reduction.png"))
    println("Saved willingness_to_supply_with_reduction.png")

    # Plot 2: Certainty equivalent — portfolio vs sum of individuals
    println("\n─── Generating Certainty Equivalent Plot ───")
    plt_b = plot(xlabel="P_sell (\$/MWh)", ylabel="Value (\$)",
                 title="Certainty equivalent & risk premium: portfolio vs. individuals",
                 legend=:outertopright, size=(1000,550),
                 left_margin=10Plots.mm, bottom_margin=5Plots.mm)

    ρ_portfolio = [r.ρ for r in results["Portfolio"]]
    ρ_sum_indiv = [results["Wind only"][i].ρ + results["SH only"][i].ρ + results["Bio only"][i].ρ
                   for i in eachindex(Ps)]
    RP_portfolio = [r.EΠ - r.CVaR for r in results["Portfolio"]]
    RP_sum_indiv = [ρ_sum_indiv[i] - results["Portfolio"][i].CVaR for i in eachindex(Ps)]

    plot!(plt_b, Ps, ρ_portfolio; label="ρ — Portfolio", color=:red, linewidth=2)
    plot!(plt_b, Ps, ρ_sum_indiv; label="ρ — Sum of individuals", color=:blue, linewidth=2)
    plot!(plt_b, Ps, RP_portfolio; label="RP — Portfolio", color=:red, linestyle=:dash, linewidth=2)
    plot!(plt_b, Ps, RP_sum_indiv; label="RP — Sum of individuals", color=:blue, linestyle=:dash, linewidth=2)

    savefig(plt_b, joinpath(RESULTS_DIR, "certainty_equivalent_with_reduction.png"))
    println("Saved certainty_equivalent_with_reduction.png")

    return Psell_grid, results
end

println("\n─── Running Task 4: LP with Reduction Option ───")
Psell_grid, results = run_task4()





###################################
### Operation logic constraints ###

using JuMP
using HiGHS
using Dates

# ------------------------------------------------------------
# Time indexing
# ------------------------------------------------------------

# Non-leap year example
start_time = DateTime(2025, 1, 1, 0, 0, 0)

T = 8760
times = [start_time + Hour(t - 1) for t in 1:T]

# ------------------------------------------------------------
# Build day/month index sets
# ------------------------------------------------------------

# Day index for each hour
day_of_hour = [Dates.dayofyear(t) for t in times]

# Month index for each hour
month_of_hour = [Dates.month(t) for t in times]

# Hours belonging to each day
days = unique(day_of_hour)

hours_in_day = Dict(
    d => findall(day_of_hour .== d)
    for d in days
)

# Hours belonging to each month
months = 1:12

hours_in_month = Dict(
    m => findall(month_of_hour .== m)
    for m in months
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

model = Model(HiGHS.Optimizer)

# y[t] = 1 if DR event active at hour t
@variable(model, y[1:T], Bin)

# s[t] = 1 if DR event starts at hour t
@variable(model, s[1:T], Bin)

# ------------------------------------------------------------
# Event start logic
# ------------------------------------------------------------

# First hour
@constraint(model, s[1] >= y[1])

# Remaining hours
@constraint(model,
    [t in 2:T],
    s[t] >= y[t] - y[t-1]
)

# ------------------------------------------------------------
# At most one event per day
# ------------------------------------------------------------

@constraint(model,
    [d in days],
    sum(s[t] for t in hours_in_day[d]) <= 1
)

# ------------------------------------------------------------
# Maximum 6 event hours per day
# ------------------------------------------------------------

@constraint(model,
    [d in days],
    sum(y[t] for t in hours_in_day[d]) <= 6
)

# ------------------------------------------------------------
# Maximum 10 events per month
# ------------------------------------------------------------

@constraint(model,
    [m in months],
    sum(s[t] for t in hours_in_month[m]) <= 10
)

# ------------------------------------------------------------
# Maximum 180 event hours per year
# ------------------------------------------------------------

@constraint(model,
    sum(y[t] for t in 1:T) <= 180
)

# ------------------------------------------------------------
# Example objective
# ------------------------------------------------------------

# Dummy hourly values
value = rand(T)

@objective(model, Max,
    sum(value[t] * y[t] for t in 1:T)
)

optimize!(model)

# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

println("Total event hours = ", value.(y) |> sum)
println("Total events = ", value.(s) |> sum)