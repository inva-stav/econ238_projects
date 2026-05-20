using JuMP
using HiGHS
using Statistics
using Plots
using CSV
using DataFrames
using Printf
using Dates

# =============================================================================
# Task 4 — 
#
# To-Do: Work in progress, not complete yet! Explain PG&E tariff this is based on, find a avg Pqr from that tariff, clean up time resolution for Henry's CA/TX data (H[t]), check function arguments, add plots (#events, #hours, sweeps, ...), add a run function.
#
# Contract format: Tolling (fixed retainer P_i * Q_i per period; TC receives
#   GSF_i * Q_i of generation and sells at spot). Demand reduction is modeled
#   through a dispatchable load reduction (Qdr_disp) incentivized by a fixed
#   monthly payment (Pdr). The optimization accounts for tolling contracts,
#   demand reduction incentives, and portfolio diversification effects.
#
# Rationale: Tolling transfers the full price-and-quantity risk to the TC,
#   making the portfolio diversification effect most visible. Demand reduction
#   adds flexibility and cost savings by reducing contracted load during high
#   price periods.
# =============================================================================

const DATA_DIR = joinpath(@__DIR__, "data")
const RESULTS_DIR = joinpath(@__DIR__, "results", "task4")
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
"""
Build the annual profit vector for arbitrary P_sell (prices P_i fixed at 100).
Under tolling:  Π_{t,ω} = (P_sell - π^NE_{t,ω}) Q_sell
                         + Σ_i [ π^{n(i)}_{t,ω} GSF^i_{t,ω} Q_i  -  P_i Q_i ]
"""
function annual_profit_vec(Psell, Qsell, QWind, QSH, QBio, Qdr_disp, Qdr_contracted;
                           PWind=100.0, PSH=100.0, PBio=100.0, Pdr=20.0)
    Π = zeros(N)
    for ω in 1:N
        for t in 1:T
            π_t = (Psell - π_NE[t,ω]) * (Qsell - Qdr_disp[t,ω]) +
                  (π_NE[t,ω] * GSF_Wind[t,ω] - PWind) * QWind +
                  (π_SE[t,ω] * GSF_SH[t,ω]   - PSH)   * QSH +
                  (π_SE[t,ω] * GSF_Bio[t,ω]   - PBio)  * QBio
            Π[ω] += Ht[t] * π_t
        end
        # Subtract the cost of demand reduction incentive
        Π[ω] -= Qdr_contracted * Pdr * 12
    end
    return Π
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Contracting LP  (model 27, tolling format)
# ─────────────────────────────────────────────────────────────────────────────
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
                               Qdr_contracted_max::Float64=100.0,
                               active=(wind=true, sh=true, bio=true),
                               PWind=100.0, PSH=100.0, PBio=100.0, Pdr=20.0)

    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, 0 <= Qsell <= Qsell_max)
    @variable(m, 0 <= QWind <= (active.wind ? EC_Wind : 0.0))
    @variable(m, 0 <= QSH   <= (active.sh   ? EC_SH   : 0.0))
    @variable(m, 0 <= QBio  <= (active.bio  ? EC_Bio  : 0.0))
    @variable(m, 0 <= Qdr_contracted <= Qdr_contracted_max) # should be smaller than Qsell
    @variable(m, Qdr_disp[1:T,1:N] >= 0)
    @variable(m, z)
    @variable(m, δ[1:N] >= 0)

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

    # y[t, ω] = 1 if DR event active at hour t
    @variable(m, y[1:T,1:N], Bin)

    # s[t, ω] = 1 if DR event starts at hour t
    @variable(m, s[1:T,1:N], Bin)

    # ------------------------------------------------------------
    # Event start logic
    # ------------------------------------------------------------

    # First hour
    @constraint(m, [ω in 1:N], s[1,ω] >= y[1,ω])

    # Remaining hours
    @constraint(m,
        [t in 2:T, ω in 1:N],
        s[t,ω] >= y[t,ω] - y[t-1,ω]
    )

    # ------------------------------------------------------------
    # At most one event per day
    # ------------------------------------------------------------

    @constraint(m,
        [d in days, ω in 1:N],
        sum(s[t,ω] for t in hours_in_day[d]) <= 1
    )

    # ------------------------------------------------------------
    # Maximum 6 event hours per day
    # ------------------------------------------------------------

    @constraint(m,
        [d in days, ω in 1:N],
        sum(y[t,ω] for t in hours_in_day[d]) <= 6
    )

    # ------------------------------------------------------------
    # Maximum 10 events per month
    # ------------------------------------------------------------

    @constraint(m,
        [m in months, ω in 1:N],
        sum(s[t,ω] for t in hours_in_month[m]) <= 10
    )

    # ------------------------------------------------------------
    # Maximum 180 event hours per year
    # ------------------------------------------------------------

    @constraint(m, [ω in 1:N],
        sum(y[t,ω] for t in 1:T) <= 180
    )

    # Π_ω as an affine expression of the decision variables
    @expression(m, Π[ω=1:N],
        sum(Ht[t] * (Psell - π_NE[t,ω]) * (Qsell - Qdr_disp[t,ω]) for t in 1:T)
        + sum(Ht[t] * (π_NE[t,ω] * GSF_Wind[t,ω] - PWind) for t in 1:T) * QWind 
        + sum(Ht[t] * (π_SE[t,ω] * GSF_SH[t,ω]   - PSH) for t in 1:T) * QSH 
        + sum(Ht[t] * (π_SE[t,ω] * GSF_Bio[t,ω]   - PBio) for t in 1:T) * QBio
        - Qdr_contracted * Pdr * 12) # Pdr is monthly incentive for DR, multiplied by Potential Load Reduction (PLR) = AverageDemand - FirmServiceLoad (FSL) = Qdr_contracted

    @constraint(m, [t in 1:T, ω in 1:N], Qdr_disp[t,ω] <= Qdr_contracted * y[t,ω])

    @constraint(m, [ω=1:N], δ[ω] >= z - Π[ω])

    @objective(m, Max,
        λ * (z - (1 / (α * N)) * sum(δ)) +
        (1 - λ) * (1 / N) * sum(Π))

    optimize!(m)

    qs  = value(Qsell)
    qw  = value(QWind)
    qsh = value(QSH)
    qb  = value(QBio)
    qdr_contracted = value(Qdr_contracted)
    qdr_dispatched = [value(Qdr_disp[t,ω]) for t in 1:T, ω in 1:N]

    profit_vec = annual_profit_vec(Psell, qs, qw, qsh, qb, qdr_contracted, qdr_dispatched;
                                   PWind=PWind, PSH=PSH, PBio=PBio)
    EΠ   = mean(profit_vec)
    CVaR = cvar(profit_vec, α)
    ρval = λ * CVaR + (1 - λ) * EΠ

    return (Qsell=qs, QWind=qw, QSH=qsh, QBio=qb, Qdr=qdr, EΠ=EΠ, CVaR=CVaR, ρ=ρval)
end

# println("Total event hours = ", value.(y) |> sum)
# println("Total events = ", value.(s) |> sum)

# # ─────────────────────────────────────────────────────────────────────────────
# # 5. Task 3(a): Willingness-to-supply plot
# # ─────────────────────────────────────────────────────────────────────────────

# function run_task3a(; Psell_grid=0.0:5.0:250.0, λ=0.5, α=0.05, Qsell_max=100.0)
#     configs = [
#         ("Wind only", (wind=true,  sh=false, bio=false)),
#         ("SH only",   (wind=false, sh=true,  bio=false)),
#         ("Bio only",  (wind=false, sh=false, bio=true)),
#         ("Portfolio",  (wind=true,  sh=true,  bio=true)),
#     ]

#     results = Dict{String, Vector{NamedTuple}}()
#     for (name, act) in configs
#         results[name] = NamedTuple[]
#     end

#     total = length(Psell_grid) * length(configs)
#     count = 0
#     for Ps in Psell_grid
#         for (name, act) in configs
#             r = solve_contracting_lp(Float64(Ps); λ=λ, α=α,
#                                      Qsell_max=Qsell_max, active=act)
#             push!(results[name], r)
#             count += 1
#             if count % 40 == 0
#                 println("  Progress: $count / $total")
#             end
#         end
#     end

#     return Psell_grid, results
# end

# println("\n─── Running Task 3(a): Willingness-to-supply sweep ───")
# Psell_grid, results = run_task3a()

# # --- Plot (a): Qsell* and total EC purchased vs P_sell ---
# colors = Dict("Wind only"=>:blue, "SH only"=>:green, "Bio only"=>:orange, "Portfolio"=>:red)
# Ps = collect(Psell_grid)

# plt_a = plot(xlabel="P_sell (\$/MWh)", ylabel="Quantity (avgMW)",
#              title="Willingness-to-supply (tolling, λ=0.5, α=0.05)",
#              legend=:outertopright, size=(900,500),
#              left_margin=8Plots.mm, bottom_margin=5Plots.mm)

# for name in ["Wind only", "SH only", "Bio only", "Portfolio"]
#     Qsell_vals = [r.Qsell for r in results[name]]
#     Qtotal_vals = [r.QWind + r.QSH + r.QBio for r in results[name]]
#     plot!(plt_a, Ps, Qsell_vals; label="Qsell — $name",
#           color=colors[name], linewidth=2)
#     plot!(plt_a, Ps, Qtotal_vals; label="ΣQi — $name",
#           color=colors[name], linewidth=2, linestyle=:dash)
# end

# savefig(plt_a, joinpath(RESULTS_DIR, "willingness_to_supply.png"))
# println("Saved willingness_to_supply.png")

# # ─────────────────────────────────────────────────────────────────────────────
# # 6. Task 3(b): Certainty equivalent — portfolio vs sum of individuals
# # ─────────────────────────────────────────────────────────────────────────────

# println("\n─── Running Task 3(b): Certainty equivalent comparison ───")

# plt_b = plot(xlabel="P_sell (\$/MWh)", ylabel="Value (\$)",
#              title="Certainty equivalent & risk premium: portfolio vs. individuals",
#              legend=:outertopright, size=(1000,550),
#              left_margin=10Plots.mm, bottom_margin=5Plots.mm)

# ρ_portfolio = [r.ρ for r in results["Portfolio"]]
# ρ_sum_indiv = [results["Wind only"][i].ρ + results["SH only"][i].ρ + results["Bio only"][i].ρ
#                for i in eachindex(Ps)]
# RP_portfolio = [r.EΠ - r.CVaR for r in results["Portfolio"]]
# RP_sum_indiv = [results["Wind only"][i].EΠ - results["Wind only"][i].CVaR +
#                 results["SH only"][i].EΠ   - results["SH only"][i].CVaR +
#                 results["Bio only"][i].EΠ   - results["Bio only"][i].CVaR
#                 for i in eachindex(Ps)]

# plot!(plt_b, Ps, ρ_portfolio;  label="ρ portfolio", color=:red, linewidth=2)
# plot!(plt_b, Ps, ρ_sum_indiv;  label="Σ ρ_i (individuals)", color=:blue, linewidth=2)
# plot!(plt_b, Ps, RP_portfolio; label="RP portfolio", color=:red, linewidth=2, linestyle=:dash)
# plot!(plt_b, Ps, RP_sum_indiv; label="Σ RP_i", color=:blue, linewidth=2, linestyle=:dash)

# savefig(plt_b, joinpath(RESULTS_DIR, "ce_portfolio_vs_individuals.png"))
# println("Saved ce_portfolio_vs_individuals.png")

# # --- Diversification benefit plot ---
# plt_div = plot(xlabel="P_sell (\$/MWh)", ylabel="Diversification benefit (\$)",
#                title="Portfolio diversification benefit (ρ_portfolio − Σ ρ_i)",
#                legend=:topleft, size=(800,400),
#                left_margin=8Plots.mm, bottom_margin=5Plots.mm)
# plot!(plt_div, Ps, ρ_portfolio .- ρ_sum_indiv;
#       label="ρ_portfolio − Σ ρ_i  (≥ 0 by superadditivity)",
#       color=:purple, linewidth=2, fill=0, fillalpha=0.2)
# savefig(plt_div, joinpath(RESULTS_DIR, "diversification_benefit.png"))
# println("Saved diversification_benefit.png")

# # ─────────────────────────────────────────────────────────────────────────────
# # 7. Risk–return frontier (P_sell = 140, sweep λ)
# # ─────────────────────────────────────────────────────────────────────────────

# println("\n─── Running risk-return frontier (P_sell=140, λ sweep) ───")

# λ_grid = 0.200:0.005:1.000
# frontier_E  = Float64[]
# frontier_RP = Float64[]
# frontier_λ  = Float64[]

# for λ_val in λ_grid
#     r = solve_contracting_lp(140.0; λ=Float64(λ_val), α=0.05,
#                              Qsell_max=100.0,
#                              active=(wind=true, sh=true, bio=true))
#     push!(frontier_E,  r.EΠ)
#     push!(frontier_RP, r.EΠ - r.CVaR)
#     push!(frontier_λ,  λ_val)
# end

# plt_rr = scatter(frontier_RP, frontier_E;
#                  xlabel="Risk  (E[Π] − CVaR₀.₀₅)",
#                  ylabel="Return  E[Π] (\$)",
#                  title="Risk-return frontier (P_sell=140, joint portfolio)",
#                  legend=:topright, size=(700,500),
#                  left_margin=8Plots.mm, bottom_margin=5Plots.mm,
#                  marker_z=frontier_λ, color=:viridis, markerstrokewidth=0,
#                  markersize=5, colorbar_title="λ",
#                  label="λ ∈ [0.20, 1.00]")

# savefig(plt_rr, joinpath(RESULTS_DIR, "risk_return_frontier.png"))
# println("Saved risk_return_frontier.png")

# # ─────────────────────────────────────────────────────────────────────────────
# # 8. Task 3(c): Snapshot at P_sell = 140 $/MWh
# # ─────────────────────────────────────────────────────────────────────────────

# println("\n─── Task 3(c): Snapshot at P_sell = 140 ───")

# configs_c = [
#     ("Wind only", (wind=true,  sh=false, bio=false)),
#     ("SH only",   (wind=false, sh=true,  bio=false)),
#     ("Bio only",  (wind=false, sh=false, bio=true)),
#     ("Portfolio",  (wind=true,  sh=true,  bio=true)),
# ]

# println("\n┌──────────────┬────────┬────────┬────────┬────────┬──────────────┬──────────────┬──────────────┐")
# println("│ Config       │ Qsell* │ QWind* │  QSH*  │  QBio* │    E[Π*]     │  CVaR₀.₀₅    │    ρ₀.₅      │")
# println("├──────────────┼────────┼────────┼────────┼────────┼──────────────┼──────────────┼──────────────┤")

# for (name, act) in configs_c
#     r = solve_contracting_lp(140.0; λ=0.5, α=0.05, Qsell_max=100.0, active=act)
#     @printf("│ %-12s │ %6.2f │ %6.2f │ %6.2f │ %6.2f │ %12.2f │ %12.2f │ %12.2f │\n",
#             name, r.Qsell, r.QWind, r.QSH, r.QBio, r.EΠ, r.CVaR, r.ρ)
# end

# println("└──────────────┴────────┴────────┴────────┴────────┴──────────────┴──────────────┴──────────────┘")

# # ─────────────────────────────────────────────────────────────────────────────
# # 9. Diagnostics: Seasonality & correlation analysis for SH inclusion
# # ─────────────────────────────────────────────────────────────────────────────

# println("\n─── Seasonality analysis (supporting Task 3c commentary) ───")

# month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# # Monthly quantiles of GSF_SH
# q05_sh  = [quantile(GSF_SH[t,:], 0.05) for t in 1:T]
# q50_sh  = [quantile(GSF_SH[t,:], 0.50) for t in 1:T]
# q95_sh  = [quantile(GSF_SH[t,:], 0.95) for t in 1:T]
# mean_sh = [mean(GSF_SH[t,:]) for t in 1:T]

# q05_wind  = [quantile(GSF_Wind[t,:], 0.05) for t in 1:T]
# q50_wind  = [quantile(GSF_Wind[t,:], 0.50) for t in 1:T]
# q95_wind  = [quantile(GSF_Wind[t,:], 0.95) for t in 1:T]

# plt_gsf = plot(1:T, q50_sh; label="SH median", color=:green, linewidth=2,
#                xlabel="Month", ylabel="GSF", xticks=(1:12, month_labels),
#                title="GSF seasonality: Small Hydro vs Wind",
#                legend=:topright, size=(800,450),
#                left_margin=8Plots.mm, bottom_margin=5Plots.mm)
# plot!(plt_gsf, 1:T, q50_wind; label="Wind median", color=:blue, linewidth=2)
# plot!(plt_gsf, 1:T, q05_sh; label="SH 5%-95%", color=:green, linewidth=1,
#       linestyle=:dash, fillrange=q95_sh, fillalpha=0.15)
# plot!(plt_gsf, 1:T, q05_wind; label="Wind 5%-95%", color=:blue, linewidth=1,
#       linestyle=:dash, fillrange=q95_wind, fillalpha=0.15)
# savefig(plt_gsf, joinpath(RESULTS_DIR, "gsf_seasonality.png"))

# # Per-scenario temporal correlation between spot price and GSF
# corr_se_sh = Float64[]
# corr_ne_wind = Float64[]
# for ω in 1:N
#     push!(corr_se_sh,   cor(π_SE[:, ω], GSF_SH[:, ω]))
#     push!(corr_ne_wind, cor(π_NE[:, ω], GSF_Wind[:, ω]))
# end

# plt_corr = histogram(corr_ne_wind; bins=50, alpha=0.5, label="Corr(π_NE, GSF_Wind)",
#                      color=:blue, normalize=:pdf,
#                      xlabel="Temporal correlation (across 12 months)",
#                      ylabel="Density",
#                      title="Per-scenario temporal correlation: spot price vs GSF",
#                      legend=:topleft, size=(800,400),
#                      left_margin=8Plots.mm, bottom_margin=5Plots.mm)
# histogram!(plt_corr, corr_se_sh; bins=50, alpha=0.5, label="Corr(π_SE, GSF_SH)",
#            color=:green, normalize=:pdf)
# savefig(plt_corr, joinpath(RESULTS_DIR, "temporal_correlation.png"))

# println("Mean Corr(π_NE, GSF_Wind) = ", round(mean(corr_ne_wind); digits=4))
# println("Mean Corr(π_SE, GSF_SH)   = ", round(mean(corr_se_sh); digits=4))

# println("\nSaved gsf_seasonality.png and temporal_correlation.png")
# println("\n═══ Task 3 complete ═══")
