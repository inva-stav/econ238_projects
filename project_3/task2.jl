# Task 2 — Wind-only tolling on the case-study data
#
# The script
#   (a) sweeps Q^sell over [0, 15] avgMW, computes E[Π], CVaR₀.₀₅[Π] and
#       ρ_{α,λ}[Π] = λ CVaR + (1-λ) E for λ ∈ {0.5, 0.99}, and locates the
#       maximizer of ρ on the grid;
#   (b) plots the 2000 NE spot-price time-series in gray and overlays the
#       100 worst-profit scenarios at Q^sell = 0 (blue) and Q^sell = 15 (red),
#       together with a summary table of the three tail populations.

import Pkg
let
    need = ["JuMP", "HiGHS", "Plots", "CSV", "DataFrames", "Statistics", "Printf"]
    have = [string(k.name) for k in values(Pkg.dependencies())]
    miss = setdiff(need, have)
    for p in miss; Pkg.add(p); end
end

using JuMP, HiGHS
ENV["GKSwstype"] = "nul"
using Plots
using CSV, DataFrames
using Statistics
using Printf

const DATA_DIR = joinpath(@__DIR__, "data")
const OUT_DIR  = joinpath(@__DIR__, "task2_out")
mkpath(OUT_DIR)

# ── Data loading ────────────────────────────────────────────────────────────
"Load a (12 × 2000) scenario block from a CSV produced by export_data.py."
function load_block(filename::AbstractString)
    df = CSV.read(joinpath(DATA_DIR, filename), DataFrame)
    # Drop the leading 'month' column; the rest are scenario columns 1..2000.
    M = Matrix{Float64}(df[:, 2:end])
    @assert size(M) == (12, 2000) "Unexpected size for $filename: $(size(M))"
    return M
end

const GSF_WIND  = load_block("wind_gsf.csv")
const PI_NE     = load_block("ne_price.csv")
const HOURS_DF  = CSV.read(joinpath(DATA_DIR, "hours.csv"), DataFrame)
const HOURS     = Float64.(HOURS_DF.hours)
@assert sum(HOURS) == 8760

# pull in the cvar function from task1.jl
include("task1.jl")

# Annual profit vector for wind-only tolling
"""
    annual_profit_wind_tolling(Qsell; Psell, Pwind, Qwind)

Return the 2000-vector of annual profits Π_ω(Q^sell) under the slide-40
configuration: only wind, in tolling. Per-month profit follows the workbook
equation

    Π_{t,ω} = H_t · [ (P^sell − π^NE_{t,ω}) Q^sell
                      + π^NE_{t,ω} GSF^Wind_{t,ω} Q^Wind − P^Wind Q^Wind ].
"""
function annual_profit_wind_tolling(Qsell::Real;
        Psell::Real = 140.0,
        Pwind::Real = 100.0,
        Qwind::Real = 11.41)
    # Per-month profit matrix (12 × 2000)
    gen     = GSF_WIND .* Qwind        # avgMW realized
    spotleg = PI_NE .* (gen .- Qsell)  # π^NE · (G − Q^sell)
    detleg  = Psell * Qsell - Pwind * Qwind  # scalar
    per_month = (detleg .+ spotleg) .* HOURS
    return vec(sum(per_month, dims = 1))   # 2000-vector
end

# ── (a) Q^sell sweep ────────────────────────────────────────────────────────
function sweep_Qsell(λ::Real;
                     grid = collect(0.0:0.01:15.0),
                     alpha = 0.05,
                     verbose = true)
    n = length(grid)
    E   = Vector{Float64}(undef, n)
    CV  = Vector{Float64}(undef, n)
    ρ   = Vector{Float64}(undef, n)
    for (i, q) in pairs(grid)
        Π   = annual_profit_wind_tolling(q)
        E[i]  = mean(Π)
        CV[i] = cvar(Π, alpha)
        ρ[i]  = λ * CV[i] + (1 - λ) * E[i]
        if verbose && (i == 1 || i == n || i % 200 == 0)
            @printf("  λ=%.2f  Q=%5.2f   E=%12.2f   CVaR=%12.2f   ρ=%12.2f\n",
                    λ, q, E[i], CV[i], ρ[i])
        end
    end
    return (; grid, E, CV, ρ)
end

function plot_sweep(s, λ; filename)
    p = plot(s.grid, s.E ./ 1e6,
             label = "E[Π]",
             lw = 2,
             xlabel = "Q^sell  (avgMW)",
             ylabel = "Annual profit  (\$M)",
             title  = "Wind-only tolling — λ = $(λ)",
             legend = :bottomleft,
             size = (820, 480))
    plot!(p, s.grid, s.CV ./ 1e6, label = "CVaR₀.₀₅[Π]", lw = 2)
    plot!(p, s.grid, s.ρ  ./ 1e6, label = "ρ_{α,λ}[Π]",  lw = 2, linestyle = :dash)
    i★ = argmax(s.ρ)
    Q★ = s.grid[i★]
    vline!(p, [Q★], label = "Q^sell* = $(round(Q★, digits = 2))",
           lw = 1, linestyle = :dot, color = :black)
    savefig(p, filename)
    Plots.closeall()
    return (Q★ = Q★, E★ = s.E[i★], CV★ = s.CV[i★], ρ★ = s.ρ[i★])
end

println("\n── Task 2(a): Q^sell sweep ─────────────────────────────────")
println("\nλ = 0.5 (slide 40):")
sweep_05 = sweep_Qsell(0.5)
res_05   = plot_sweep(sweep_05, 0.5; filename = joinpath(OUT_DIR, "task2a_lambda050.png"))
@printf("  → Q^sell* = %.2f avgMW   E = %.2f   CVaR = %.2f   ρ = %.2f\n",
        res_05.Q★, res_05.E★, res_05.CV★, res_05.ρ★)
@printf("    (slide 40 reference: Q^sell* ≈ 9.88)\n")

println("\nλ = 0.99 (slide 41):")
sweep_99 = sweep_Qsell(0.99)
res_99   = plot_sweep(sweep_99, 0.99; filename = joinpath(OUT_DIR, "task2a_lambda099.png"))
@printf("  → Q^sell* = %.2f avgMW   E = %.2f   CVaR = %.2f   ρ = %.2f\n",
        res_99.Q★, res_99.E★, res_99.CV★, res_99.ρ★)
@printf("    (slide 41 reference: Q^sell* ≈ 9.12)\n")

# Persist the swept curves for the report.
CSV.write(joinpath(OUT_DIR, "task2a_sweep_lambda050.csv"),
          DataFrame(Qsell = sweep_05.grid, E = sweep_05.E,
                    CVaR = sweep_05.CV, rho = sweep_05.ρ))
CSV.write(joinpath(OUT_DIR, "task2a_sweep_lambda099.csv"),
          DataFrame(Qsell = sweep_99.grid, E = sweep_99.E,
                    CVaR = sweep_99.CV, rho = sweep_99.ρ))

# ── (b) Which spot-price scenarios drive the tail? ──────────────────────────
println("\n── Task 2(b): tail scenarios at Q^sell = 0 and Q^sell = 15 ─")

"Indices of the αN worst-profit scenarios (the CVaR composition)."
function tail_indices(profit::AbstractVector{<:Real}, alpha::Real)
    N = length(profit)
    k = Int(round(alpha * N))
    return partialsortperm(profit, 1:k)   # ascending by profit
end

Π0  = annual_profit_wind_tolling(0.0)
Π15 = annual_profit_wind_tolling(15.0)
tail_blue = tail_indices(Π0,  0.05)    # 100 worst at Q^sell = 0
tail_red  = tail_indices(Π15, 0.05)    # 100 worst at Q^sell = 15

# Sanity-check the LP function once: mean of the 100 worst profits at Q^sell=0
# must equal the LP value (αN integer ⇒ LP = mean of the αN worst).
let lp = cvar(Π0, 0.05), mn = mean(Π0[tail_blue])
    @assert isapprox(lp, mn; rtol = 1e-6) "CVaR LP and mean-of-tail disagree: $lp vs $mn"
    @printf("  Sanity check: CVaR LP = %.2f  ≈  mean of 100 worst = %.2f\n", lp, mn)
end

# Time-series overlay plot.
t = 1:12
p = plot(size = (900, 540),
         xlabel = "Month (t)",
         ylabel = "π^NE  (\$/MWh)",
         title  = "NE spot-price scenarios — tails of CVaR₀.₀₅ for Q^sell ∈ {0, 15}",
         xticks = (1:12, ["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"]),
         legend = :topleft)
# 2000 gray paths
for ω in 1:2000
    plot!(p, t, PI_NE[:, ω], label = false, color = :gray80, lw = 0.4, alpha = 0.5)
end
# Blue overlay: tail at Q^sell = 0 (long the spot)
for (j, ω) in pairs(tail_blue)
    plot!(p, t, PI_NE[:, ω],
          label = j == 1 ? "tail @ Q^sell = 0   (long)"  : false,
          color = :royalblue, lw = 1.0, alpha = 0.7)
end
# Red overlay: tail at Q^sell = 15 (short the spot)
for (j, ω) in pairs(tail_red)
    plot!(p, t, PI_NE[:, ω],
          label = j == 1 ? "tail @ Q^sell = 15  (short)" : false,
          color = :firebrick, lw = 1.0, alpha = 0.7)
end
savefig(p, joinpath(OUT_DIR, "task2b_price_overlay.png"))
Plots.closeall()

# Summary table: annual-average π^NE, 5% & 95% quantiles, mean annual GSF^Wind.
function tail_summary(idx_set, name)
    # annual-average π^NE for each ω in idx_set: 12-month mean
    π_ann = vec(mean(PI_NE[:, idx_set], dims = 1))   # length = |idx_set|
    g_ann = vec(mean(GSF_WIND[:, idx_set], dims = 1))
    return (
        population = name,
        n          = length(idx_set),
        mean_πNE   = mean(π_ann),
        q05_πNE    = quantile(π_ann, 0.05),
        q95_πNE    = quantile(π_ann, 0.95),
        mean_GSF   = mean(g_ann),
    )
end

summary = DataFrame([
    tail_summary(tail_red,  "red (tail @ Q^sell = 15)"),
    tail_summary(tail_blue, "blue (tail @ Q^sell = 0)"),
    tail_summary(1:2000,    "all 2000"),
])
CSV.write(joinpath(OUT_DIR, "task2b_tail_summary.csv"), summary)
println()
println(summary)

println("\nOutputs in $(OUT_DIR)/")
