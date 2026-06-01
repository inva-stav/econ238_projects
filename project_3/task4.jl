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

const EC_Wind  = 2.0
const EC_Solar = 2.0

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

    @constraint(m, [i in 1:n], s[1,i] >= y[1,i])
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
    qdr_d = [value(Qdr_disp[t,i]) for t in 1:T, i in 1:n]
    y_vals = [round(Int, value(y[t,i])) for t in 1:T, i in 1:n]
    s_vals = [round(Int, value(s[t,i])) for t in 1:T, i in 1:n]

    profit_vec = compute_profit_vec(Psell, qs, qw, qsol, qdr_c, qdr_d;
                                    PWind=PWind, PSolar=PSolar, Pdr=Pdr,
                                    scenario_indices=scenario_indices)
    EΠ       = mean(profit_vec)
    CVaR_val = cvar(profit_vec, α)
    ρval     = λ * CVaR_val + (1-λ) * EΠ

    return (Qsell=qs, QWind=qw, QSolar=qsol, Qdr_contracted=qdr_c, Qdr_disp=qdr_d,
            EΠ=EΠ, CVaR=CVaR_val, ρ=ρval,
            y_vals=y_vals, s_vals=s_vals, profit_vec=profit_vec)
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Sweep helpers
# ─────────────────────────────────────────────────────────────────────────────

function run_psell_sweep(; Psell_grid=20.0:10.0:120.0, λ=0.5, α=0.05,
                           Qsell_max=10.0, Qdr_contracted_max=5.0)
    configs = [
        ("Wind+Solar", (wind=true,  solar=true)),
        ("Wind only",  (wind=true,  solar=false)),
        ("Solar only", (wind=false, solar=true)),
    ]
    results = Dict(name => NamedTuple[] for (name, _) in configs)
    total   = length(Psell_grid) * length(configs)
    count   = 0
    for (name, act) in configs
        for Ps in Psell_grid
            r = solve_contracting_lp(Float64(Ps); λ=λ, α=α,
                                      Qsell_max=Qsell_max,
                                      Qdr_contracted_max=Qdr_contracted_max,
                                      active=act)
            push!(results[name], r)
            count += 1
            println("  Psell sweep $count/$total  ($(name), Ps=\$$(Int(Ps)))")
        end
    end
    return collect(Psell_grid), results
end

function run_lambda_sweep(Psell; λ_grid=0.1:0.1:1.0, α=0.05,
                           Qsell_max=10.0, Qdr_contracted_max=5.0)
    rr_results = NamedTuple[]
    for (k, λv) in enumerate(λ_grid)
        r = solve_contracting_lp(Float64(Psell); λ=Float64(λv), α=α,
                                  Qsell_max=Qsell_max,
                                  Qdr_contracted_max=Qdr_contracted_max)
        push!(rr_results, r)
        println("  λ sweep $(k)/$(length(λ_grid))  (λ=$(λv))")
    end
    return collect(λ_grid), rr_results
end

# ─────────────────────────────────────────────────────────────────────────────
# 6. Plot functions
# ─────────────────────────────────────────────────────────────────────────────

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
    colors = Dict("Wind+Solar"=>:red, "Wind only"=>:blue, "Solar only"=>:darkorange)
    mean_price = mean(vec(π_ERCOT))

    plt = plot(xlabel="P_sell (\$/MWh)", ylabel="Quantity (MW)",
               title="Willingness-to-supply  (λ=0.5, α=0.05, N=40 scenarios)",
               legend=:outertopright, size=(950,500),
               left_margin=8Plots.mm, bottom_margin=5Plots.mm)

    for name in ["Wind+Solar", "Wind only", "Solar only"]
        Qsell_vals = [r.Qsell         for r in sweep_results[name]]
        QEC_vals   = [r.QWind+r.QSolar for r in sweep_results[name]]
        plot!(plt, Ps_vals, Qsell_vals; label="Qsell — $name",
              color=colors[name], linewidth=2)
        plot!(plt, Ps_vals, QEC_vals;   label="QWind+QSolar — $name",
              color=colors[name], linewidth=2, linestyle=:dash)
    end

    vline!(plt, [mean_price];
           label=@sprintf("Mean price \$%.0f/MWh", mean_price),
           color=:black, linestyle=:dot, linewidth=1.5)

    savefig(plt, joinpath(RESULTS_DIR, "willingness_to_supply.png"))
    println("Saved willingness_to_supply.png")
    return plt
end

# ── Plot 3: Certainty equivalent & diversification benefit ────────────────────

function plot_certainty_equivalent(Ps_vals, sweep_results)
    ρ_joint = [r.ρ for r in sweep_results["Wind+Solar"]]
    ρ_wind  = [r.ρ for r in sweep_results["Wind only"]]
    ρ_solar = [r.ρ for r in sweep_results["Solar only"]]
    ρ_sum   = ρ_wind .+ ρ_solar
    div_ben = ρ_joint .- ρ_sum

    plt_ce = plot(xlabel="P_sell (\$/MWh)", ylabel="Certainty equivalent ρ (\$)",
                  title="Certainty equivalent  (λ=0.5)",
                  legend=:outertopright, size=(800,450),
                  left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    plot!(plt_ce, Ps_vals, ρ_joint; label="ρ  Wind+Solar", color=:red,       linewidth=2)
    plot!(plt_ce, Ps_vals, ρ_sum;   label="ρ_Wind + ρ_Solar", color=:blue,   linewidth=2)
    plot!(plt_ce, Ps_vals, ρ_wind;  label="ρ  Wind only",  color=:steelblue, linewidth=1.5, linestyle=:dash)
    plot!(plt_ce, Ps_vals, ρ_solar; label="ρ  Solar only", color=:darkorange,linewidth=1.5, linestyle=:dash)

    plt_div = plot(Ps_vals, div_ben;
                   xlabel="P_sell (\$/MWh)", ylabel="Diversification benefit (\$)",
                   title="ρ_joint − (ρ_Wind + ρ_Solar)",
                   label="Benefit", color=:purple, linewidth=2,
                   fill=0, fillalpha=0.2, legend=:topleft,
                   left_margin=8Plots.mm, bottom_margin=5Plots.mm)
    hline!(plt_div, [0.0]; color=:black, linestyle=:dot, label="")

    plt = plot(plt_ce, plt_div; layout=(1,2), size=(1400,450),
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
    hod_labels = [string(h-1) for h in 1:HOURS_PER_DAY]  # 0–23 UTC

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

Psell              = 70.0
λ                  = 0.5
α                  = 0.05
Qsell_max          = 10.0   # 10
Qdr_contracted_max = 5.0   # 5
PWind              = 0.0    # 0
PSolar             = 0.0    # 0
Pdr                = 0.0    # 0

# ── Single-scenario smoke test ───────────────────────────────────────────────
println("\nRunning single-scenario test (scenario 1: $(SCENARIO_LABELS[1]))...")
t0 = time()
result_test = solve_contracting_lp(Psell; λ=λ, α=α,
    Qsell_max=Qsell_max, Qdr_contracted_max=Qdr_contracted_max, Pdr=Pdr,
    scenario_indices=1:1)
elapsed = time() - t0
println("Done in $(round(elapsed; digits=1))s")
@printf("  Qsell=%.2f MW  QWind=%.2f MW  QSolar=%.2f MW  Qdr_contracted=%.2f MW\n",
        result_test.Qsell, result_test.QWind, result_test.QSolar, result_test.Qdr_contracted)
@printf("  E[Π]=\$%.2f  CVaR=\$%.2f  ρ=\$%.2f\n",
        result_test.EΠ, result_test.CVaR, result_test.ρ)




# # ── Full 40-scenario run ─────────────────────────────────────────────────────
# println("\nRunning full 40-scenario optimization...")
# t1 = time()
# result_full = solve_contracting_lp(Psell; λ=λ, α=α,
#     Qsell_max=Qsell_max, Qdr_contracted_max=Qdr_contracted_max, Pdr=Pdr)
# println("Done in $(round(time()-t1; digits=1))s")
# @printf("  Qsell=%.2f MW  QWind=%.2f MW  QSolar=%.2f MW  Qdr_contracted=%.2f MW\n",
#         result_full.Qsell, result_full.QWind, result_full.QSolar, result_full.Qdr_contracted)
# @printf("  E[Π]=\$%.2f  CVaR=\$%.2f  ρ=\$%.2f\n",
#         result_full.EΠ, result_full.CVaR, result_full.ρ)

# # ── Plot 4: Seasonality (fast — no optimization) ─────────────────────────────
# println("\nGenerating seasonality plot...")
# plot_seasonality()

# # ── Plot 5: DR dispatch (fast — uses result_full) ────────────────────────────
# println("\nGenerating DR dispatch plots...")
# plot_dr_dispatch(result_full)




# # ── Psell sweep → Plots 1 & 3 (~10 min for 11 pts × 3 configs × 18s each) ───
# println("\nRunning Psell sweep (≈10 min)...")
# t2 = time()
# Ps_vals, sweep_results = run_psell_sweep(; Psell_grid=20.0:10.0:120.0, λ=λ, α=α,
#                                            Qsell_max=Qsell_max,
#                                            Qdr_contracted_max=Qdr_contracted_max,
#                                            Pdr=Pdr)
# println("Psell sweep done in $(round(time()-t2; digits=0))s")

# println("\nGenerating willingness-to-supply plot...")
# plot_willingness_to_supply(Ps_vals, sweep_results)

# println("Generating certainty equivalent plot...")
# plot_certainty_equivalent(Ps_vals, sweep_results)

# # ── λ sweep → Plot 2 (~3 min for 10 pts × 18s each) ─────────────────────────
# println("\nRunning λ sweep (≈3 min)...")
# t3 = time()
# λ_vals, rr_results = run_lambda_sweep(Psell; λ_grid=0.1:0.1:1.0, α=α,
#                                        Qsell_max=Qsell_max,
#                                        Qdr_contracted_max=Qdr_contracted_max
#                                        Pdr=Pdr)
# println("λ sweep done in $(round(time()-t3; digits=0))s")

# println("\nGenerating risk-return frontier plot...")
# plot_risk_return_frontier(λ_vals, rr_results)

# ── Qdr_contracted_max sweep  ─────────────────────────────────────────────────
function sweep_Qdr_contracted_max(; Qdr_grid=range(0, stop=10, length=11), Psell=70.0, λ=0.5, α=0.05)
    results = []
    total_points = length(Qdr_grid)

    for (i, Qdr_contracted_max) in enumerate(Qdr_grid)
        println("Running Qdr_contracted_max = ", Qdr_contracted_max, " (", i, "/", total_points, ")")
        result = solve_contracting_lp(Psell; λ=λ, α=α, Qsell_max=10.0, Qdr_contracted_max=Qdr_contracted_max, Pdr=0.0)

        # Collect metrics
        total_s_events = sum(result.s_vals)
        total_y_hours = sum(result.y_vals)
        total_Qdisp = sum(result.Qdr_disp)

        push!(results, (
            Qdr_contracted_max,
            result.EΠ, result.CVaR, result.ρ,
            result.Qsell, result.QWind, result.QSolar, result.Qdr_contracted,
            total_s_events, total_y_hours, total_Qdisp
        ))
    end

    # Save results to CSV
    csv_path = joinpath(RESULTS_DIR, "sweep_Qdr_contracted_max.csv")
    open(csv_path, "w") do io
        println(io, "Qdr_contracted_max,Profit,CVaR,Certainty_Equivalent,Qsell,QWind,QSolar,Qdr_contracted,DR_Events,DR_Hours,DR_Energy")
        for r in results
            println(io, join(r, ","))
        end
    end

    println("Saved sweep_Qdr_contracted_max.csv")
    return results
end

# Add call to the sweep function
println("\nRunning Qdr_contracted_max sweep ...")
sweep_Qdr_contracted_max()

# ── Plot results from Qdr_contracted_max sweep ──────────────────────────────
function plot_qdr_contracted_max_results()
    # Read the CSV file
    csv_path = joinpath(RESULTS_DIR, "sweep_Qdr_contracted_max.csv")
    data = CSV.read(csv_path, DataFrame)

    # Extract data
    x = data.Qdr_contracted_max
    profit = data.Profit
    cvar = data.CVaR
    ce = data.Certainty_Equivalent
    qsell = data.Qsell
    qwind = data.QWind
    qsolar = data.QSolar
    qdr_contracted = data.Qdr_contracted
    dr_events = data.DR_Events
    dr_hours = data.DR_Hours
    dr_energy = data.DR_Energy

    # First plot: Profit, CVaR, Certainty Equivalent, and quantities
    plt1 = plot(x, profit; label="Profit", color=:blue, linewidth=2, xlabel="Qdr_contracted_max", ylabel="Profit / CVaR / Certainty Eq.")
    plot!(plt1, x, cvar; label="CVaR", color=:red, linewidth=2)
    plot!(plt1, x, ce; label="Certainty Eq.", color=:green, linewidth=2, linestyle=:dash)

    twin = twinx()
    plot!(twin, x, qsell; label="Qsell", color=:purple, linewidth=2)
    plot!(twin, x, qwind; label="QWind", color=:orange, linewidth=2)
    plot!(twin, x, qsolar; label="QSolar", color=:cyan, linewidth=2)
    plot!(twin, x, qdr_contracted; label="Qdr_contracted", color=:magenta, linewidth=2)
    ylabel!(twin, "Quantities")
    #legend(:topright)
    plot!(plt1, legend=:topright)
    savefig(plt1, joinpath(RESULTS_DIR, "qdr_contracted_max_plot1.png"))
    println("Saved qdr_contracted_max_plot1.png")

    # Second plot: DR Events, DR Hours, DR Energy
    plt2 = plot(layout=(1,3), size=(1200,400))
    bar!(plt2[1], x, dr_events; xlabel="Qdr_contracted_max", ylabel="DR Events", color=:blue, legend=false)
    bar!(plt2[2], x, dr_hours; xlabel="Qdr_contracted_max", ylabel="DR Hours", color=:green, legend=false)
    bar!(plt2[3], x, dr_energy; xlabel="Qdr_contracted_max", ylabel="DR Energy", color=:red, legend=false)
    savefig(plt2, joinpath(RESULTS_DIR, "qdr_contracted_max_plot2.png"))
    println("Saved qdr_contracted_max_plot2.png")
end

function plot_qdr_contracted_max_results2()

    # Read the CSV file
    csv_path = joinpath(RESULTS_DIR, "sweep_Qdr_contracted_max.csv")
    data = CSV.read(csv_path, DataFrame)

    # Extract data
    x = data.Qdr_contracted_max

    profit = data.Profit
    cvar = data.CVaR
    ce = data.Certainty_Equivalent

    qsell = data.Qsell
    qwind = data.QWind
    qsolar = data.QSolar
    qdr_contracted = data.Qdr_contracted

    dr_events = data.DR_Events
    dr_hours = data.DR_Hours
    dr_energy = data.DR_Energy

    # ─────────────────────────────────────────────────────────────
    # First Plot
    # ─────────────────────────────────────────────────────────────

    plt1 = plot(
        x, profit;
        label = "Profit",
        color = :blue,
        linewidth = 2,
        linestyle = :dash,
        xlabel = "Qdr_contracted_max",
        ylabel = "Profit / CVaR / Certainty Eq.",
        legend = :topleft,
        size = (1000,600),
        margin = 10Plots.mm
    )

    plot!(
        plt1,
        x, cvar;
        label = "CVaR",
        color = :red,
        linewidth = 2,
        linestyle = :dash
    )

    plot!(
        plt1,
        x, ce;
        label = "Certainty Eq.",
        color = :green,
        linewidth = 2,
        linestyle = :dash
    )

    # Twin axis for quantities
    twin = twinx()

    plot!(
        twin,
        x, qsell;
        label = "Qsell",
        color = :purple,
        linewidth = 2
    )

    plot!(
        twin,
        x, qwind;
        label = "QWind",
        color = :orange,
        linewidth = 2
    )

    plot!(
        twin,
        x, qsolar;
        label = "QSolar",
        color = :cyan,
        linewidth = 2
    )

    plot!(
        twin,
        x, qdr_contracted;
        label = "Qdr_contracted",
        color = :magenta,
        linewidth = 2,
        ylabel = "Quantities",
        legend = :topright
    )

    savefig(
        plt1,
        joinpath(RESULTS_DIR, "qdr_contracted_max_plot1.png")
    )

    println("Saved qdr_contracted_max_plot1.png")

    # ─────────────────────────────────────────────────────────────
    # Second Plot
    # ─────────────────────────────────────────────────────────────

    plt2 = plot(
        layout = (1,3),
        size = (1400,500),
        margin = 10Plots.mm
    )

    bar!(
        plt2[1],
        x,
        dr_events;
        xlabel = "Qdr_contracted_max",
        ylabel = "DR Events",
        color = :blue,
        legend = false
    )

    bar!(
        plt2[2],
        x,
        dr_hours;
        xlabel = "Qdr_contracted_max",
        ylabel = "DR Hours",
        color = :green,
        legend = false
    )

    bar!(
        plt2[3],
        x,
        dr_energy;
        xlabel = "Qdr_contracted_max",
        ylabel = "DR Energy",
        color = :red,
        legend = false
    )

    savefig(
        plt2,
        joinpath(RESULTS_DIR, "qdr_contracted_max_plot2.png")
    )

    println("Saved qdr_contracted_max_plot2.png")
end

# Call the plotting function
println("\nGenerating plots for Qdr_contracted_max sweep results...")
plot_qdr_contracted_max_results2()

println("\n═══ Task 4 complete — plots saved to $(RESULTS_DIR) ═══")
