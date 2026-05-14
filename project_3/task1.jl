using JuMP
using HiGHS
using Statistics
using Plots

# -----------------------------------------------------------------------------
# (a) CVaR function — Rockafellar–Uryasev LP 
# -----------------------------------------------------------------------------

function cvar(profit::AbstractVector{<:Real}, alpha::Real)
    # TODO: build JuMP model with HiGHS optimizer
    model = Model(HiGHS.Optimizer)
    # TODO: declare variables z (free) and δ[1:N] (≥ 0)
    @variable(model, z)
    @variable(model, delta[1:length(profit)] >= 0) 

    # TODO: add constraints δ[ω] ≥ z - Π[ω]
    for ω in 1:length(profit)
        @constraint(model, delta[ω] >= z - profit[ω]) #epigraph variable delta to capture the positive part of the (z-profit) terms
    end
    # TODO: set objective  max z - (1/(αN)) * sum(δ)
    @objective(model, Max, z - (1/(alpha * length(profit))) * sum(delta))
    # TODO: optimize! and return objective_value
    optimize!(model)
    return objective_value(model)
end


# -----------------------------------------------------------------------------
# Two-scenario primitive of Section 2.4
# -----------------------------------------------------------------------------

π_scenarios = Float64[0, 100]   # e.g. [π_sunny, π_cloudy]
g_scenarios = Float64[15, 5]   # e.g. [g_sunny, g_cloudy]

# Profit as a function of Q for a given contract price P
# Π(ω; Q) = (P - π(ω)) * Q + π(ω) * g(ω)
function profit_vector(Q::Real, P::Real; π::Vector{Float64}=π_scenarios, g::Vector{Float64}=g_scenarios)
            # Profit is fixed price (P * Q) from selling forward contract
            # plus the settlement price * (generation minus forward quantity) from the spot settlement
    return [(P * Q) + (π[ω] * (g[ω] - Q)) for ω in 1:length(π)]
end


# -----------------------------------------------------------------------------
# (b) Reproduce Figure 1: sweep Q over [0, 15] and plot
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# (b.1) Single-α plot (α = 0.05, reproduces Figure 1 of Section 2.4)
# -----------------------------------------------------------------------------

scenario1 = Float64[] # profit in scenario 1 (sunny)
scenario2 = Float64[]   # profit in scenario 2 (cloudy)
expected = Float64[] # expected profit E[Π](Q) = (Π(ω₁; Q) + Π(ω₂; Q)) / 2
cvar_curves_single = Float64[] # CVaR_α(Q) for each Q in Q_grid

P = 50.0
Q_grid = range(0, 15; length=100) # Q values from 0 to 15
alpha = 0.05 # tail level for CVaR

# loop over Q in Q_grid, fill curves
for Q in Q_grid
    Π = profit_vector(Q, P)
    push!(scenario1, Π[1])
    push!(scenario2, Π[2])
    push!(expected, (Π[1] + Π[2]) / 2)
    push!(cvar_curves_single, cvar(Π, alpha))
end

# plot the four curves on one axes
plt_single = plot(Q_grid, scenario1, label="Scenario 1 (sunny)",
                  xlabel="Q (MW)", ylabel="Profit (\$)",
                  left_margin = 8Plots.mm, bottom_margin = 4Plots.mm,
                  legend = :outerright, linewidth = 2,
                  ylim = (-Inf, 600))
plot!(plt_single, Q_grid, scenario2, label="Scenario 2 (cloudy)", linewidth = 2)
plot!(plt_single, Q_grid, expected, label="E[Π](Q)", linewidth = 2)
plot!(plt_single, Q_grid, cvar_curves_single,
      label="CVaR α=$(alpha)", linestyle=:dash, linewidth=2)

savefig(plt_single, "results/task1/figure1_P$(Int(P)).png")


# -----------------------------------------------------------------------------
# (b.2) α-sweep plot: overlay CVaR curves for α ∈ {0.1, 0.2, …, 1.0}
# -----------------------------------------------------------------------------

alpha_grid = 0.1:0.1:1.0                     # sweep α from 0.1 to 1.0 in steps of 0.1

# CVaR curve for each α (scenario1/scenario2/expected from (b.1) reused)
cvar_curves = Dict(α => [cvar(profit_vector(Q, P), α)       for Q in Q_grid]          for α in alpha_grid)

# Plot: scenario lines + expected profit + one CVaR curve per α
plt = plot(Q_grid, scenario1, label="Scenario 1 (sunny)",
           xlabel="Q (MW)", ylabel="Profit (\$)",
           left_margin = 8Plots.mm, bottom_margin = 4Plots.mm,
           legend = :outerright, linewidth = 2, color = :black,
           ylim = (-Inf, 600))
plot!(plt, Q_grid, scenario2, label="Scenario 2 (cloudy)",
      linewidth = 2, color = :gray)
plot!(plt, Q_grid, expected, label="E[Π](Q)",
      linewidth = 2, color = :blue)

for (i, α) in enumerate(alpha_grid)
    plot!(plt, Q_grid, cvar_curves[α],
          label = "CVaR α=$(round(α, digits=2))",
          linestyle = :dash, linewidth = 1.5,
          color = cgrad(:viridis)[i / length(alpha_grid)])
end

savefig(plt, "results/task1/figure1_P$(Int(P))_alpha_sweep.png")




# -----------------------------------------------------------------------------
# (c) Add risk premium and standard deviation
# -----------------------------------------------------------------------------
#   RP(Q)   = E[Π](Q) - CVaR_α(Π(Q))
#   σ_Π(Q)  = sqrt(Var[Π(Q)])     (use uncorrected variance: equiprobable)
# -----------------------------------------------------------------------------

function plot_figure1_extended(P::Real; alpha::Real=0.05, Q_grid=range(0, 15; length=100))
    # RP(Q) and σ_Π(Q) on the same grid.

    scenario1 = Float64[] # profit in scenario 1 (sunny)
    scenario2 = Float64[]   # profit in scenario 2 (cloudy)
    expected = Float64[] # expected profit E[Π](Q) = (Π(ω₁; Q) + Π(ω₂; Q)) / 2
    cvar_curves_single = Float64[] # CVaR_α(Q) for each Q in Q_grid
    RP_curve = Float64[] # risk premium curve RP(Q) = E[Π](Q) - CVaR_α(Q)
    sigma_curve = Float64[] # standard deviation curve σ_Π(Q) = sqrt(Var[Π(Q)])


    RP_curve = Float64[]
    sigma_curve = Float64[]
    for Q in Q_grid
        Π = profit_vector(Q, P)
        push!(scenario1, Π[1])
        push!(scenario2, Π[2])
        push!(expected, (Π[1] + Π[2]) / 2)
        push!(cvar_curves_single, cvar(Π, alpha))
        push!(RP_curve, mean(Π) - cvar(Π, alpha))
        μ = mean(Π)
        push!(sigma_curve, sqrt(sum((Π .- μ).^2) / length(Π)))
    end

    # plot
    plt = plot(Q_grid, scenario1, label="Scenario 1 (sunny)",
               xlabel="Q (MW)", ylabel="Profit (\$)",
               left_margin = 8Plots.mm, bottom_margin = 4Plots.mm,
               legend = :outerright, linewidth = 2, color = :black,
               ylim = (-Inf, 600));
    plot!(plt, Q_grid, scenario2, label="Scenario 2 (cloudy)", linewidth = 2, color = :gray);
    plot!(plt, Q_grid, expected, label="E[Π](Q)", linewidth = 2, color = :blue);
    plot!(plt, Q_grid, cvar_curves_single, label="CVaR α=$(alpha)", linestyle=:dash, linewidth=2, color=:red);
    plot!(plt, Q_grid, RP_curve, label="Risk Premium (E[Π] − CVaR)", linestyle=:dot, linewidth=2, color=:green);
    plot!(plt, Q_grid, sigma_curve, label="σ_Π(Q)", linestyle=:dashdot, linewidth=2, color=:orange);
    savefig(plt, "results/task1/figure1_P$(Int(P))_extended.png");
    return plt
end

plot_figure1_extended(P)


function rho(profit::AbstractVector{<:Real}, alpha::Real, lambda::Real)
    return lambda * cvar(profit, alpha) + (1 - lambda) * mean(profit)
end

function tail_scenarios(profit::AbstractVector{<:Real}, alpha::Real)
    #       (i.e. the worst ceil(αN) scenarios — here αN < 1 means the single worst)
    N = length(profit)
    num_tail = ceil(Int, alpha * N) # number of scenarios in the tail
    sorted_indices = sortperm(profit) # indices of scenarios sorted by profit
    return sorted_indices[1:num_tail] # return indices of the worst num_tail scenarios
end

P = 50.0
# ρ_{α,λ}(Q) curves for each λ on the same axes, with Q★ marked
plt_rho = plot(xlabel = "Q (MW)",
               ylabel = "ρ_{α,λ}(Q) (\$)",
               title  = "Certainty equivalent vs Q  (P=$(Int(P)), α=$(alpha))",
               left_margin = 8Plots.mm, bottom_margin = 4Plots.mm,
               legend = :outerright, linewidth = 2,
               ylim = (-10, 600))

for (i, λ) in enumerate([0.0, 0.5, 0.6, 1.0])
    # ρ_{α,λ}(Q) over Q_grid
    ρ_values = [rho(profit_vector(Q, P), alpha, λ) for Q in Q_grid]
    Q_star = Q_grid[argmax(ρ_values)]
    Π_star = profit_vector(Q_star, P)
    E_Π        = mean(Π_star)
    CVaR_value = cvar(Π_star, alpha)
    RP_value   = E_Π - CVaR_value
    σ_value    = sqrt(sum((Π_star .- E_Π).^2) / length(Π_star))
    println("$(λ)\t$(round(Q_star, digits=2))\t$(round(E_Π, digits=2))\t$(round(CVaR_value, digits=2))\t$(round(RP_value, digits=2))\t$(round(σ_value, digits=2))")

    # ρ curve + Q★ marker
    color = cgrad(:plasma)[i / 3]
    plot!(plt_rho, Q_grid, ρ_values,
          label = "λ = $(λ)", color = color, linewidth = 2)
    scatter!(plt_rho, [Q_star], [maximum(ρ_values)],
             label = "Q★(λ=$(λ)) = $(round(Q_star, digits=2))",
             color = color, markersize = 6, markershape = :star5)
end

savefig(plt_rho, "results/task1/rho_vs_Q_P$(Int(P)).png")


# -----------------------------------------------------------------------------
# (e) Repeat with P = 55
# -----------------------------------------------------------------------------
# Re-run (b), (c), (d) with the contract price raised from 50 to 55.
# Expectation: E[Π](Q) is no longer flat — slope (P - E[π]) * Q = 5 * Q.
# The ρ_{α,λ}-optimal Q★ now shifts with λ.
# -----------------------------------------------------------------------------
P = 65.0
plt = plot_figure1_extended(P)
savefig(plt, "results/task1/figure1_P$(Int(P))_extended.png")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
# Uncomment to run when executing this file directly:
# run_task1()
