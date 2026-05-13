# =============================================================================
# Project 3 — Task 1: PQ-Risk illustration and CVaR linear program
# ECON-138, Prof. Alexandre Street
# =============================================================================
#
# Task overview:
#   (a) Implement CVaR via Rockafellar-Uryasev LP using JuMP + HiGHS
#   (b) Reproduce Figure 1 of Section 2.4 (two-scenario PQ illustration)
#   (c) Add risk premium and standard deviation curves
#   (d) Comment on results
#   (e) Repeat (a)-(d) with P = 55
#
# Two-scenario primitive (from Section 2.4 of theory document):
#   - Two equiprobable spot-price scenarios (sunny / cloudy) at the single period
#   - One generator with generation g(ω) under each scenario
#   - Forward contract sells Q at fixed price P; residual settles at spot
#   - Profit:  Π(ω; Q) = (P - π(ω)) * Q + π(ω) * g(ω)
#
# Fill in the scenario primitive parameters below to match the slides.
# =============================================================================

using JuMP
using HiGHS
using Statistics
using Plots

# -----------------------------------------------------------------------------
# (a) CVaR function — Rockafellar–Uryasev LP 
# -----------------------------------------------------------------------------
#
#   CVaR_α(Π) = max  z - (1/(αN)) * Σ_ω δ_ω
#               s.t. δ_ω ≥ z - Π_ω,  δ_ω ≥ 0,  ∀ ω = 1,...,N
#
# Inputs:
#   profit :: Vector{<:Real}   — N equiprobable profit realizations
#   alpha  :: Real             — tail level in (0, 1]
# Output:
#   CVaR_α(Π) as a scalar Float64
# -----------------------------------------------------------------------------

function cvar(profit::AbstractVector{<:Real}, alpha::Real)
    # TODO: build JuMP model with HiGHS optimizer
    model = Model(HiGHS.Optimizer)
    # TODO: declare variables z (free) and δ[1:N] (≥ 0)
    @variable(model, z)
    @variable(model, delta[1:length(profit)] >= 0) 

    # TODO: add constraints δ[ω] ≥ z - Π[ω]
    for omega in 1:length(profit)
        @constraint(model, delta[omega] >= z - profit[omega]) #epigraph variable delta to capture the positive part of the (z-profit) terms
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
# TODO: set the scenario data from the slides
#   π_scenarios :: Vector{Float64}   — spot prices in the two scenarios
#   g_scenarios :: Vector{Float64}   — generation in the two scenarios
#   (the slides imply E[g] = 10 and E[π] = 50; pick the two-scenario values
#    that reproduce Figure 1 — e.g. (π, g) pairs for "sunny" vs "cloudy")

π_scenarios = Float64[]   # e.g. [π_sunny, π_cloudy]
g_scenarios = Float64[]   # e.g. [g_sunny, g_cloudy]

# Profit as a function of Q for a given contract price P
# Π(ω; Q) = (P - π(ω)) * Q + π(ω) * g(ω)
function profit_vector(Q::Real, P::Real;
                       π::Vector{Float64}=π_scenarios,
                       g::Vector{Float64}=g_scenarios)
    # TODO: return Vector{Float64} of length length(π) with Π(ω; Q)
end


# -----------------------------------------------------------------------------
# (b) Reproduce Figure 1: sweep Q over [0, 15] and plot
# -----------------------------------------------------------------------------
# Curves on the same axes:
#   - scenario profit lines Π(ω₁; Q), Π(ω₂; Q)
#   - expected profit E[Π](Q)
#   - CVaR_0.05(Q) via cvar(profit_vector(Q, P), 0.05)
# -----------------------------------------------------------------------------

function plot_figure1(P::Real; alpha::Real=0.05, Q_grid=range(0, 15; length=301))
    # TODO: allocate arrays for scenario1, scenario2, expected, cvar curves
    # TODO: loop over Q in Q_grid, fill curves
    # TODO: plot the four curves on one axes (label, xlabel = "Q", ylabel = "\$")
    # TODO: return the plot object (and optionally the curve arrays)
end


# -----------------------------------------------------------------------------
# (c) Add risk premium and standard deviation
# -----------------------------------------------------------------------------
#   RP(Q)   = E[Π](Q) - CVaR_α(Π(Q))
#   σ_Π(Q)  = sqrt(Var[Π(Q)])     (use uncorrected variance: equiprobable)
# -----------------------------------------------------------------------------

function plot_figure1_extended(P::Real; alpha::Real=0.05, Q_grid=range(0, 15; length=301))
    # TODO: as in plot_figure1, plus:
    # TODO: compute RP(Q) and σ_Π(Q) on the same grid
    # TODO: overlay them on the figure with their own labels
    # TODO: return plot (and the curve arrays for inspection)
end


# -----------------------------------------------------------------------------
# (d) Commentary helpers (for your own inspection — not required deliverables)
# -----------------------------------------------------------------------------
# Use these to back the talking points:
#   (i)   why E[Π] is flat in Q when P = E[π]
#   (ii)  CVaR / RP / σ all optimal at the same Q★ = 5; Q★ < E[g] = 10
#   (iii) ρ_{α,λ}-maximizer for λ ∈ {0, 0.5, 1}
#   (iv)  which scenario(s) compose CVaR at Q = 0, 5, 10
# -----------------------------------------------------------------------------

function rho(profit::AbstractVector{<:Real}, alpha::Real, lambda::Real)
    # TODO: return λ * CVaR_α(Π) + (1 - λ) * E[Π]
end

function tail_scenarios(profit::AbstractVector{<:Real}, alpha::Real)
    # TODO: return the indices of the scenarios composing the α-tail
    #       (i.e. the worst ceil(αN) scenarios — here αN < 1 means the single worst)
end


# -----------------------------------------------------------------------------
# (e) Repeat with P = 55
# -----------------------------------------------------------------------------
# Re-run (b), (c), (d) with the contract price raised from 50 to 55.
# Expectation: E[Π](Q) is no longer flat — slope (P - E[π]) * Q = 5 * Q.
# The ρ_{α,λ}-optimal Q★ now shifts with λ.
# -----------------------------------------------------------------------------

function run_task1(; P_values=(50.0, 55.0), alpha::Real=0.05)
    # TODO: for each P in P_values:
    # TODO:   produce plot_figure1_extended(P)
    # TODO:   save to file (e.g. "task1_figure1_P50.png", "task1_figure1_P55.png")
    # TODO:   print the table of (Q★, E[Π], CVaR, RP, σ) for λ ∈ {0, 0.5, 1}
end


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
# Uncomment to run when executing this file directly:
# run_task1()
