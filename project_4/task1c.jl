# =============================================================================
# Project 4 — Task 1c
# Deterministic economic dispatch: primal, dual, and strong duality
#
# Single-node system: 3 generators, 1 demand node.
#   c = (100, 150, 500) $/MWh   energy offers
#   Q = (100,  50, 120) MWh     capacities
#   d = 120              MWh    demand
#
# Run:  julia task1c.jl
# =============================================================================

using JuMP, HiGHS
using CSV, DataFrames, Dates
using Printf

###################################
### Data
###################################

const c_base = [100.0, 150.0, 500.0]
const Q_base = [100.0,  50.0, 120.0]
const d_base = 120.0

###################################
### Primal model
#
# min   Σ c_i g_i
# s.t.  Σ g_i  = d      [balance, dual π  (free)]
#       g_i   ≤ Q_i    [capacity, dual β_i ≥ 0]
#       g_i   ≥ 0
#
# JuMP dual-sign convention for a MIN problem:
#   ==  constraint  →  dual() is the shadow price, free sign.
#                       dual(balance) = π   (rate of change of min-cost w.r.t. d)
#   <=  constraint  →  dual() ≤ 0.
#                       Relaxing an upper bound in a min can only reduce cost,
#                       so the shadow price is non-positive.
#                       dual(cap[i]) = -β_i  ≤ 0   ⟹   β_i = -dual(cap[i]) ≥ 0
###################################

function solve_dispatch(c, Q, d)
    n = length(c)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, g[i=1:n] >= 0)
    @constraint(m, balance, sum(g) == d)
    @constraint(m, cap[i=1:n], g[i] <= Q[i])
    @objective(m, Min, sum(c[i] * g[i] for i in 1:n))

    optimize!(m)
    @assert termination_status(m) == OPTIMAL "Primal did not reach OPTIMAL"

    g_opt  = value.(g)
    C_opt  = objective_value(m)
    π_opt  = dual(balance)                      # free; positive: more demand costs more
    β_opt  = [-dual(cap[i]) for i in 1:n]      # flip sign: β_i = -dual(cap[i]) ≥ 0

    return g_opt, C_opt, π_opt, β_opt
end

###################################
### Dual model
#
# max   π d - Σ β_i Q_i
# s.t.  π - β_i ≤ c_i   ∀i
#       β_i ≥ 0,  π free
#
# Derivation: Lagrangian of the primal (multiplying each constraint by its dual):
#   L = Σ c_i g_i  +  π (d - Σ g_i)  +  Σ β_i (Q_i - g_i)
# Collecting g_i terms: coefficient = c_i - π - β_i.
# For inf_{g≥0} L to be bounded, need c_i - π - β_i ≥ 0 for all i,
# i.e., π - β_i ≤ c_i.  The dual objective is the lower envelope evaluated
# at g=0: π d - Σ β_i Q_i... wait — actually at g=0, L = πd + Σβ_i Q_i,
# but we define β_i via the (+)-signed multiplier on (Q_i - g_i), so the
# dual objective is πd - Σβ_i Q_i only after absorbing the sign convention
# β_i → -μ_i where μ_i ≤ 0 is the raw JuMP dual.  This matches the LP duality
# table: for a min primal with <= constraint, the dual variable is ≤ 0 (μ_i),
# and the dual objective picks up b_i μ_i = Q_i μ_i = -Q_i β_i.
###################################

function solve_dual_model(c, Q, d)
    n = length(c)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, π_var)                  # free
    @variable(m, β[i=1:n] >= 0)
    @constraint(m, [i=1:n], π_var - β[i] <= c[i])
    @objective(m, Max, π_var * d - sum(β[i] * Q[i] for i in 1:n))

    optimize!(m)
    @assert termination_status(m) == OPTIMAL "Dual model did not reach OPTIMAL"

    return value(π_var), value.(β), objective_value(m)
end

###################################
### Main runner
###################################

function run_task1c(; save::Bool = true)
    n = length(c_base)
    tol = 1e-6

    println("\n" * "="^60)
    println("Task 1c: Deterministic Economic Dispatch")
    println("="^60)

    # --- Solve primal and explicit dual ---
    g_opt, C_opt, π_primal, β_primal = solve_dispatch(c_base, Q_base, d_base)
    π_dual, β_dual, _                 = solve_dual_model(c_base, Q_base, d_base)

    # --- Print primal solution ---
    println("\n── Primal solution ──")
    @printf("  %-6s  %8s  %8s  %8s\n", "gen", "g* (MWh)", "Q (MWh)", "c (\$/MWh)")
    println("  " * "-"^38)
    for i in 1:n
        @printf("  %-6d  %8.4f  %8.1f  %8.1f\n", i, g_opt[i], Q_base[i], c_base[i])
    end
    @printf("\n  C* = %.4f \$/h\n", C_opt)

    # --- Print dual variables (two sources) ---
    println("\n── Dual variables ──")
    @printf("  %-22s  %10s  %10s  %10s  %10s\n",
            "source", "pi* (\$/MWh)", "beta*_1", "beta*_2", "beta*_3")
    println("  " * "-"^66)
    @printf("  %-22s  %10.4f  %10.4f  %10.4f  %10.4f\n",
            "primal model duals", π_primal, β_primal[1], β_primal[2], β_primal[3])
    @printf("  %-22s  %10.4f  %10.4f  %10.4f  %10.4f\n",
            "explicit dual LP", π_dual, β_dual[1], β_dual[2], β_dual[3])

    # --- Verification ---
    println("\n── Verification ──")

    # 1. Dual variables match
    @assert abs(π_primal - π_dual) < tol       "π mismatch: primal duals vs dual model"
    for i in 1:n
        @assert abs(β_primal[i] - β_dual[i]) < tol  "β_$i mismatch"
    end
    println("  ✓  π* and β* from primal duals match explicit dual LP")

    # 2. Strong duality: C* = π* d − Σ β_i* Q_i
    strong_val = π_dual * d_base - sum(β_dual[i] * Q_base[i] for i in 1:n)
    @assert abs(C_opt - strong_val) < tol  "Strong duality violated"
    @printf("  ✓  Strong duality: C* = %.4f  =  π*d − Σβ_iQ_i = %.4f\n", C_opt, strong_val)

    # 3. Known-answer sanity checks
    @assert isapprox(g_opt,  [100.0, 20.0, 0.0], atol=tol)  "g* sanity failed"
    @assert isapprox(C_opt,  13000.0,             atol=tol)  "C* sanity failed"
    @assert isapprox(π_dual, 150.0,               atol=tol)  "π* sanity failed"
    @assert isapprox(β_dual, [50.0, 0.0, 0.0],   atol=tol)  "β* sanity failed"
    println("  ✓  Sanity: g*=(100,20,0), C*=13000, π*=150, β*=(50,0,0)")

    # --- Complementary slackness ---
    # At optimality: β_i · (Q_i − g_i) = 0  for all i.
    #   Gen 1: capacity binding (g_1 = Q_1 = 100)  → β_1 = 50 > 0 is consistent.
    #   Gen 2: partially dispatched (0 < g_2 = 20 < Q_2 = 50)  → β_2 = 0 must hold.
    #   Gen 3: out of merit (g_3 = 0 < Q_3 = 120)  → the dual constraint
    #          π − β_3 ≤ c_3 is slack (150 ≤ 500), so β_3 = 0 from primal CS.
    println("\n── Complementary slackness  β_i · (Q_i − g_i) = 0 ──")
    @printf("  %-6s  %8s  %10s  %12s\n", "gen", "β_i", "slack", "product")
    println("  " * "-"^42)
    for i in 1:n
        slack = Q_base[i] - g_opt[i]
        prod  = β_primal[i] * slack
        @printf("  %-6d  %8.4f  %10.4f  %12.2e\n", i, β_primal[i], slack, prod)
        @assert abs(prod) < tol  "CS violated for gen $i"
    end
    println("  ✓  Complementary slackness holds")

    # --- Perturbation demo: raise c_2 from 150 to 200 ---
    println("\n── Perturbation: raise c_2 from 150 to 200 \$/MWh ──")
    c_pert = copy(c_base); c_pert[2] = 200.0
    g_pert, C_pert, π_pert, _ = solve_dispatch(c_pert, Q_base, d_base)
    @printf("  g* = (%.1f, %.1f, %.1f),  C* = %.4f,  π* = %.4f\n",
            g_pert[1], g_pert[2], g_pert[3], C_pert, π_pert)

    # --- Save CSVs ---
    if save
        println("\nSaving outputs...")
        dir = joinpath(@__DIR__, "results", "task1c")
        mkpath(dir)

        # dispatch.csv
        df_disp = DataFrame(
            gen      = 1:n,
            g_opt    = g_opt,
            Q        = Q_base,
            c        = c_base,
            status   = [g_opt[i] ≈ Q_base[i] ? "at_capacity" :
                        g_opt[i] > 0          ? "partial"     : "out_of_merit"
                        for i in 1:n]
        )
        p = joinpath(dir, "dispatch.csv")
        CSV.write(p, df_disp); println("  saved: $p")

        # duals.csv
        df_dual = DataFrame(
            source = ["primal_model", "dual_model"],
            pi     = [π_primal, π_dual],
            beta_1 = [β_primal[1], β_dual[1]],
            beta_2 = [β_primal[2], β_dual[2]],
            beta_3 = [β_primal[3], β_dual[3]]
        )
        p = joinpath(dir, "duals.csv")
        CSV.write(p, df_dual); println("  saved: $p")

        # strong_duality.csv
        df_sd = DataFrame(
            quantity = ["primal_cost_C_star", "dual_obj_pi_d_minus_betaQ",
                        "difference", "strong_duality_holds"],
            value    = [C_opt, strong_val, abs(C_opt - strong_val),
                        abs(C_opt - strong_val) < tol ? 1.0 : 0.0]
        )
        p = joinpath(dir, "strong_duality.csv")
        CSV.write(p, df_sd); println("  saved: $p")

        # complementary_slackness.csv
        df_cs = DataFrame(
            gen     = 1:n,
            beta_i  = β_primal,
            slack_i = [Q_base[i] - g_opt[i] for i in 1:n],
            product = [β_primal[i] * (Q_base[i] - g_opt[i]) for i in 1:n]
        )
        p = joinpath(dir, "complementary_slackness.csv")
        CSV.write(p, df_cs); println("  saved: $p")

        # metadata.csv
        df_meta = DataFrame(
            key   = ["n", "d", "C_star", "pi_star", "timestamp"],
            value = [string(n), string(d_base), @sprintf("%.4f", C_opt),
                     @sprintf("%.4f", π_dual), string(now())]
        )
        p = joinpath(dir, "metadata.csv")
        CSV.write(p, df_meta); println("  saved: $p")
    end
end

run_task1c()
