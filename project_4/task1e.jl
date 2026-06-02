# =============================================================================
# Project 4 — Task 1e
# n−1 security (reserve): monolithic vs. robust counterpart via LP duality
#
# Single-node system: 3 generators, K=1 (any single generator can fail).
#   c  = (100, 150, 500) $/MWh   energy offers
#   cr = 0.2 * c         $/MWh   reserve costs  = (20, 30, 100)
#   Q  = (100,  50, 120) MWh     capacities
#   d  = 120             MWh     demand
#
# Run:  julia task1e.jl
# =============================================================================

using JuMP, HiGHS
using CSV, DataFrames, Dates
using Printf

###################################
### Data
###################################

const c  = [100.0, 150.0, 500.0]
const Q  = [100.0,  50.0, 120.0]
const d  = 120.0
const cr = 0.2 .* c                  # reserve cost = (20, 30, 100) $/MWh
const n  = length(c)

###################################
### (A) Monolithic formulation
#
# Enumerate all n contingencies explicitly.
#
# min   Σ (c_i g_i + cr_i r_i)
# s.t.  Σ g_i  = d                           [energy balance]
#       g_i + r_i ≤ Q_i          ∀i          [capacity including reserve]
#       Σ_{j≠k} (g_j + r_j) ≥ d  ∀k         [n-1 security: each contingency]
#       g_i ≥ 0,  r_i ≥ 0
#
# The k-th constraint says: if generator k trips, the surviving units
# (at their maximum output g_j + r_j) can still serve demand.
# As the contingency set grows, the number of security constraints
# grows as C(n,K): n for K=1, n(n-1)/2 for K=2, etc.
###################################

function solve_monolithic(c, Q, d, cr)
    n = length(c)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, g[i=1:n] >= 0)
    @variable(m, r[i=1:n] >= 0)

    @constraint(m, balance, sum(g) == d)
    @constraint(m, cap[i=1:n], g[i] + r[i] <= Q[i])

    # Contingency k: all generators except k serve demand at full output
    @constraint(m, security[k=1:n],
        sum(g[j] + r[j] for j in 1:n if j != k) >= d)

    @objective(m, Min, sum(c[i]*g[i] + cr[i]*r[i] for i in 1:n))

    optimize!(m)
    @assert termination_status(m) == OPTIMAL "Monolithic model did not reach OPTIMAL"

    return value.(g), value.(r), objective_value(m)
end

###################################
### (B) Robust counterpart via LP duality
#
# Step 1 — rewrite the n security constraints as one minimax:
#
#   ∀k: Σ_{j≠k}(g_j+r_j) ≥ d
#   ⟺  Σ_j(g_j+r_j) − max_k(g_k+r_k) ≥ d
#
#   (The tightest contingency is the one where the largest unit trips.)
#
# Step 2 — write the inner adversarial LP (given g, r fixed):
#
#   max_{z}  Σ_i z_i (g_i + r_i)
#   s.t.     Σ_i z_i = 1,   z_i ≥ 0          [probability simplex]
#
#   LP relaxation is exact: the simplex is compact and the objective is
#   linear, so the max is attained at a vertex z = e_k for the k with the
#   largest (g_k + r_k).  Hence max_LP = max_k(g_k+r_k).
#
# Step 3 — take the dual of the inner LP:
#
#   min_{λ}  λ
#   s.t.     λ ≥ g_i + r_i   ∀i              [one per z_i]
#
# Step 4 — invoke strong LP duality  <<<
#
#   Inner LP is feasible (z = (1/n)·1 is strictly feasible) and bounded
#   (objective ≤ max Q_i < ∞).  By strong LP duality:
#
#       max_{z ∈ Δ} Σ z_i (g_i+r_i)  =  min{λ : λ ≥ g_i+r_i ∀i}
#
#   Substituting into the security requirement gives the robust counterpart:
#
# Step 5 — absorb the dual variable λ into the outer program:
#
# min   Σ (c_i g_i + cr_i r_i)
# s.t.  Σ g_i  = d                           [energy balance]
#       g_i + r_i ≤ Q_i          ∀i          [capacity]
#       Σ_i(g_i+r_i) − λ ≥ d                 [security via duality  ← strong LP duality]
#       λ ≥ g_i + r_i            ∀i          [dual constraints of inner LP]
#       g_i ≥ 0,  r_i ≥ 0,  λ ≥ 0
#
# This compact LP has 2n+2 constraints regardless of K, versus C(n,K) in (A).
###################################

function solve_robust(c, Q, d, cr)
    n = length(c)
    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, g[i=1:n] >= 0)
    @variable(m, r[i=1:n] >= 0)
    @variable(m, λ >= 0)             # dual variable of inner adversarial LP

    @constraint(m, balance, sum(g) == d)
    @constraint(m, cap[i=1:n], g[i] + r[i] <= Q[i])

    # <<< strong LP duality applied here >>>
    # max_k(g_k+r_k) replaced by λ from the dual of the adversarial LP
    @constraint(m, security_dual, sum(g[i] + r[i] for i in 1:n) - λ >= d)
    @constraint(m, dual_con[i=1:n], λ >= g[i] + r[i])

    @objective(m, Min, sum(c[i]*g[i] + cr[i]*r[i] for i in 1:n))

    optimize!(m)
    @assert termination_status(m) == OPTIMAL "Robust counterpart did not reach OPTIMAL"

    return value.(g), value.(r), objective_value(m)
end

###################################
### Main runner
###################################

function run_task1e(; save::Bool = true)
    tol = 1e-6

    println("\n" * "="^60)
    println("Task 1e: n−1 Security — Monolithic vs. Robust Counterpart")
    println("="^60)
    @printf("\n  Reserve costs: cr = (%.0f, %.0f, %.0f) \$/MWh  (= 0.2 × c)\n",
            cr[1], cr[2], cr[3])

    g_A, r_A, obj_A = solve_monolithic(c, Q, d, cr)
    g_B, r_B, obj_B = solve_robust(c, Q, d, cr)

    # --- Side-by-side results ---
    println("\n── Results: Monolithic (A) vs. Robust Counterpart (B) ──")
    @printf("  %-8s  %14s  %14s\n", "var", "Monolithic (A)", "Robust (B)")
    println("  " * "-"^40)
    for i in 1:n
        @printf("  g_%-5d  %14.4f  %14.4f  MWh\n", i, g_A[i], g_B[i])
    end
    for i in 1:n
        @printf("  r_%-5d  %14.4f  %14.4f  MWh\n", i, r_A[i], r_B[i])
    end
    println("  " * "-"^40)

    energy_A  = sum(c[i]*g_A[i]  for i in 1:n)
    reserve_A = sum(cr[i]*r_A[i] for i in 1:n)
    energy_B  = sum(c[i]*g_B[i]  for i in 1:n)
    reserve_B = sum(cr[i]*r_B[i] for i in 1:n)

    @printf("  %-8s  %14.4f  %14.4f  \$/h\n", "energy",  energy_A,  energy_B)
    @printf("  %-8s  %14.4f  %14.4f  \$/h\n", "reserve", reserve_A, reserve_B)
    @printf("  %-8s  %14.4f  %14.4f  \$/h\n", "TOTAL",   obj_A,     obj_B)

    # --- Verification ---
    println("\n── Verification ──")
    @assert abs(obj_A - obj_B) < tol        "Objectives differ: $(obj_A) vs $(obj_B)"
    @assert isapprox(g_A, g_B, atol=tol)   "Dispatch g* differs between formulations"
    @assert isapprox(r_A, r_B, atol=tol)   "Reserve r* differs between formulations"
    println("  ✓  Both formulations agree to tol $tol")

    # Contingency check: verify each security constraint at optimum
    println("\n── Security constraint check (monolithic solution) ──")
    @printf("  %-12s  %14s  %8s\n", "contingency", "Σ_{j≠k}(g+r)", "≥ d?")
    println("  " * "-"^38)
    for k in 1:n
        coverage = sum(g_A[j] + r_A[j] for j in 1:n if j != k)
        ok = coverage >= d - tol ? "✓" : "✗"
        @printf("  k=%-9d  %14.4f     %s\n", k, coverage, ok)
        @assert coverage >= d - tol  "Contingency $k violated"
    end

    # --- Save CSVs ---
    if save
        println("\nSaving outputs...")
        dir = joinpath(@__DIR__, "results", "task1e")
        mkpath(dir)

        # dispatch_reserve.csv
        df_dr = DataFrame(
            gen            = 1:n,
            g_monolithic   = g_A,
            r_monolithic   = r_A,
            g_robust       = g_B,
            r_robust       = r_B,
            c_energy       = c,
            cr_reserve     = cr,
            Q_capacity     = Q
        )
        p = joinpath(dir, "dispatch_reserve.csv")
        CSV.write(p, df_dr); println("  saved: $p")

        # objectives.csv
        df_obj = DataFrame(
            formulation  = ["monolithic", "robust_counterpart"],
            energy_cost  = [energy_A,  energy_B],
            reserve_cost = [reserve_A, reserve_B],
            total_cost   = [obj_A,     obj_B]
        )
        p = joinpath(dir, "objectives.csv")
        CSV.write(p, df_obj); println("  saved: $p")

        # security_check.csv
        df_sec = DataFrame(
            k        = 1:n,
            coverage = [sum(g_A[j]+r_A[j] for j in 1:n if j != k) for k in 1:n],
            demand   = fill(d, n),
            slack    = [sum(g_A[j]+r_A[j] for j in 1:n if j != k) - d for k in 1:n]
        )
        p = joinpath(dir, "security_check.csv")
        CSV.write(p, df_sec); println("  saved: $p")

        # metadata.csv
        df_meta = DataFrame(
            key   = ["n", "d", "K", "obj_monolithic", "obj_robust",
                     "agree", "timestamp"],
            value = [string(n), string(d), "1",
                     @sprintf("%.4f", obj_A), @sprintf("%.4f", obj_B),
                     string(abs(obj_A - obj_B) < tol), string(now())]
        )
        p = joinpath(dir, "metadata.csv")
        CSV.write(p, df_meta); println("  saved: $p")
    end
end

run_task1e()
