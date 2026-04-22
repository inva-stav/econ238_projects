# Problem 6: 2-node power system dispatch via Kelley with balance-epigraph cuts.
# Also solves the direct LP as a reference benchmark.

include(joinpath(@__DIR__, "algorithms.jl"))

# ── Power system data ───────────────────────────────────────────
n_gen    = 3
n_nodes  = 2
n_lines  = 1

n_demand = 2
t_steps  = 24 #can modify to make problem larger

g_opex   = [30, 20, 40]
g_capex = [100, 200, 80]

g_max_cap = [100, 50, 500]   # Increased capacity for generator 3 to ensure feasibility

incidence = [1, -1]

f_lim = 1000   # 1 GW line limit

# Base 24-hour daily shape (MW) per demand node.
base_demand = [
    [60, 60, 50, 50, 70, 90, 110, 120, 120, 110, 100, 100, 100, 90, 110, 130, 150, 140, 130, 120, 100, 80, 70, 60],
    [150, 140, 130, 120, 150, 180, 200, 220, 210, 200, 190, 180, 170, 160, 180, 200, 230, 250, 240, 220, 190, 170, 160, 150]
]
hours_per_day = length(base_demand[1])

# Build t_steps-long demand per node by tiling the daily shape and scaling
# each day by a small random multiplier so days are similar but not identical.
using Random
Random.seed!(42)
day_noise_std = 0.05   # ~5% day-to-day variability
n_days = ceil(Int, t_steps / hours_per_day)
demand = [Float64[] for _ in 1:n_demand]
for nd in 1:n_demand
    for _ in 1:n_days
        day_scale = 1.0 + day_noise_std * randn()
        append!(demand[nd], base_demand[nd] .* day_scale)
    end
    resize!(demand[nd], t_steps)
end

# # ── Expansion Planning Problem ───────────────────────────────────
expansion_model = Model(HiGHS.Optimizer)

@variable(expansion_model, g[i=1:n_gen, t=1:t_steps] >= 0)
@variable(expansion_model, -1 * f_lim <= f[l=1:n_lines, t=1:t_steps] <= f_lim)
@variable(expansion_model, 0 <= Capacity_gen[i=1:n_gen] <= 1e8) # Generator Capacity Limits, do we want to make this an integer variable? or maybe make the lines integer variables?

@objective(expansion_model, Min, sum(g_opex[i]*g[i,t] for i=1:n_gen, t=1:t_steps) + sum(g_capex[i] * Capacity_gen[i] for i=1:n_gen))

@constraint(expansion_model, gen_op_lim[i=1:n_gen,t=1:t_steps], g[i,t] <= Capacity_gen[i])

@constraint(expansion_model, demand_balance_n1[t=1:t_steps],
    g[1,t] + g[2,t] - demand[1][t] + f[1,t] == 0)

@constraint(expansion_model, demand_balance_n2[t=1:t_steps],
    g[3,t] - demand[2][t] - f[1,t] == 0)

optimize!(expansion_model)

if termination_status(expansion_model) == MOI.OPTIMAL
    println("Reference LP optimal!  obj = ", objective_value(expansion_model))
    g_values   = value.(g)
    f_values   = value.(f)
    cap_values = value.(Capacity_gen)
    println("\nGenerator outputs (g):")
    for i in 1:n_gen
        println("  Generator $(i): ", g_values[i,:])
    end
    println("\nLine flows (f):")
    for i in 1:n_lines
        println("  Line $(i): ", f_values[i,:])
    end
    println("\nInstalled capacities: ", cap_values)
else
    println("Reference LP failed: ", termination_status(expansion_model))
end

# Generator dispatch plot (reference): all generators + capacity lines
prob_num = 6
out_dir  = joinpath(@__DIR__, "results", "problem$(prob_num)")
mkpath(out_dir)

if @isdefined(g_values)
    time_steps = 1:t_steps
    p = plot(xlabel="Time Step", ylabel="Generation (MW)",
             title="Generator Dispatch — Reference LP", legend=:outertopright)
    for i in 1:n_gen
        color = palette(:tab10)[i]
        plot!(p, time_steps, g_values[i,:], label="Gen $(i)", lw=2, color=color)
        hline!(p, [cap_values[i]], label="Gen $(i) capacity",
               linestyle=:dash, color=color)
    end
    savefig(p, joinpath(out_dir, "ref_dispatch.png"))
    Plots.closeall()
end

# ── Kelley formulation (Benders-style) ──────────────────────────
# Outer decision x = installed generator capacities  (length n = n_gen)
# f(x) = g_capex' x + Q(x)
#   where Q(x) = min dispatch cost s.t. g[i,t] <= x[i] + balance + flow limits
# Envelope theorem:  ∂Q/∂x[i] = sum_t dual(gen_op_lim[i,t])   (≤ 0 in JuMP's
# convention for a ≤-constraint in a min problem).
alg = "kelley"
n   = n_gen

x_lb = zeros(n)
x_ub = fill(1.0e5, n)   # loose upper bound on capacity

# ── Parameterized inner dispatch LP (built once) ────────────────
# Slack s[nd,t] with VOLL penalty keeps the LP feasible at any x ≥ 0.
VOLL = 1.0e6
oracle_model = Model(HiGHS.Optimizer)
set_silent(oracle_model)

@variable(oracle_model, g_d[i=1:n_gen, t=1:t_steps] >= 0)
@variable(oracle_model, -f_lim <= f_d[l=1:n_lines, t=1:t_steps] <= f_lim)
@variable(oracle_model, s_d[nd=1:n_nodes, t=1:t_steps] >= 0)

@objective(oracle_model, Min,
    sum(g_opex[i] * g_d[i,t]   for i=1:n_gen,   t=1:t_steps)
  + sum(VOLL     * s_d[nd,t]   for nd=1:n_nodes, t=1:t_steps))

# Capacity constraint — RHS (0.0 placeholder) updated each oracle call via set_normalized_rhs.
@constraint(oracle_model, gen_op_lim[i=1:n_gen, t=1:t_steps], g_d[i,t] <= 0.0)

@constraint(oracle_model, balance_n1[t=1:t_steps],
    g_d[1,t] + g_d[2,t] + f_d[1,t] + s_d[1,t] == demand[1][t])
@constraint(oracle_model, balance_n2[t=1:t_steps],
    g_d[3,t]            - f_d[1,t] + s_d[2,t] == demand[2][t])

# ── Oracle ──────────────────────────────────────────────────────
function functionAndGradient(x)
    # Push current capacity into the RHS of the capacity constraints.
    for i in 1:n_gen, t in 1:t_steps
        set_normalized_rhs(gen_op_lim[i,t], x[i])
    end
    optimize!(oracle_model)
    @assert termination_status(oracle_model) == MOI.OPTIMAL "dispatch oracle failed at x=$x"

    Q = objective_value(oracle_model)  # dispatch cost + VOLL·slack
    capex = sum(g_capex[i] * x[i] for i in 1:n_gen)
    f_val = capex + Q

    # Calculate gradients by extracting from duals
    grad = zeros(n)
    for i in 1:n_gen
        grad[i] = g_capex[i] + sum(dual(gen_op_lim[i,t]) for t in 1:t_steps)
    end
    return f_val, grad
end


prob_num = "6benders"
out_dir  = joinpath(@__DIR__, "results", "problem$(prob_num)")
mkpath(out_dir)

run_algorithm(logscale = true)
