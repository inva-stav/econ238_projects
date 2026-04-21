# Problem 6: 2-node power system dispatch via Kelley with balance-epigraph cuts.
# Also solves the direct LP as a reference benchmark.

include(joinpath(@__DIR__, "algorithms.jl"))

# ── Power system data ───────────────────────────────────────────
n_gen    = 3
n_nodes  = 2
n_lines  = 1

n_demand = 2
t_steps  = 72 #can modify to make problem larger

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

# ── Reference direct LP solve ───────────────────────────────────
ref_model = Model(HiGHS.Optimizer)

@variable(ref_model, g[i=1:n_gen, t=1:t_steps] >= 0)
@variable(ref_model, -1 * f_lim <= f[l=1:n_lines, t=1:t_steps] <= f_lim)
@variable(ref_model, 0 <= Capacity_gen[i=1:n_gen] <= g_max_cap[i]) # Generator Capacity Limits, do we want to make this an integer variable? or maybe make the lines integer variables?

@objective(ref_model, Min, sum(g_opex[i]*g[i,t] for i=1:n_gen, t=1:t_steps) + sum(g_capex[i] * Capacity_gen[i] for i=1:n_gen))

@constraint(ref_model, gen_op_lim[i=1:n_gen,t=1:t_steps], g[i,t] <= Capacity_gen[i])

@constraint(ref_model, demand_balance_n1[t=1:t_steps],
    g[1,t] + g[2,t] - demand[1][t] + f[1,t] == 0)

@constraint(ref_model, demand_balance_n2[t=1:t_steps],
    g[3,t] - demand[2][t] - f[1,t] == 0)

optimize!(ref_model)

if termination_status(ref_model) == MOI.OPTIMAL
    println("Reference LP optimal!  obj = ", objective_value(ref_model))
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
    println("Reference LP failed: ", termination_status(ref_model))
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

# ── Kelley formulation ───────────────────────────────────────────
# Decision vector x layout (length n):
#   generation g[i,t]        → x[g_idx(i,t)]    for i=1:n_gen,   t=1:t_steps
#   line flow f[l,t]         → x[f_idx(l,t)]    for l=1:n_lines, t=1:t_steps
#   balance epigraph θ[nd,t] → x[epi_idx(nd,t)] for nd=1:n_nodes,t=1:t_steps
alg   = "kelley"
n_epi = n_nodes * t_steps
n     = t_steps*n_gen + t_steps*n_lines + n_epi

g_idx(i, t)    = (t-1)*n_gen   + i
f_idx(l, t)    = t_steps*n_gen + (t-1)*n_lines + l
epi_idx(nd, t) = t_steps*n_gen + t_steps*n_lines + (t-1)*n_nodes + nd

# Bounds: gens in [0, g_op_max], flows in [-f_lim, f_lim], epigraphs ≥ 0
x_lb = zeros(n)
x_ub = Vector{Float64}(undef, n)
for i in 1:n_gen
    for t in 1:t_steps
        x_ub[g_idx(i, t)] = g_op_max[i]
    end
end
for l in 1:n_lines
    for t in 1:t_steps
        x_lb[f_idx(l, t)] = -f_lim
        x_ub[f_idx(l, t)] =  f_lim
    end
end
for nd in 1:n_nodes
    for t in 1:t_steps
        x_ub[epi_idx(nd, t)] = 1e8   # epigraph UB: just needs to be large
    end
end

# Linear objective: generator cost + large penalty on balance epigraphs
BAL_PENALTY = 1e6
cost = zeros(n)
for i in 1:n_gen
    for t in 1:t_steps
        cost[g_idx(i, t)] = g_opex[i]
    end
end
for nd in 1:n_nodes
    for t in 1:t_steps
        cost[epi_idx(nd, t)] = BAL_PENALTY
    end
end

functionAndGradient(x) = (cost' * x, copy(cost))

# Indexed demand-balance constraints
#   node 1, t : g[1,t] + g[2,t] + f[1,t] − demand[1][t] == 0
#   node 2, t : g[3,t]          − f[1,t] − demand[2][t] == 0
balance_indices = [(nd, t) for nd in 1:n_nodes for t in 1:t_steps]

function balance_oracle(x, idx)
    (nd, t) = idx
    gv = zeros(n)
    if nd == 1 # node 1
        v = x[g_idx(1,t)] + x[g_idx(2,t)] + x[f_idx(1,t)] - demand[1][t]
        gv[g_idx(1,t)] =  1.0
        gv[g_idx(2,t)] =  1.0
        gv[f_idx(1,t)] =  1.0
    else  # nd == 2
        v = x[g_idx(3,t)] - x[f_idx(1,t)] - demand[2][t]
        gv[g_idx(3,t)] =  1.0
        gv[f_idx(1,t)] = -1.0   # negative injection due to withdraw from line 1
    end
    return (v, gv, epi_idx(nd, t))
end

balance_kind = :eq

dispatch(balance_indices = balance_indices,
         balance_oracle  = balance_oracle,
         balance_kind    = balance_kind)
