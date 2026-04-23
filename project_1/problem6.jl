# Problem 6: 2-node power system dispatch via Kelley with balance-epigraph cuts.
# Also solves the direct LP as a reference benchmark.

include(joinpath(@__DIR__, "algorithms.jl"))

# ── Power system data ───────────────────────────────────────────
n_gen    = 3
n_nodes  = 2
n_lines  = 1

n_demand = 2
t_steps  = 600 #can modify to make problem larger

g_opex   = [40, 30, 45]
g_capex = [60, 325, 55]

f_lim = 100

# Shared optimality/feasibility tolerance — used for the HiGHS reference solve,
# the inner dispatch oracle, and Kelley's UB–LB gap stopping criterion.
OPT_TOL = 1e-5

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
set_optimizer_attribute(expansion_model, "primal_feasibility_tolerance", OPT_TOL)
set_optimizer_attribute(expansion_model, "dual_feasibility_tolerance",   OPT_TOL)
set_optimizer_attribute(expansion_model, "ipm_optimality_tolerance",     OPT_TOL)

@variable(expansion_model, g[i=1:n_gen, t=1:t_steps] >= 0)
@variable(expansion_model, -1 * f_lim <= f[l=1:n_lines, t=1:t_steps] <= f_lim)
@variable(expansion_model, 0 <= Capacity_gen[i=1:n_gen] <= 1e4) # Generator Capacity Limits, do we want to make this an integer variable? or maybe make the lines integer variables?

@objective(expansion_model, Min, sum(g_opex[i]*g[i,t] for i=1:n_gen, t=1:t_steps) + sum(g_capex[i] * Capacity_gen[i] for i=1:n_gen))

@constraint(expansion_model, gen_op_lim[i=1:n_gen,t=1:t_steps], g[i,t] <= Capacity_gen[i])

@constraint(expansion_model, demand_balance_n1[t=1:t_steps],
    g[1,t] + g[2,t] - demand[1][t] + f[1,t] == 0)

@constraint(expansion_model, demand_balance_n2[t=1:t_steps],
    g[3,t] - demand[2][t] - f[1,t] == 0)

highs_time = @elapsed optimize!(expansion_model)

prob_num = "6highs"
out_dir  = joinpath(@__DIR__, "results", "problem$(prob_num)")
mkpath(out_dir)

if termination_status(expansion_model) == MOI.OPTIMAL
    obj_val    = objective_value(expansion_model)
    g_values   = value.(g)
    f_values   = value.(f)
    cap_values = value.(Capacity_gen)
    capex_cost = sum(g_capex[i] * cap_values[i] for i in 1:n_gen)
    opex_cost  = obj_val - capex_cost

    println("Reference LP optimal!  obj = ", obj_val, "  CPU = ", round(highs_time, digits=4), "s")
    println("\nGenerator outputs (g):")
    for i in 1:n_gen; println("  Generator $(i): ", g_values[i,:]); end
    println("\nLine flows (f):")
    for l in 1:n_lines; println("  Line $(l): ", f_values[l,:]); end
    println("\nInstalled capacities: ", cap_values)

    # Summary — same columns as Kelley's summary CSV.
    CSV.write(joinpath(out_dir, "summary_highs.csv"),
        DataFrame(Algorithm=["highs"], FinalObjective=[obj_val],
                  FinalGap=[0.0], Iterations=[1],
                  CPUTime_s=[round(highs_time, digits=4)],
                  CapexCost=[capex_cost], OpexCost=[opex_cost]))

    # Installed capacities (the decision variable x* in Kelley terms).
    CSV.write(joinpath(out_dir, "capacities_highs.csv"),
        DataFrame(Generator=1:n_gen, Capacity=collect(cap_values),
                  Capex=g_capex, Opex=g_opex))

    # Hourly dispatch.
    dispatch_df = DataFrame(Hour=1:t_steps)
    for i in 1:n_gen;   dispatch_df[!, "Gen$(i)"]  = g_values[i, :]; end
    for l in 1:n_lines; dispatch_df[!, "Flow$(l)"] = f_values[l, :]; end
    CSV.write(joinpath(out_dir, "dispatch_highs.csv"), dispatch_df)
else
    println("Reference LP failed: ", termination_status(expansion_model))
end

if @isdefined(g_values)
    time_steps = 1:t_steps
    p = plot(xlabel="Time Step", ylabel="Generation (MW)",
             title="Generator Dispatch — HiGHS Reference", legend=:outertopright)
    for i in 1:n_gen
        color = palette(:tab10)[i]
        plot!(p, time_steps, g_values[i,:], label="Gen $(i)", lw=2, color=color)
        hline!(p, [cap_values[i]], label="Gen $(i) capacity",
               linestyle=:dash, color=color)
    end
    savefig(p, joinpath(out_dir, "dispatch_highs.png"))
    Plots.closeall()
end

# ── Kelley formulation (Benders-style) ──────────────────────────
# Outer decision x = installed generator capacities  (length n = n_gen)
# f(x) = g_capex' x + Q(x)
#   where Q(x) = min dispatch cost s.t. g[i,t] <= x[i] + balance + flow limits
# Envelope theorem:  ∂Q/∂x[i] = sum_t dual(gen_op_lim[i,t])   (≤ 0 in JuMP's
# convention for a ≤-constraint in a min problem).
n   = n_gen

x_lb = fill(0.0, n)
x_ub = fill(1.0e6, n)   # loose upper bound on capacity

# ── Parameterized inner dispatch LP (built once) ────────────────
# Slack s[nd,t] with VOLL penalty keeps the LP feasible at any x ≥ 0.
VOLL = 1.0e6
oracle_model = Model(HiGHS.Optimizer)
set_silent(oracle_model)
set_optimizer_attribute(oracle_model, "primal_feasibility_tolerance", OPT_TOL)
set_optimizer_attribute(oracle_model, "dual_feasibility_tolerance",   OPT_TOL)
set_optimizer_attribute(oracle_model, "ipm_optimality_tolerance",     OPT_TOL)

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
        dual_sum = sum(dual(gen_op_lim[i,t]) for t in 1:t_steps)
        # println("Duals for gen_op_lim[$i,t]: ", [dual(gen_op_lim[i,t]) for t in 1:t_steps])
        grad[i] = g_capex[i] + dual_sum
    end
    # println("Function value: f(x)=$(round(f_val,digits=1))")
    # println("Gradient components: ", grad)
    return f_val, grad
end

alg = "both"
prob_num = "6$(alg)"
out_dir  = joinpath(@__DIR__, "results", "problem$(prob_num)")
mkpath(out_dir)

dispatch(logscale = true, tol = OPT_TOL)

# ── HiGHS vs Kelley vs Subgradient comparison ──────────────────
kelley_caps = CSV.read(joinpath(out_dir, "capacities_kelley.csv"), DataFrame).Value
functionAndGradient(kelley_caps)         # re-populate oracle_model at x*
kelley_g = value.(g_d)
kelley_f = value.(f_d)

subgrad_caps = CSV.read(joinpath(out_dir, "capacities_subgradient.csv"), DataFrame).Value
functionAndGradient(subgrad_caps)
subgrad_g = value.(g_d)
subgrad_f = value.(f_d)

compare_dir = joinpath(@__DIR__, "results", "problem6compare")
mkpath(compare_dir)

highs_caps = collect(cap_values)
CSV.write(joinpath(compare_dir, "capacities_compare.csv"),
    DataFrame(Generator=1:n_gen, HiGHS=highs_caps, Kelley=kelley_caps,
              Subgradient=subgrad_caps,
              AbsDiff_Kelley=abs.(highs_caps .- kelley_caps),
              AbsDiff_Subgradient=abs.(highs_caps .- subgrad_caps)))

# Capacity bar chart (side-by-side).
p_cap = bar((1:n_gen) .- 0.25, highs_caps, bar_width=0.25, label="HiGHS",
            xlabel="Generator", ylabel="Capacity (MW)",
            title="Installed Capacity — HiGHS vs Kelley vs Subgradient",
            xticks=(1:n_gen, ["Gen $(i)" for i in 1:n_gen]),
            legend=:topleft)
bar!(p_cap, (1:n_gen), kelley_caps, bar_width=0.25, label="Kelley")
bar!(p_cap, (1:n_gen) .+ 0.25, subgrad_caps, bar_width=0.25, label="Subgradient")
savefig(p_cap, joinpath(compare_dir, "capacity_compare.png"))
Plots.closeall()

# Dispatch overlay: solid = HiGHS, dashed = Kelley, dotted = Subgradient.
ts = 1:t_steps
p_dis = plot(xlabel="Time Step", ylabel="Generation (MW)",
             title="Dispatch — HiGHS (solid), Kelley (dashed), Subgradient (dotted)",
             legend=:outertopright)
for i in 1:n_gen
    color = palette(:tab10)[i]
    plot!(p_dis, ts, g_values[i,:],   label="Gen $(i) HiGHS",       lw=2, color=color)
    plot!(p_dis, ts, kelley_g[i,:],   label="Gen $(i) Kelley",      lw=2, color=color, linestyle=:dash)
    plot!(p_dis, ts, subgrad_g[i,:],  label="Gen $(i) Subgradient", lw=2, color=color, linestyle=:dot)
end
savefig(p_dis, joinpath(compare_dir, "dispatch_compare.png"))
Plots.closeall()

println("Comparison written to $(compare_dir)/")
