# Shared algorithms: Kelley cutting-plane and projected subgradient.
# Each problem file sets globals (n, x_lb, x_ub, functionAndGradient,
# out_dir, prob_num, alg) before invoking run_kelley / run_subgradient.

using JuMP
using HiGHS
ENV["GKSwstype"] = "nul"

include(joinpath(@__DIR__, "save_outputs.jl"))

# ── Kelley cutting-plane algorithm ──────────────────────────────
# Optional kwargs let us add indexed balance-style epigraph constraints as
# cutting planes.  The epigraph *variables themselves* live inside x (baked
# into the problem definition, e.g. problem 6), so the objective is untouched.
#
#   balance_indices : iterable of identifiers, e.g. [(node, t) for node in 1:n_nodes, t in 1:t_steps]
#   balance_oracle  : function (x, idx) -> (value, gradient, j_epi)
#                       value     = residual c_idx(x)
#                       gradient  = ∇c_idx(x)           (length n)
#                       j_epi     = index into x of the epigraph variable for idx
#   balance_kind    : :eq for c(x) == 0  (cuts on both +c and -c; drives x[j] → 0)
#                     :le for c(x) ≤ 0   (single cut)
function run_kelley(; tol=1e-4, MaxIteration=1000,
                      balance_indices=nothing,
                      balance_oracle=nothing,
                      balance_kind=:eq)
    t_start = time()
    k  = 1
    x1 = copy(x_lb)
    f1, g1 = functionAndGradient(x1)

    x_best = copy(x1)
    LB = [-1.0e6]
    UB = [f1]
    F  = [f1];  G = [g1];  X = [x1]

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x_lb[i] <= x[i=1:n] <= x_ub[i])
    @variable(model, θ)
    @objective(model, Min, θ)

    # Initial cut on the main objective
    @constraint(model, θ >= F[1] + G[1]' * (x .- X[1]))

    has_balance = balance_indices !== nothing && balance_oracle !== nothing

    # Helper: add an epigraph cut for a single index at a given iterate
    add_balance_cut! = (idx, x_ref) -> begin
        v, gv, j = balance_oracle(x_ref, idx)
        @constraint(model, x[j] >=  v + gv' * (x .- x_ref))
        if balance_kind == :eq
            @constraint(model, x[j] >= -v - gv' * (x .- x_ref))
        end
    end

    # Initial cuts at x1
    if has_balance
        for idx in balance_indices
            add_balance_cut!(idx, x1)
        end
    end

    while (UB[end] - LB[end] > tol) && (k < MaxIteration)
        k += 1
        optimize!(model)
        x_k      = value.(x)
        θ_k      = objective_value(model)
        f_k, g_k = functionAndGradient(x_k)

        push!(LB, θ_k)
        if f_k < UB[end]; x_best = copy(x_k); end
        push!(UB, min(UB[end], f_k))

        println("k=$(lpad(k,4))  x_k=$(round.(x_k,digits=5))  LB=$(round(θ_k,digits=6))  UB=$(round(UB[end],digits=6))  gap=$(round(UB[end]-LB[end],digits=6))")

        push!(F, f_k); push!(G, g_k); push!(X, x_k)
        @constraint(model, θ >= F[end] + G[end]' * (x .- X[end]))

        if has_balance
            for idx in balance_indices
                add_balance_cut!(idx, x_k)
            end
        end
    end

    save_outputs(X, F, G, LB, UB, x_best, k, time()-t_start, "kelley")
end

# ── Subgradient algorithm (TODO) ────────────────────────────────
function run_subgradient(; tol=1e-4, MaxIteration=1000)
    # TODO: implement projected subgradient method
    # Outline:
    #   1. x1 = copy(x_lb); f1, g1 = functionAndGradient(x1); UB = f1; x_best = x1
    #   2. For k = 1, ..., MaxIteration:
    #        a. Gradient step:   y       = x_k - α_k * g_k
    #        b. Project onto X:  x_{k+1} = clamp.(y, x_lb, x_ub)
    #        c. f_{k+1}, g_{k+1} = functionAndGradient(x_{k+1})
    #        d. UB = min(UB, f_{k+1}); update x_best if improved
    #        e. Step size: e.g. α_k = α₀/√k  or Polyak step
    #   3. No clean lower bound — LB stays at -1e6
    #   4. Call save_outputs(..., "subgradient") when done
    error("Subgradient method not yet implemented")
end

# ── Dispatch helper ──────────────────────────────────────────────
# Each problem file calls this after defining its globals.
function dispatch(; balance_indices=nothing, balance_oracle=nothing, balance_kind=:eq)
    if alg in ("kelley", "both")
        run_kelley(balance_indices = balance_indices,
                   balance_oracle  = balance_oracle,
                   balance_kind    = balance_kind)
    end
    if alg in ("subgradient", "both"); run_subgradient(); end
end
