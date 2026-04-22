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
function run_kelley(; tol=1e-4, MaxIteration=1000, logscale::Bool=false)
    t_start = time()
    k  = 1
    x1 = copy(x_lb)
    f1, g1 = functionAndGradient(x1)

    x_best = copy(x1)
    LB = [-1.0e7]
    UB = [f1]
    F  = [f1];  G = [g1];  X = [x1]

    model = Model(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, x_lb[i] <= x[i=1:n] <= x_ub[i])
    @variable(model, θ)
    @objective(model, Min, θ)

    # Initial cut on the main objective
    @constraint(model, θ >= F[1] + G[1]' * (x .- X[1]))

    println("k=$(lpad(k,4))")
    println("x_k=$(round.(x1,digits=3))")
    println("Objective Values: LB=$(round(LB[1],digits=3))  UB=$(round(UB[end],digits=3))  gap=$(round(UB[end]-LB[end],digits=3))")

    while (UB[end] - LB[end] > tol) && (k < MaxIteration)
        k += 1
        optimize!(model)
        x_k      = value.(x)
        θ_k      = objective_value(model)
        f_k, g_k = functionAndGradient(x_k)

        push!(LB, θ_k)
        if f_k < UB[end]; x_best = copy(x_k); end
        push!(UB, min(UB[end], f_k))

        println("k=$(lpad(k,4))")
        println("x_k=$(round.(x_k,digits=3))")
        println("Objective Values: LB=$(round(θ_k,digits=3))  UB=$(round(UB[end],digits=3))  gap=$(round(UB[end]-LB[end],digits=3))")

        push!(F, f_k); push!(G, g_k); push!(X, x_k)
        @constraint(model, θ >= F[end] + G[end]' * (x .- X[end]))

    end

    save_outputs(X, F, G, LB, UB, x_best, k, time()-t_start, "kelley"; logscale=logscale)
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
function run_algorithm(; logscale::Bool=false, tol=1e-4)
    if alg in ("kelley", "both")
        run_kelley(logscale=logscale,  tol=1e-4)
    end
    if alg in ("subgradient", "both"); run_subgradient( tol=1e-4); end
end
