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

        if has_balance
            for idx in balance_indices
                add_balance_cut!(idx, x_k)
            end
        end
    end

    save_outputs(X, F, G, LB, UB, x_best, k, time()-t_start, "kelley")
end

# ── Projected subgradient algorithm ─────────────────────────────
# step_rule = :polyak  →  α_k = max((f_k − UB) / ‖g_k‖², α₀/√k)
#           = :diminishing →  α_k = α₀ / √k
# Subgradient gives no LB, so LB stays at −1e6 throughout
# and the run terminates only when k reaches MaxIteration.
function run_subgradient(; tol=1e-4, MaxIteration=1000, α₀=1.0, step_rule=:polyak,
                           ub_tol=-Inf)
    t_start = time()
    k   = 1
    x_k = copy(x_lb)
    f_k, g_k = functionAndGradient(x_k)

    x_best = copy(x_k)
    LB = [-1.0e6]
    UB = [f_k]
    F  = [f_k];  G = [g_k];  X = [x_k]

    # ub_tol: stop early when UB ≤ ub_tol (e.g. 1e-6 when f*=0); -Inf disables.
    while (UB[end] - LB[end] > tol) && (k < MaxIteration) && (UB[end] > ub_tol)
        # Polyak step using current best UB as f* estimate; fall back to
        # α₀/√k when the current point already achieves the best UB (step = 0).
        g_sq = g_k'g_k
        if step_rule == :polyak && g_sq > 1e-14
            α_k = max((f_k - UB[end]) / g_sq, α₀ / sqrt(k))
        else
            α_k = α₀ / sqrt(k)
        end

        # subgradient step and projection onto box X
        x_k = clamp.(x_k .- α_k .* g_k, x_lb, x_ub)
        k  += 1
        f_k, g_k = functionAndGradient(x_k)

        push!(LB, -1.0e6)
        if f_k < UB[end]; x_best = copy(x_k); end
        push!(UB, min(UB[end], f_k))
        push!(F, f_k); push!(G, g_k); push!(X, x_k)

        # suppresses print statements for n > 4, otherwise it'll be too much
        if n <= 4
            println("k=$(lpad(k,4))  x_k=$(round.(x_k,digits=5))  UB=$(round(UB[end],digits=6))  α=$(round(α_k,digits=6))")
        else
            println("k=$(lpad(k,4))  UB=$(round(UB[end],digits=6))  α=$(round(α_k,digits=6))")
        end
    end

    save_outputs(X, F, G, LB, UB, x_best, k, time()-t_start, "subgradient")
end

# ── Dispatch helper ──────────────────────────────────────────────
# Each problem file calls this after defining its globals.
# ub_tol is forwarded to run_subgradient for early stopping when f*=0.
function dispatch(; balance_indices=nothing, balance_oracle=nothing, balance_kind=:eq,
                    ub_tol=-Inf)
    if alg in ("kelley", "both")
        run_kelley(logscale=logscale, tol=tol)
    end
    if alg in ("subgradient", "both"); run_subgradient(ub_tol=ub_tol); end
end
