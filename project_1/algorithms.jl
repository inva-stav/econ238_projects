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


        # suppresses print statements for n > 4, otherwise it'll be too much
        if n <= 4
            println("k=$(lpad(k,4))  x_k=$(round.(x_k,digits=3))  LB=$(round(LB[end],digits=3))  UB=$(round(UB[end],digits=3))  gap=$(round(UB[end]-LB[end],digits=3))")
        else
            println("k=$(lpad(k,4))  LB=$(round(LB[end],digits=3))  UB=$(round(UB[end],digits=3))  gap=$(round(UB[end]-LB[end],digits=3))")
        end


        push!(F, f_k); push!(G, g_k); push!(X, x_k)
        @constraint(model, θ >= F[end] + G[end]' * (x .- X[end]))
    end

    save_outputs(X, F, G, LB, UB, x_best, k, time()-t_start, "kelley"; logscale=logscale)
end

# ── Projected subgradient algorithm ─────────────────────────────
# step_rule = :polyak  →  α_k = max((f_k − UB) / ‖g_k‖², α₀/√k)
#           = :diminishing →  α_k = α₀ / √k
# Subgradient gives no LB, so LB stays at −1e6 throughout
# and the run terminates only when k reaches MaxIteration.
function run_subgradient_polyak(; tol=1e-4, MaxIteration=1000, α₀=1.0, step_rule=:polyak, ub_tol=-Inf)
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


# ── Subgradient algorithm (diminishing step, used by problem 6) ─
function run_subgradient_diminishing(; tol=1e-4, MaxIteration=2000)
    # write large print line to signify start of subgradient algorithm
    println("\n" * "="^80)
    println("Starting subgradient algorithm with tol=$(tol) and MaxIteration=$(MaxIteration)")
    println("="^80 * "\n")
    N_DIGITS_PRINT = 1  # for rounding in print statements
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
    
    # Start timing the algorithm
    t_start = time()
    
    # ── Step 1: Initialize ──────────────────────────────────────
    # Start at the lower bound of the feasible region
    x_k = copy(x_lb)  # Midpoint initialization (can also try x_lb or random)
    
    # Evaluate the function and gradient at the initial point
    f_k, g_k = functionAndGradient(x_k)
    println("Function value at initial point: f(x_k)=$(round(f_k,digits=N_DIGITS_PRINT))")
    println("Gradient at initial point: g_k=$(round.(g_k,digits=N_DIGITS_PRINT))")
    
    # Initialize the best solution found so far
    # Upper bound (UB) is the best objective value we've seen
    x_best = copy(x_k)
    f_best = f_k
    
    # Storage for iteration history
    # X: points visited, F: function values, G: (sub)gradients
    X = [copy(x_k)]
    F = [f_k]
    G = [copy(g_k)]
    
    # Upper bound track: best objective found so far
    UB = [f_k]
    
    # Lower bound track: subgradient method doesn't provide clean LB,
    # so we keep it at a sentinel value for compatibility with plotting
    LB = [-1.0e6]
    
    # ── Step 2: Choose step-size rule parameters ────────────────
    # We'll use a diminishing step size: α_k = α₀ / sqrt(k)
    # This guarantees convergence for convex functions
    α₀ = 1e-1  # Initial step size (tune this if needed)
    
    # Alternative: Polyak step size (commented out, but you can try it)
    # Requires an estimate of the optimal value f_opt
    # α_k = (f_k - f_opt) / ||g_k||²
    
    # Iteration counter (starts at 1 since we've already done initialization)
    k = 1
    
    # ── Step 3: Main iteration loop ─────────────────────────────

    while k < MaxIteration
        # Increment iteration counter
        k += 1
        
        # ── (a) Compute step size ───────────────────────────────
        # Diminishing step size rule: α_k = α₀ / sqrt(k)
        α_k = α₀ / sqrt(k)
        
        # ── (b) Take a subgradient step ─────────────────────────
        # Move in the negative subgradient direction (descent)
        # For minimization: x_new = x_old - α * g
        # println("x_k=$(round.(x_k,digits=N_DIGITS_PRINT))  g_k=$(round.(g_k,digits=N_DIGITS_PRINT))  α_k=$(round(α_k,digits=N_DIGITS_PRINT))  ")
        clamped_g_k = clamp.(g_k, -1.0e2, 1.0e2)  # Optional: prevent extreme steps from huge gradients
        y = x_k - α_k * clamped_g_k
        
        # ── (c) Project back onto feasible region ───────────────
        # The feasible region is a box: [x_lb, x_ub]
        # Projection is simply clamping each coordinate
        # println("y=$(round.(y,digits=N_DIGITS_PRINT))  ")
        x_next = clamp.(y, x_lb, x_ub)
        # println("x_next=$(round.(x_next,digits=N_DIGITS_PRINT))  ")
        
        # ── (d) Evaluate function at new point ──────────────────
        f_next, g_next = functionAndGradient(x_next)
        
        # ── (e) Update best solution found ──────────────────────
        # Subgradient method doesn't monotonically decrease objective,
        # so we track the best value seen so far
        if f_next < f_best
            f_best = f_next
            x_best = copy(x_next)
        end
        
        # ── (f) Store iteration data ────────────────────────────
        # IMPORTANT: Push to all arrays to keep them synchronized
        push!(X, copy(x_next))
        push!(F, f_next)
        push!(G, copy(g_next))
        push!(UB, f_best)  # UB is the best objective seen so far
        push!(LB, -1.0e6)  # No meaningful LB from subgradient method
        
        # ── (g) Print progress ──────────────────────────────────
        g_norm = sqrt(sum(g_next .^ 2))
        println("k=$(lpad(k,4))" *
                "f(x_k)=$(round(f_next,digits=N_DIGITS_PRINT))  " *
                "f_best=$(round(f_best,digits=N_DIGITS_PRINT))  " *
                "α_k=$(round(α_k,digits=6))  " *
                "||g||=$(round(g_norm,digits=N_DIGITS_PRINT))")
        
        # ── (h) Check stopping criteria ────────────────────────
        # Stop if gradient is very small (near stationary point)
        if sqrt(sum(g_next .^ 2)) < tol
            println("Stopped: gradient norm < tol")
            break
        end
        
        # Stop if step size becomes too small
        if α_k < 1e-5
            println("Stopped: step size too small")
            break
        end
        
        # Stop if objective hasn't improved in a while (optional)
        if k > 50 && abs(UB[end] - UB[end-50]) < tol
            println("Stopped: no improvement in 50 iterations")
            break
        end

        # Stop if f_best has been the same for many iterations (optional)
        if k > 50 && abs(f_best - UB[end-5]) < tol
            println("Stopped: f_best hasn't improved in 50 iterations")
            break
        end
        
        # ── (i) Prepare for next iteration ──────────────────────
        x_k = x_next
        f_k = f_next
        g_k = g_next
    end
    
    # ── Step 4: Save results ────────────────────────────────────
    # Record final iteration count and CPU time
    cpu_time = time() - t_start
    
    # Save all outputs (CSV files and plots)
    # k now correctly represents the number of iterations (including initial point)
    save_outputs(X, F, G, LB, UB, x_best, k, cpu_time, "subgradient")
    
    println("\nSubgradient algorithm completed:")
    println("  Best objective: $(f_best)")
    println("  Best solution:  $(x_best)")
    println("  Iterations:     $(k)")
    println("  CPU time:       $(round(cpu_time, digits=N_DIGITS_PRINT))s")
end

# ── Dispatch helper ──────────────────────────────────────────────
# Each problem file calls this after defining its globals.
# ub_tol is accepted for API compatibility (used by problem 4's f*=0 early stop);
# the current subgradient implementation doesn't consume it.
function dispatch(; logscale::Bool=false, tol=1e-4, ub_tol=-Inf)
    if alg in ("kelley", "both")
        run_kelley(logscale=logscale, tol=tol)
    end
    if alg in ("subgradient", "both")
        # Problem 6 uses the diminishing-step subgradient; all other problems
        # use the Polyak-step variant. prob_num may be a String ("6both") or Int.
        if startswith(string(prob_num), "6")
            run_subgradient_diminishing(tol=tol)
        else
            run_subgradient_polyak(tol=tol, ub_tol=ub_tol)
        end
    end
end
