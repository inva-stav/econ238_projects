# Problem 4: min ‖x‖² over x ∈ [-0.5, 1]ⁿ, n ∈ {2, 5, 10, 100, 500}
include(joinpath(@__DIR__, "algorithms.jl"))

prob_num = 4
alg      = "both"   # run Kelley and subgradient for each n

ns = [2, 5, 10, 100, 500]

scaling_kelley     = DataFrame(n=Int[], Iterations=Int[], CPUTime_s=Float64[])
scaling_subgradient = DataFrame(n=Int[], Iterations=Int[], CPUTime_s=Float64[])

for n_val in ns
    global n, x_lb, x_ub, functionAndGradient, out_dir

    n    = n_val
    x_lb = fill(-0.5, n)
    x_ub = fill(1.0,  n)
    functionAndGradient(x) = (x'x, 2 .* x)

    out_dir = joinpath(@__DIR__, "results", "problem$(prob_num)", "n$(n)")
    mkpath(out_dir)

    println("\n" * "="^50)
    println("Problem 4 — n = $n")
    println("="^50)

    dispatch(ub_tol=1e-6)   # f* = 0 for ‖x‖²; stop subgradient once UB ≈ 0

    # Collect scaling data for both algorithms
    for (a, df_scaling) in (("kelley", scaling_kelley), ("subgradient", scaling_subgradient))
        csv_path = joinpath(out_dir, "summary_$(a).csv")
        if isfile(csv_path)
            df = CSV.read(csv_path, DataFrame)
            push!(df_scaling, (n, df.Iterations[1], df.CPUTime_s[1]))
        end
    end
end

results_dir = joinpath(@__DIR__, "results", "problem$(prob_num)")

# Scaling plots: iterations vs n (both methods on same axes)
p_iter = plot(xlabel="n", ylabel="Iterations", xscale=:log10,
              title="Iterations vs n — P4", legend=:topleft)
nrow(scaling_kelley)      > 0 && plot!(p_iter, scaling_kelley.n,      scaling_kelley.Iterations,      marker=:circle, lw=2, label="Kelley")
nrow(scaling_subgradient) > 0 && plot!(p_iter, scaling_subgradient.n, scaling_subgradient.Iterations, marker=:square, lw=2, label="Subgradient")
savefig(p_iter, joinpath(results_dir, "scaling_iterations.png"))
Plots.closeall()

# CPU time vs n (log-log)
p_cpu = plot(xlabel="n", ylabel="CPU time (s)", xscale=:log10, yscale=:log10,
             title="CPU time vs n — P4", legend=:topleft)
nrow(scaling_kelley)      > 0 && plot!(p_cpu, scaling_kelley.n,      scaling_kelley.CPUTime_s,      marker=:circle, lw=2, label="Kelley")
nrow(scaling_subgradient) > 0 && plot!(p_cpu, scaling_subgradient.n, scaling_subgradient.CPUTime_s, marker=:square, lw=2, label="Subgradient")
savefig(p_cpu, joinpath(results_dir, "scaling_cpu.png"))
Plots.closeall()

println("\nScaling summary — Kelley:")
println(scaling_kelley)
println("\nScaling summary — Subgradient:")
println(scaling_subgradient)
