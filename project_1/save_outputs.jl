# Persist results from an algorithm run: summary CSV, iteration CSV,
# convergence plot, and (for n=1) a function + cutting-planes plot.
# Reads globals: out_dir, prob_num, n, x_lb, x_ub, functionAndGradient.

using Plots
using CSV
using DataFrames

function save_outputs(X, F, G, LB, UB, x_best, k, cpu_time, alg; logscale::Bool=false)
    # Summary table
    CSV.write(joinpath(out_dir, "summary_$(alg).csv"),
        DataFrame(Algorithm=[alg], FinalObjective=[UB[end]], FinalGap=[UB[end]-LB[end]],
                  Iterations=[k], CPUTime_s=[round(cpu_time, digits=4)]))

    # Iteration data CSV
    CSV.write(joinpath(out_dir, "iteration_data_$(alg).csv"),
        DataFrame(Iteration=1:k, LowerBound=LB, UpperBound=UB, OracleValue=F, Gap=UB.-LB))

    # Best-iterate decision vector (x*) — mirrors capacities CSV from HiGHS run.
    CSV.write(joinpath(out_dir, "capacities_$(alg).csv"),
        DataFrame(Index=1:length(x_best), Value=collect(x_best)))

    # Convergence plot: UB/LB on left axis, gap on right axis (approaches 0).
    ks = 2:k
    yscale = logscale ? :log10 : :identity
    # On log scale, non-positive values become NaN so Plots just leaves gaps.
    safe(ys) = logscale ? [y > 0 ? y : NaN for y in ys] : ys
    p_conv = plot(ks, safe(UB[2:end]), label="UB", lw=2, color=1,
                  xlabel="Iteration", ylabel="Objective", yscale=yscale,
                  title="Convergence — $(titlecase(alg)) P$(prob_num)", legend=:topleft)
    plot!(p_conv, ks, safe(LB[2:end]), label="LB", lw=2, color=2)
    p_right = twinx(p_conv)
    plot!(p_right, ks, safe((UB .- LB)[2:end]), label="Gap", lw=2,
          linestyle=:dash, color=:red, yscale=yscale, ylabel="Gap", legend=:topright)
    savefig(p_conv, joinpath(out_dir, "convergence_$(alg).png"))
    Plots.closeall()

    # Univariate function + cutting planes plot (n=1 only)
    if n == 1
        x_range = range(x_lb[1] - 0.1, x_ub[1] + 0.1, length=200)
        p = plot(x_range, [functionAndGradient([xi])[1] for xi in x_range],
                 label="f(x)", lw=2, color=:black,
                 xlabel="x", ylabel="f(x)", title="$(titlecase(alg)) — P$(prob_num)",
                 legend=:outertopright, size=(800, 400))
        for i in eachindex(F)
            plot!(p, x_range, [F[i] + G[i]' * ([xi] .- X[i]) for xi in x_range],
                  label=false, linestyle=:dash, alpha=0.5)
        end
        scatter!(p, [xi[1] for xi in X], F, label="Test points", color=:red, marker=:circle)
        for i in eachindex(X)
            annotate!(p, X[i][1], F[i] + 0.15, text("$i", 8, :center, :red))
        end
        redirect_stderr(devnull) do
            scatter!(p, [x_best[1]], [UB[end]], label="x*", color=:green, marker=:star5, ms=8)
        end
        savefig(p, joinpath(out_dir, "function_plot_$(alg).png"))
        Plots.closeall()
    end

    println("Saved to $(out_dir)/  |  f(x*) = $(UB[end])  gap = $(UB[end]-LB[end])  k = $k")
end
