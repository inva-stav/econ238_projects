###################################
### Master runner — reproduces all figures and tables for P1–P5.
### Usage:  julia runner.jl
###################################

# P1–P3 share algorithms.jl and save_outputs.jl via their own includes.
# The PROGRAM_FILE guards in each file prevent auto-execution on include.

include("problem1.jl")
include("problem2.jl")
include("problem3.jl")

println("\n" * "#"^60)
println("# P1: Warm-up (n=2, T=2, perfect anti-correlation)")
println("#"^60)
run_p1()

println("\n" * "#"^60)
println("# P2: Scaling (n=3 and n=10)")
println("#"^60)
run_p2(3)
run_p2(10)

println("\n" * "#"^60)
println("# P3: Correlation study (n=2, T=168)")
println("#"^60)
run_p3()

# P4 has its own includes (algorithms.jl, save_outputs.jl) so we
# include the sweep file which itself includes problem4.jl.
include("problem4_sweep.jl")

println("\n" * "#"^60)
println("# P4: Real renewables.ninja data + 12-month sweep")
println("#"^60)
run_p4(; save=true)       # single-month default (January)
run_p4_sweep(; save=true) # full 12-month sweep

# P5 includes problem5.jl (which includes algorithms.jl again — Julia
# deduplicates top-level function definitions, so this is safe).
include("problem5.jl")
include("problem5_synthetic_sweep.jl")

println("\n" * "#"^60)
println("# P5: Demand-side heterogeneity")
println("#"^60)
run_p5(; save=true)
run_p5_synthetic_sweep(; save=true)

# Python plotting scripts (network diagrams, coalition analysis, P4/P5 figures)
println("\n" * "#"^60)
println("# Generating plots (Python)")
println("#"^60)
for script in ["plot_networks.py", "plot_coalition_analysis.py", "plot_p4.py",
               "plot_p5.py", "plot_p5_demand_comparison.py"]
    if isfile(joinpath(@__DIR__, script))
        println("\n  running $script ...")
        run(`python3 $script`)
    else
        println("  skipping $script (not found)")
    end
end

println("\n" * "="^60)
println("All problems complete. Results in results/")
println("="^60)
