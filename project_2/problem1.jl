include("algorithms.jl")
include("save_outputs.jl")
using LinearAlgebra
using Printf

###################################
### P1 data setup
### n=2, T=2, perfect anti-correlation, hardcoded INV
###################################

function run_p1(; save::Bool = true)
    println("\n" * "="^60)
    println("P1: n = 2,  T = 2,  perfect anti-correlation")
    println("="^60)

    # Fixed problem data (calibrated so C({1})≈90, C({2})≈100, C({1,2})≈120)
    n   = 2
    N   = [1, 2]
    T   = [1, 2]
    g   = Matrix{Float64}(I, n, n)  # g[i,t] = 1 if i==t, 0 otherwise
    L   = [(1, 0), (2, 0), (1, 2)]
    INV = Dict((1, 0) => 90.0, (2, 0) => 100.0, (1, 2) => 50.0)
    P   = 0.0

    # Compute coalition costs
    println("\nComputing $(2^n) coalition costs...")
    C   = compute_all_costs(n, N, T, g, L, INV; P=P)
    C_N = C[collect(1:n)]

    # Print coalition cost table
    sorted_coalitions = sort(collect(keys(C)), by = s -> (length(s), s))
    println("\nCoalition cost table:")
    @printf("  %-20s  %10s  %10s\n", "members", "C(s)", "C(s)/C(N)")
    println("  " * "-"^46)
    for s in sorted_coalitions
        @printf("  %-20s  %10.4f  %10.6f\n", string(s), C[s], C[s] / C_N)
    end

    # Compute nucleolus
    println("\nRunning sequential LP nucleolus:")
    x_star = nucleolus_sequential_lp(n, C)

    # Print nucleolus allocation
    println("\nNucleolus allocation x*:")
    @printf("  %-8s  %10s  %10s\n", "player", "x*", "x*/C(N)")
    println("  " * "-"^34)
    for i in 1:n
        @printf("  %-8d  %10.4f  %10.6f\n", i, x_star[i], x_star[i] / C_N)
    end
    @printf("  C(N) = %.4f,  sum(x*) = %.4f\n", C_N, sum(x_star))

    # Save outputs
    if save
        println("\nSaving outputs...")
        dir = joinpath("results", "problem1")
        save_metadata(; n=n, T=length(T), seed=nothing, C_N=C_N, problem_dir=dir)
        save_line_costs(INV, dir)
        save_coalition_costs(C, C_N, dir)
        save_nucleolus(x_star, C_N, dir)
    end

    return x_star, C
end

################# Written explanation of the solution process ###
# Both players win as their excess is each 35, with player 1's cost falling 
# from 90 to 55 and player 2's cost falling from 100 to 65. In a two 
# player game, the excess is always equal. The perfect anti-correlation 
# in generation leads to the grand coalition only needing 0.5 MW of capacity on each line. 