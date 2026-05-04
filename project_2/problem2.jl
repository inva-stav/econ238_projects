include("algorithms.jl")
include("save_outputs.jl")
using Random
using LinearAlgebra
using Printf

###################################
### P2 data generation
### n players, T=n timesteps, Euclidean INV, seed=238
###################################

function generate_data(n::Int; seed::Int = 238)
    # Set the random seed for reproducibility
    Random.seed!(seed)

    # Nodes with renewables (excluding substation at 0)
    N = collect(1:n)
    # Timesteps
    T = collect(1:n)

    # Generation matrix: g[i,t] = 1.0 if i==t, 0.0 otherwise (identity matrix = perfect anticorrelation)
    g = Matrix{Float64}(I, n, n)

    # Node positions: substation node 0 fixed at origin, others random in 100x100 square
    pos = Dict{Int, Tuple{Float64,Float64}}(0 => (0.0, 0.0))
    for i in N
        pos[i] = (rand() * 100, rand() * 100)
    end

    # All nodes including substation 0
    all_nodes = collect(0:n)
    # Line set: all unordered pairs (i, j) for i < j
    L = [(i, j) for i in all_nodes for j in all_nodes if i < j]
    # Investment cost per MW: Euclidean distance between nodes
    INV = Dict(
        (i, j) => sqrt((pos[j][1]-pos[i][1])^2 + (pos[j][2]-pos[i][2])^2)
        for (i, j) in L
    )

    return N, T, g, L, INV, pos
end

###################################
### P2 runner
###################################

function run_p2(n::Int; save::Bool = true)
    println("\n" * "="^60)
    println("P2: n = $n,  T = $n,  seed = 238")
    println("="^60)

    N, T, g, L, INV, pos = generate_data(n)

    println("\nNode positions (node 0 fixed at origin):")
    for i in 0:n
        @printf("  node %2d  (%.2f, %.2f)\n", i, pos[i][1], pos[i][2])
    end

    println("\nLine investment costs (\$/MW = Euclidean distance):")
    for (l, v) in sort(collect(INV), by = x -> x[1])
        @printf("  (%2d,%2d)  %.4f\n", l[1], l[2], v)
    end

    println("\nComputing $(2^n) coalition costs...")
    C   = compute_all_costs(n, N, T, g, L, INV; verbose = (n >= 10))
    C_N = C[collect(1:n)]

    # Sort coalitions by size then lexicographically for a clean table
    sorted_coalitions = sort(collect(keys(C)), by = s -> (length(s), s))

    println("\nCoalition cost table:")
    @printf("  %-20s  %10s  %10s\n", "members", "C(s)", "C(s)/C(N)")
    println("  " * "-"^46)
    for s in sorted_coalitions
        @printf("  %-20s  %10.4f  %10.6f\n", string(s), C[s], C[s] / C_N)
    end

    println("\nRunning sequential LP nucleolus:")
    x_star = nucleolus_sequential_lp(n, C)

    println("\nNucleolus allocation x*:")
    @printf("  %-8s  %10s  %10s\n", "player", "x*", "x*/C(N)")
    println("  " * "-"^34)
    for i in 1:n
        @printf("  %-8d  %10.4f  %10.6f\n", i, x_star[i], x_star[i] / C_N)
    end
    @printf("  C(N) = %.4f,  sum(x*) = %.4f\n", C_N, sum(x_star))

    if save
        println("\nSaving outputs...")
        dir = joinpath("results", "problem2", "n$n")
        save_metadata(; n=n, T=n, seed=238, C_N=C_N, problem_dir=dir)
        save_node_positions(pos, dir)
        save_line_costs(INV, dir)
        save_coalition_costs(C, C_N, dir)
        save_nucleolus(x_star, C_N, dir)
    end

    return x_star, C
end


################# Written explanation of the solution process ###
# In both the n=3 and n=10 cases, the grand coalition cost (122.28 and 184.50 respectively) is well
# below the sum of standalone costs, because perfect anti-correlation lets all generators share
# transmission capacity. For n=3, the nucleolus allocates costs as x*=(52.81, 55.45, 14.01): player 3
# pays the least because it is closest to the substation (standalone cost only 26.70), while players 1
# and 2 split the bulk of the cost reflecting their greater distance. For n=10, the cost signal is
# highly location-dependent, with generators 6, 4, 9, and 8 paying the largest shares due to their
# position in the network. In both cases the first sequential-LP floor (ε*=12.69 for n=3, ε*=11.81
# for n=10) guarantees every proper coalition is strictly better off in the grand coalition than alone.