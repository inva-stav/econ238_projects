using CSV
using DataFrames
using Dates
using Printf

###################################
### Save outputs to results/ directory
###################################

function _ensure_dir(dir::String)
    mkpath(dir)
end

function save_metadata(; n, T, seed, C_N, problem_dir::String)
    _ensure_dir(problem_dir)
    df = DataFrame(
        key   = ["n", "T", "seed", "C_N", "timestamp"],
        value = [string(n), string(T), string(seed === nothing ? "none" : seed),
                 @sprintf("%.6f", C_N), string(now())]
    )
    path = joinpath(problem_dir, "metadata.csv")
    CSV.write(path, df)
    println("  saved: $path")
end

function save_node_positions(pos::Dict, problem_dir::String)
    _ensure_dir(problem_dir)
    nodes = sort(collect(keys(pos)))
    df = DataFrame(
        node = nodes,
        x    = [pos[i][1] for i in nodes],
        y    = [pos[i][2] for i in nodes]
    )
    path = joinpath(problem_dir, "node_positions.csv")
    CSV.write(path, df)
    println("  saved: $path")
end

function save_line_costs(INV::Dict, problem_dir::String)
    _ensure_dir(problem_dir)
    lines = sort(collect(keys(INV)))
    df = DataFrame(
        from_node = [l[1] for l in lines],
        to_node   = [l[2] for l in lines],
        inv_cost  = [INV[l] for l in lines]
    )
    path = joinpath(problem_dir, "line_costs.csv")
    CSV.write(path, df)
    println("  saved: $path")
end

function save_coalition_costs(C::Dict{Vector{Int},Float64}, C_N::Float64, problem_dir::String)
    _ensure_dir(problem_dir)
    # Sort coalitions by size then lexicographically for a clean table
    sorted = sort(collect(keys(C)), by = s -> (length(s), s))
    df = DataFrame(
        members        = [string(s) for s in sorted],
        C_s            = [C[s] for s in sorted],
        C_s_over_C_N   = [C[s] / C_N for s in sorted]
    )
    path = joinpath(problem_dir, "coalition_costs.csv")
    CSV.write(path, df)
    println("  saved: $path")
end

function save_nucleolus(x_star::Vector{Float64}, C_N::Float64, problem_dir::String)
    _ensure_dir(problem_dir)
    n  = length(x_star)
    df = DataFrame(
        player  = collect(1:n),
        x_star  = x_star,
        share   = x_star ./ C_N
    )
    path = joinpath(problem_dir, "nucleolus.csv")
    CSV.write(path, df)
    println("  saved: $path")
end
