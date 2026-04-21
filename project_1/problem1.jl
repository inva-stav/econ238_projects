# Problem 1: min x² over x ∈ [-0.5, 1.0]
include(joinpath(@__DIR__, "algorithms.jl"))

prob_num = 1
alg      = "kelley"    # "kelley" | "subgradient" | "both"

out_dir  = joinpath(@__DIR__, "results", "problem$(prob_num)")
mkpath(out_dir)

n    = 1
x_lb = [-0.5]
x_ub = [1.0]
functionAndGradient(x) = (x'x, 2*x)

dispatch()
