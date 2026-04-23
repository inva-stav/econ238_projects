# Problem 2: min x² over x ∈ [1.0, 2.0]
include(joinpath(@__DIR__, "algorithms.jl"))

prob_num = 2
alg      = "both"

out_dir  = joinpath(@__DIR__, "results", "problem$(prob_num)")
mkpath(out_dir)

n    = 1
x_lb = [1.0]
x_ub = [2.0]
functionAndGradient(x) = (x'x, 2*x)

dispatch()
