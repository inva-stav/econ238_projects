# Problem 5: min max_i (aᵢ'x + bᵢ) over x ∈ [-10,10]ⁿ
#   n ∈ {1, 2, 10, 100}, m = 20, aᵢ ~ N(0,Iₙ), bᵢ ~ N(0,1)
include(joinpath(@__DIR__, "algorithms.jl"))

using Random

prob_num = 5
alg      = "both"

m_affine = 20
seed     = 42

for n_val in [1, 2, 10, 100]
    global n, x_lb, x_ub, out_dir

    n    = n_val
    x_lb = fill(-10.0, n)
    x_ub = fill( 10.0, n)

    out_dir = joinpath(@__DIR__, "results", "problem$(prob_num)", "n$(n)")
    mkpath(out_dir)

    rng = MersenneTwister(seed)
    A   = randn(rng, m_affine, n)   # each row is aᵢ'
    b   = randn(rng, m_affine)      # each entry is bᵢ

    global function functionAndGradient(x)
        vals = A * x .+ b           # m-vector of aᵢ'x + bᵢ
        i    = argmax(vals)
        return (vals[i], A[i, :])
    end

    println("\n===  Problem 5, n = $n  ===")
    dispatch()
end
