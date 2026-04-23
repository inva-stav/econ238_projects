# Problem 3: min max(e^-x, x, x²) over x ∈ [-0.5, 2.0]
include(joinpath(@__DIR__, "algorithms.jl"))

prob_num = 3
alg      = "both"

out_dir  = joinpath(@__DIR__, "results", "problem$(prob_num)")
mkpath(out_dir)

n    = 1
x_lb = [-0.5]
x_ub = [2.0]

function functionAndGradient(x)
    x_value = x[1]
    vals    = [exp(-x_value), x_value, x_value^2]       # the three component values
    i       = argmax(vals)                               # index of active component
    grads   = [-exp(-x_value), 1.0, 2.0*x_value]        # gradients of e^-x, x, x²
    return (vals[i], [grads[i]])                         # return scalar f and length-1 gradient
end

dispatch()
