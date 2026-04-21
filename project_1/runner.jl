# Runs all 6 problems sequentially. Failures in one problem do not abort the rest.

for i in 1:6
    println("\n\n" * "="^60)
    println("  Problem $(i)")
    println("="^60)
    try
        include(joinpath(@__DIR__, "problem$(i).jl"))
    catch e
        println("Problem $(i) failed: $(e)")
    end
end
