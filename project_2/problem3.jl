include("algorithms.jl")
include("save_outputs.jl")
using Random

###################################
### P3: Correlation study
### n=2, T=168 hours, bivariate generation with prescribed correlation ρ
###################################

function generate_data_p3(ρ::Float64; seed::Int = 238, T::Int = 168)
    error("P3 generate_data not yet implemented")
end

function run_p3(; rho_values = range(-1.0, 1.0, length = 21), seed::Int = 238)
    error("P3 run_p3 not yet implemented")
end
