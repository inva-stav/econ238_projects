using JuMP
using HiGHS

###################
#### Data Setup ###
###################

# Sets
N = [1, 2] # nodes with renewable units
T = [1, 2] # timesteps

# Generation (MW) (perfect anti-correlation)
g = Dict(
    (1,1)=>0.0, (1,2)=>1.0,
    (2,1)=>1.0, (2,2)=>0.0
)

# Lines: complete graph between {0,1,2}
# Represent as (from,to)
L = [(1,0), (2,0), (1,2)]

# Investment cost per MW 
INV = Dict(
    (1,0)=>90, #54.1436
    (2,0)=>100, #64.1436
    (1,2)=>50.0
)

# Tariff
P = 0   # 35.8564


###################
#### Function to compute C(s) ###
###################

function compute_cost(s::Dict{Int,Int})
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    # Variables
    @variable(model, F[l in L] >= 0) # line capacities
    @variable(model, f[l in L, t in T]) # flow on line l at time t
    @variable(model, G >= 0)

    # Helper: incidence
    function incoming(i) # returns all lines that flow into node i
        [l for l in L if l[2] == i]
    end
    function outgoing(i) # returns all lines that flow out of node i
        [l for l in L if l[1] == i]
    end

    # --- Constraints ---

    # Generation nodes
    for i in N, t in T
        @constraint(model,
            sum(f[l,t] for l in outgoing(i)) -
            sum(f[l,t] for l in incoming(i))
            == s[i] * g[(i,t)]
        )
    end

    # Include collector nodes later
    
    # Substation node 0
    for t in T
        @constraint(model,
            sum(f[l,t] for l in incoming(0)) -
            sum(f[l,t] for l in outgoing(0))
            == sum(s[i]*g[(i,t)] for i in N)
        )
    end

    # Line capacities
    for l in L, t in T
        #@constraint(model, -F[l] <= f[l,t] <= F[l])
        @constraint(model, f[l,t] <= F[l])
        @constraint(model, f[l,t] >= -F[l])
    end

    # Peak injection
    for t in T
        @constraint(model,
            G >= sum(s[i]*g[(i,t)] for i in N) # at no time can we inject more than G at the substation
        )
    end

    # Objective
    @objective(model, Min,
        sum(INV[l]*F[l] for l in L) + P*G
    )

    optimize!(model)

    return objective_value(model)
end

#####################
### Compute all coalition costs ###
#####################

# Coalitions
coalitions = [
    Dict(1=>0, 2=>0),
    Dict(1=>1, 2=>0),
    Dict(1=>0, 2=>1),
    Dict(1=>1, 2=>1)
]

C = Dict()
for s in coalitions
    C[s] = compute_cost(s)
end

println("Coalition costs:")
for (s,c) in C
    println(s, " => ", c)
end

######################
### Nucleolus (first LP step) ###
######################

model = Model(HiGHS.Optimizer)
set_silent(model)

@variable(model, x[i in N]) # allocate costs to players (unrestricted in sign)
@variable(model, ε)

# Efficiency
@constraint(model, sum(x[i] for i in N) == C[Dict(1=>1,2=>1)]) # total cost of grand coalition case must be allocated

# Excess constraints (exclude empty and grand coalitions:
# empty has no players; grand coalition's excess is identically 0 by efficiency)
for s in coalitions
    if s != Dict(1=>0,2=>0) && s != Dict(1=>1,2=>1)
        @constraint(model,
            C[s] - sum(s[i]*x[i] for i in N) >= ε
        )
    end
end

@objective(model, Max, ε)

optimize!(model)

println("Nucleolus solution (1st step):")
println("x = ", value.(x))
println("ε = ", value(ε))

#################
### Sequential LP (full nucleolus) ###
#################

# For two players first LP already gives nuceolus. For larger n, iteratively fix tight constraints and re-solve.


########################################################################
########################################################################
### Observed Output for P1 on 4/30:
########################################################################
# Coalition costs:
# Dict(2 => 0, 1 => 1) => 90.0
# Dict(2 => 1, 1 => 1) => 120.0
# Dict(2 => 1, 1 => 0) => 100.0
# Dict(2 => 0, 1 => 0) => 0.0
# Nucleolus solution (1st step):
# x = 1-dimensional DenseAxisArray{Float64,1,...} with index sets:
#     Dimension 1, [1, 2]
# And data, a 2-element Vector{Float64}:
#  55.0
#  65.0
# ε = 35.0
########################################################################
########################################################################
########################################################################