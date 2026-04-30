# Created for project 2, problem_1.jl
# If INV and P are not known, use this script to calibrate them to match the target costs C(s) for the 3 coalitions.

function coalition_costs(INV::Dict, P::Float64)
    # reuse compute_cost(s) from problem_1.jl but pass INV and P in
    function compute_cost(s)
        model = Model(HiGHS.Optimizer)
        set_silent(model)

        @variable(model, F[l in keys(INV)] >= 0)
        @variable(model, f[l in keys(INV), t in 1:2])
        @variable(model, G >= 0)

        # generation
        g = Dict(
            (1,1)=>0.0, (1,2)=>1.0,
            (2,1)=>1.0, (2,2)=>0.0
        )

        N = [1,2]

        incoming(i) = [l for l in keys(INV) if l[2]==i]
        outgoing(i) = [l for l in keys(INV) if l[1]==i]

        # node balance
        for i in N, t in 1:2
            @constraint(model,
                sum(f[l,t] for l in outgoing(i)) -
                sum(f[l,t] for l in incoming(i))
                == s[i]*g[(i,t)]
            )
        end

        # node 0
        for t in 1:2
            @constraint(model,
                sum(f[l,t] for l in incoming(0)) -
                sum(f[l,t] for l in outgoing(0))
                == sum(s[i]*g[(i,t)] for i in N)
            )
        end

        # capacity constraints (FIXED VERSION)
        for l in keys(INV), t in 1:2
            @constraint(model, f[l,t] <= F[l])
            @constraint(model, f[l,t] >= -F[l])
        end

        # peak
        for t in 1:2
            @constraint(model, G >= sum(s[i]*g[(i,t)] for i in N))
        end

        @objective(model, Min,
            sum(INV[l]*F[l] for l in keys(INV)) + P*G
        )

        optimize!(model)
        return objective_value(model)
    end

    C1  = compute_cost(Dict(1=>1,2=>0))
    C2  = compute_cost(Dict(1=>0,2=>1))
    C12 = compute_cost(Dict(1=>1,2=>1))

    return C1, C2, C12
end

function calibration_error(INV, P)
    C1, C2, C12 = coalition_costs(INV, P)

    return (C1-90)^2 + (C2-100)^2 + (C12-120)^2
end


##########################################
#### Use simple grid search ##############
function run_grid_search()
    best_err = Inf
    best_params = nothing

    for P in 40.0:5.0:100.0
        for inv10 in 10.0:5.0:80.0
            for inv20 in 10.0:5.0:80.0
                for inv12 in 0.0:5.0:40.0

                    INV = Dict(
                        (1,0)=>inv10,
                        (2,0)=>inv20,
                        (1,2)=>inv12
                    )

                    err = calibration_error(INV, P)

                    if err < best_err
                        best_err = err
                        best_params = (INV=INV, P=P)
                    end
                end
            end
        end
    end

    println("Best error: ", best_err)
    println("Best params: ", best_params)
    println("Costs: ", coalition_costs(best_params.INV, best_params.P))
end

#run_grid_search()


##########################################
#### Use nonlin optimizer ##############

#import Pkg
#Pkg.add("Optim")
using Optim

function objective(x)
    INV = Dict(
        (1,0)=>x[1],
        (2,0)=>x[2],
        (1,2)=>x[3]
    )
    P = x[4]

    return calibration_error(INV, P)
end

res = optimize(objective, [50.0, 60.0, 40.0, 40.0])
println(res.minimizer)