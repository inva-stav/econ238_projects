using JuMP
using HiGHS
using Printf

###################################
### Coalition cost LP
###################################

function compute_cost(s::Vector{Int}, N, T, g, L, INV; P::Float64 = 0.0)
    # s = binary membership vector: s[i] = 1 if player i is in the coalition, 0 otherwise
    # Decision variables: F[l] = capacity built on line l (MW),
    #                     f[l,t] = flow on line l at time t (unrestricted sign = bidirectional),
    #                     G = peak injection capacity at the substation.
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, F[l in L] >= 0)
    @variable(model, f[l in L, t in T])
    @variable(model, G >= 0)

    # Lines are stored as (from, to) pairs; flow is positive in the named direction.
    incoming(i) = [l for l in L if l[2] == i]
    outgoing(i) = [l for l in L if l[1] == i]

    # Flow conservation at each generation node:
    # net flow leaving node i = s[i] * g[i,t]  (zero for non-members since s[i]=0)
    for i in N, t in T
        @constraint(model,
            sum(f[l, t] for l in outgoing(i)) -
            sum(f[l, t] for l in incoming(i)) == s[i] * g[i, t]
        )
    end

    # Flow conservation at substation node 0:
    # net flow arriving at node 0 = total generation injected by the coalition
    # (redundant given the generation-node constraints, but explicit for clarity)
    for t in T
        @constraint(model,
            sum(f[l, t] for l in incoming(0)) -
            sum(f[l, t] for l in outgoing(0)) == sum(s[i] * g[i, t] for i in N)
        )
    end

    # Bidirectional line capacity: flow in either direction cannot exceed built capacity
    for l in L, t in T
        @constraint(model, f[l, t] <=  F[l])
        @constraint(model, f[l, t] >= -F[l])
    end

    # Peak injection capacity: G must cover the maximum total generation across all timesteps
    for t in T
        @constraint(model, G >= sum(s[i] * g[i, t] for i in N))
    end

    # Minimize total investment: line capacity costs + substation injection capacity cost
    @objective(model, Min, sum(INV[l] * F[l] for l in L) + P * G)
    optimize!(model)
    return objective_value(model)
end

###################################
### Coalition enumeration
###################################

# Returns all 2^n subsets of players 1..n as a list of member vectors.
# e.g. n=2 => [[], [1], [2], [1,2]]
# Built iteratively: start with {[]}, then for each new player i add i to every existing subset.
function all_subsets(n::Int)::Vector{Vector{Int}}
    subsets = [Int[]]
    for i in 1:n
        new_subsets = Vector{Vector{Int}}()
        for s in subsets
            new_s = copy(s)
            push!(new_s, i)
            push!(new_subsets, new_s)
        end
        append!(subsets, new_subsets)
    end
    return subsets
end

# Converts a member list to a binary membership vector of length n.
# e.g. coalition_vector([1,3], 4) => [1, 0, 1, 0]
# This is the s vector used by compute_cost: s[i] = 1 means player i is in the coalition.
function coalition_vector(members::Vector{Int}, n::Int)::Vector{Int}
    s = zeros(Int, n)
    for i in members
        s[i] = 1
    end
    return s
end

# Solves the coalition-cost LP for every subset and returns C keyed by member list.
# e.g. C[[1,3]] = cost for coalition {1,3}
function compute_all_costs(n::Int, N, T, g, L, INV; P::Float64 = 0.0, verbose::Bool = false)
    coalitions = all_subsets(n)
    C          = Dict{Vector{Int}, Float64}()
    total      = length(coalitions)
    for (k, coalition) in enumerate(coalitions)
        verbose && k % 128 == 0 &&
            println("    progress: $k / $total coalitions computed")
        s        = coalition_vector(coalition, n)
        C[coalition] = compute_cost(s, N, T, g, L, INV; P=P)
    end
    return C
end

###################################
### Sequential LP nucleolus
###################################

function nucleolus_sequential_lp(n::Int, C::Dict{Vector{Int},Float64}; tol::Float64 = 1e-6)
    # The nucleolus is found by lexicographically maximizing the sorted vector of coalition excesses.
    # excess e(s, x) = C(s) - x(s) = how much coalition s saves by joining the grand coalition.
    # We iteratively: (1) maximize the minimum excess over all unsettled coalitions,
    #                 (2) pin the coalitions that achieved that minimum (they are now "settled"),
    #                 (3) repeat on the remaining coalitions until all are settled.
    grand  = collect(1:n)                   # grand coalition = [1, 2, ..., n]
    active = Set{Vector{Int}}(             # proper non-empty coalitions (excludes [] and grand)
        s for s in keys(C) if !isempty(s) && s != grand
    )
    fixed  = Dict{Vector{Int}, Float64}()  # coalition => locked excess value from a prior iteration
    x_star = zeros(n)

    for iter in 1:n
        isempty(active) && break

        model = Model(HiGHS.Optimizer)
        set_silent(model)

        # x[i] = cost share allocated to player i (unrestricted in sign)
        @variable(model, x[1:n])
        # ε = the minimum excess across all active coalitions (what we maximize)
        @variable(model, ε)

        # Efficiency: full cost of grand coalition must be allocated, no more no less
        @constraint(model, sum(x[i] for i in 1:n) == C[grand])

        # Active coalitions: their excess must be >= ε (we are maximizing ε)
        for s in active
            @constraint(model, C[s] - sum(x[i] for i in s) >= ε)
        end

        # Fixed coalitions: their excess is pinned to the value from the iteration that settled them
        for (s, val) in fixed
            @constraint(model, C[s] - sum(x[i] for i in s) == val)
        end

        @objective(model, Max, ε)
        optimize!(model)

        ε_star = value(ε)
        x_vals = [value(x[i]) for i in 1:n]
        x_star = x_vals

        # Tight coalitions are those whose excess exactly equals ε* — their relative ordering is now settled
        tight = Set{Vector{Int}}(
            s for s in active
            if abs(C[s] - sum(x_vals[i] for i in s) - ε_star) < tol
        )

        @printf("  iter %d | ε* = %8.4f | tight = %d | active remaining = %d\n",
            iter, ε_star, length(tight), length(active) - length(tight))

        isempty(tight) && break

        # Pin tight coalitions and remove them from the active set for the next iteration
        for s in tight
            fixed[s] = ε_star
            delete!(active, s)
        end
    end

    return x_star
end
