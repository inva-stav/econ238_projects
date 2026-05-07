using JuMP
using HiGHS
using Printf
using LinearAlgebra

###################################
### Coalition cost LP
###################################

# Solves the coalition's transmission-investment LP and returns the full solution
# (cost split into investment vs curtailment-penalty parts, line capacities,
# peak injection, and per-(player, time) curtailment).
#
# Curtailment: when `λ_curtail` is finite, each member i is allowed to curtail
# c[i,t] ∈ [0, g[i,t]] of its generation at hour t. The flow-conservation and
# peak constraints see the *delivered* quantity g[i,t] − c[i,t], and the
# objective gains a `λ_curtail · Σ c[i,t]` term so the LP doesn't degenerate
# to "curtail everything." `λ_curtail = Inf` (default) skips the curtailment
# variables entirely, recovering the original investment-only model.
function solve_coalition_lp(s::Vector{Int}, N, T, g, L, INV;
                            P::Float64 = 0.0, λ_curtail::Float64 = Inf)
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, F[l in L] >= 0)
    @variable(model, f[l in L, t in T])
    @variable(model, G >= 0)

    use_curt = isfinite(λ_curtail)
    if use_curt
        @variable(model, c[i in N, t in T] >= 0)
        # Non-members (s[i]=0) get c[i,t] <= 0 which forces c[i,t] = 0.
        for i in N, t in T
            @constraint(model, c[i, t] <= s[i] * g[i, t])
        end
    end

    incoming(i) = [l for l in L if l[2] == i]
    outgoing(i) = [l for l in L if l[1] == i]

    delivered(i, t) = use_curt ? (s[i] * g[i, t] - c[i, t]) : (s[i] * g[i, t])

    for i in N, t in T
        @constraint(model,
            sum(f[l, t] for l in outgoing(i)) -
            sum(f[l, t] for l in incoming(i)) == delivered(i, t)
        )
    end

    for t in T
        @constraint(model,
            sum(f[l, t] for l in incoming(0)) -
            sum(f[l, t] for l in outgoing(0)) == sum(delivered(i, t) for i in N)
        )
    end

    for l in L, t in T
        @constraint(model, f[l, t] <=  F[l])
        @constraint(model, f[l, t] >= -F[l])
    end

    for t in T
        @constraint(model, G >= sum(delivered(i, t) for i in N))
    end

    obj = sum(INV[l] * F[l] for l in L) + P * G
    if use_curt
        obj += λ_curtail * sum(c[i, t] for i in N, t in T)
    end
    @objective(model, Min, obj)
    optimize!(model)

    cost = objective_value(model)
    F_vals = Dict(l => value(F[l]) for l in L)
    G_val = value(G)
    invest_cost = sum(INV[l] * F_vals[l] for l in L) + P * G_val
    if use_curt
        c_vals = Dict((i, t) => value(c[i, t]) for i in N, t in T)
        curt_cost = cost - invest_cost
    else
        c_vals = Dict{Tuple{Int,Int}, Float64}()
        curt_cost = 0.0
    end
    return (cost = cost, invest_cost = invest_cost, curt_cost = curt_cost,
            F = F_vals, G = G_val, c = c_vals, λ_curtail = λ_curtail)
end

# Backward-compatible scalar-cost wrapper.
function compute_cost(s::Vector{Int}, N, T, g, L, INV;
                      P::Float64 = 0.0, λ_curtail::Float64 = Inf)
    return solve_coalition_lp(s, N, T, g, L, INV; P=P, λ_curtail=λ_curtail).cost
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
function compute_all_costs(n::Int, N, T, g, L, INV;
                           P::Float64 = 0.0, verbose::Bool = false,
                           λ_curtail::Float64 = Inf)
    coalitions = all_subsets(n)
    C          = Dict{Vector{Int}, Float64}()
    total      = length(coalitions)
    for (k, coalition) in enumerate(coalitions)
        verbose && k % 128 == 0 &&
            println("    progress: $k / $total coalitions computed")
        s        = coalition_vector(coalition, n)
        C[coalition] = compute_cost(s, N, T, g, L, INV; P=P, λ_curtail=λ_curtail)
    end
    return C
end

# Same as compute_all_costs but keeps the full per-coalition solution
# (cost, line caps, peak, curtailment matrix) for diagnostics.
function compute_all_solutions(n::Int, N, T, g, L, INV;
                               P::Float64 = 0.0, verbose::Bool = false,
                               λ_curtail::Float64 = Inf)
    coalitions = all_subsets(n)
    sols       = Dict{Vector{Int}, NamedTuple}()
    total      = length(coalitions)
    for (k, coalition) in enumerate(coalitions)
        verbose && k % 128 == 0 &&
            println("    progress: $k / $total coalitions solved")
        s = coalition_vector(coalition, n)
        sols[coalition] = solve_coalition_lp(s, N, T, g, L, INV; P=P, λ_curtail=λ_curtail)
    end
    return sols
end

###################################
### Sequential LP nucleolus
###################################

# Rank of the system formed by the efficiency constraint plus the incidence rows of
# every pinned coalition.  When this equals n the allocation x is uniquely determined.
function coalition_rank(n::Int, fixed_coalitions)
    rows = Vector{Vector{Float64}}()
    push!(rows, ones(Float64, n))          # efficiency: sum(x) = C(N)
    for s in fixed_coalitions
        row = zeros(Float64, n)
        for i in s
            row[i] = 1.0
        end
        push!(rows, row)
    end
    return rank(reduce(vcat, transpose.(rows)))
end

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

    iter = 0
    while !isempty(active) && coalition_rank(n, keys(fixed)) < n
        iter += 1

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

        @printf("  iter %d | ε* = %8.4f | tight = %d | active remaining = %d | rank = %d/%d\n",
            iter, ε_star, length(tight), length(active) - length(tight),
            coalition_rank(n, keys(fixed)), n)

        isempty(tight) && break

        # Pin tight coalitions and remove them from the active set for the next iteration
        for s in tight
            fixed[s] = ε_star
            delete!(active, s)
        end
    end

    return x_star
end
