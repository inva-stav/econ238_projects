include("algorithms.jl")
include("save_outputs.jl")

using Random
using Distributions
using LinearAlgebra
using Statistics
using Printf
using Plots

###################################
### P3 data generation
### n=2, T=168 hours, bivariate generation with varying Pearson correlation ρ.
###################################

function generate_correlated_pair_anchored(ρ::Float64, T::Int; seed::Int = 238)
    # need Gaussian copula: generates joint distribution and we can decompose the correlations structure and marginal distribution separately
    Random.seed!(seed)
    z1 = randn(T)
    ε  = randn(T)
    ρ_safe = clamp(ρ, -0.999, 0.999) # avoid clamp at +-1 so forumla always works
    z2 = ρ_safe .* z1 .+ sqrt(1 - ρ_safe^2) .* ε # building z2 to have correlation with z1

    Φ = Normal(0.0, 1.0)
    marginal = Beta(2.0, 5.0) # arbitrarily chosen to be bounded in [0,1] MW and right-skewed, a reasonable representation of capacity factors of intermittent renewable generation
    g1 = quantile.(marginal, cdf.(Φ, z1)) # pushes z1 through phi to get uniformly distributed values in [0,1]
    g2 = quantile.(marginal, cdf.(Φ, z2))
    return g1, g2
end

###################################
### Build network for n=2 and run all three coalitions
###################################
function p3_costs_at_rho(ρ::Float64; T::Int = 168, seed::Int = 238)
    g1, g2 = generate_correlated_pair_anchored(ρ, T; seed=seed)

    # Same network topology and cost magnitudes as P1
    n   = 2
    N   = [1, 2]
    Tset = collect(1:T)
    L   = [(1, 0), (2, 0), (1, 2)]
    INV = Dict((1, 0) => 90.0, (2, 0) => 100.0, (1, 2) => 50.0)
    P   = 0.0

    # Pack generation into a matrix indexable as g[i, t]
    g = zeros(Float64, n, T)
    g[1, :] = g1
    g[2, :] = g2

    C = compute_all_costs(n, N, Tset, g, L, INV; P=P)
    return C, g
end

###################################
### Run the sweep
###################################
function run_p3(; ρ_grid = -1.0:0.1:1.0, T::Int = 168, seed::Int = 238,
                   save::Bool = true)
    println("\n" * "="^60)
    println("P3: n = 2,  T = $T,  correlation sweep,  seed = $seed")
    println("="^60)

    rhos          = collect(ρ_grid)
    realized_rho  = Float64[]
    savings       = Float64[]
    share_1       = Float64[]
    share_2       = Float64[]
    eps_star      = Float64[]
    C1_vec        = Float64[]
    C2_vec        = Float64[]
    C12_vec       = Float64[]

    for ρ in rhos
        @printf("\nρ = %+0.2f\n", ρ)
        C, g = p3_costs_at_rho(ρ; T=T, seed=seed)

        C1   = C[[1]]
        C2   = C[[2]]
        C12  = C[[1, 2]]
        save_amt = C1 + C2 - C12

        # sample correlation, within ~0.05 of target for the copula
        ρ_hat = cor(g[1, :], g[2, :])

        @printf("  realized ρ̂ = %+0.4f\n", ρ_hat)
        @printf("  C({1}) = %.4f,  C({2}) = %.4f,  C({1,2}) = %.4f\n", C1, C2, C12)
        @printf("  savings = C({1}) + C({2}) - C({1,2}) = %.4f\n", save_amt)

        # nucleolus
        x_star = nucleolus_sequential_lp(2, C)
        @printf("  x* = (%.4f, %.4f),  shares = (%.4f, %.4f)\n",
            x_star[1], x_star[2], x_star[1]/C12, x_star[2]/C12)

        # Excess for either singleton, equal in 2-player case, so use coalition 1
        ε = C1 - x_star[1]

        push!(realized_rho, ρ_hat)
        push!(savings,       save_amt)
        push!(share_1,       x_star[1] / C12)
        push!(share_2,       x_star[2] / C12)
        push!(eps_star,      ε)
        push!(C1_vec,  C1)
        push!(C2_vec,  C2)
        push!(C12_vec, C12)
    end

    # plotting
    p1 = plot(rhos, savings,
        marker = :circle, lw = 2, label = "savings",
        xlabel = "target ρ", ylabel = "C({1}) + C({2}) − C({1,2})",
        title  = "Cooperative savings vs. correlation",
        legend = :topright)

    p2 = plot(rhos, share_1,
        marker = :circle, lw = 2, label = "x*₁ / C(N)",
        xlabel = "target ρ", ylabel = "nucleolus share",
        title  = "Nucleolus shares vs. correlation",
        legend = :right)
    plot!(p2, rhos, share_2, marker = :square, lw = 2, label = "x*₂ / C(N)")

    p3 = plot(rhos, realized_rho,
        marker = :circle, lw = 2, label = "realized ρ̂",
        xlabel = "target ρ", ylabel = "sample correlation",
        title  = "Copula sanity check: realized vs. target ρ",
        legend = :topleft)
    plot!(p3, rhos, rhos, ls = :dash, label = "y = x")

    if save
        dir = joinpath("results", "problem3")
        mkpath(dir)
        savefig(p1, joinpath(dir, "savings_vs_rho.png"))
        savefig(p2, joinpath(dir, "shares_vs_rho.png"))
        savefig(p3, joinpath(dir, "rho_sanity_check.png"))

        # Also dump the numerical sweep so it's reproducible without re-running
        open(joinpath(dir, "sweep.csv"), "w") do io
            println(io, "rho_target,rho_realized,C_1,C_2,C_12,savings,share_1,share_2,epsilon")
            for k in 1:length(rhos)
                @printf(io, "%.4f,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    rhos[k], realized_rho[k],
                    C1_vec[k], C2_vec[k], C12_vec[k],
                    savings[k], share_1[k], share_2[k], eps_star[k])
            end
        end
        println("\nSaved results to $dir")
    end

    return (
        rhos = rhos,
        realized_rho = realized_rho,
        savings = savings,
        share_1 = share_1,
        share_2 = share_2,
        eps_star = eps_star,
    )
end

# to run
run_p3()

############################################################
# written explanation of the solution process
# Regarding the savings plot: it is monotone non-increasing. 
# Between ρ ∈ [−1, −0.5] we see a sharp decrease, at ρ ∈ [−0.5, +0.5] we see a plateau
# and at ρ ∈ [+0.5, +1] we see a sharp decrease again. When ρ ∈ [−0.5, +0.5], the correlation in this range
# has no effect on the cost of cooperation. This is likely because our Beta(2,5) distributed marginals is dominated 
# by whichever one of g1 or g2 happens to be near its individual peak at some hour while the other is at its typical
# low value. Moderate correlation doesn't really impact cooperation when the generators are highly variable. 
# Only at extreme correlations (strong anti or positive correlation) do we see the cooperation being impacted (anti:
# higher cooperative savings, positive: lower cooperative savings, which intuitively aligns with our expectations).

# Regarding the shares plot: it is NOT monotone. 
# For most of the sweep, G2 pays the larger share, peaking at near 58% when ρ = 0.3. At ρ < -0.55, we see the shares
# crossing over, where G1 pays more. This is likely due to the anchoring method we used to generate correlated pairs.
# G1 is held fixed across ρ, so the change in C2 at strong anti-correlation can outpace the underlying asymmetry in 
# connection costs. This theory can be confirmed by rerunning the sweep with different seeds and comparing the share curves.
#
#
############################################################