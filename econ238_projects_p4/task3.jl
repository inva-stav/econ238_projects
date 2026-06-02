# =============================================================================
# Project 4 — Task 3
# Robust capacity-adequacy procurement from CAISO's active interconnection queue
#
# Question: which projects sitting in CAISO's *active* interconnection queue
# (the public queue report from caiso.com, status = ACTIVE, all pre-GIA, all in
# study) should the planner advance to meet a capacity-adequacy target D at
# least cost, when the actual capacity contribution of intermittents (solar,
# wind, 4-hr storage, solar+storage hybrids) at the system's critical hour may
# fall short of the nominal Net Qualifying Capacity (NQC) ratings used by
# CAISO's Resource Adequacy framework?
#
# Duality enters as a BUILDING BLOCK (Bertsimas & Sim, 2004):
#   - Inner adversarial LP picks which intermittents derate, subject to a
#     budget Γ.
#   - Strong LP duality replaces the inner max with its dual min.
#   - The dual variables (λ, μ) are absorbed into the outer build-selection LP.
#
# The single equation where strong LP duality is invoked is marked below with
# the comment   # <<< strong LP duality applied here >>>
# =============================================================================

using JuMP, HiGHS
using DataFrames, CSV, XLSX
using Plots, Random
using Printf

const OUTDIR = joinpath(@__DIR__, "task3_output")
isdir(OUTDIR) || mkdir(OUTDIR)

# =============================================================================
# 1. DATA — CAISO active interconnection queue, June 1, 2026
#
#   Source: https://www.caiso.com/documents/publicqueuereport.xlsx
#           Sheet "Grid GenerationQueue", filtered to Application Status =
#           ACTIVE. Each row is one real CAISO interconnection request that is
#           currently in study (pre-GIA, pre-construction). Updated daily.
#
#   Per-project fields used:
#     - "Net MWs to Grid"  → nameplate P̄_i (we use this rather than the sum of
#                            MW-1/2/3 because for hybrids it is the actual
#                            grid-injection capacity after inverter limit).
#     - "Fuel-1", "Fuel-2", "Fuel-3" → composed into a fuel tag, mapped to one
#                            of the model classes below.
#
#   Class-level parameters:
#     - ρ (nominal capacity credit): CAISO Resource Adequacy NQC class values
#       — solar ~13%, wind ~30%, 4-hr storage ~95% (CAISO RA is far more
#       generous to storage than PJM ELCC because the slice-of-day framework
#       gives full credit if duration matches the peak window), hybrid
#       solar+storage ~60%, gas/geothermal/firm hydro ~85–95%.
#     - ρ̂ (max downward deviation under a multi-hour stress event): full
#       evening dropout for solar, regional wind drought for wind, energy-budget
#       exhaustion for storage in a 6+ hour stress event (Aug 2020-style).
#     - c (annualized fixed cost, $/MW-yr): NREL ATB 2024 moderate scenario,
#       CRF = 0.08 (≈ 30-yr life @ 7% WACC) applied to overnight CAPEX, plus
#       fixed O&M.
# =============================================================================

# (class, ρ, ρ̂, c_$/MW-yr, intermittent?)
const CAISO_CLASSES = Dict(
    :solar         => (rho = 0.13, rho_hat = 0.13, cost = 115_000.0, V = true),
    :wind          => (rho = 0.30, rho_hat = 0.25, cost = 150_000.0, V = true),
    :storage       => (rho = 0.95, rho_hat = 0.55, cost = 135_000.0, V = true),   # 4-hr battery
    :hybrid_solar  => (rho = 0.60, rho_hat = 0.40, cost = 160_000.0, V = true),
    :hybrid_wind   => (rho = 0.50, rho_hat = 0.35, cost = 175_000.0, V = true),
    :gas           => (rho = 0.95, rho_hat = 0.00, cost = 120_000.0, V = false),
    :geothermal    => (rho = 0.95, rho_hat = 0.00, cost = 400_000.0, V = false),
    :hydro         => (rho = 0.85, rho_hat = 0.00, cost = 200_000.0, V = false),  # pumped + ROR
    :other         => (rho = 0.50, rho_hat = 0.00, cost = 200_000.0, V = false),
)

"Map a CAISO fuel tag (e.g. `Solar+Battery`, `Battery`, `Wind Turbine+Battery`)
to one of the model classes in `CAISO_CLASSES`."
function classify_caiso(tag::AbstractString)
    t = lowercase(tag)
    # firm / non-intermittent first
    occursin("natural gas", t)         && return :gas
    occursin("geothermal", t)          && return :geothermal
    occursin("pumped-storage", t) ||
        occursin("pumped storage", t)  && return :hydro
    occursin("water", t)               && return :hydro
    # hybrids (multi-fuel tags with battery)
    if occursin("battery", t)
        if occursin("solar", t)
            return :hybrid_solar       # solar dominant + battery, in either order
        elseif occursin("wind", t)
            return :hybrid_wind
        else
            return :storage             # battery-only, battery+battery, battery+other
        end
    end
    # pure intermittents
    occursin("solar", t)               && return :solar
    occursin("wind", t)                && return :wind
    return :other
end

"""
Build a project-level dataset from the CAISO public queue Excel.
Filters to Application Status = ACTIVE and drops zero-MW rows.
"""
function build_caiso_dataset(path::AbstractString)
    xf = XLSX.readxlsx(path)
    sh = xf["Grid GenerationQueue"]
    rows = sh[:]
    # Header is on row 4 (rows 1–3 are merged-cell title banners).
    hdr_row = 4
    header = [ismissing(x) ? "" : strip(replace(string(x), "\n" => " "))
              for x in rows[hdr_row, :]]
    df = DataFrame(rows[hdr_row+1:end, :], header, makeunique = true)
    df = df[.!(map(r -> all(ismissing, r), eachrow(df))), :]
    df = df[df."Application Status" .=== "ACTIVE", :]

    parse_mw(x) = ismissing(x) ? 0.0 :
                  (x isa Number ? Float64(x) :
                   (try parse(Float64, string(x)); catch; 0.0; end))

    # Nameplate = "Net MWs to Grid" if present, else sum(MW-1..MW-3).
    net = parse_mw.(df."Net MWs to Grid")
    mw_sum = parse_mw.(df."MW-1") .+ parse_mw.(df."MW-2") .+ parse_mw.(df."MW-3")
    nameplate = ifelse.(net .> 0, net, mw_sum)

    # Build fuel tag from up to three fuel columns.
    function fueltag(r)
        fuels = filter(x -> !ismissing(x) && !isempty(strip(string(x))),
                       [r."Fuel-1", r."Fuel-2", r."Fuel-3"])
        return isempty(fuels) ? "Unknown" : join(string.(fuels), "+")
    end
    tags = fueltag.(eachrow(df))

    out = NamedTuple[]
    for (i, (tag, mw)) in enumerate(zip(tags, nameplate))
        mw <= 0 && continue
        cls = classify_caiso(tag)
        p   = CAISO_CLASSES[cls]
        push!(out, (
            id = length(out) + 1,
            caiso_qpos = string(df."Queue Position"[i]),
            name = string(df."Project Name"[i]),
            fuel_tag = tag,
            class = cls,
            nameplate = mw,
            rho = p.rho,
            rho_hat = p.rho_hat,
            cost_ann = p.cost,
            is_intermittent = p.V,
            county = string(df."County"[i]),
        ))
    end
    return DataFrame(out)
end

# =============================================================================
# 2. MODEL — robust counterpart with the inner LP dualized in-place
#
# Variables of the OUTER problem:
#   x_i ∈ [0,1]                fraction of project i built
#
# Dual variables of the INNER adversarial LP, lifted to the outer problem:
#   λ   ≥ 0                    multiplier on the Γ budget
#   μ_i ≥ 0  for i ∈ V         multiplier on the z_i ≤ 1 upper bound
#
# Outer constraints (after dualization):
#
#   ∑_i ρ_i P̄_i x_i  −  Γ λ  −  ∑_{i∈V} μ_i  ≥  D            (π)
#   λ + μ_i  ≥  ρ̂_i P̄_i x_i                  ∀ i ∈ V        (η_i)
#
# The first constraint replaces "nominal adequacy minus worst-case derating".
# # <<< strong LP duality applied here >>>
# =============================================================================

"Build the robust counterpart for a given Γ. Returns (model, refs)."
function build_robust(projects::DataFrame, D::Real, Γ::Real)
    n  = nrow(projects)
    V  = findall(projects.is_intermittent)
    ρ  = projects.rho
    ρ̂  = projects.rho_hat
    P̄  = projects.nameplate
    c  = projects.cost_ann

    m = Model(HiGHS.Optimizer)
    set_silent(m)

    @variable(m, 0 <= x[1:n] <= 1)
    @variable(m, λ >= 0)
    @variable(m, μ[V] >= 0)

    @constraint(m, adequacy,
        sum(ρ[i] * P̄[i] * x[i] for i in 1:n) - Γ * λ - sum(μ[i] for i in V) >= D)

    @constraint(m, dualfeas[i in V],
        λ + μ[i] >= ρ̂[i] * P̄[i] * x[i])

    @objective(m, Min, sum(c[i] * P̄[i] * x[i] for i in 1:n))
    return m, (; x, λ, μ, adequacy, dualfeas, V)
end

"Deterministic baseline (Γ = 0): no derating, no dual variables needed."
function build_deterministic(projects::DataFrame, D::Real)
    n = nrow(projects)
    ρ  = projects.rho
    P̄  = projects.nameplate
    c  = projects.cost_ann
    m = Model(HiGHS.Optimizer); set_silent(m)
    @variable(m, 0 <= x[1:n] <= 1)
    @constraint(m, adequacy, sum(ρ[i] * P̄[i] * x[i] for i in 1:n) >= D)
    @objective(m, Min, sum(c[i] * P̄[i] * x[i] for i in 1:n))
    return m, (; x, adequacy)
end

# =============================================================================
# 3. SANITY CHECK — the inner LP solved on its own, for one (x, Γ) pair, must
# match the dualized value (strong-duality verification on the building block).
# =============================================================================

"""
Solve the inner adversarial LP directly:
    max_z   ∑_{i∈V} ρ̂_i P̄_i x_i z_i
    s.t.    ∑_{i∈V} z_i ≤ Γ,   0 ≤ z_i ≤ 1
This is the maximum derating against a fixed build x. Returns (value, z*).
"""
function solve_inner(projects::DataFrame, x::Vector{Float64}, Γ::Real)
    V  = findall(projects.is_intermittent)
    ρ̂  = projects.rho_hat
    P̄  = projects.nameplate
    m = Model(HiGHS.Optimizer); set_silent(m)
    @variable(m, 0 <= z[V] <= 1)
    @constraint(m, sum(z[i] for i in V) <= Γ)
    @objective(m, Max, sum(ρ̂[i] * P̄[i] * x[i] * z[i] for i in V))
    optimize!(m)
    return objective_value(m), Dict(i => value(z[i]) for i in V)
end

# =============================================================================
# 4. SOLVE + REPORTING HELPERS
# =============================================================================

struct RunResult
    Γ::Float64
    cost::Float64
    π::Float64
    λ::Float64
    by_class::Dict{Symbol,Float64}
    n_built::Dict{Symbol,Int}
    n_binding::Int
    x::Vector{Float64}
    μ::Dict{Int,Float64}
end

function summarize_with(m::Model, projects::DataFrame, refs, Γ::Real)
    xv = value.(refs.x)
    by_class = Dict{Symbol,Float64}()
    n_built  = Dict{Symbol,Int}()
    for r in eachrow(projects)
        by_class[r.class] = get(by_class, r.class, 0.0) + r.nameplate * xv[r.id]
        n_built[r.class] = get(n_built, r.class, 0) + (xv[r.id] > 1e-6 ? 1 : 0)
    end
    μv = Dict(i => value(refs.μ[i]) for i in refs.V)
    return RunResult(
        Float64(Γ),
        objective_value(m),
        dual(refs.adequacy),
        value(refs.λ),
        by_class, n_built,
        count(>(1e-6), values(μv)),
        xv, μv,
    )
end

# =============================================================================
# 5. EXPERIMENT — load CAISO data, run sweep, verify duality, plot.
# =============================================================================

function main()
    caiso_path = joinpath(@__DIR__, "caiso_public_queue.xlsx")
    projects = build_caiso_dataset(caiso_path)
    n  = nrow(projects)
    V  = findall(projects.is_intermittent)
    D  = 10_000.0   # capacity-adequacy target, MW NQC-equivalent

    println("=" ^ 78)
    println("Instance summary  —  CAISO active interconnection queue")
    println("=" ^ 78)
    @printf("Source:  caiso.com/documents/publicqueuereport.xlsx\n")
    @printf("Status filter:  Application Status == ACTIVE  (pre-GIA, in study)\n")
    @printf("Projects:                    %d\n", n)
    @printf("Intermittent projects |V|:   %d\n", length(V))
    @printf("Capacity-adequacy target D:  %.0f MW NQC\n", D)
    @printf("Total nameplate in queue:    %.1f GW\n", sum(projects.nameplate)/1e3)
    println()
    println("Class breakdown:")
    @printf("  %-14s %5s %10s %8s %8s %10s\n",
            "class", "n", "Σ MW", "ρ", "ρ̂", "c (\$/MW-yr)")
    for cls in keys(CAISO_CLASSES)
        sub = projects[projects.class .== cls, :]
        isempty(sub) && continue
        p = CAISO_CLASSES[cls]
        @printf("  %-14s %5d %10.0f %8.2f %8.2f %10.0f  %s\n",
                cls, nrow(sub), sum(sub.nameplate),
                p.rho, p.rho_hat, p.cost,
                p.V ? "(intermittent)" : "")
    end
    println()

    # Save the dataset for the report.
    CSV.write(joinpath(OUTDIR, "projects_caiso.csv"), projects)

    # -------------------------------------------------------------------------
    # (a) Deterministic baseline (Γ = 0).
    # -------------------------------------------------------------------------
    m0, refs0 = build_deterministic(projects, D)
    optimize!(m0)
    π0   = dual(refs0.adequacy)
    cost0 = objective_value(m0)
    println("Deterministic (Γ = 0):")
    @printf("  cost  = \$%.3e /yr     π* (adequacy shadow price) = \$%.2f /MW-yr\n",
            cost0, π0)
    println()

    # -------------------------------------------------------------------------
    # (b) Robust sweep over Γ.
    # -------------------------------------------------------------------------
    Γs = [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 100.0,
          125.0, 130.0, 135.0, 140.0, 145.0, 150.0,
          175.0, 200.0, 250.0, Float64(length(V))]
    results = RunResult[]
    println("Γ-sweep:")
    @printf("  %6s  %12s  %14s  %14s  %8s\n",
            "Γ", "cost (\$/yr)", "π* (\$/MW-yr)", "λ* (MW)", "#μ>0")
    for Γ in Γs
        m, refs = build_robust(projects, D, Γ)
        optimize!(m)
        r = summarize_with(m, projects, refs, Γ)
        push!(results, r)
        @printf("  %6.1f  %12.3e  %14.2f  %14.2f  %8d\n",
                Γ, r.cost, r.π, r.λ, r.n_binding)
    end
    println()

    # -------------------------------------------------------------------------
    # (c) Strong-duality sanity check.
    # -------------------------------------------------------------------------
    println("Strong-duality verification (inner LP vs absorbed dual):")
    @printf("  %6s  %16s  %16s  %12s\n",
            "Γ", "inner LP opt", "Γλ* + ∑μ*", "rel err")
    for r in results[2:end-1]
        inner_val, _ = solve_inner(projects, r.x, r.Γ)
        absorbed = r.Γ * r.λ + sum(values(r.μ))
        rel = abs(inner_val - absorbed) / max(1.0, abs(inner_val))
        @printf("  %6.1f  %16.2f  %16.2f  %12.2e\n",
                r.Γ, inner_val, absorbed, rel)
    end
    println()

    # -------------------------------------------------------------------------
    # (d) Cost / mix tables and plots.
    # -------------------------------------------------------------------------
    summary_df = DataFrame(
        Γ       = [r.Γ for r in results],
        cost    = [r.cost for r in results],
        π_star  = [r.π for r in results],
        λ_star  = [r.λ for r in results],
        n_bind  = [r.n_binding for r in results],
    )
    classes = sort(collect(keys(CAISO_CLASSES)))
    for cls in classes
        summary_df[!, Symbol("MW_$(cls)")] = [get(r.by_class, cls, 0.0) for r in results]
    end
    CSV.write(joinpath(OUTDIR, "summary_caiso.csv"), summary_df)

    # ----- Queue composition (descriptive: what's in the queue) ---------------
    # Two-panel bar plot: nameplate GW (left) and project count (right),
    # sorted by GW descending. Colored by intermittent vs firm.
    queue_rows = NamedTuple[]
    for cls in keys(CAISO_CLASSES)
        sub = projects[projects.class .== cls, :]
        isempty(sub) && continue
        push!(queue_rows, (
            class = cls,
            n = nrow(sub),
            mw = sum(sub.nameplate),
            intermittent = CAISO_CLASSES[cls].V,
        ))
    end
    sort!(queue_rows, by = r -> r.mw, rev = true)
    qlabels = [String(r.class) for r in queue_rows]
    qgw     = [r.mw / 1e3 for r in queue_rows]
    qcounts = [r.n for r in queue_rows]
    int_mask = [r.intermittent for r in queue_rows]
    gw_int   = [m ? qgw[i]   : 0.0 for (i, m) in enumerate(int_mask)]
    gw_firm  = [m ? 0.0      : qgw[i]   for (i, m) in enumerate(int_mask)]
    n_int    = [m ? qcounts[i] : 0   for (i, m) in enumerate(int_mask)]
    n_firm   = [m ? 0          : qcounts[i] for (i, m) in enumerate(int_mask)]
    BLUE   = RGB(0.20, 0.50, 0.80)
    ORANGE = RGB(0.90, 0.45, 0.15)

    pq_mw = bar(qlabels, gw_int,
                color = BLUE, lw = 0, label = "",
                ylabel = "Nameplate (GW)", title = "Nameplate capacity",
                xrotation = 30,
                bottom_margin = 8Plots.mm, left_margin = 6Plots.mm,
                ylims = (0, maximum(qgw) * 1.12))
    bar!(pq_mw, qlabels, gw_firm, color = ORANGE, lw = 0, label = "")
    for (i, v) in enumerate(qgw)
        annotate!(pq_mw, i, v + maximum(qgw)*0.03,
                  text(v >= 1 ? @sprintf("%.1f", v) : @sprintf("%.2f", v),
                       :center, 8, :black))
    end

    pq_n  = bar(qlabels, n_int,
                color = BLUE, lw = 0, label = "",
                ylabel = "# projects", title = "Project count",
                xrotation = 30,
                bottom_margin = 8Plots.mm, left_margin = 6Plots.mm,
                ylims = (0, maximum(qcounts) * 1.12))
    bar!(pq_n, qlabels, n_firm, color = ORANGE, lw = 0, label = "")
    for (i, v) in enumerate(qcounts)
        annotate!(pq_n, i, v + maximum(qcounts)*0.03,
                  text(string(v), :center, 8, :black))
    end
    # Legend strip (tiny extra panel)
    pq_leg = plot(
        [0], [0], seriestype = :scatter,
        color = BLUE, markersize = 9, markershape = :rect,
        label = "intermittent  (∈ V)",
        framestyle = :none, xticks = false, yticks = false,
        legend = :left, foreground_color_legend = nothing,
        xlims = (0, 1), ylims = (0, 1),
    )
    plot!(pq_leg, [0], [0], seriestype = :scatter,
        color = ORANGE, markersize = 9, markershape = :rect,
        label = "firm  (∉ V)")
    pq = plot(pq_mw, pq_n, pq_leg,
              layout = @layout([a b; c{0.08h}]),
              size = (1000, 520),
              plot_title = "CAISO active interconnection queue — composition (June 1, 2026)",
              plot_titlefontsize = 13)
    savefig(pq, joinpath(OUTDIR, "queue_composition.pdf"))
    savefig(pq, joinpath(OUTDIR, "queue_composition.png"))

    # Price-of-robustness curve.
    p1 = plot(
        [r.Γ for r in results], [r.cost ./ 1e9 for r in results],
        marker = :circle, lw = 2, legend = false,
        xlabel = "Robustness budget Γ  (# intermittents allowed to derate)",
        ylabel = "Total annualized cost  (B\$ / yr)",
        title  = "Price of robustness — CAISO active queue procurement",
    )
    savefig(p1, joinpath(OUTDIR, "cost_vs_gamma.pdf"))
    savefig(p1, joinpath(OUTDIR, "cost_vs_gamma.png"))

    # Resource-mix curves.
    cls_order = [:gas, :geothermal, :hydro, :storage, :hybrid_solar,
                 :hybrid_wind, :solar, :wind, :other]
    p2 = plot(
        xlabel = "Robustness budget Γ",
        ylabel = "Built capacity (GW nameplate)",
        title  = "Resource mix vs Γ — CAISO queue",
        legend = :topright,
    )
    for c in cls_order
        vals = [get(r.by_class, c, 0.0) / 1e3 for r in results]
        all(v -> v < 1e-6, vals) && continue   # skip classes that are never built
        plot!(p2, [r.Γ for r in results], vals,
              label = String(c), marker = :circle, lw = 2)
    end
    savefig(p2, joinpath(OUTDIR, "mix_vs_gamma.pdf"))
    savefig(p2, joinpath(OUTDIR, "mix_vs_gamma.png"))

    # λ* and #μ>0 vs Γ — the dual-variable story.
    p3 = plot(
        [r.Γ for r in results], [r.λ for r in results],
        marker = :circle, lw = 2, label = "λ* (MW)",
        xlabel = "Robustness budget Γ",
        ylabel = "λ*  (MW per contingency)",
        title = "Dual variables vs Γ",
    )
    plot!(twinx(p3), [r.Γ for r in results], [r.n_binding for r in results],
          marker = :diamond, lw = 2, label = "#{μ_i > 0}", color = :red,
          ylabel = "# of binding μ_i (right axis)", legend = :topright)
    savefig(p3, joinpath(OUTDIR, "duals_vs_gamma.pdf"))
    savefig(p3, joinpath(OUTDIR, "duals_vs_gamma.png"))

    println("Wrote outputs to: $OUTDIR")
    return summary_df, results, projects
end

isinteractive() || main()
