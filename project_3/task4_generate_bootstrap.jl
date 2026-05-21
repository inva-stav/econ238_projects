using CSV
using DataFrames
using Random
using Statistics
using Dates
using Printf

# =============================================================================
# Generate bootstrap scenario data for Task 4.
#
# Method: day-level block bootstrap
#   - Split the 40 historical months into 1,200 day-blocks (24 h each).
#   - Each bootstrap scenario draws 30 blocks with replacement → 720 hours.
#   - Price, wind CF, and solar CF are always drawn together so within-day
#     correlations (e.g. hot afternoon = high price + high solar) are preserved.
#   - Cross-day structure is randomised, giving genuine new scenarios.
#
# Output (saved to task4/task4_data/):
#   prices_bootstrap_<N>.csv   — columns: scenario, t (1-720), lmp_usd_mwh
#   wind_cf_bootstrap_<N>.csv  — columns: scenario, t, wind_cf
#   solar_cf_bootstrap_<N>.csv — columns: scenario, t, solar_cf
# =============================================================================

const DATA_DIR = joinpath(@__DIR__, "task4", "task4_data")

N_BOOTSTRAP = 1000   # number of bootstrap scenarios to generate
BLOCK_SIZE  = 24     # hours per block (one day)
T           = 720    # hours per scenario (30 days)
SEED        = 42

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load original data
# ─────────────────────────────────────────────────────────────────────────────

function load_original()
    prices_df = CSV.read(joinpath(DATA_DIR, "prices_summer.csv"), DataFrame)
    wind_df   = CSV.read(joinpath(DATA_DIR, "wind_cf_summer.csv"), DataFrame)
    solar_df  = CSV.read(joinpath(DATA_DIR, "solar_cf_summer.csv"), DataFrame)

    parse_dt(s) = DateTime(string(s)[1:19], "yyyy-mm-dd HH:MM:SS")
    for df in (prices_df, wind_df, solar_df)
        df[!, :dt]    = parse_dt.(df[!, :datetime_utc])
        df[!, :year]  = year.(df[!, :dt])
        df[!, :month] = month.(df[!, :dt])
    end

    df = leftjoin(select(prices_df, :dt, :lmp_usd_mwh => :price),
                  select(wind_df,   :dt, :wind_cf),  on=:dt)
    df = leftjoin(df, select(solar_df, :dt, :solar_cf), on=:dt)
    df[!, :wind_cf]  = coalesce.(df[!, :wind_cf],  0.0)
    df[!, :solar_cf] = coalesce.(df[!, :solar_cf], 0.0)
    df[!, :year]  = year.(df[!, :dt])
    df[!, :month] = month.(df[!, :dt])
    sort!(df, :dt)

    scenario_keys = sort(unique(collect(zip(df[!, :year], df[!, :month]))))
    N_orig = length(scenario_keys)

    π_mat     = Matrix{Float64}(undef, T, N_orig)
    wind_mat  = Matrix{Float64}(undef, T, N_orig)
    solar_mat = Matrix{Float64}(undef, T, N_orig)

    for (ω, (yr, mo)) in enumerate(scenario_keys)
        rows = sort(df[(df[!, :year] .== yr) .& (df[!, :month] .== mo), :], :dt)
        π_mat[:,     ω] = rows[1:T, :price]
        wind_mat[:,  ω] = rows[1:T, :wind_cf]
        solar_mat[:, ω] = rows[1:T, :solar_cf]
    end

    return π_mat, wind_mat, solar_mat, scenario_keys
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

function bootstrap(π_orig, wind_orig, solar_orig;
                   n_bootstrap=N_BOOTSTRAP, block_size=BLOCK_SIZE, seed=SEED)
    Random.seed!(seed)
    N_orig           = size(π_orig, 2)
    n_blocks_per_scen = T ÷ block_size        # 30
    n_total_blocks    = N_orig * n_blocks_per_scen  # 1200

    # Stack all historical day-blocks
    bp = Matrix{Float64}(undef, block_size, n_total_blocks)
    bw = Matrix{Float64}(undef, block_size, n_total_blocks)
    bs = Matrix{Float64}(undef, block_size, n_total_blocks)
    idx = 1
    for ω in 1:N_orig, d in 1:n_blocks_per_scen
        tr = ((d-1)*block_size + 1):(d*block_size)
        bp[:, idx] = π_orig[tr, ω]
        bw[:, idx] = wind_orig[tr, ω]
        bs[:, idx] = solar_orig[tr, ω]
        idx += 1
    end

    new_π    = Matrix{Float64}(undef, T, n_bootstrap)
    new_wind = Matrix{Float64}(undef, T, n_bootstrap)
    new_sol  = Matrix{Float64}(undef, T, n_bootstrap)

    for s in 1:n_bootstrap
        drawn = rand(1:n_total_blocks, n_blocks_per_scen)
        for (d, b) in enumerate(drawn)
            tr = ((d-1)*block_size + 1):(d*block_size)
            new_π[tr, s]    = bp[:, b]
            new_wind[tr, s] = bw[:, b]
            new_sol[tr, s]  = bs[:, b]
        end
    end

    return new_π, new_wind, new_sol
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Save
# ─────────────────────────────────────────────────────────────────────────────

function save_bootstrap(π_bs, wind_bs, solar_bs; n=N_BOOTSTRAP)
    suffix = "bootstrap_$(n)"
    n_scen = size(π_bs, 2)

    # Wide format: rows = hours (t=1..720), columns = scenario_1 .. scenario_N
    scen_cols = ["s$(lpad(i,4,'0'))" for i in 1:n_scen]

    for (mat, name, col) in (
            (π_bs,    "prices",  "lmp_usd_mwh"),
            (wind_bs, "wind_cf", "wind_cf"),
            (solar_bs,"solar_cf","solar_cf"))
        df = DataFrame(mat, scen_cols)
        insertcols!(df, 1, :t => 1:T)
        path = joinpath(DATA_DIR, "$(name)_$(suffix).csv")
        CSV.write(path, df)
        println("  Saved $(basename(path))  ($(T) rows × $(n_scen) scenarios)")
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Run
# ─────────────────────────────────────────────────────────────────────────────

println("Loading original data...")
π_orig, wind_orig, solar_orig, scenario_keys = load_original()
N_orig = size(π_orig, 2)
println("  $(N_orig) original scenarios, $(T) hours each")
@printf("  Price: mean=\$%.1f  std=\$%.1f  min=\$%.1f  max=\$%.1f\n",
        mean(π_orig), std(π_orig), minimum(π_orig), maximum(π_orig))

tag = "bootstrap_$(N_BOOTSTRAP)"
p_out = joinpath(DATA_DIR, "prices_$(tag).csv")

if isfile(p_out)
    println("\nBootstrap data already exists — delete to regenerate:")
    println("  $(p_out)")
else
    println("\nGenerating $(N_BOOTSTRAP) bootstrap scenarios (seed=$(SEED))...")
    t0 = time()
    π_bs, wind_bs, solar_bs = bootstrap(π_orig, wind_orig, solar_orig)
    elapsed = time() - t0
    println("  Done in $(round(elapsed; digits=1))s")

    @printf("  Bootstrap price: mean=\$%.1f  std=\$%.1f  min=\$%.1f  max=\$%.1f\n",
            mean(π_bs), std(π_bs), minimum(π_bs), maximum(π_bs))

    println("\nSaving...")
    save_bootstrap(π_bs, wind_bs, solar_bs)
end

println("\n═══ Done — bootstrap data in $(DATA_DIR) ═══\n")
