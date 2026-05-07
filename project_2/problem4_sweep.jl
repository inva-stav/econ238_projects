include("problem4.jl")

using Printf, Statistics, CSV, DataFrames

###################################
### P4 extension: 12-month sweep
### Runs run_p4 for each month, collects results, saves sweep CSV.
###################################

function run_p4_sweep(; save::Bool = true)
    println("\n" * "="^60)
    println("P4 SWEEP: 12-month analysis of real PV vs. wind")
    println("="^60)

    rows = []
    for m in 1:12
        T_hours = P4_MONTH_HOURS[m]
        res = run_p4(; month=m, T_hours=T_hours, save=false)
        push!(rows, (
            month      = m,
            month_name = P4_MONTH_NAMES[m],
            T          = res.T,
            rho_hat    = res.rho_hat,
            mean_cf_pv = mean(load_ninja_capacity_factors(P4_PV_PATH)[month_window(m, T_hours)]),
            mean_cf_wind = mean(load_ninja_capacity_factors(P4_WIND_PATH)[month_window(m, T_hours)]),
            C_1        = res.C1,
            C_2        = res.C2,
            C_12       = res.C12,
            savings    = res.savings,
            share_1    = res.share_1,
            share_2    = res.share_2,
            epsilon    = res.eps,
        ))
    end

    println("\n" * "="^60)
    println("MONTHLY SUMMARY")
    println("="^60)
    @printf("  %-5s  %6s  %7s  %7s  %7s  %7s  %7s  %7s  %7s\n",
        "Month", "ρ̂", "CF_PV", "CF_W", "C({1})", "C({2})", "C(N)", "Save", "ε")
    println("  " * "-"^72)
    for r in rows
        @printf("  %-5s  %+6.3f  %7.4f  %7.4f  %7.2f  %7.2f  %7.2f  %7.2f  %7.4f\n",
            r.month_name, r.rho_hat, r.mean_cf_pv, r.mean_cf_wind,
            r.C_1, r.C_2, r.C_12, r.savings, r.epsilon)
    end

    if save
        dir = joinpath(@__DIR__, "results", "problem4")
        mkpath(dir)

        df = DataFrame(rows)
        path = joinpath(dir, "monthly_sweep.csv")
        CSV.write(path, df)
        println("\n  saved: $path")

        pv_full   = load_ninja_capacity_factors(P4_PV_PATH)
        wind_full = load_ninja_capacity_factors(P4_WIND_PATH)
        gen_df = DataFrame(hour = 1:8760, pv = pv_full, wind = wind_full)
        gen_path = joinpath(dir, "annual_generation.csv")
        CSV.write(gen_path, gen_df)
        println("  saved: $gen_path")
    end

    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_p4_sweep()
end
