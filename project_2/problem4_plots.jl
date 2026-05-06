###################################
### Problem 4 plotting helpers
### Loaded from problem4.jl after the data constants are defined; relies on
### P4_PV_LAT/P4_PV_LON/P4_WIND_LAT/P4_WIND_LON being in scope.
###################################

using Plots
using Printf
using Statistics
using Downloads

###################################
### Map of generator locations (OSM tiles stitched + annotated by ffmpeg)
###################################

# Slippy-map tile coords: returns (tile_x_float, tile_y_float) at the given zoom.
# Integer floor gives the tile index; fractional part times 256 gives the pixel
# offset inside that tile. https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
function lonlat_to_tile_xy(lon::Float64, lat::Float64, z::Int)
    n = 2.0^z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = lat * π / 180.0
    y = (1.0 - asinh(tan(lat_rad)) / π) / 2.0 * n
    return (x, y)
end

# Haversine distance in km between two lat/lon points.
function haversine_km(lat1, lon1, lat2, lon2)
    R = 6371.0
    φ1, φ2 = lat1*π/180, lat2*π/180
    Δφ = (lat2 - lat1) * π/180
    Δλ = (lon2 - lon1) * π/180
    a = sin(Δφ/2)^2 + cos(φ1)*cos(φ2)*sin(Δλ/2)^2
    return 2 * R * asin(sqrt(a))
end

# Try to render a real basemap (OSM tiles + ffmpeg compositing). On any failure
# (no network, ffmpeg missing, etc.) fall back to a Plots.jl scatter.
# Default zoom 7 with a California-tight bbox shows the state outline plus
# clearer county-level admin boundaries.
function make_locations_map(out_path::String;
        zoom::Int = 7,
        lat_bounds::Tuple{Float64,Float64} = (32.3, 42.2),
        lon_bounds::Tuple{Float64,Float64} = (-124.5, -114.0))
    pts = [(name="PV (player 1)",   lat=P4_PV_LAT,   lon=P4_PV_LON,   color="red"),
           (name="Wind (player 2)", lat=P4_WIND_LAT, lon=P4_WIND_LON, color="dodgerblue")]

    pt_xy = [lonlat_to_tile_xy(p.lon, p.lat, zoom) for p in pts]

    # Tile bounds covering the requested lat/lon box AND all markers.
    bx_w, by_n = lonlat_to_tile_xy(lon_bounds[1], lat_bounds[2], zoom)  # NW corner
    bx_e, by_s = lonlat_to_tile_xy(lon_bounds[2], lat_bounds[1], zoom)  # SE corner
    tx_min = min(floor(Int, bx_w), floor(Int, minimum(p[1] for p in pt_xy)))
    tx_max = max(floor(Int, bx_e), floor(Int, maximum(p[1] for p in pt_xy)))
    ty_min = min(floor(Int, by_n), floor(Int, minimum(p[2] for p in pt_xy)))
    ty_max = max(floor(Int, by_s), floor(Int, maximum(p[2] for p in pt_xy)))
    nx = tx_max - tx_min + 1
    ny = ty_max - ty_min + 1

    return try
        tmpdir = mktempdir()
        tile_files = String[]
        for j in 0:ny-1, i in 0:nx-1
            tx, ty = tx_min + i, ty_min + j
            url = "https://tile.openstreetmap.org/$zoom/$tx/$ty.png"
            f = joinpath(tmpdir, "tile_$(j)_$(i).png")
            Downloads.download(url, f;
                headers = ["User-Agent" => "econ238-project2/1.0 (academic)"])
            push!(tile_files, f)
        end

        # Pixel coords of each marker on the stitched image (origin = top-left).
        marker_pixels = [(p.name, p.color,
                          (pt_xy[k][1] - tx_min) * 256, (pt_xy[k][2] - ty_min) * 256)
                         for (k, p) in enumerate(pts)]

        # Build the xstack layout string in row-major order so it matches the
        # download loop above. (Doing this via a 2-D comprehension is wrong:
        # Julia linearizes matrices column-major, which transposes the mosaic.)
        layout_parts = String[]
        for j in 0:ny-1, i in 0:nx-1
            push!(layout_parts, "$(i*256)_$(j*256)")
        end
        layout = join(layout_parts, "|")
        filt   = "xstack=inputs=$(nx*ny):layout=$layout[m]"
        chain  = "[m]"

        # At state-level zoom the two markers are only ~2 px apart, so the labels
        # have to splay vertically (PV above, Wind below) to stay legible.
        for (idx, (name, color, px, py)) in enumerate(marker_pixels)
            r = 5
            chain *= "drawbox=x=$(px-r):y=$(py-r):w=$(2r):h=$(2r):color=$color@1.0:t=fill"
            chain *= ","
            chain *= "drawbox=x=$(px-r-1):y=$(py-r-1):w=$(2r+2):h=$(2r+2):color=white@1:t=2"
            chain *= ","
            label_dx = 12
            label_dy = idx == 1 ? -32 : 14
            label_x  = px + label_dx
            label_y  = py + label_dy
            chain *= "drawtext=text='$name':x=$label_x:y=$label_y:fontcolor=white:fontsize=16:" *
                     "box=1:boxcolor=$color@0.9:boxborderw=4"
            if idx < length(marker_pixels)
                chain *= ","
            end
        end
        chain *= ",drawtext=text='© OpenStreetMap contributors':x=8:y=h-18:" *
                 "fontcolor=black:fontsize=10:box=1:boxcolor=white@0.75:boxborderw=2"

        full_filter = filt * ";" * chain

        cmd = `ffmpeg -y -hide_banner -loglevel error`
        for f in tile_files
            cmd = `$cmd -i $f`
        end
        cmd = `$cmd -filter_complex $full_filter -frames:v 1 $out_path`
        run(cmd)
        true
    catch e
        @warn "OSM/ffmpeg map render failed; using fallback scatter" exception=(e, catch_backtrace())
        false
    end
end

# Plots.jl fallback: lat/lon scatter with cos-corrected aspect, line + distance.
function make_locations_scatter(out_path::String)
    d_km = haversine_km(P4_PV_LAT, P4_PV_LON, P4_WIND_LAT, P4_WIND_LON)
    lats = [P4_PV_LAT, P4_WIND_LAT]
    lons = [P4_PV_LON, P4_WIND_LON]
    aspect = 1 / cos(mean(lats) * π/180)   # 1 deg lon ≈ cos(lat) deg lat in km

    pad = 0.25
    xlim = (minimum(lons) - pad, maximum(lons) + pad)
    ylim = (minimum(lats) - pad, maximum(lats) + pad)

    p = plot(lons, lats, lw = 1.5, ls = :dash, color = :gray,
        xlabel = "longitude (°E)", ylabel = "latitude (°N)",
        title  = @sprintf("Generator locations (Tehachapi Pass, CA)\ngreat-circle distance = %.1f km", d_km),
        titlefontsize = 10,
        xlim = xlim, ylim = ylim, aspect_ratio = aspect,
        legend = :topright, label = "transmission link (1,2)")
    scatter!(p, [P4_PV_LON],   [P4_PV_LAT],
        ms = 9, mc = :gold, msw = 1.5, msc = :black, label = "PV (player 1)")
    scatter!(p, [P4_WIND_LON], [P4_WIND_LAT],
        ms = 9, mc = :steelblue, msw = 1.5, msc = :black, label = "wind (player 2)")
    annotate!(p, P4_PV_LON,   P4_PV_LAT   + 0.05, text(@sprintf("(%.4f, %.4f)", P4_PV_LAT,   P4_PV_LON),   8, :left))
    annotate!(p, P4_WIND_LON, P4_WIND_LAT + 0.05, text(@sprintf("(%.4f, %.4f)", P4_WIND_LAT, P4_WIND_LON), 8, :left))
    savefig(p, out_path)
end

###################################
### Per-run plots (called from run_p4)
###################################

function plot_generation_timeseries(g_pv, g_wind, period_label::String,
                                    is_full_year::Bool, month, dir::String)
    T = length(g_pv)
    x_unit = is_full_year ? "hour of year" : (month === nothing ? "hour" : "hour of month")
    ts = plot(1:T, g_pv,
        lw = is_full_year ? 0.4 : 1.2, label = "PV (player 1)",
        alpha = is_full_year ? 0.7 : 1.0,
        xlabel = x_unit, ylabel = "capacity factor",
        title  = "$period_label — renewables.ninja (MERRA-2)",
        titlefontsize = 10,
        size = is_full_year ? (1100, 380) : (700, 400),
        legend = :topright)
    plot!(ts, 1:T, g_wind,
        lw = is_full_year ? 0.4 : 1.2,
        alpha = is_full_year ? 0.7 : 1.0,
        label = "wind (player 2)")
    out = joinpath(dir, "generation_timeseries.png")
    savefig(ts, out)
    println("  saved: $out")
end

function plot_distribution_marginals(g_pv, g_wind, dir::String)
    edges = 0:0.025:1.0
    h_pv = histogram(g_pv, bins = edges,
        label = "PV", color = :gold, lw = 0,
        xlabel = "capacity factor", ylabel = "hours",
        title  = @sprintf("PV capacity factor distribution\nmean = %.3f, std = %.3f",
                          mean(g_pv), std(g_pv)),
        titlefontsize = 10)
    h_wind = histogram(g_wind, bins = edges,
        label = "wind", color = :steelblue, lw = 0,
        xlabel = "capacity factor", ylabel = "hours",
        title  = @sprintf("Wind capacity factor distribution\nmean = %.3f, std = %.3f",
                          mean(g_wind), std(g_wind)),
        titlefontsize = 10)
    h = plot(h_pv, h_wind, layout = (2, 1), size = (700, 600))
    out = joinpath(dir, "distribution_marginals.png")
    savefig(h, out)
    println("  saved: $out")
end

function plot_distribution_joint(g_pv, g_wind, period_label::String, corr, dir::String)
    is_day = g_pv .> 0.0
    sc = scatter(g_pv[.!is_day], g_wind[.!is_day],
        label = "night (PV=0)", ms = 2.5, mc = :royalblue, msw = 0, alpha = 0.55,
        xlabel = "PV capacity factor", ylabel = "wind capacity factor",
        title = @sprintf("Joint distribution — %s\nPearson all = %+.3f, daytime = %+.3f, Spearman = %+.3f",
            period_label, corr.pearson_all, corr.pearson_daytime, corr.spearman),
        titlefontsize = 10,
        legend = :topright, xlims = (0, 1), ylims = (0, 1))
    scatter!(sc, g_pv[is_day], g_wind[is_day],
        label = "daytime (PV>0)", ms = 2.5, mc = :firebrick, msw = 0, alpha = 0.45)
    out = joinpath(dir, "distribution_joint.png")
    savefig(sc, out)
    println("  saved: $out")
end

function plot_locations_map(dir::String)
    out = joinpath(dir, "locations_map.png")
    ok = make_locations_map(out)
    if !ok
        make_locations_scatter(out)
    end
    println("  saved: $out")
end

function plot_daily_mean_timeseries(g_pv, g_wind, period_label::String, dir::String)
    T = length(g_pv)
    T < 48 && return
    n_full_days = T ÷ 24
    days  = 1:n_full_days
    pv_d   = [mean(g_pv[(d-1)*24+1 : d*24])   for d in days]
    wind_d = [mean(g_wind[(d-1)*24+1 : d*24]) for d in days]
    dp = plot(days, pv_d,
        lw = 1.5, label = "PV daily mean", color = :gold,
        xlabel = "day of period", ylabel = "daily-mean capacity factor",
        title  = "Daily-mean capacity factors — $period_label",
        titlefontsize = 10, size = (1100, 380), legend = :topright)
    plot!(dp, days, wind_d, lw = 1.5, label = "wind daily mean", color = :steelblue)
    out = joinpath(dir, "daily_mean_timeseries.png")
    savefig(dp, out)
    println("  saved: $out")
end

function plot_all_p4(g_pv, g_wind, period_label::String,
                    is_full_year::Bool, month, corr, dir::String)
    plot_generation_timeseries(g_pv, g_wind, period_label, is_full_year, month, dir)
    plot_distribution_marginals(g_pv, g_wind, dir)
    plot_distribution_joint(g_pv, g_wind, period_label, corr, dir)
    plot_locations_map(dir)
    plot_daily_mean_timeseries(g_pv, g_wind, period_label, dir)
end

###################################
### Curtailment-sweep plots
###################################

# Each row is a NamedTuple from run_p4_curtailment_sweep. The Inf baseline (no
# curtailment) is shown as a horizontal dashed reference line on each panel.
function plot_curtailment_sweep(rows, dir::String)
    finite = sort([r for r in rows if isfinite(r.λ_curtail)], by = r -> r.λ_curtail)
    inf_row = let i = findfirst(r -> !isfinite(r.λ_curtail), rows)
        i === nothing ? nothing : rows[i]
    end
    isempty(finite) && return

    λs = [r.λ_curtail for r in finite]

    # Panel 1: cooperative savings (raw and investment-only) vs λ.
    p1 = plot(λs, [r.savings for r in finite],
        marker = :circle, lw = 2, label = "savings (incl. λ·Σc)",
        xlabel = "λ_curtail", ylabel = "savings = C₁ + C₂ − C₁₂",
        title = "Cooperative savings vs curtailment penalty",
        titlefontsize = 10, legend = :bottomright)
    plot!(p1, λs, [r.invest_savings for r in finite],
        marker = :square, lw = 2, label = "investment savings only", ls = :dash)
    if inf_row !== nothing
        hline!(p1, [inf_row.savings], ls = :dot, color = :gray,
            label = @sprintf("λ→∞ baseline (savings = %.3f)", inf_row.savings))
    end

    # Panel 2: nucleolus shares vs λ.
    p2 = plot(λs, [r.share_1 for r in finite],
        marker = :circle, lw = 2, label = "PV share x*₁/C₁₂", color = :gold,
        xlabel = "λ_curtail", ylabel = "nucleolus share",
        title = "Nucleolus shares vs curtailment penalty",
        titlefontsize = 10, legend = :right, ylim = (0, 1))
    plot!(p2, λs, [r.share_2 for r in finite],
        marker = :square, lw = 2, label = "wind share x*₂/C₁₂", color = :steelblue)
    if inf_row !== nothing
        hline!(p2, [inf_row.share_1], ls = :dot, color = :gold,
            label = @sprintf("PV baseline (%.3f)", inf_row.share_1))
        hline!(p2, [inf_row.share_2], ls = :dot, color = :steelblue,
            label = @sprintf("wind baseline (%.3f)", inf_row.share_2))
    end

    # Panel 3: curtailment fractions per player, standalone vs grand-coalition.
    p3 = plot(λs, [100 * r.curt_pv_alone for r in finite],
        marker = :circle, lw = 2, label = "PV — standalone", color = :gold,
        xlabel = "λ_curtail", ylabel = "curtailed energy (% of available)",
        title  = "Curtailment fractions vs penalty",
        titlefontsize = 10, legend = :topright)
    plot!(p3, λs, [100 * r.curt_pv_grand for r in finite],
        marker = :diamond, lw = 2, label = "PV — grand coalition", color = :gold, ls = :dash)
    plot!(p3, λs, [100 * r.curt_wind_alone for r in finite],
        marker = :square, lw = 2, label = "wind — standalone", color = :steelblue)
    plot!(p3, λs, [100 * r.curt_wind_grand for r in finite],
        marker = :utriangle, lw = 2, label = "wind — grand coalition", color = :steelblue, ls = :dash)

    # Panel 4: line capacities (grand coalition) vs λ.
    p4 = plot(λs, [r.F_pv_grand for r in finite],
        marker = :circle, lw = 2, label = "F (PV→0)", color = :gold,
        xlabel = "λ_curtail", ylabel = "line capacity built",
        title = "Grand-coalition line capacities vs penalty",
        titlefontsize = 10, legend = :right)
    plot!(p4, λs, [r.F_wind_grand for r in finite],
        marker = :square, lw = 2, label = "F (wind→0)", color = :steelblue)
    plot!(p4, λs, [r.F_link_grand for r in finite],
        marker = :diamond, lw = 2, label = "F (PV↔wind)", color = :firebrick)
    if inf_row !== nothing
        hline!(p4, [inf_row.F_pv_grand],   ls = :dot, color = :gold,       label = "")
        hline!(p4, [inf_row.F_wind_grand], ls = :dot, color = :steelblue,  label = "")
        hline!(p4, [inf_row.F_link_grand], ls = :dot, color = :firebrick,  label = "")
    end

    grid = plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 900))
    out = joinpath(dir, "curtailment_sweep.png")
    savefig(grid, out)
    println("  saved: $out")
end

# Detailed dispatch / pattern plots for the grand coalition. Designed to make
# the "peak shaving" intuition visible:
#   - Duration curves: independently sort g and (g − c) descending; the gap is
#     the curtailed energy, concentrated in the high-CF tail.
#   - Hour-of-day intensity: shows when in a day curtailment hits.
#   - Two-week dispatch sample: g overlay with delivered (g − c) in the same
#     window, so you can see the worst peaks getting clipped.
function plot_curtailment_dispatch_panels(
        g_pv::Vector{Float64}, g_wind::Vector{Float64},
        c_pv::Vector{Float64}, c_wind::Vector{Float64},
        λ_focus::Float64,
        overlay_curts::Dict{Float64, Tuple{Vector{Float64}, Vector{Float64}}},
        dir::String)
    T = length(g_pv)

    # ---------- Duration curves with multi-λ overlay (PV and wind panels) ----------
    pv_sorted   = sort(g_pv,   rev = true)
    wind_sorted = sort(g_wind, rev = true)

    # Sort overlay λ smallest → largest curtailment so the most-shaved curve is on top.
    overlay_λs_sorted = sort(collect(keys(overlay_curts)), rev = true)
    overlay_colors    = palette(:viridis, max(length(overlay_λs_sorted), 2))

    dc_pv = plot(1:T, pv_sorted,
        lw = 2.5, color = :black, label = "available (no curtailment)",
        xlabel = "rank (hours, descending CF)", ylabel = "capacity factor",
        title  = "PV duration curve — peak shaving by λ_curtail",
        titlefontsize = 10, legend = :topright)
    for (k, λ) in enumerate(overlay_λs_sorted)
        c_pv_λ, _ = overlay_curts[λ]
        delivered_sorted = sort(g_pv .- c_pv_λ, rev = true)
        plot!(dc_pv, 1:T, delivered_sorted,
            lw = 1.6, color = overlay_colors[k],
            label = @sprintf("delivered (λ = %.3f, %.1f%% curtailed)",
                             λ, 100 * sum(c_pv_λ) / sum(g_pv)))
    end

    dc_wind = plot(1:T, wind_sorted,
        lw = 2.5, color = :black, label = "available (no curtailment)",
        xlabel = "rank (hours, descending CF)", ylabel = "capacity factor",
        title  = "Wind duration curve — peak shaving by λ_curtail",
        titlefontsize = 10, legend = :topright)
    for (k, λ) in enumerate(overlay_λs_sorted)
        _, c_wind_λ = overlay_curts[λ]
        delivered_sorted = sort(g_wind .- c_wind_λ, rev = true)
        plot!(dc_wind, 1:T, delivered_sorted,
            lw = 1.6, color = overlay_colors[k],
            label = @sprintf("delivered (λ = %.3f, %.1f%% curtailed)",
                             λ, 100 * sum(c_wind_λ) / sum(g_wind)))
    end

    # ---------- Hour-of-day curtailment intensity at λ_focus ----------
    hod = [(t - 1) % 24 for t in 1:T]
    pv_avail_by_hod   = [mean(g_pv[hod .== h])   for h in 0:23]
    pv_curt_by_hod    = [mean(c_pv[hod .== h])   for h in 0:23]
    wind_avail_by_hod = [mean(g_wind[hod .== h]) for h in 0:23]
    wind_curt_by_hod  = [mean(c_wind[hod .== h]) for h in 0:23]

    hod_pv = plot(0:23, pv_avail_by_hod,
        lw = 2, label = "available (avg)", color = :gold,
        xlabel = "hour of day (UTC)", ylabel = "mean capacity factor",
        title = @sprintf("PV hour-of-day pattern at λ = %.3f", λ_focus),
        titlefontsize = 10, legend = :topright)
    plot!(hod_pv, 0:23, pv_avail_by_hod .- pv_curt_by_hod,
        lw = 2, label = "delivered (avg)", color = :firebrick, ls = :dash)
    plot!(hod_pv, 0:23, pv_curt_by_hod,
        lw = 1.5, label = "curtailed (avg)", color = :darkred, fillrange = 0, fillalpha = 0.3)

    hod_wind = plot(0:23, wind_avail_by_hod,
        lw = 2, label = "available (avg)", color = :steelblue,
        xlabel = "hour of day (UTC)", ylabel = "mean capacity factor",
        title = @sprintf("Wind hour-of-day pattern at λ = %.3f", λ_focus),
        titlefontsize = 10, legend = :topright)
    plot!(hod_wind, 0:23, wind_avail_by_hod .- wind_curt_by_hod,
        lw = 2, label = "delivered (avg)", color = :navy, ls = :dash)
    plot!(hod_wind, 0:23, wind_curt_by_hod,
        lw = 1.5, label = "curtailed (avg)", color = :midnightblue, fillrange = 0, fillalpha = 0.3)

    # ---------- Two-week dispatch sample (mid-June, hours 3961–4296) ----------
    win_lo = min(T, 24*162 + 1)            # ~ Jun 12
    win_hi = min(T, win_lo + 24*14 - 1)    # +14 days
    window = win_lo:win_hi

    ts_pv = plot(window, g_pv[window],
        lw = 1.4, label = "available", color = :gold,
        xlabel = "hour of year", ylabel = "capacity factor",
        title = @sprintf("PV two-week dispatch sample (mid-June) at λ = %.3f", λ_focus),
        titlefontsize = 10, legend = :topright)
    plot!(ts_pv, window, g_pv[window] .- c_pv[window],
        lw = 1.4, label = "delivered", color = :firebrick)

    ts_wind = plot(window, g_wind[window],
        lw = 1.0, label = "available", color = :steelblue, alpha = 0.85,
        xlabel = "hour of year", ylabel = "capacity factor",
        title = @sprintf("Wind two-week dispatch sample (mid-June) at λ = %.3f", λ_focus),
        titlefontsize = 10, legend = :topright)
    plot!(ts_wind, window, g_wind[window] .- c_wind[window],
        lw = 1.0, label = "delivered", color = :navy)

    grid = plot(dc_pv, dc_wind, hod_pv, hod_wind, ts_pv, ts_wind,
                layout = (3, 2), size = (1400, 1200))
    out = joinpath(dir, "curtailment_dispatch_patterns.png")
    savefig(grid, out)
    println("  saved: $out")
end
