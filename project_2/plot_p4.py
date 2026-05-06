"""
P4 visualizations — run after `julia problem4_sweep.jl` has produced
results/problem4/monthly_sweep.csv and results/problem4/annual_generation.csv.

Generates:
  1. Real data overlaid on P3 synthetic savings-vs-ρ curve
  2. Monthly bar charts (correlation, savings, nucleolus shares)
  3. PV vs. wind generation scatter for representative months
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import csv, os

RESULTS = "results"
P4_DIR  = os.path.join(RESULTS, "problem4")
P3_DIR  = os.path.join(RESULTS, "problem3")

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ── 1. Overlay on P3 curve ────────────────────────────────────────────────────

def plot_overlay():
    """Plot each month's (ρ, savings) on top of the P3 synthetic curve."""
    p3 = read_csv(os.path.join(P3_DIR, "sweep.csv"))
    p4 = read_csv(os.path.join(P4_DIR, "monthly_sweep.csv"))

    p3_rho = [float(r["rho_target"]) for r in p3]
    p3_sav = [float(r["savings"]) for r in p3]

    p4_rho = [float(r["rho_hat"]) for r in p4]
    p4_sav = [float(r["savings"]) for r in p4]
    p4_names = [r["month_name"] for r in p4]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(p3_rho, p3_sav, "o-", color="#abd9e9", lw=2, markersize=5,
            label="P3 synthetic (Beta copula, T=168)", zorder=2)
    ax.fill_between(p3_rho, 0, p3_sav, alpha=0.08, color="#2c7bb6")

    cmap = plt.cm.coolwarm
    for i, (rho, sav, name) in enumerate(zip(p4_rho, p4_sav, p4_names)):
        month_frac = i / 11
        color = cmap(month_frac)
        ax.scatter(rho, sav, s=90, c=[color], edgecolors="black", linewidths=0.6, zorder=4)
        offset_x = 0.02 if i % 2 == 0 else -0.02
        offset_y = 0.8
        ax.annotate(name, (rho, sav), textcoords="offset points",
                    xytext=(8, 6 if i % 2 == 0 else -10), fontsize=7,
                    color=color, fontweight="bold")

    ax.set_xlabel("Correlation  ρ", fontsize=11)
    ax.set_ylabel("Cooperative savings  C({1})+C({2})−C(N)  [$/MW·yr]", fontsize=11)
    ax.set_title("P4: Real Solar+Wind Savings vs. P3 Synthetic Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-1.05, 1.05)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 12))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[1, 4, 7, 10, 12], shrink=0.8, pad=0.02)
    cbar.ax.set_yticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"], fontsize=8)
    cbar.set_label("Month", fontsize=9)

    fig.tight_layout()
    out = os.path.join(P4_DIR, "overlay_on_p3.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ── 2. Monthly bar charts ────────────────────────────────────────────────────

def plot_monthly_bars():
    """Three-panel figure: monthly ρ, savings, and nucleolus shares."""
    p4 = read_csv(os.path.join(P4_DIR, "monthly_sweep.csv"))

    months = [r["month_name"] for r in p4]
    rhos   = [float(r["rho_hat"]) for r in p4]
    savings = [float(r["savings"]) for r in p4]
    share1 = [float(r["share_1"]) for r in p4]
    share2 = [float(r["share_2"]) for r in p4]
    x = np.arange(12)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("P4: Monthly Analysis — Real PV vs. Wind (Tehachapi, CA, 2019)",
                 fontsize=14, fontweight="bold", y=1.01)

    # Panel 1: correlation
    colors_rho = ["#2c7bb6" if r < 0 else "#d7191c" for r in rhos]
    axes[0].bar(x, rhos, color=colors_rho, edgecolor="white", linewidth=0.5)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("Realized  ρ̂", fontsize=11)
    axes[0].set_title("PV–Wind Correlation", fontsize=11)
    for i, v in enumerate(rhos):
        axes[0].text(i, v + (0.01 if v >= 0 else -0.03), f"{v:+.2f}",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    axes[0].set_ylim(min(rhos) - 0.08, max(rhos) + 0.08)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: savings
    axes[1].bar(x, savings, color="#2ca25f", edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("Savings  [$/MW·yr]", fontsize=11)
    axes[1].set_title("Cooperative Savings = C({1}) + C({2}) − C(N)", fontsize=11)
    for i, v in enumerate(savings):
        axes[1].text(i, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: nucleolus shares
    w = 0.35
    axes[2].bar(x - w/2, share1, w, label="PV (player 1)", color="#fdae61", edgecolor="white")
    axes[2].bar(x + w/2, share2, w, label="Wind (player 2)", color="#2c7bb6", edgecolor="white")
    axes[2].axhline(0.5, color="gray", ls=":", lw=1)
    axes[2].set_ylabel("Nucleolus share  x*/C(N)", fontsize=11)
    axes[2].set_title("Nucleolus Shares", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(months, fontsize=9)
    axes[2].set_xlabel("Month (2019)", fontsize=11)
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].set_ylim(0, 0.75)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(P4_DIR, "monthly_bars.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ── 3. Generation scatter for representative months ───────────────────────────

def plot_generation_scatter():
    """Scatter PV vs. wind capacity factor for 4 representative months."""
    p4 = read_csv(os.path.join(P4_DIR, "monthly_sweep.csv"))
    gen = read_csv(os.path.join(P4_DIR, "annual_generation.csv"))

    pv_all   = np.array([float(r["pv"]) for r in gen])
    wind_all = np.array([float(r["wind"]) for r in gen])

    month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    pick = [0, 3, 6, 9]  # Jan, Apr, Jul, Oct
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle("P4: Hourly PV vs. Wind Capacity Factors (2019)",
                 fontsize=14, fontweight="bold", y=1.01)

    for idx, m in enumerate(pick):
        ax = axes[idx // 2][idx % 2]
        start = sum(month_hours[:m])
        stop = start + month_hours[m]
        pv_m = pv_all[start:stop]
        wind_m = wind_all[start:stop]
        rho = float(p4[m]["rho_hat"])

        ax.scatter(pv_m, wind_m, s=8, alpha=0.4, c="#2c7bb6", edgecolors="none")
        ax.set_xlabel("PV capacity factor", fontsize=9)
        ax.set_ylabel("Wind capacity factor", fontsize=9)
        ax.set_title(f"{month_names[m]}   (ρ̂ = {rho:+.3f})", fontsize=11, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.plot([0, 1], [0, 1], ":", color="gray", alpha=0.4)
        ax.grid(alpha=0.2)

        ax.text(0.95, 0.95, f"ρ̂ = {rho:+.3f}\nn = {len(pv_m)}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

    fig.tight_layout()
    out = os.path.join(P4_DIR, "generation_scatter.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ── 4. Annual generation time series ──────────────────────────────────────────

def plot_annual_timeseries():
    """Full-year PV and wind capacity factors with monthly shading."""
    gen = read_csv(os.path.join(P4_DIR, "annual_generation.csv"))
    pv   = np.array([float(r["pv"]) for r in gen])
    wind = np.array([float(r["wind"]) for r in gen])
    hours = np.arange(len(pv))

    month_hours = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle("P4: Annual Generation Profiles — Tehachapi, CA (2019)",
                 fontsize=13, fontweight="bold", y=1.01)

    ax1.plot(hours, pv, lw=0.3, color="#fdae61", alpha=0.7)
    window = 24
    pv_smooth = np.convolve(pv, np.ones(window)/window, mode="same")
    ax1.plot(hours, pv_smooth, lw=1.2, color="#e08214", label="PV (24h avg)")
    ax1.set_ylabel("PV capacity factor", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(-0.02, 1.02)

    ax2.plot(hours, wind, lw=0.3, color="#abd9e9", alpha=0.7)
    wind_smooth = np.convolve(wind, np.ones(window)/window, mode="same")
    ax2.plot(hours, wind_smooth, lw=1.2, color="#2c7bb6", label="Wind (24h avg)")
    ax2.set_ylabel("Wind capacity factor", fontsize=10)
    ax2.set_xlabel("Hour of year", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_ylim(-0.02, 1.02)

    cum = 0
    for i, (h, name) in enumerate(zip(month_hours, month_names)):
        for ax in (ax1, ax2):
            if i % 2 == 0:
                ax.axvspan(cum, cum + h, alpha=0.04, color="gray")
            ax.axvline(cum, color="gray", lw=0.3, alpha=0.5)
        mid = cum + h / 2
        ax2.text(mid, -0.08, name, ha="center", fontsize=7, color="gray",
                 transform=ax2.get_xaxis_transform())
        cum += h

    fig.tight_layout()
    out = os.path.join(P4_DIR, "annual_timeseries.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    print("Generating P4 plots...")
    print()

    required = [
        os.path.join(P4_DIR, "monthly_sweep.csv"),
        os.path.join(P4_DIR, "annual_generation.csv"),
        os.path.join(P3_DIR, "sweep.csv"),
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("ERROR: missing data files. Run the Julia scripts first:")
        for f in missing:
            print(f"  - {f}")
        if os.path.join(P4_DIR, "monthly_sweep.csv") in missing:
            print("\n  julia problem4_sweep.jl")
        if os.path.join(P3_DIR, "sweep.csv") in missing:
            print("\n  julia problem3.jl")
        exit(1)

    plot_overlay()
    plot_monthly_bars()
    plot_generation_scatter()
    plot_annual_timeseries()
    print("\nDone.")
