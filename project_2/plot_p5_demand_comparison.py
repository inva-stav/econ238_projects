"""
P5 Demand-Heterogeneity Comparison Plot

Compares:
  1. P3 synthetic Beta copula savings curve (NO demand)
  2. P5 synthetic Beta copula savings curve (WITH demand heterogeneity)
  3. Real monthly data points (P4) — actual wind+solar savings

Goal: see if incorporating demand heterogeneity into the synthetic copula
prediction brings predictions closer to the actual monthly savings.

Run after:
  julia problem5_synthetic_sweep.jl
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv, os

RESULTS = "results"
P3_DIR  = os.path.join(RESULTS, "problem3")
P4_DIR  = os.path.join(RESULTS, "problem4")
P5_DIR  = os.path.join(RESULTS, "problem5")

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_demand_copula_comparison():
    """
    Main comparison figure:
    - Panel 1: Overlay of savings-vs-ρ curves (no demand vs with demand) + real monthly points
    - Panel 2: Per-month bar chart comparing predicted savings (from each curve) vs actual
    """
    p3 = read_csv(os.path.join(P3_DIR, "sweep.csv"))
    p4 = read_csv(os.path.join(P4_DIR, "monthly_sweep.csv"))

    synth_demand_path = os.path.join(P5_DIR, "synthetic_sweep_with_demand.csv")
    if not os.path.exists(synth_demand_path):
        print(f"ERROR: {synth_demand_path} not found.")
        print("  Run: julia problem5_synthetic_sweep.jl")
        return
    p5_synth = read_csv(synth_demand_path)

    # P3 synthetic curve (no demand)
    p3_rho = np.array([float(r["rho_target"]) for r in p3])
    p3_sav = np.array([float(r["savings"]) for r in p3])

    # P5 synthetic curve (with demand)
    p5s_rho = np.array([float(r["rho_target"]) for r in p5_synth])
    p5s_sav = np.array([float(r["savings"]) for r in p5_synth])

    # Real monthly data (P4)
    p4_rho   = np.array([float(r["rho_hat"]) for r in p4])
    p4_sav   = np.array([float(r["savings"]) for r in p4])
    p4_names = [r["month_name"] for r in p4]

    # Also load P5 monthly with-demand results for actual demand-adjusted savings
    p5_wd = read_csv(os.path.join(P5_DIR, "monthly_with_demand.csv"))
    p5_wd_sav = np.array([float(r["savings"]) for r in p5_wd])

    # ─── Figure setup ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 11))
    fig.suptitle(
        "Demand Heterogeneity in Synthetic Beta Copula Predictions\n"
        "Wind + Solar Cooperative Savings",
        fontsize=14, fontweight="bold", y=0.98
    )

    # ─── Panel 1: Savings-vs-ρ overlay ────────────────────────────────────────
    ax = axes[0]

    ax.plot(p3_rho, p3_sav, "o-", color="#abd9e9", lw=2.5, markersize=5,
            label="Synthetic: No demand (P3, Beta copula, T=168)", zorder=2)
    ax.fill_between(p3_rho, 0, p3_sav, alpha=0.06, color="#2c7bb6")

    ax.plot(p5s_rho, p5s_sav, "s-", color="#2ca25f", lw=2.5, markersize=5,
            label="Synthetic: With demand (Beta copula + Ind/Res, T=744)", zorder=3)
    ax.fill_between(p5s_rho, 0, p5s_sav, alpha=0.06, color="#2ca25f")

    cmap = plt.cm.coolwarm
    for i, (rho, sav, name) in enumerate(zip(p4_rho, p4_sav, p4_names)):
        color = cmap(i / 11)
        ax.scatter(rho, sav, s=100, c=[color], edgecolors="black",
                   linewidths=0.7, zorder=5)
        ax.annotate(name, (rho, sav), textcoords="offset points",
                    xytext=(8, 6 if i % 2 == 0 else -10), fontsize=7.5,
                    color=color, fontweight="bold")

    ax.set_xlabel("Correlation  ρ", fontsize=11)
    ax.set_ylabel("Cooperative Savings  [$/MW·yr]", fontsize=11)
    ax.set_title("Savings vs. Correlation: Synthetic Curves + Real Monthly Data",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-1.05, 1.05)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 12))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[1, 4, 7, 10, 12], shrink=0.8, pad=0.02)
    cbar.ax.set_yticklabels(["Jan", "Apr", "Jul", "Oct", "Dec"], fontsize=8)
    cbar.set_label("Month (real data)", fontsize=9)

    # ─── Panel 2: Per-month comparison bars ───────────────────────────────────
    ax2 = axes[1]

    # Interpolate synthetic predictions at each month's realized ρ
    p3_pred = np.interp(p4_rho, p3_rho, p3_sav)
    p5_pred = np.interp(p4_rho, p5s_rho, p5s_sav)

    x = np.arange(12)
    w = 0.22

    bars_actual = ax2.bar(x - 1.5*w, p4_sav, w,
                          label="Actual (real data, no demand)",
                          color="#d73027", edgecolor="white", linewidth=0.5)
    bars_p5wd   = ax2.bar(x - 0.5*w, p5_wd_sav, w,
                          label="Actual (real data, with demand)",
                          color="#fc8d59", edgecolor="white", linewidth=0.5)
    bars_p3     = ax2.bar(x + 0.5*w, p3_pred, w,
                          label="Predicted: synth no demand (at ρ̂)",
                          color="#abd9e9", edgecolor="white", linewidth=0.5)
    bars_p5s    = ax2.bar(x + 1.5*w, p5_pred, w,
                          label="Predicted: synth with demand (at ρ̂)",
                          color="#2ca25f", edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(p4_names, fontsize=9)
    ax2.set_xlabel("Month (2019)", fontsize=11)
    ax2.set_ylabel("Savings  [$/MW·yr]", fontsize=11)
    ax2.set_title("Monthly Savings: Actual vs. Synthetic Predictions (interpolated at realized ρ̂)",
                  fontsize=12)
    ax2.legend(fontsize=8, loc="upper left", ncol=2)
    ax2.grid(axis="y", alpha=0.3)

    # Add RMSE annotations
    rmse_p3 = np.sqrt(np.mean((p4_sav - p3_pred)**2))
    rmse_p5 = np.sqrt(np.mean((p4_sav - p5_pred)**2))
    rmse_p5wd = np.sqrt(np.mean((p5_wd_sav - p5_pred)**2))

    textstr = (f"RMSE (actual vs synth-no-demand): {rmse_p3:.2f} $/MW·yr\n"
               f"RMSE (actual vs synth-with-demand): {rmse_p5:.2f} $/MW·yr\n"
               f"RMSE (actual+demand vs synth+demand): {rmse_p5wd:.2f} $/MW·yr")
    ax2.text(0.98, 0.95, textstr, transform=ax2.transAxes, fontsize=8,
             verticalalignment="top", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(P5_DIR, "demand_copula_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


def plot_prediction_error_by_month():
    """
    Apples-to-apples comparison:
      - No-demand world:   actual (P4) vs synthetic (P3) at each month's ρ
      - With-demand world: actual (P5 with demand) vs synthetic+demand at each month's ρ
    Shows whether demand heterogeneity improves the copula's predictive power.
    """
    p3 = read_csv(os.path.join(P3_DIR, "sweep.csv"))
    p4 = read_csv(os.path.join(P4_DIR, "monthly_sweep.csv"))

    synth_demand_path = os.path.join(P5_DIR, "synthetic_sweep_with_demand.csv")
    if not os.path.exists(synth_demand_path):
        return
    p5_synth = read_csv(synth_demand_path)
    p5_wd = read_csv(os.path.join(P5_DIR, "monthly_with_demand.csv"))

    p3_rho = np.array([float(r["rho_target"]) for r in p3])
    p3_sav = np.array([float(r["savings"]) for r in p3])
    p5s_rho = np.array([float(r["rho_target"]) for r in p5_synth])
    p5s_sav = np.array([float(r["savings"]) for r in p5_synth])

    p4_rho   = np.array([float(r["rho_hat"]) for r in p4])
    p4_sav   = np.array([float(r["savings"]) for r in p4])
    p5wd_sav = np.array([float(r["savings"]) for r in p5_wd])
    p4_names = [r["month_name"] for r in p4]

    p3_pred = np.interp(p4_rho, p3_rho, p3_sav)
    p5_pred = np.interp(p4_rho, p5s_rho, p5s_sav)

    # Apples-to-apples errors
    err_no_demand   = p4_sav   - p3_pred   # no-demand actual vs no-demand synthetic
    err_with_demand = p5wd_sav - p5_pred    # with-demand actual vs with-demand synthetic

    x = np.arange(12)
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Prediction Error: Does Demand Heterogeneity Improve\n"
                 "the Synthetic Beta Copula's Monthly Predictions?",
                 fontsize=13, fontweight="bold", y=0.98)

    # Panel 1: side-by-side error bars
    axes[0].bar(x - w/2, np.abs(err_no_demand), w,
                label="| Error |  no-demand model",
                color="#abd9e9", edgecolor="white")
    axes[0].bar(x + w/2, np.abs(err_with_demand), w,
                label="| Error |  with-demand model",
                color="#2ca25f", edgecolor="white")

    axes[0].set_ylabel("|Actual − Predicted|  [$/MW·yr]", fontsize=10)
    axes[0].set_title("Absolute Prediction Error per Month (apples-to-apples)", fontsize=11)
    axes[0].legend(fontsize=9, loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)

    rmse_nd = np.sqrt(np.mean(err_no_demand**2))
    rmse_wd = np.sqrt(np.mean(err_with_demand**2))
    mae_nd  = np.mean(np.abs(err_no_demand))
    mae_wd  = np.mean(np.abs(err_with_demand))
    improvement = (1 - rmse_wd / rmse_nd) * 100

    axes[0].text(0.02, 0.95,
                 f"No-demand model:   RMSE={rmse_nd:.2f}, MAE={mae_nd:.2f}\n"
                 f"With-demand model: RMSE={rmse_wd:.2f}, MAE={mae_wd:.2f}\n"
                 f"Improvement: {improvement:+.1f}% RMSE reduction",
                 transform=axes[0].transAxes, fontsize=9,
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

    # Panel 2: signed errors showing systematic bias
    axes[1].bar(x - w/2, err_no_demand, w,
                label="No-demand: actual − predicted",
                color="#abd9e9", edgecolor="white")
    axes[1].bar(x + w/2, err_with_demand, w,
                label="With-demand: actual − predicted",
                color="#2ca25f", edgecolor="white")
    axes[1].axhline(0, color="black", lw=0.5)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(p4_names, fontsize=9)
    axes[1].set_xlabel("Month (2019)", fontsize=11)
    axes[1].set_ylabel("Actual − Predicted  [$/MW·yr]", fontsize=10)
    axes[1].set_title("Signed Prediction Error (+ = underpredicts, − = overpredicts)",
                      fontsize=11)
    axes[1].legend(fontsize=9, loc="lower right")
    axes[1].grid(axis="y", alpha=0.3)

    bias_nd = np.mean(err_no_demand)
    bias_wd = np.mean(err_with_demand)
    axes[1].text(0.02, 0.05,
                 f"Mean bias (no-demand): {bias_nd:+.2f}\n"
                 f"Mean bias (with-demand): {bias_wd:+.2f}",
                 transform=axes[1].transAxes, fontsize=9,
                 va="bottom", ha="left",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.9))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(P5_DIR, "prediction_error_by_month.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    print("Generating P5 demand-copula comparison plots...\n")

    required = [
        os.path.join(P3_DIR, "sweep.csv"),
        os.path.join(P4_DIR, "monthly_sweep.csv"),
        os.path.join(P5_DIR, "synthetic_sweep_with_demand.csv"),
        os.path.join(P5_DIR, "monthly_with_demand.csv"),
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("ERROR: missing data files:")
        for f in missing:
            print(f"  - {f}")
        if os.path.join(P5_DIR, "synthetic_sweep_with_demand.csv") in missing:
            print("\n  Run: julia problem5_synthetic_sweep.jl")
        exit(1)

    plot_demand_copula_comparison()
    plot_prediction_error_by_month()
    print("\nDone.")
