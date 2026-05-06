"""
P5 visualizations — run after `julia problem5.jl` produces CSVs in results/problem5/.

Generates:
  1. Demand profile shapes (one-week view)
  2. Pairing comparison bar chart
  3. Demand-scale sweep (savings & C(N) vs α)
  4. Monthly comparison: with demand vs without demand
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv, os

P5_DIR = os.path.join("results", "problem5")

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ── 1. Demand profile shapes ─────────────────────────────────────────────────

def plot_demand_profiles():
    rows = read_csv(os.path.join(P5_DIR, "demand_profiles.csv"))
    hours = [int(r["hour"]) for r in rows]
    ind   = [float(r["industrial"]) for r in rows]
    res   = [float(r["residential"]) for r in rows]
    flat  = [float(r["flat"]) for r in rows]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(hours, ind,  lw=1.8, label="Industrial", color="#e08214")
    ax.plot(hours, res,  lw=1.8, label="Residential", color="#2c7bb6")
    ax.plot(hours, flat, lw=1.8, label="Flat",        color="#999999", ls="--")

    for day in range(0, 168, 24):
        ax.axvline(day, color="gray", lw=0.3, alpha=0.5)
    ax.set_xlabel("Hour of week", fontsize=10)
    ax.set_ylabel("Demand [MW / MW capacity]", fontsize=10)
    ax.set_title("Synthetic Demand Profiles (one week)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(1, 168)
    ax.set_ylim(-0.02, 1.0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(P5_DIR, "demand_profiles.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ── 2. Pairing comparison ────────────────────────────────────────────────────

def plot_pairings():
    rows = read_csv(os.path.join(P5_DIR, "pairings.csv"))

    labels  = [r["label"] for r in rows]
    savings = [float(r["savings"]) for r in rows]
    c1      = [float(r["C1"]) for r in rows]
    c2      = [float(r["C2"]) for r in rows]
    c12     = [float(r["C12"]) for r in rows]
    sh1     = [float(r["share_1"]) for r in rows]
    sh2     = [float(r["share_2"]) for r in rows]
    sc1     = [float(r["sc_1"]) for r in rows]
    sc2     = [float(r["sc_2"]) for r in rows]

    short_labels = [l.split("(")[-1].rstrip(")") if "(" in l else l for l in labels]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    fig.suptitle("P5: Effect of Demand-Profile Pairing (July, scale=0.5)",
                 fontsize=14, fontweight="bold", y=1.01)

    # Panel 1: coalition costs
    w = 0.25
    axes[0].bar(x - w, c1,  w, label="C({1}) — PV", color="#fdae61", edgecolor="white")
    axes[0].bar(x,     c2,  w, label="C({2}) — Wind", color="#2c7bb6", edgecolor="white")
    axes[0].bar(x + w, c12, w, label="C(N)",          color="#2ca25f", edgecolor="white")
    axes[0].set_ylabel("Cost [$/MW·yr]", fontsize=10)
    axes[0].set_title("Coalition Costs", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: savings
    colors = ["#2ca25f" if s > 0 else "#d7191c" for s in savings]
    axes[1].bar(x, savings, color=colors, edgecolor="white")
    for i, v in enumerate(savings):
        axes[1].text(i, v + 0.2, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_ylabel("Savings [$/MW·yr]", fontsize=10)
    axes[1].set_title("Cooperative Savings = C({1}) + C({2}) − C(N)", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: self-consumption rates
    w2 = 0.35
    axes[2].bar(x - w2/2, sc1, w2, label="PV self-consumption", color="#fdae61", edgecolor="white")
    axes[2].bar(x + w2/2, sc2, w2, label="Wind self-consumption", color="#2c7bb6", edgecolor="white")
    axes[2].set_ylabel("Self-consumption rate", fontsize=10)
    axes[2].set_title("Fraction of Generation Consumed Locally", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_labels, fontsize=8, rotation=18, ha="right")
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 1.0)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(P5_DIR, "pairing_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ── 3. Scale sweep ───────────────────────────────────────────────────────────

def plot_scale_sweep():
    rows = read_csv(os.path.join(P5_DIR, "scale_sweep.csv"))

    alpha   = [float(r["demand_scale"]) for r in rows]
    savings = [float(r["savings"]) for r in rows]
    c12     = [float(r["C12"]) for r in rows]
    c1      = [float(r["C1"]) for r in rows]
    c2      = [float(r["C2"]) for r in rows]
    sc1     = [float(r["sc_1"]) for r in rows]
    sc2     = [float(r["sc_2"]) for r in rows]
    sh1     = [float(r["share_1"]) for r in rows]
    sh2     = [float(r["share_2"]) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("P5: Demand-Scale Sweep (July, PV+Industrial / Wind+Residential)",
                 fontsize=13, fontweight="bold", y=1.01)

    # Panel 1: costs
    axes[0, 0].plot(alpha, c1,  "o-", ms=4, label="C({1}) — PV",  color="#fdae61")
    axes[0, 0].plot(alpha, c2,  "s-", ms=4, label="C({2}) — Wind", color="#2c7bb6")
    axes[0, 0].plot(alpha, c12, "^-", ms=4, label="C(N)",          color="#2ca25f")
    axes[0, 0].set_ylabel("Cost [$/MW·yr]", fontsize=10)
    axes[0, 0].set_title("Coalition Costs vs. Demand Scale", fontsize=11)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    # Panel 2: savings
    axes[0, 1].plot(alpha, savings, "o-", ms=5, color="#2ca25f", lw=2)
    axes[0, 1].fill_between(alpha, 0, savings, alpha=0.15, color="#2ca25f")
    axes[0, 1].set_ylabel("Savings [$/MW·yr]", fontsize=10)
    axes[0, 1].set_title("Cooperative Savings vs. Demand Scale", fontsize=11)
    axes[0, 1].grid(alpha=0.3)

    # Panel 3: self-consumption
    axes[1, 0].plot(alpha, sc1, "o-", ms=4, label="PV",   color="#fdae61", lw=2)
    axes[1, 0].plot(alpha, sc2, "s-", ms=4, label="Wind", color="#2c7bb6", lw=2)
    axes[1, 0].set_xlabel("Demand scale  α", fontsize=10)
    axes[1, 0].set_ylabel("Self-consumption rate", fontsize=10)
    axes[1, 0].set_title("Self-Consumption vs. Demand Scale", fontsize=11)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].grid(alpha=0.3)

    # Panel 4: nucleolus shares
    axes[1, 1].plot(alpha, sh1, "o-", ms=4, label="PV share",   color="#fdae61", lw=2)
    axes[1, 1].plot(alpha, sh2, "s-", ms=4, label="Wind share", color="#2c7bb6", lw=2)
    axes[1, 1].axhline(0.5, color="gray", ls=":", lw=1)
    axes[1, 1].set_xlabel("Demand scale  α", fontsize=10)
    axes[1, 1].set_ylabel("Nucleolus share  x*/C(N)", fontsize=10)
    axes[1, 1].set_title("Nucleolus Shares vs. Demand Scale", fontsize=11)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_ylim(0, 0.8)
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(P5_DIR, "scale_sweep.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ── 4. Monthly comparison: with vs without demand ────────────────────────────

def plot_monthly_comparison():
    wd = read_csv(os.path.join(P5_DIR, "monthly_with_demand.csv"))
    nd = read_csv(os.path.join(P5_DIR, "monthly_no_demand.csv"))

    months   = [r["month_name"] for r in wd]
    sav_wd   = [float(r["savings"]) for r in wd]
    sav_nd   = [float(r["savings"]) for r in nd]
    c12_wd   = [float(r["C12"]) for r in wd]
    c12_nd   = [float(r["C12"]) for r in nd]
    sh1_wd   = [float(r["share_1"]) for r in wd]
    sh1_nd   = [float(r["share_1"]) for r in nd]

    x = np.arange(12)
    w = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("P5: Monthly Comparison — With vs. Without Local Demand\n"
                 "(PV+Industrial / Wind+Residential, scale=0.5)",
                 fontsize=13, fontweight="bold", y=1.02)

    # Panel 1: C(N)
    axes[0].bar(x - w/2, c12_nd, w, label="No demand", color="#abd9e9", edgecolor="white")
    axes[0].bar(x + w/2, c12_wd, w, label="With demand", color="#2c7bb6", edgecolor="white")
    axes[0].set_ylabel("C(N) [$/MW·yr]", fontsize=10)
    axes[0].set_title("Grand Coalition Cost", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)

    # Panel 2: savings
    axes[1].bar(x - w/2, sav_nd, w, label="No demand", color="#b2df8a", edgecolor="white")
    axes[1].bar(x + w/2, sav_wd, w, label="With demand", color="#2ca25f", edgecolor="white")
    axes[1].set_ylabel("Savings [$/MW·yr]", fontsize=10)
    axes[1].set_title("Cooperative Savings", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)

    # Panel 3: PV nucleolus share
    axes[2].plot(x, sh1_nd, "o-", ms=6, label="No demand",   color="#abd9e9", lw=2)
    axes[2].plot(x, sh1_wd, "s-", ms=6, label="With demand", color="#2c7bb6", lw=2)
    axes[2].axhline(0.5, color="gray", ls=":", lw=1)
    axes[2].set_ylabel("PV nucleolus share  x*₁/C(N)", fontsize=10)
    axes[2].set_title("PV Player's Share of Grand Coalition Cost", fontsize=11)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(months, fontsize=9)
    axes[2].set_xlabel("Month (2019)", fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].set_ylim(0.2, 0.7)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(P5_DIR, "monthly_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    print("Generating P5 plots...\n")

    required = [
        os.path.join(P5_DIR, "pairings.csv"),
        os.path.join(P5_DIR, "scale_sweep.csv"),
        os.path.join(P5_DIR, "monthly_with_demand.csv"),
        os.path.join(P5_DIR, "monthly_no_demand.csv"),
        os.path.join(P5_DIR, "demand_profiles.csv"),
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("ERROR: missing data. Run `julia problem5.jl` first.")
        for f in missing:
            print(f"  - {f}")
        exit(1)

    plot_demand_profiles()
    plot_pairings()
    plot_scale_sweep()
    plot_monthly_comparison()
    print("\nDone.")
