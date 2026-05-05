import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import csv, os, re

RESULTS = "results"

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def parse_members(s):
    s = s.strip()
    if s in ("Int64[]", "[]"):
        return []
    nums = re.findall(r"\d+", s)
    return [int(x) for x in nums]

def load_problem(subdir):
    coalitions = read_csv(os.path.join(subdir, "coalition_costs.csv"))
    nuc = read_csv(os.path.join(subdir, "nucleolus.csv"))
    meta = read_csv(os.path.join(subdir, "metadata.csv"))
    C_N = float([r for r in meta if r["key"] == "C_N"][0]["value"])
    n = int([r for r in meta if r["key"] == "n"][0]["value"])

    sizes, costs = [], []
    for r in coalitions:
        members = parse_members(r["members"])
        if len(members) == 0:
            continue
        sizes.append(len(members))
        costs.append(float(r["C_s"]))

    x_star = {int(r["player"]): float(r["x_star"]) for r in nuc}
    standalone = {}
    for r in coalitions:
        members = parse_members(r["members"])
        if len(members) == 1:
            standalone[members[0]] = float(r["C_s"])

    return n, C_N, np.array(sizes), np.array(costs), x_star, standalone


def plot_cost_vs_size(n, C_N, sizes, costs, subdir, label):
    """Scatter + box plot of C(s) vs |s|."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [3, 2]})
    fig.suptitle(f"{label}: Coalition Cost Analysis (n={n})", fontsize=14, fontweight="bold", y=1.02)

    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(sizes))
    ax1.scatter(sizes + jitter, costs, alpha=0.35, s=18, c="#2c7bb6", edgecolors="none")
    ax1.axhline(C_N, color="#d7191c", ls="--", lw=1.2, label=f"C(N) = {C_N:.2f}")

    unique_sizes = sorted(set(sizes))
    means = [costs[sizes == s].mean() for s in unique_sizes]
    ax1.plot(unique_sizes, means, "o-", color="#fdae61", lw=2, markersize=6, label="Mean C(s)", zorder=5)

    ax1.set_xlabel("Coalition size |s|", fontsize=11)
    ax1.set_ylabel("Coalition cost C(s)  [$/MW·yr]", fontsize=11)
    ax1.set_title("C(s) vs. coalition size", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.set_xticks(unique_sizes)
    ax1.grid(axis="y", alpha=0.3)

    grouped = [costs[sizes == s] for s in unique_sizes]
    bp = ax2.boxplot(grouped, positions=unique_sizes, widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor="#abd9e9", edgecolor="#2c7bb6"),
                     medianprops=dict(color="#d7191c", lw=2),
                     whiskerprops=dict(color="#2c7bb6"),
                     capprops=dict(color="#2c7bb6"),
                     flierprops=dict(marker=".", markersize=3, alpha=0.4))
    ax2.axhline(C_N, color="#d7191c", ls="--", lw=1.2)
    ax2.set_xlabel("Coalition size |s|", fontsize=11)
    ax2.set_ylabel("C(s)  [$/MW·yr]", fontsize=11)
    ax2.set_title("Distribution by size", fontsize=12)
    ax2.set_xticks(unique_sizes)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(subdir, "coalition_cost_vs_size.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


def plot_per_capita(n, C_N, sizes, costs, subdir, label):
    """Per-capita cost C(s)/|s| vs coalition size -- shows economies of scale."""
    per_capita = costs / sizes
    unique_sizes = sorted(set(sizes))

    fig, ax = plt.subplots(figsize=(7, 5))

    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(sizes))
    ax.scatter(sizes + jitter, per_capita, alpha=0.3, s=18, c="#2c7bb6", edgecolors="none")

    means = [per_capita[sizes == s].mean() for s in unique_sizes]
    ax.plot(unique_sizes, means, "o-", color="#d7191c", lw=2, markersize=6, label="Mean per-capita cost", zorder=5)

    sum_standalone = sum(costs[sizes == 1])
    ax.axhline(sum_standalone / n, color="#fdae61", ls=":", lw=1.5,
               label=f"Avg standalone = {sum_standalone/n:.1f}")
    ax.axhline(C_N / n, color="#2ca25f", ls="--", lw=1.5,
               label=f"Grand coalition per-capita = {C_N/n:.1f}")

    ax.set_xlabel("Coalition size |s|", fontsize=11)
    ax.set_ylabel("Per-capita cost  C(s)/|s|  [$/MW·yr]", fontsize=11)
    ax.set_title(f"{label}: Per-Capita Cost vs. Coalition Size (n={n})", fontsize=13, fontweight="bold")
    ax.set_xticks(unique_sizes)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(subdir, "per_capita_cost.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


def plot_standalone_vs_nucleolus(n, C_N, x_star, standalone, subdir, label):
    """Bar chart comparing standalone cost vs nucleolus allocation per player."""
    players = sorted(x_star.keys())
    stand_vals = [standalone[p] for p in players]
    nuc_vals = [x_star[p] for p in players]
    savings = [standalone[p] - x_star[p] for p in players]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, n * 0.9), 9),
                                    gridspec_kw={"height_ratios": [3, 2]})
    fig.suptitle(f"{label}: Standalone Cost vs. Nucleolus Allocation (n={n})",
                 fontsize=13, fontweight="bold", y=1.01)

    x = np.arange(len(players))
    w = 0.35
    bars1 = ax1.bar(x - w/2, stand_vals, w, label="Standalone C({i})", color="#fdae61", edgecolor="#e08214")
    bars2 = ax1.bar(x + w/2, nuc_vals, w, label="Nucleolus x*_i", color="#2c7bb6", edgecolor="#1a5276")

    ax1.set_xlabel("Player", fontsize=11)
    ax1.set_ylabel("Cost  [$/MW·yr]", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(p) for p in players])
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars1, stand_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="#333")
    for bar, val in zip(bars2, nuc_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=7, color="#333")

    colors = ["#2ca25f" if s > 0 else "#d7191c" for s in savings]
    bars3 = ax2.bar(x, savings, 0.5, color=colors, edgecolor=[c for c in colors])
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Player", fontsize=11)
    ax2.set_ylabel("Savings  [$/MW·yr]", fontsize=11)
    ax2.set_title("Individual savings = C({i}) − x*_i", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(p) for p in players])
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars3, savings):
        va = "bottom" if val >= 0 else "top"
        offset = 0.5 if val >= 0 else -0.5
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                 f"{val:.1f}", ha="center", va=va, fontsize=7, color="#333")

    fig.tight_layout()
    out = os.path.join(subdir, "standalone_vs_nucleolus.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    for problem, subdir, label in [
        ("n3",  os.path.join(RESULTS, "problem2", "n3"),  "P2"),
        ("n10", os.path.join(RESULTS, "problem2", "n10"), "P2"),
    ]:
        print(f"\n--- {label} (n={problem[1:]}) ---")
        n, C_N, sizes, costs, x_star, standalone = load_problem(subdir)
        plot_cost_vs_size(n, C_N, sizes, costs, subdir, label)
        plot_per_capita(n, C_N, sizes, costs, subdir, label)
        plot_standalone_vs_nucleolus(n, C_N, x_star, standalone, subdir, label)

    print("\nDone.")
