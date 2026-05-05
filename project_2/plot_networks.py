import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import csv, os

RESULTS = "results"

def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return list(reader)

# ── Problem 1 ────────────────────────────────────────────────────────────────

def plot_p1():
    nuc = read_csv(os.path.join(RESULTS, "problem1", "nucleolus.csv"))
    x_star = {int(r["player"]): float(r["x_star"]) for r in nuc}

    pos = {0: (0.5, 0.0), 1: (0.0, 0.9), 2: (1.0, 0.9)}
    labels = {0: "0\n(substation)", 1: "Gen 1", 2: "Gen 2"}
    edges = [(1, 0, 90.0), (2, 0, 100.0), (1, 2, 50.0)]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_aspect("equal")
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.35, 1.25)
    ax.axis("off")
    ax.set_title("P1: n = 2, T = 2  —  Network & Nucleolus", fontsize=13, fontweight="bold", pad=12)

    for i, j, inv in edges:
        xi, yi = pos[i]
        xj, yj = pos[j]
        ax.plot([xi, xj], [yi, yj], "k-", lw=1.5, zorder=1)
        mx, my = (xi + xj) / 2, (yi + yj) / 2
        dx, dy = xj - xi, yj - yi
        length = np.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        offset = 0.07
        ax.text(mx + nx * offset, my + ny * offset,
                f"INV = {inv:.0f}", fontsize=8, ha="center", va="center",
                color="dimgray", fontstyle="italic")

    for node, (x, y) in pos.items():
        if node == 0:
            ax.plot(x, y, "s", color="#2c7bb6", markersize=18, zorder=3)
            ax.text(x, y - 0.15, labels[node], fontsize=9, ha="center", va="top",
                    fontweight="bold", color="#2c7bb6")
        else:
            ax.plot(x, y, "o", color="#d7191c", markersize=18, zorder=3)
            ax.text(x, y + 0.10, labels[node], fontsize=9, ha="center", va="bottom",
                    fontweight="bold", color="#d7191c")
            ax.text(x, y - 0.12,
                    f"x* = {x_star[node]:.1f}  ({x_star[node]/120*100:.1f}%)",
                    fontsize=8, ha="center", va="top", color="#333")
        ax.text(x, y, str(node), fontsize=9, ha="center", va="center",
                color="white", fontweight="bold", zorder=4)

    legend_items = [
        mpatches.Patch(color="#2c7bb6", label="Substation (node 0)"),
        mpatches.Patch(color="#d7191c", label="Generator node"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8, framealpha=0.9)
    ax.text(0.5, -0.30, f"C(N) = 120.0    x* = (55.0, 65.0)    ε* = 35.0",
            fontsize=9, ha="center", va="top", color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#ccc"))

    out = os.path.join(RESULTS, "problem1", "network_diagram.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


# ── Problem 2 helper ─────────────────────────────────────────────────────────

def plot_p2(n):
    subdir = os.path.join(RESULTS, "problem2", f"n{n}")
    nodes = read_csv(os.path.join(subdir, "node_positions.csv"))
    lines = read_csv(os.path.join(subdir, "line_costs.csv"))
    nuc   = read_csv(os.path.join(subdir, "nucleolus.csv"))
    meta  = read_csv(os.path.join(subdir, "metadata.csv"))
    C_N   = float([r for r in meta if r["key"] == "C_N"][0]["value"])

    pos = {int(r["node"]): (float(r["x"]), float(r["y"])) for r in nodes}
    edges = [(int(r["from_node"]), int(r["to_node"]), float(r["inv_cost"])) for r in lines]
    x_star = {int(r["player"]): float(r["x_star"]) for r in nuc}

    fig, ax = plt.subplots(figsize=(9, 8) if n >= 10 else (7, 6.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"P2: n = {n}, T = {n}  —  Network & Nucleolus", fontsize=13,
                 fontweight="bold", pad=12)

    inv_values = [e[2] for e in edges]
    inv_min, inv_max = min(inv_values), max(inv_values)

    for i, j, inv in edges:
        xi, yi = pos[i]
        xj, yj = pos[j]
        norm = (inv - inv_min) / (inv_max - inv_min + 1e-9)
        alpha = 0.08 + 0.35 * (1 - norm) if n >= 10 else 0.15 + 0.6 * (1 - norm)
        lw = 0.4 + 1.6 * (1 - norm) if n >= 10 else 0.6 + 2.0 * (1 - norm)
        ax.plot([xi, xj], [yi, yj], "-", color="gray", lw=lw, alpha=alpha, zorder=1)

    if n <= 5:
        for i, j, inv in edges:
            xi, yi = pos[i]
            xj, yj = pos[j]
            mx, my = (xi + xj) / 2, (yi + yj) / 2
            ax.text(mx, my, f"{inv:.1f}", fontsize=6, ha="center", va="center",
                    color="gray", alpha=0.7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    ms_sub = 22 if n <= 5 else 18
    ms_gen = 20 if n <= 5 else 16
    fs_id  = 10 if n <= 5 else 8

    x0, y0 = pos[0]
    ax.plot(x0, y0, "s", color="#2c7bb6", markersize=ms_sub, zorder=3)
    ax.text(x0, y0, "0", fontsize=fs_id, ha="center", va="center",
            color="white", fontweight="bold", zorder=4)
    ax.text(x0, y0 - 4, "substation", fontsize=7, ha="center", va="top",
            fontweight="bold", color="#2c7bb6")

    _nudge = {}
    if n >= 10:
        placed = []
        for node_id in range(1, n + 1):
            x, y = pos[node_id]
            for px, py, pid in placed:
                if np.hypot(x - px, y - py) < 12:
                    _nudge[node_id] = (15, 10 if node_id % 2 == 0 else -12)
                    break
            placed.append((x, y, node_id))

    max_share = max(x_star[i] / C_N for i in x_star)
    cmap = plt.cm.YlOrRd

    for node_id in range(1, n + 1):
        x, y = pos[node_id]
        share = x_star[node_id] / C_N
        color = cmap(0.2 + 0.7 * share / max_share)
        ax.plot(x, y, "o", color=color, markersize=ms_gen, zorder=3,
                markeredgecolor="black", markeredgewidth=0.5)
        ax.text(x, y, str(node_id), fontsize=fs_id, ha="center", va="center",
                color="white" if share / max_share > 0.5 else "black",
                fontweight="bold", zorder=4)

        label = f"x*={x_star[node_id]:.1f}\n({share*100:.1f}%)"
        offset_y = 5 if n <= 5 else 4
        if n >= 10 and node_id in _nudge:
            nx_off, ny_off = _nudge[node_id]
            ax.annotate(label, xy=(x, y), xytext=(x + nx_off, y + ny_off),
                        fontsize=6, ha="center", va="bottom", color="#333",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.5))
        else:
            ax.text(x, y + offset_y, label, fontsize=7 if n <= 5 else 6,
                    ha="center", va="bottom", color="#333",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.85))

    legend_items = [
        mpatches.Patch(color="#2c7bb6", label="Substation (node 0)"),
        mpatches.Patch(color=cmap(0.25), label="Gen node (low share)"),
        mpatches.Patch(color=cmap(0.75), label="Gen node (high share)"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=8, framealpha=0.9)

    summary = f"C(N) = {C_N:.2f}    seed = 238"
    pad = 10
    xmin = min(p[0] for p in pos.values()) - pad
    xmax = max(p[0] for p in pos.values()) + pad
    ax.text((xmin + xmax) / 2, min(p[1] for p in pos.values()) - 8,
            summary, fontsize=9, ha="center", va="top", color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#ccc"))

    out = os.path.join(subdir, "network_diagram.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    print("Generating network diagrams...")
    plot_p1()
    plot_p2(3)
    plot_p2(10)
    print("Done.")
