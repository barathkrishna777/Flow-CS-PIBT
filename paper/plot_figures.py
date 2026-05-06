#!/usr/bin/env python3
"""
Publication-quality figures for continuous-space MAPF paper.
Generates PDF (vector) and high-DPI PNG for each figure.

Requirements: matplotlib, numpy
Run: python paper/plot_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- Color palette (colorblind-safe, grayscale-readable) ---
C = {
    "orca":           "#7F7F7F",  # neutral gray
    "po_orca":        "#5F6B7A",  # muted slate
    "straight_epibt": "#2A9D8F",  # teal
    "flow_orca":      "#D4890E",  # amber
    "flow_epibt":     "#264653",  # deep navy/indigo
}

MARKERS = {
    "orca": "s",
    "po_orca": "D",
    "straight_epibt": "^",
    "flow_orca": "v",
    "flow_epibt": "o",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.2,
    "lines.markersize": 6,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  -> {name}.pdf / .png")


# ====================================================================
# Figure 1: Mechanism Decomposition (connected dot / slope)
# ====================================================================
def fig1_mechanism_decomposition():
    methods = ["ORCA", "PO-ORCA", "Straight\n+EPIBT", "Flow\n+ORCA", "Flow\n+EPIBT"]
    at_goal = [0.168, 0.149, 0.774, 0.486, 0.874]
    coll =    [4168.9, 2626.3, 1306.9, 3996.0, 633.4]
    colors = [C["orca"], C["po_orca"], C["straight_epibt"], C["flow_orca"], C["flow_epibt"]]
    markers = ["s", "D", "^", "v", "o"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.2), height_ratios=[1, 1])
    fig.subplots_adjust(hspace=0.45)

    xs = np.arange(len(methods))

    # Top panel: AtGoal
    for i in range(len(methods) - 1):
        ax1.plot([xs[i], xs[i+1]], [at_goal[i], at_goal[i+1]],
                 color="#BBBBBB", lw=0.7, zorder=1)
    for i, (x, y) in enumerate(zip(xs, at_goal)):
        ax1.scatter(x, y, c=colors[i], marker=markers[i], s=50, zorder=3, edgecolors="white", linewidths=0.4)
        offset = 0.04 if y < 0.8 else -0.05
        ax1.text(x, y + offset, f"{y:.3f}", ha="center", va="bottom" if offset > 0 else "top", fontsize=7)

    ax1.set_ylabel("AtGoal")
    ax1.set_ylim(0, 1.02)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(methods, fontsize=7)
    ax1.axhline(0, color="#CCCCCC", lw=0.4)
    ax1.set_title("random-32-32-10, N=100, 512 steps", fontsize=9, fontweight="bold", pad=8)

    # Annotations
    ax1.annotate("", xy=(2, 0.774), xytext=(0, 0.168),
                 arrowprops=dict(arrowstyle="->", color=C["straight_epibt"], lw=1.0))
    ax1.text(1.0, 0.50, "Shield:\n+60.6 pp", fontsize=6.5, color=C["straight_epibt"],
             ha="center", style="italic")
    ax1.text(3.5, 0.92, "+10.0 pp\n(learning)", fontsize=6.5, color=C["flow_epibt"],
             ha="center", style="italic")

    # Bottom panel: Collisions
    for i in range(len(methods) - 1):
        ax2.plot([xs[i], xs[i+1]], [coll[i], coll[i+1]],
                 color="#BBBBBB", lw=0.7, zorder=1)
    for i, (x, y) in enumerate(zip(xs, coll)):
        ax2.scatter(x, y, c=colors[i], marker=markers[i], s=50, zorder=3, edgecolors="white", linewidths=0.4)
        offset = 150 if y < 3000 else -250
        ax2.text(x, y + offset, f"{int(y)}", ha="center", va="bottom" if offset > 0 else "top", fontsize=7)
    ax2.set_ylabel("Collisions")
    ax2.set_ylim(0, 5000)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(methods, fontsize=7)
    ax2.text(4.0, 1000, "84.8% fewer\nvs ORCA", fontsize=6.5, color=C["flow_epibt"],
             ha="center", style="italic")

    save(fig, "fig1_mechanism_decomposition")


# ====================================================================
# Figure 2: Generalization / OOD (dumbbell small multiples)
# ====================================================================
def fig2_generalization():
    maps = [
        ("random-64\nN=50", {"orca": 0.048, "straight_epibt": 0.505, "flow_epibt": 0.614}),
        ("room-32\nN=50",   {"orca": 0.012, "straight_epibt": 0.114, "flow_epibt": 0.262}),
        ("warehouse\nN=50", {"orca": 0.132, "straight_epibt": 0.411, "flow_epibt": 0.255}),
        ("warehouse\nN=100",{"orca": 0.125, "straight_epibt": 0.394, "flow_epibt": 0.217}),
    ]
    method_order = ["orca", "straight_epibt", "flow_epibt"]
    labels = {"orca": "ORCA", "straight_epibt": "Straight+EPIBT", "flow_epibt": "Flow+EPIBT"}

    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.0), sharey=True)
    fig.subplots_adjust(wspace=0.12)

    for ax, (title, data) in zip(axes, maps):
        ys = np.arange(len(method_order))
        vals = [data[m] for m in method_order]
        cols = [C[m] for m in method_order]

        # Horizontal dumbbell connecting all three
        ax.plot(vals, ys, color="#CCCCCC", lw=1.0, zorder=1)
        for i, (v, c) in enumerate(zip(vals, cols)):
            ax.scatter(v, i, color=c, s=55, zorder=3, edgecolors="white", linewidths=0.4)
            ax.text(v + 0.02, i + 0.15, f"{v:.3f}", fontsize=6.5, va="bottom")

        ax.set_xlim(-0.02, 0.75)
        ax.set_xlabel("AtGoal", fontsize=7.5)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.axvline(0, color="#DDDDDD", lw=0.4)
        if ax == axes[0]:
            ax.set_yticks(ys)
            ax.set_yticklabels([labels[m] for m in method_order], fontsize=7)
        else:
            ax.set_yticks([])
        ax.grid(axis="x", alpha=0.3, lw=0.3)

    # Annotation on warehouse panels
    axes[2].text(0.35, -0.7, "OOD reversal:\nlearned prior hurts",
                 fontsize=6, color=C["flow_epibt"], ha="center",
                 transform=axes[2].get_xaxis_transform())

    save(fig, "fig2_generalization_ood")


# ====================================================================
# Figure 3: Integration Steps Tradeoff
# ====================================================================
def fig3_integration_steps():
    steps = [3, 5, 10, 20]
    at_goal = {
        "empty N=50":  [0.989, 0.984, 0.982, 0.984],
        "empty N=100": [0.968, 0.961, 0.941, 0.956],
        "random N=50": [0.866, 0.893, 0.856, 0.870],
        "random N=100":[0.874, 0.879, 0.856, 0.868],
    }
    arr_plr = {
        "empty N=50":  [1.24, 1.26, 1.29, 1.31],
        "empty N=100": [1.28, 1.30, 1.34, 1.33],
        "random N=50": [1.37, 1.42, 1.48, 1.49],
        "random N=100":[1.462, 1.592, 1.644, 1.659],
    }
    line_styles = {
        "empty N=50":  (C["straight_epibt"], "-", "o"),
        "empty N=100": (C["straight_epibt"], "--", "s"),
        "random N=50": (C["flow_epibt"], "-", "o"),
        "random N=100":(C["flow_epibt"], "--", "s"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.2))
    fig.subplots_adjust(wspace=0.35)

    for key in at_goal:
        color, ls, marker = line_styles[key]
        ax1.plot(steps, at_goal[key], color=color, ls=ls, marker=marker, markersize=4, label=key)
        ax2.plot(steps, arr_plr[key], color=color, ls=ls, marker=marker, markersize=4, label=key)

    ax1.set_xlabel("Integration steps (k)")
    ax1.set_ylabel("AtGoal")
    ax1.set_xticks(steps)
    ax1.set_ylim(0.82, 1.01)
    ax1.set_title("Completion", fontsize=9, fontweight="bold")
    ax1.axvline(3, color="#EEEEEE", lw=5, zorder=0)
    ax1.text(3, 1.005, "default", ha="center", fontsize=6, color="#666666")

    ax2.set_xlabel("Integration steps (k)")
    ax2.set_ylabel("ArrPLR")
    ax2.set_xticks(steps)
    ax2.set_ylim(1.15, 1.75)
    ax2.set_title("Route efficiency", fontsize=9, fontweight="bold")
    ax2.axvline(3, color="#EEEEEE", lw=5, zorder=0)
    ax2.text(3, 1.76, "default", ha="center", fontsize=6, color="#666666")

    ax2.annotate("+13.5% longer\nat k=20 vs k=3", xy=(20, 1.659), xytext=(15, 1.72),
                 fontsize=6, color=C["flow_epibt"], ha="center",
                 arrowprops=dict(arrowstyle="->", color=C["flow_epibt"], lw=0.6))

    ax1.legend(loc="lower left", fontsize=6.5, frameon=False, ncol=1)

    save(fig, "fig3_integration_steps")


# ====================================================================
# Figure 4: Multi-Seed Robustness (dot + interval)
# ====================================================================
def fig4_multiseed():
    configs = [
        ("empty N=50",  [0.984, 0.965, 0.974]),
        ("empty N=100", [0.981, 0.948, 0.926]),
        ("random N=50", [0.868, 0.781, 0.801]),
        ("random N=100",[0.866, 0.789, 0.782]),
    ]
    straight_epibt_ref = {
        "empty N=50": 1.000,
        "empty N=100": 1.000,
        "random N=50": 0.774,
        "random N=100": 0.774,
    }
    seed_labels = ["s42", "s123", "s456"]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    ys = np.arange(len(configs))[::-1]
    for i, (label, seeds) in enumerate(configs):
        y = ys[i]
        mean = np.mean(seeds)
        std = np.std(seeds)

        # Interval bar
        ax.plot([min(seeds), max(seeds)], [y, y], color=C["flow_epibt"], lw=2.0, solid_capstyle="round")
        # Individual seeds
        for j, s in enumerate(seeds):
            ax.scatter(s, y, color=C["flow_epibt"], s=25, zorder=4,
                       edgecolors="white", linewidths=0.3,
                       marker=["o", "^", "s"][j])
        # Mean marker (larger)
        ax.scatter(mean, y, color=C["flow_epibt"], s=70, zorder=5,
                   edgecolors="white", linewidths=0.6, marker="D")
        # Straight+EPIBT reference
        ref = straight_epibt_ref[label]
        if ref < 0.99:
            ax.axvline(ref, color=C["straight_epibt"], lw=0.6, ls=":", alpha=0.7)

        # Annotation
        ax.text(max(seeds) + 0.012, y, f"{mean:.3f}±{std:.3f}", fontsize=6.5, va="center")

    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in configs], fontsize=7.5)
    ax.set_xlabel("AtGoal")
    ax.set_xlim(0.7, 1.05)
    ax.set_title("Training seed robustness (3 seeds)", fontsize=9, fontweight="bold")

    # Reference line label
    ax.text(0.774, ys[-1] + 0.35, "Straight+EPIBT", fontsize=6,
            color=C["straight_epibt"], ha="center")

    # Seed legend
    for j, sl in enumerate(seed_labels):
        ax.scatter([], [], color=C["flow_epibt"], marker=["o", "^", "s"][j], s=20, label=sl)
    ax.scatter([], [], color=C["flow_epibt"], marker="D", s=40, label="mean")
    ax.legend(loc="lower left", fontsize=6, frameon=False, ncol=4)

    save(fig, "fig4_multiseed_robustness")


# ====================================================================
# Figure 5: Qualitative Trajectory (schematic)
# ====================================================================
def fig5_qualitative_schematic():
    """
    Schematic qualitative figure. In a real paper this would use
    actual trajectory data; here we generate a representative layout
    to demonstrate the visual design and annotation style.
    """
    np.random.seed(42)

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
    fig.subplots_adjust(wspace=0.15)
    titles = ["ORCA", "Straight+EPIBTShield", "Flow+EPIBTShield"]
    congestion = [0.85, 0.35, 0.10]  # schematic congestion level

    for ax, title, cong in zip(axes, titles, congestion):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw obstacles (random blocks)
        for _ in range(8):
            ox = np.random.uniform(1, 8)
            oy = np.random.uniform(1, 8)
            w = np.random.uniform(0.3, 0.8)
            h = np.random.uniform(0.3, 0.8)
            ax.add_patch(plt.Rectangle((ox, oy), w, h, color="#E0E0E0", ec="#AAAAAA", lw=0.4))

        # Schematic agent trajectories
        n_agents = 8
        for a in range(n_agents):
            start = np.random.uniform(0.5, 2.0, 2)
            goal = np.random.uniform(8.0, 9.5, 2)

            # Generate path with congestion-dependent noise
            t = np.linspace(0, 1, 30)
            noise_scale = cong * 1.5
            path_x = start[0] + (goal[0] - start[0]) * t + np.cumsum(np.random.randn(30) * 0.05) * noise_scale
            path_y = start[1] + (goal[1] - start[1]) * t + np.cumsum(np.random.randn(30) * 0.05) * noise_scale

            # Truncate path for ORCA (deadlock)
            if cong > 0.5 and a > 2:
                cutoff = np.random.randint(8, 15)
                path_x = path_x[:cutoff]
                path_y = path_y[:cutoff]

            alpha = 0.4 if cong > 0.5 and a > 2 else 0.6
            ax.plot(path_x, path_y, color=C["flow_epibt"], alpha=alpha, lw=0.6)
            ax.scatter(path_x[0], path_y[0], color="#2A9D8F", s=12, zorder=5, marker="o")
            ax.scatter(path_x[-1], path_y[-1], color="#E76F51", s=12, zorder=5, marker="x")

        # Annotation
        if cong > 0.5:
            ax.text(5, 0.3, "deadlock region", fontsize=6, ha="center", color="#AA0000", style="italic")

    # Caption annotation
    fig.text(0.5, -0.02,
             "○ start   × end/stuck   gray: obstacles.  "
             "ORCA deadlocks in congestion; Flow+EPIBT resolves and reaches goals.",
             ha="center", fontsize=6.5, color="#555555")

    save(fig, "fig5_qualitative_trajectories")


# ====================================================================
if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_mechanism_decomposition()
    fig2_generalization()
    fig3_integration_steps()
    fig4_multiseed()
    fig5_qualitative_schematic()
    print(f"\nAll figures saved to: {OUT}/")
    print("Note: Figure 5 uses schematic data. Replace with actual")
    print("trajectory logs for the camera-ready version.")
