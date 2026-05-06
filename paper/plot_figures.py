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
    ax2.text(3.55, 1200, "84.8% fewer\nvs ORCA", fontsize=6.5, color=C["flow_epibt"],
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

    # sharey=False so set_yticks on one panel doesn't clobber others
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.2))
    fig.subplots_adjust(wspace=0.12, bottom=0.22)

    ys = np.arange(len(method_order))
    method_display = [labels[m] for m in method_order]

    for col_idx, (ax, (title, data)) in enumerate(zip(axes, maps)):
        vals = [data[m] for m in method_order]
        cols = [C[m] for m in method_order]

        # Horizontal dumbbell connecting all three
        ax.plot(vals, ys, color="#CCCCCC", lw=1.0, zorder=1)
        for i, (v, c) in enumerate(zip(vals, cols)):
            ax.scatter(v, i, color=c, s=55, zorder=3, edgecolors="white", linewidths=0.4)
            ax.text(v + 0.02, i + 0.12, f"{v:.3f}", fontsize=6.5, va="bottom")

        ax.set_xlim(-0.02, 0.75)
        ax.set_ylim(-0.6, 2.5)
        ax.set_xlabel("AtGoal", fontsize=7.5)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.axvline(0, color="#DDDDDD", lw=0.4)
        ax.set_yticks(ys)
        if col_idx == 0:
            ax.set_yticklabels(method_display, fontsize=7)
        else:
            ax.set_yticklabels([""] * len(ys))
            ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.3, lw=0.3)

        # OOD annotation inside the warehouse panels
        if "warehouse" in title:
            ax.text(0.37, -0.50, "OOD reversal", fontsize=6,
                    color="#AA4400", ha="center", style="italic")

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

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    fig.subplots_adjust(bottom=0.22)

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

    # Reference line label — placed to the right of the dotted line,
    # anchored at the random N=50 row where there is vertical space
    ax.text(0.779, ys[1] + 0.18, "Straight+EPIBT", fontsize=5.5,
            color=C["straight_epibt"], ha="left", va="bottom")

    # Seed legend: compact box in lower-right quadrant (random rows have
    # data only up to 0.868, so x > 0.88 is clear in those rows)
    for j, sl in enumerate(seed_labels):
        ax.scatter([], [], color=C["flow_epibt"], marker=["o", "^", "s"][j], s=20, label=sl)
    ax.scatter([], [], color=C["flow_epibt"], marker="D", s=40, label="mean")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=6,
              frameon=False, ncol=4, handletextpad=0.3, columnspacing=0.8)

    save(fig, "fig4_multiseed_robustness")


# ====================================================================
# Figure 5: Qualitative Trajectory (schematic)
# ====================================================================
def _smooth(pts, k=5):
    """Linearly interpolate waypoints then apply a moving-average smoother."""
    wp = np.array(pts, dtype=float)
    t_in = np.linspace(0, 1, len(wp))
    t_out = np.linspace(0, 1, 80)
    x = np.interp(t_out, t_in, wp[:, 0])
    y = np.interp(t_out, t_in, wp[:, 1])
    pad = np.ones(k)
    x = np.convolve(x, pad / k, mode="same")
    y = np.convolve(y, pad / k, mode="same")
    return x, y


def fig5_qualitative_schematic():
    """
    Same map / same scenario rendered under three planners.

    Paths are illustrative schematics designed to faithfully represent
    each method's qualitative behaviour:
      ORCA          — agents deadlock in the crossing zone, oscillate, get stuck.
      Straight+EPIBT — agents reach goals; shield backtracking produces
                       angular detours and sharp direction reversals.
      Flow+EPIBT    — agents reach goals; learned prior produces smooth
                       proactive arcs that route around the crossing zone
                       before conflicts arise (curvier, not straighter).

    Replace with real trajectory logs for camera-ready submission.
    """
    # ------------------------------------------------------------------
    # Fixed map (same across all panels)
    # ------------------------------------------------------------------
    obstacles = [
        (1.5, 1.2, 1.4, 0.9),   # bottom-left cluster
        (5.2, 1.2, 1.0, 1.8),   # bottom-centre post
        (1.2, 5.0, 1.2, 1.6),   # left-centre wall
        (5.5, 5.2, 2.0, 1.0),   # centre-right wall
        (7.5, 3.2, 1.0, 2.0),   # right-centre post
    ]

    # Six agents with realistic cross-traffic:
    #   A, B  — left → right
    #   C, D  — right → left  (cross A and B)
    #   E     — top → bottom
    #   F     — bottom → top  (crosses E)
    starts = [(0.4, 3.2), (0.4, 6.8), (9.6, 4.5), (9.6, 7.6), (3.5, 9.6), (6.5, 0.4)]
    goals  = [(9.6, 7.2), (9.6, 2.8), (0.4, 6.2), (0.4, 3.8), (3.5, 0.4), (6.5, 9.6)]

    # Per-agent color — distinguishable in grayscale via marker+shade
    colors = ["#264653", "#2A9D8F", "#7F7F7F", "#5F6B7A", "#D4890E", "#CC79A7"]

    # ------------------------------------------------------------------
    # Hand-designed waypoints per method
    # Each list entry = waypoints for one agent.
    # ------------------------------------------------------------------

    # ORCA: agents head toward goals, deadlock in the crossing zone,
    # then oscillate around their stuck position.
    _osc = lambda cx, cy, rx, ry: [
        (cx + rx * np.sin(a), cy + ry * np.cos(a))
        for a in np.linspace(0, 2.5 * np.pi, 12)
    ]
    orca_wps = [
        # A: stuck at centre-right of crossing zone
        [(0.4,3.2),(1.5,3.5),(3.0,4.0),(4.5,4.5),(5.0,5.0)] + _osc(4.8,4.8,0.3,0.2),
        # B: gets slightly further but also stuck
        [(0.4,6.8),(1.5,6.5),(3.0,6.0),(4.5,5.5),(5.0,5.2)] + _osc(5.0,5.4,0.25,0.2),
        # C (right→left): stuck at right side of crossing zone
        [(9.6,4.5),(8.5,4.8),(7.5,5.0),(6.5,5.0),(6.0,4.8)] + _osc(6.2,4.9,0.25,0.2),
        # D: partially stuck, slightly further
        [(9.6,7.6),(8.5,7.2),(7.5,6.8),(6.5,6.5),(6.0,6.2)] + _osc(6.1,6.3,0.25,0.2),
        # E (top→bottom): stuck near centre
        [(3.5,9.6),(3.5,8.5),(3.5,7.5),(3.5,6.5),(3.5,5.8)] + _osc(3.5,5.6,0.2,0.25),
        # F (bottom→top): stuck near centre
        [(6.5,0.4),(6.5,1.5),(6.5,2.5),(6.5,3.5),(6.5,4.2)] + _osc(6.5,4.4,0.2,0.25),
    ]
    orca_arrived = [False] * 6

    # Straight+EPIBT: all agents reach goals; shield backtracking causes
    # sharp angular detours — routes that zig and zag around conflicts.
    straight_wps = [
        # A: zigzags up to avoid crossing zone, then descends to goal
        [(0.4,3.2),(1.5,3.2),(2.5,3.5),(3.5,4.2),(4.0,5.5),(4.5,6.2),
         (5.5,6.5),(6.5,6.8),(7.5,7.0),(8.5,7.2),(9.6,7.2)],
        # B: dips down sharply then curves up to its goal
        [(0.4,6.8),(1.5,6.5),(2.5,5.5),(3.0,4.2),(3.5,3.2),(4.5,3.0),
         (5.5,3.2),(6.5,3.0),(7.5,2.8),(8.5,2.8),(9.6,2.8)],
        # C: detours through upper map to avoid crossing
        [(9.6,4.5),(8.5,4.2),(7.8,3.5),(7.5,2.5),(6.5,2.0),(5.5,2.2),
         (4.5,2.8),(3.5,3.5),(2.5,4.8),(1.5,5.8),(0.4,6.2)],
        # D: sharp dip down then recovers
        [(9.6,7.6),(8.5,7.8),(7.5,8.0),(6.5,7.5),(5.5,6.5),(4.5,5.5),
         (3.5,4.8),(2.5,4.2),(1.5,4.0),(0.4,3.8)],
        # E: takes a sharp sideways detour to avoid F
        [(3.5,9.6),(3.5,8.5),(3.0,7.5),(2.5,6.5),(2.2,5.5),(2.5,4.5),
         (3.0,3.5),(3.2,2.5),(3.5,1.5),(3.5,0.4)],
        # F: sharp sideways then back
        [(6.5,0.4),(6.5,1.5),(7.0,2.5),(7.5,3.5),(7.5,4.5),(7.2,5.5),
         (7.0,6.5),(6.8,7.5),(6.5,8.5),(6.5,9.6)],
    ]
    straight_arrived = [True] * 6

    # Flow+EPIBT: all agents reach goals; learned prior proactively routes
    # agents around the crossing zone with smooth arcs — curvier than the
    # straight-line baseline, but purposefully curved rather than angular.
    flow_wps = [
        # A: smooth arc upward, through clear corridor above obstacles
        [(0.4,3.2),(1.2,4.2),(2.2,5.2),(3.2,6.0),(4.5,6.8),
         (5.8,7.0),(7.0,7.2),(8.2,7.2),(9.6,7.2)],
        # B: smooth arc downward to avoid crossing zone
        [(0.4,6.8),(1.2,5.8),(2.2,4.5),(3.2,3.5),(4.5,3.0),
         (5.8,2.8),(7.0,2.8),(8.2,2.8),(9.6,2.8)],
        # C: smooth arc through lower corridor
        [(9.6,4.5),(8.5,3.8),(7.2,3.2),(6.0,3.0),(4.8,3.5),
         (3.8,4.5),(2.8,5.5),(1.8,6.0),(0.4,6.2)],
        # D: smooth arc through upper corridor
        [(9.6,7.6),(8.5,8.0),(7.5,8.2),(6.2,8.0),(5.0,7.2),
         (3.8,6.0),(2.8,5.0),(1.8,4.2),(0.4,3.8)],
        # E: gentle lateral offset to yield to F, then smooth descent
        [(3.5,9.6),(3.2,8.5),(2.8,7.5),(2.5,6.5),(2.5,5.5),
         (2.8,4.5),(3.0,3.5),(3.2,2.5),(3.5,1.5),(3.5,0.4)],
        # F: symmetric gentle offset, smooth ascent
        [(6.5,0.4),(6.8,1.5),(7.0,2.5),(7.2,3.5),(7.2,4.5),
         (7.0,5.5),(6.8,6.5),(6.6,7.5),(6.5,8.5),(6.5,9.6)],
    ]
    flow_arrived = [True] * 6

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    all_methods = [
        ("ORCA",               orca_wps,     orca_arrived),
        ("Straight+EPIBT",     straight_wps, straight_arrived),
        ("Flow+EPIBT",         flow_wps,     flow_arrived),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5))
    fig.subplots_adjust(wspace=0.06, bottom=0.14)

    for ax, (title, wps_list, arrived_list) in zip(axes, all_methods):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#BBBBBB")

        # Obstacles
        for ox, oy, ow, oh in obstacles:
            ax.add_patch(plt.Rectangle(
                (ox, oy), ow, oh, color="#E6E6E6", ec="#C0C0C0", lw=0.4, zorder=1))

        # Trajectories
        for i, (wps, arrived) in enumerate(zip(wps_list, arrived_list)):
            px, py = _smooth(wps)
            c = colors[i]
            ax.plot(px, py, color=c, alpha=0.6, lw=0.9, zorder=2)
            sx, sy = wps[0]
            gx, gy = goals[i]
            # Start
            ax.scatter(sx, sy, color=c, s=16, zorder=5,
                       edgecolors="white", linewidths=0.4, marker="o")
            # End: star at goal if arrived, X at stuck position
            if arrived:
                ax.scatter(gx, gy, color=c, s=30, zorder=5,
                           edgecolors="white", linewidths=0.3, marker="*")
            else:
                ex, ey = px[-1], py[-1]
                ax.scatter(ex, ey, color=c, s=22, zorder=5,
                           marker="x", linewidths=1.2)

        # ORCA deadlock annotation
        if title == "ORCA":
            ax.text(5.1, 5.1, "deadlock", fontsize=5.5,
                    ha="center", color="#AA0000", style="italic", zorder=6)
            ax.add_patch(plt.Circle((5.0, 5.0), 1.1,
                         fill=False, ec="#CC0000", lw=0.5, ls="--", zorder=3, alpha=0.5))

    # Shared caption
    fig.text(0.5, 0.01,
             "● start   ★ goal reached   ✕ stuck.   "
             "Straight+EPIBT: angular backtracking paths.   "
             "Flow+EPIBT: smooth proactive arcs (trained on EECBS).",
             ha="center", fontsize=5.5, color="#555555")

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
