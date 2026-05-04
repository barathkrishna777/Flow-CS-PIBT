#!/usr/bin/env python3
"""Generate paper-style small-multiple metric panels by map and agent count."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.dont_write_bytecode = True

from report_data import CSV_PATHS, FIG_DIR, REPO_ROOT, RISHI12_MAPS, TABLE_DIR, ensure_dirs


PRIMARY8_GROUPS = [
    (
        "Well-connected / low coordination load",
        [
            "Paris_1_256",
            "empty-48-48",
            "den520d",
            "random-64-64-10",
        ],
    ),
    (
        "Topology and coordination stress",
        [
            "den312d",
            "random-32-32-10",
            "maze-128-128-2",
            "warehouse-10-20-10-2-1",
        ],
    ),
]

PRIMARY8_DENSITY_ORDER = [
    "Paris_1_256",
    "empty-48-48",
    "den520d",
    "random-64-64-10",
    "den312d",
    "random-32-32-10",
    "maze-128-128-2",
    "warehouse-10-20-10-2-1",
]


METHODS = {
    "flow": {
        "label": "FLOMAP",
        "short": "FLOMAP",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "ssil": {
        "label": "SSIL classifier",
        "short": "SSIL",
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
    },
}


PREVIEW_PATHS = {
    "Paris_1_256": REPO_ROOT / "presentation/map_images/heldout/Paris_1_256.png",
    "empty-48-48": REPO_ROOT / "presentation/map_images/heldout/empty-48-48.png",
    "den312d": REPO_ROOT / "presentation/map_images/heldout/den312d.png",
    "maze-128-128-2": REPO_ROOT / "presentation/map_images/heldout/maze-128-128-2.png",
    "random-64-64-10": REPO_ROOT / "presentation/map_images/heldout/random-64-64-10.png",
    "warehouse-10-20-10-2-1": REPO_ROOT
    / "presentation/map_images/heldout/warehouse-10-20-10-2-1.png",
    "Berlin_1_256": REPO_ROOT / "presentation/map_images/seen/Berlin_1_256.png",
    "empty-32-32": REPO_ROOT / "presentation/map_images/seen/empty-32-32.png",
    "maze-32-32-4": REPO_ROOT / "presentation/map_images/seen/maze-32-32-4.png",
    "random-64-64-20": REPO_ROOT / "presentation/map_images/seen/random-64-64-20.png",
    "room-64-64-16": REPO_ROOT / "presentation/map_images/seen/room-64-64-16.png",
    "warehouse-20-40-10-2-1": REPO_ROOT
    / "presentation/map_images/seen/warehouse-20-40-10-2-1.png",
}


DISPLAY_NAMES = {
    "warehouse-10-20-10-2-1": "warehouse-10-20",
    "warehouse-20-40-10-2-1": "warehouse-20-40",
}


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def group_metric(
    rows: list[dict[str, str]],
    metric: str,
    cost_column: str,
) -> dict[tuple[str, int], dict[str, float]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["mapName"], int(number(row, "agentNum")))].append(row)

    out: dict[tuple[str, int], dict[str, float]] = {}
    for key, value_rows in groups.items():
        if metric == "success":
            successes = sum(truthy(row["success"]) for row in value_rows)
            n = len(value_rows)
            value = 100.0 * successes / n
            if n:
                z = 1.96
                phat = successes / n
                denom = 1.0 + z * z / n
                center = (phat + z * z / (2.0 * n)) / denom
                half = z * ((phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
                low = 100.0 * max(0.0, center - half)
                high = 100.0 * min(1.0, center + half)
                err_low = value - low
                err_high = high - value
            else:
                err_low = err_high = 0.0
        elif metric == "cost":
            samples = [
                number(row, cost_column) / max(number(row, "agentNum"), 1.0)
                for row in value_rows
            ]
            value = mean(samples)
            err_low = err_high = stdev(samples) / (len(samples) ** 0.5) if len(samples) > 1 else 0.0
        elif metric == "runtime":
            samples = [number(row, "runtime") for row in value_rows]
            value = mean(samples)
            err_low = err_high = stdev(samples) / (len(samples) ** 0.5) if len(samples) > 1 else 0.0
        else:
            raise ValueError(f"unknown metric: {metric}")
        out[key] = {
            "value": value,
            "rows": len(value_rows),
            "err_low": err_low,
            "err_high": err_high,
        }
    return out


def all_agents(*datasets: dict[tuple[str, int], dict[str, float]]) -> list[int]:
    values = sorted({agent for data in datasets for (_map_name, agent) in data})
    return values


def write_metric_table(
    path: Path,
    maps: list[str],
    flow_data: dict[tuple[str, int], dict[str, float]],
    ssil_data: dict[tuple[str, int], dict[str, float]],
) -> None:
    agents = all_agents(flow_data, ssil_data)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["map", "agent_count", "flow", "ssil", "flow_rows", "ssil_rows"])
        for map_name in maps:
            for agent in agents:
                frow = flow_data.get((map_name, agent), {})
                srow = ssil_data.get((map_name, agent), {})
                writer.writerow(
                    [
                        map_name,
                        agent,
                        f'{frow.get("value", "")}',
                        f'{srow.get("value", "")}',
                        frow.get("rows", ""),
                        srow.get("rows", ""),
                    ]
                )


def nice_ylim(metric: str, values: list[float]) -> tuple[float, float]:
    if metric == "success":
        return (-2.0, 105.0)
    if not values:
        return (0.0, 1.0)
    ymax = max(values)
    if metric == "cost":
        return (0.0, max(10.0, ymax * 1.10))
    return (0.0, max(5.0, ymax * 1.15))


def load_preview(map_name: str):
    import numpy as np
    import matplotlib.image as mpimg

    path = PREVIEW_PATHS.get(map_name)
    if path is not None and path.exists():
        img = mpimg.imread(path)
        if img.ndim == 3:
            return img[..., :3]
        return img

    # Repository fallback: schematic thumbnails are used only when no map image
    # or MovingAI .map file is available locally.
    rng = np.random.default_rng(abs(hash(map_name)) % (2**32))
    if "random" in map_name:
        size = 48 if "64" in map_name else 32
        img = rng.random((size, size)) > (0.82 if "20" in map_name else 0.90)
        return 1.0 - img.astype(float)
    if "empty" in map_name:
        return np.ones((48, 48)) * 0.92
    if "warehouse" in map_name:
        img = np.zeros((42, 62)) + 0.92
        img[4:10, 7:55] = 0.10
        for y in range(16, 37, 6):
            img[y : y + 2, 8:54] = 0.25
        return img
    if "maze" in map_name:
        img = np.ones((48, 48)) * 0.92
        img[2:-2:5, 2:-2] = 0.12
        img[2:-2, 2:-2:9] = 0.12
        img[7:-2:10, 5:43:12] = 0.92
        return img
    if map_name.startswith("den"):
        img = rng.random((48, 48)) > 0.72
        img = 1.0 - img.astype(float)
        img[8:38, 8:38] = np.maximum(img[8:38, 8:38], 0.85)
        img[12:35:7, 12:35] = 0.18
        return img
    return np.ones((48, 48)) * 0.88


def draw_preview(ax, map_name: str) -> None:
    ax.imshow(load_preview(map_name), cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#444444")


def metric_values_for_map(
    map_name: str,
    flow_data: dict[tuple[str, int], dict[str, float]],
    ssil_data: dict[tuple[str, int], dict[str, float]],
) -> list[float]:
    values: list[float] = []
    for data in [flow_data, ssil_data]:
        values.extend(
            float(row["value"])
            for (cur_map, _agent), row in data.items()
            if cur_map == map_name
        )
    return values


def plot_metric_axis(
    ax,
    map_name: str,
    flow_data: dict[tuple[str, int], dict[str, float]],
    ssil_data: dict[tuple[str, int], dict[str, float]],
    metric: str,
    ylabel: str | None,
    show_xlabel: bool,
) -> None:
    agents = all_agents(flow_data, ssil_data)
    for key, data in [("flow", flow_data), ("ssil", ssil_data)]:
        xs: list[int] = []
        ys: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for agent in agents:
            row = data.get((map_name, agent))
            if row is None:
                continue
            xs.append(agent)
            ys.append(float(row["value"]))
            lows.append(float(row.get("err_low", 0.0)))
            highs.append(float(row.get("err_high", 0.0)))
        if not xs:
            continue
        style = METHODS[key]
        ax.errorbar(
            xs,
            ys,
            yerr=[lows, highs],
            label=style["short"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.2,
            markersize=2.7,
            markeredgewidth=0.4,
            markeredgecolor="white",
            elinewidth=0.55,
            capsize=1.4,
            capthick=0.55,
        )

    values = metric_values_for_map(map_name, flow_data, ssil_data)
    ax.set_ylim(*nice_ylim(metric, values))
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.55)
    ax.grid(axis="x", color="#F0F0F0", linewidth=0.4)
    ax.tick_params(axis="both", labelsize=6, length=2.2, pad=1)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    if show_xlabel:
        ax.set_xlabel("# agents", fontsize=7, labelpad=1)
    else:
        ax.set_xlabel("")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.7)


def plot_grouped_metric_plate(
    out_path: Path,
    groups: list[tuple[str, list[str]]],
    flow_metrics: dict[str, dict[tuple[str, int], dict[str, float]]],
    ssil_metrics: dict[str, dict[tuple[str, int], dict[str, float]]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit(
                "matplotlib is required for grouped metric-panel PNG generation. "
                "Use `.venv-plot/bin/python research_report/scripts/plot_density_panels.py` "
                "from the repo root, or install matplotlib in the active Python."
            ) from exc
        raise

    ncols = max(len(maps) for _name, maps in groups)
    rows_per_group = 4
    gap_rows = max(0, len(groups) - 1)
    total_rows = rows_per_group * len(groups) + gap_rows
    height_ratios: list[float] = []
    for group_idx in range(len(groups)):
        if group_idx:
            height_ratios.append(0.24)
        height_ratios.extend([0.32, 1.0, 1.0, 1.0])

    fig = plt.figure(figsize=(2.45 * ncols, 1.10 * total_rows + 0.25), dpi=300)
    gs = GridSpec(
        total_rows,
        ncols,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.32,
        wspace=0.28,
    )

    first_plot_ax = None
    metric_specs = [
        ("success", "Success", "Success (%)"),
        ("runtime", "Runtime", "Runtime (s)"),
        ("cost", "Cost", "Cost / agent"),
    ]

    row0 = 0
    for group_idx, (group_name, maps) in enumerate(groups):
        if group_idx:
            for col in range(ncols):
                fig.add_subplot(gs[row0, col]).axis("off")
            row0 += 1
        for col in range(ncols):
            if col >= len(maps):
                for r in range(rows_per_group):
                    fig.add_subplot(gs[row0 + r, col]).axis("off")
                continue

            map_name = maps[col]
            header_ax = fig.add_subplot(gs[row0, col])
            header_ax.axis("off")
            preview_ax = header_ax.inset_axes([0.76, 0.02, 0.22, 0.96])
            draw_preview(preview_ax, map_name)
            header_ax.text(
                0.00,
                0.48,
                DISPLAY_NAMES.get(map_name, map_name),
                ha="left",
                va="center",
                fontsize=7.2,
                fontstyle="italic",
                transform=header_ax.transAxes,
            )
            if col == 0:
                header_ax.text(
                    -0.02,
                    1.25,
                    group_name,
                    ha="left",
                    va="bottom",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#333333",
                    transform=header_ax.transAxes,
                    clip_on=False,
                )

            for metric_idx, (metric, _metric_title, ylabel) in enumerate(metric_specs):
                ax = fig.add_subplot(gs[row0 + metric_idx + 1, col])
                if first_plot_ax is None:
                    first_plot_ax = ax
                plot_metric_axis(
                    ax,
                    map_name,
                    flow_metrics[metric],
                    ssil_metrics[metric],
                    metric,
                    ylabel if col == 0 else None,
                    show_xlabel=metric_idx == len(metric_specs) - 1,
                )
        row0 += rows_per_group

    if first_plot_ax is not None:
        handles, labels = first_plot_ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=True,
            framealpha=1.0,
            edgecolor="#999999",
            fontsize=8.0,
            bbox_to_anchor=(0.5, 0.008),
        )

    fig.subplots_adjust(left=0.060, right=0.995, top=0.975, bottom=0.065)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panel(
    out_path: Path,
    maps: list[str],
    flow_data: dict[tuple[str, int], dict[str, float]],
    ssil_data: dict[tuple[str, int], dict[str, float]],
    metric: str,
    title: str,
    ylabel: str,
    ncols: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            raise SystemExit(
                "matplotlib is required for density-panel PNG generation. "
                "Use `.venv-plot/bin/python research_report/scripts/plot_density_panels.py` "
                "from the repo root, or install matplotlib in the active Python."
            ) from exc
        raise

    agents = all_agents(flow_data, ssil_data)
    nrows = (len(maps) + ncols - 1) // ncols
    fig_w = 3.2 * ncols
    fig_h = 2.25 * nrows + 0.55
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), sharex=False)
    flat_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    all_values: list[float] = []
    for data in [flow_data, ssil_data]:
        all_values.extend(float(row["value"]) for row in data.values())
    ylim = nice_ylim(metric, all_values)

    for idx, map_name in enumerate(maps):
        ax = flat_axes[idx]
        map_values: list[float] = []
        for data in [flow_data, ssil_data]:
            map_values.extend(
                float(row["value"])
                for (cur_map, _agent), row in data.items()
                if cur_map == map_name
            )
        panel_ylim = ylim if metric == "success" else nice_ylim(metric, map_values)
        for key, data in [("flow", flow_data), ("ssil", ssil_data)]:
            xs: list[int] = []
            ys: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for agent in agents:
                row = data.get((map_name, agent))
                if row is None:
                    continue
                xs.append(agent)
                ys.append(float(row["value"]))
                lows.append(float(row.get("err_low", 0.0)))
                highs.append(float(row.get("err_high", 0.0)))
            if not xs:
                continue
            style = METHODS[key]
            ax.errorbar(
                xs,
                ys,
                yerr=[lows, highs],
                label=style["short"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=4.4,
                markeredgewidth=0.7,
                markeredgecolor="white",
                elinewidth=0.75,
                capsize=1.8,
                capthick=0.75,
            )
        ax.set_title(map_name, fontsize=9.5, fontstyle="italic", pad=5)
        ax.set_ylim(*panel_ylim)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.8)
        ax.grid(axis="x", color="#EFEFEF", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=8)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("# agents", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(maps), len(flat_axes)):
        flat_axes[idx].axis("off")

    handles, labels = flat_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(title, y=0.99, fontsize=11, fontweight="normal")
    fig.tight_layout(rect=(0, 0.055, 1, 0.955))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_panel_set(
    name: str,
    display_name: str,
    maps: list[str],
    flow_path: Path,
    ssil_path: Path,
    cost_column: str,
    groups: list[tuple[str, list[str]]] | None = None,
) -> bool:
    if not flow_path.exists() or not ssil_path.exists():
        print(f"Skipping {name}: missing CSV(s)")
        if not flow_path.exists():
            print(f"  missing: {flow_path}")
        if not ssil_path.exists():
            print(f"  missing: {ssil_path}")
        return False

    flow_rows = load_rows(flow_path)
    ssil_rows = load_rows(ssil_path)
    metrics = [
        ("success", "All-Agent Success by Map and Agent Count", "Success (%)"),
        ("cost", "Path Cost per Agent by Map and Agent Count", "Cost / agent"),
        ("runtime", "Solution Time by Map and Agent Count", "Runtime (s)"),
    ]

    ncols = 4 if len(maps) <= 8 else 3
    flow_metric_data: dict[str, dict[tuple[str, int], dict[str, float]]] = {}
    ssil_metric_data: dict[str, dict[tuple[str, int], dict[str, float]]] = {}
    for metric, title_suffix, ylabel in metrics:
        flow_data = group_metric(flow_rows, metric, cost_column)
        ssil_data = group_metric(ssil_rows, metric, cost_column)
        flow_metric_data[metric] = flow_data
        ssil_metric_data[metric] = ssil_data
        write_metric_table(TABLE_DIR / f"{name}_{metric}_by_density.csv", maps, flow_data, ssil_data)
        plot_panel(
            FIG_DIR / f"{name}_{metric}_by_density.png",
            maps,
            flow_data,
            ssil_data,
            metric,
            f"{display_name} {title_suffix}",
            ylabel,
            ncols=ncols,
        )
    if groups:
        plot_grouped_metric_plate(
            FIG_DIR / f"{name}_grouped_metric_panel.png",
            groups,
            flow_metric_data,
            ssil_metric_data,
        )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cost-column",
        default="total_cost_not_resting_at_goal",
        choices=["total_cost_true", "total_cost_not_resting_at_goal"],
    )
    parser.add_argument(
        "--include-12-map",
        dest="include_12_map",
        action="store_true",
        help="Also generate 12-map density panels when the remote flow CSV is present.",
    )
    parser.add_argument(
        "--include-rishi12",
        dest="include_12_map",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    ensure_dirs()
    generated = generate_panel_set(
        "primary8",
        "8-map benchmark",
        PRIMARY8_DENSITY_ORDER,
        CSV_PATHS["rishi8_flow"],
        CSV_PATHS["rishi8_ssil"],
        args.cost_column,
        groups=PRIMARY8_GROUPS,
    )
    if args.include_12_map:
        generate_panel_set(
            "extended12",
            "12-map benchmark",
            RISHI12_MAPS,
            CSV_PATHS["rishi12_flow"],
            CSV_PATHS["rishi12_ssil"],
            args.cost_column,
        )
    if not generated:
        return 1
    print(f"Wrote density-panel figures to {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
