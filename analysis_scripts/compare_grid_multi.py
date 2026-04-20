"""Create a multi-run Rishi-style grid eval comparison figure.

This is the same visual style as ``compare_grid_1v1.py`` but accepts two or
more CSVs so a single figure can compare Rishi, Flow-CS, Parth, etc.

Example:
    python -m analysis_scripts.compare_grid_multi \
        evals/rishi12_ssil_classifier.csv \
        evals/rishi12_wave10_newdata_clean_20260416_224144.csv \
        evals/rishi12_parth_wave2_ft_v2_epoch14.csv \
        --labels "SSIL classifier rishi12" "Flow-CS new data rishi12" "Parth wave2 rishi12" \
        --out-dir evals/final_slides/overview
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean


DEFAULT_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#17becf",
]
DEFAULT_MARKERS = ["o", "x", "^", "s", "D", "v"]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def _load_rows(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        raise SystemExit(f"Missing CSV: {path}")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class Summary:
    rows: int
    success_count: int
    success_rate: float
    at_goal_rate: float
    runtime: float
    cost_per_agent: float


def _summarize(rows: list[dict[str, str]], cost_column: str) -> Summary:
    if not rows:
        return Summary(0, 0, 0.0, 0.0, 0.0, 0.0)
    successes = [_truthy(row.get("success", "")) for row in rows]
    at_goal = [
        _int(row, "num_agents_at_goal") / max(_int(row, "agentNum"), 1) * 100.0
        for row in rows
    ]
    runtimes = [_float(row, "runtime") for row in rows]
    costs = [
        _float(row, cost_column) / max(_int(row, "agentNum"), 1)
        for row in rows
    ]
    return Summary(
        rows=len(rows),
        success_count=sum(successes),
        success_rate=sum(successes) / len(rows) * 100.0,
        at_goal_rate=mean(at_goal),
        runtime=mean(runtimes),
        cost_per_agent=mean(costs),
    )


def _group(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
) -> dict[tuple[object, ...], list[dict[str, str]]]:
    grouped: dict[tuple[object, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        group_key = []
        for key in keys:
            value: object = row[key]
            if key == "agentNum":
                value = _int(row, key)
            group_key.append(value)
        grouped[tuple(group_key)].append(row)
    return grouped


def _safe_name(label: str) -> str:
    return (
        label.lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace(",", "")
        .replace("__", "_")
    )


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(
    out_path: str,
    csvs: list[str],
    labels: list[str],
    datasets: list[list[dict[str, str]]],
    cost_column: str,
    figure_path: str,
) -> None:
    overall = [_summarize(rows, cost_column) for rows in datasets]
    lines = [
        "# Grid Eval Multi-Run Comparison",
        "",
        *[f"- {label}: `{csv}`" for label, csv in zip(labels, csvs)],
        f"- Cost-per-agent column: `{cost_column}`",
        f"- Figure: `{figure_path}`",
        "",
        "## Overall",
        "",
        _md_table(
            [
                "Run",
                "Rows",
                "Success",
                "Mean agents at goal",
                "Avg runtime",
                "Avg cost / agent",
            ],
            [
                [
                    label,
                    str(summary.rows),
                    f"{summary.success_rate:.2f}% ({summary.success_count}/{summary.rows})",
                    f"{summary.at_goal_rate:.2f}%",
                    f"{summary.runtime:.2f}s",
                    f"{summary.cost_per_agent:.2f}",
                ]
                for label, summary in zip(labels, overall)
            ],
        ),
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _plot(
    out_path: str,
    datasets: list[list[dict[str, str]]],
    labels: list[str],
    cost_column: str,
    title: str | None,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    maps = sorted({row["mapName"] for rows in datasets for row in rows})
    grouped = [_group(rows, ("mapName", "agentNum")) for rows in datasets]

    fig, axes = plt.subplots(
        3,
        len(maps),
        figsize=(max(18, 2.8 * len(maps)), 8.5),
        sharex=False,
    )
    if len(maps) == 1:
        axes = [[axes[0]], [axes[1]], [axes[2]]]

    metrics = [
        ("Success Rate", lambda s: s.success_rate / 100.0, (0.0, 1.05)),
        ("Runtime", lambda s: s.runtime, None),
        ("Solution Cost per Agent", lambda s: s.cost_per_agent, None),
    ]

    for col, map_name in enumerate(maps):
        agent_counts = sorted({
            key[1]
            for grouped_rows in grouped
            for key in grouped_rows
            if key[0] == map_name
        })
        for row_idx, (metric_name, getter, ylim) in enumerate(metrics):
            ax = axes[row_idx][col]
            for run_idx, (rows_by_key, label) in enumerate(zip(grouped, labels)):
                xs = []
                ys = []
                for agent in agent_counts:
                    rows = rows_by_key.get((map_name, agent), [])
                    if not rows:
                        continue
                    xs.append(agent)
                    ys.append(getter(_summarize(rows, cost_column)))
                ax.plot(
                    xs,
                    ys,
                    label=label,
                    color=DEFAULT_COLORS[run_idx % len(DEFAULT_COLORS)],
                    marker=DEFAULT_MARKERS[run_idx % len(DEFAULT_MARKERS)],
                    linewidth=1.7,
                    markersize=4,
                )

            if row_idx == 0:
                ax.set_title(map_name, fontsize=10)
            if col == 0:
                ax.set_ylabel(metric_name)
            if row_idx == 2:
                ax.set_xlabel("# Agents")
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25, linewidth=0.6)

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=min(len(labels), 4),
        frameon=True,
    )
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    else:
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a multi-run grid eval comparison report")
    parser.add_argument("csvs", nargs="+", help="Simulator-format CSVs")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels matching the CSV order")
    parser.add_argument("--out-dir", default="evals/grid_multi", help="Output directory")
    parser.add_argument("--name", default=None, help="Output filename stem")
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--cost-column",
        default="total_cost_not_resting_at_goal",
        choices=["total_cost_true", "total_cost_not_resting_at_goal"],
        help="CSV cost column used for solution cost per agent",
    )
    args = parser.parse_args()

    if len(args.csvs) < 2:
        parser.error("Pass at least two CSVs")
    labels = args.labels or [
        os.path.splitext(os.path.basename(path))[0] for path in args.csvs
    ]
    if len(labels) != len(args.csvs):
        parser.error("--labels must have the same length as csvs")

    datasets = [_load_rows(path) for path in args.csvs]
    os.makedirs(args.out_dir, exist_ok=True)
    stem = args.name or "_vs_".join(_safe_name(label) for label in labels)
    figure_path = os.path.join(args.out_dir, f"{stem}.png")
    report_path = os.path.join(args.out_dir, f"{stem}.md")

    _plot(figure_path, datasets, labels, args.cost_column, args.title, args.dpi)
    _write_report(report_path, args.csvs, labels, datasets, args.cost_column, figure_path)

    print(f"Wrote report: {report_path}")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    main()
