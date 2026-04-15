"""Create a one-on-one Rishi-style comparison from two grid eval CSVs.

Example:
    python -m analysis_scripts.compare_grid_1v1 \
        evals/rishi_full_flow_best.csv \
        evals/rishi_full_ssil_classifier.csv \
        --labels "Flow best" "SSIL classifier" \
        --out-dir evals/rishi_1v1
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean


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


def _group(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[object, ...], list[dict[str, str]]]:
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


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _winner(a: float, b: float, a_label: str, b_label: str, lower_is_better: bool = False) -> str:
    if abs(a - b) < 1e-9:
        return "tie"
    if lower_is_better:
        return a_label if a < b else b_label
    return a_label if a > b else b_label


def _write_report(
    out_path: str,
    csv_a: str,
    csv_b: str,
    label_a: str,
    label_b: str,
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    cost_column: str,
    figure_path: str,
) -> None:
    overall_a = _summarize(rows_a, cost_column)
    overall_b = _summarize(rows_b, cost_column)

    maps = sorted({row["mapName"] for row in rows_a + rows_b})
    agents = sorted({_int(row, "agentNum") for row in rows_a + rows_b})
    map_groups_a = _group(rows_a, ("mapName",))
    map_groups_b = _group(rows_b, ("mapName",))
    agent_groups_a = _group(rows_a, ("agentNum",))
    agent_groups_b = _group(rows_b, ("agentNum",))

    lines = [
        "# Grid Eval 1v1 Comparison",
        "",
        f"- A: `{label_a}` from `{csv_a}`",
        f"- B: `{label_b}` from `{csv_b}`",
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
                    label_a,
                    str(overall_a.rows),
                    f"{overall_a.success_rate:.2f}% ({overall_a.success_count}/{overall_a.rows})",
                    f"{overall_a.at_goal_rate:.2f}%",
                    f"{overall_a.runtime:.2f}s",
                    f"{overall_a.cost_per_agent:.2f}",
                ],
                [
                    label_b,
                    str(overall_b.rows),
                    f"{overall_b.success_rate:.2f}% ({overall_b.success_count}/{overall_b.rows})",
                    f"{overall_b.at_goal_rate:.2f}%",
                    f"{overall_b.runtime:.2f}s",
                    f"{overall_b.cost_per_agent:.2f}",
                ],
            ],
        ),
        "",
        "## Overall Delta",
        "",
        _md_table(
            ["Metric", f"{label_b} - {label_a}", "Winner"],
            [
                [
                    "Success",
                    f"{overall_b.success_rate - overall_a.success_rate:+.2f} pp",
                    _winner(overall_a.success_rate, overall_b.success_rate, label_a, label_b),
                ],
                [
                    "Mean agents at goal",
                    f"{overall_b.at_goal_rate - overall_a.at_goal_rate:+.2f} pp",
                    _winner(overall_a.at_goal_rate, overall_b.at_goal_rate, label_a, label_b),
                ],
                [
                    "Avg runtime",
                    f"{overall_b.runtime - overall_a.runtime:+.2f}s",
                    _winner(overall_a.runtime, overall_b.runtime, label_a, label_b, lower_is_better=True),
                ],
                [
                    "Avg cost / agent",
                    f"{overall_b.cost_per_agent - overall_a.cost_per_agent:+.2f}",
                    _winner(overall_a.cost_per_agent, overall_b.cost_per_agent, label_a, label_b, lower_is_better=True),
                ],
            ],
        ),
        "",
        "## Per Map",
        "",
    ]

    per_map_rows = []
    for map_name in maps:
        summary_a = _summarize(map_groups_a.get((map_name,), []), cost_column)
        summary_b = _summarize(map_groups_b.get((map_name,), []), cost_column)
        per_map_rows.append([
            map_name,
            f"{summary_a.success_rate:.2f}%",
            f"{summary_b.success_rate:.2f}%",
            f"{summary_b.success_rate - summary_a.success_rate:+.2f} pp",
            f"{summary_a.at_goal_rate:.2f}%",
            f"{summary_b.at_goal_rate:.2f}%",
            f"{summary_a.runtime:.2f}s",
            f"{summary_b.runtime:.2f}s",
            f"{summary_a.cost_per_agent:.2f}",
            f"{summary_b.cost_per_agent:.2f}",
        ])
    lines.append(_md_table(
        [
            "Map",
            f"{label_a} success",
            f"{label_b} success",
            "Success delta",
            f"{label_a} at-goal",
            f"{label_b} at-goal",
            f"{label_a} runtime",
            f"{label_b} runtime",
            f"{label_a} cost/agent",
            f"{label_b} cost/agent",
        ],
        per_map_rows,
    ))

    lines.extend(["", "## Per Agent Count", ""])
    per_agent_rows = []
    for agent in agents:
        summary_a = _summarize(agent_groups_a.get((agent,), []), cost_column)
        summary_b = _summarize(agent_groups_b.get((agent,), []), cost_column)
        per_agent_rows.append([
            str(agent),
            f"{summary_a.success_rate:.2f}%",
            f"{summary_b.success_rate:.2f}%",
            f"{summary_b.success_rate - summary_a.success_rate:+.2f} pp",
            f"{summary_a.runtime:.2f}s",
            f"{summary_b.runtime:.2f}s",
            f"{summary_a.cost_per_agent:.2f}",
            f"{summary_b.cost_per_agent:.2f}",
        ])
    lines.append(_md_table(
        [
            "Agents",
            f"{label_a} success",
            f"{label_b} success",
            "Success delta",
            f"{label_a} runtime",
            f"{label_b} runtime",
            f"{label_a} cost/agent",
            f"{label_b} cost/agent",
        ],
        per_agent_rows,
    ))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _plot(
    out_path: str,
    rows_a: list[dict[str, str]],
    rows_b: list[dict[str, str]],
    label_a: str,
    label_b: str,
    cost_column: str,
) -> None:
    import matplotlib.pyplot as plt

    maps = sorted({row["mapName"] for row in rows_a + rows_b})
    grouped_a = _group(rows_a, ("mapName", "agentNum"))
    grouped_b = _group(rows_b, ("mapName", "agentNum"))

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
            for key in set(grouped_a) | set(grouped_b)
            if key[0] == map_name
        })
        for row_idx, (metric_name, getter, ylim) in enumerate(metrics):
            ax = axes[row_idx][col]
            for rows_by_key, label, color, marker in [
                (grouped_a, label_a, "#1f77b4", "o"),
                (grouped_b, label_b, "#d62728", "x"),
            ]:
                xs = []
                ys = []
                for agent in agent_counts:
                    rows = rows_by_key.get((map_name, agent), [])
                    if not rows:
                        continue
                    xs.append(agent)
                    ys.append(getter(_summarize(rows, cost_column)))
                ax.plot(xs, ys, label=label, color=color, marker=marker, linewidth=1.7, markersize=4)

            if row_idx == 0:
                ax.set_title(map_name, fontsize=10)
            if col == 0:
                ax.set_ylabel(metric_name)
            if row_idx == 2:
                ax.set_xlabel("# Agents")
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25, linewidth=0.6)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a 1v1 grid eval comparison report")
    parser.add_argument("csv_a", help="First simulator-format CSV")
    parser.add_argument("csv_b", help="Second simulator-format CSV")
    parser.add_argument("--labels", nargs=2, default=None, metavar=("A", "B"))
    parser.add_argument("--out-dir", default="evals/grid_1v1", help="Output directory")
    parser.add_argument(
        "--cost-column",
        default="total_cost_not_resting_at_goal",
        choices=["total_cost_true", "total_cost_not_resting_at_goal"],
        help="CSV cost column used for solution cost per agent",
    )
    args = parser.parse_args()

    label_a, label_b = args.labels or [
        os.path.splitext(os.path.basename(args.csv_a))[0],
        os.path.splitext(os.path.basename(args.csv_b))[0],
    ]
    rows_a = _load_rows(args.csv_a)
    rows_b = _load_rows(args.csv_b)

    os.makedirs(args.out_dir, exist_ok=True)
    safe_a = label_a.lower().replace(" ", "_")
    safe_b = label_b.lower().replace(" ", "_")
    figure_path = os.path.join(args.out_dir, f"{safe_a}_vs_{safe_b}.png")
    report_path = os.path.join(args.out_dir, f"{safe_a}_vs_{safe_b}.md")

    try:
        _plot(figure_path, rows_a, rows_b, label_a, label_b, args.cost_column)
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        figure_path = "not written; matplotlib is not installed"
        print("WARNING: matplotlib is not installed, skipping figure generation")

    _write_report(
        report_path,
        args.csv_a,
        args.csv_b,
        label_a,
        label_b,
        rows_a,
        rows_b,
        args.cost_column,
        figure_path,
    )

    print(f"Wrote report: {report_path}")
    print(f"Wrote figure: {figure_path}")


if __name__ == "__main__":
    main()
