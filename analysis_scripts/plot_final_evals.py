"""Generate slide-ready plots and tables for the final grid eval CSVs.

This script intentionally uses only the Python standard library. It writes
SVG figures so the artifacts stay sharp in slides without needing matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import html
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


ROOT = Path("evals/final_evals")


@dataclass(frozen=True)
class RunConfig:
    key: str
    label: str
    short_label: str
    panel: str
    csv_name: str
    color: str
    train_note: str


RUNS = [
    RunConfig(
        key="heldout_rishi",
        label="Rishi classifier",
        short_label="Rishi",
        panel="8 heldout maps",
        csv_name="rishi_full_ssil_classifier.csv",
        color="#2D7DD2",
        train_note="classifier baseline",
    ),
    RunConfig(
        key="heldout_ours",
        label="Ours, trained off heldout",
        short_label="Ours",
        panel="8 heldout maps",
        csv_name="rishi8_wave10_newdata_clean_20260416_224144.csv",
        color="#E76F51",
        train_note="trained on non-heldout maps",
    ),
    RunConfig(
        key="heldout_parth",
        label="Parth, trained on all data",
        short_label="Parth",
        panel="8 heldout maps",
        csv_name="rishi8_parth_wave2_ft_v2_epoch14.csv",
        color="#2A9D8F",
        train_note="trained on full dataset",
    ),
    RunConfig(
        key="all12_rishi",
        label="Rishi classifier",
        short_label="Rishi",
        panel="12-map panel",
        csv_name="rishi12_ssil_classifier.csv",
        color="#2D7DD2",
        train_note="classifier baseline",
    ),
    RunConfig(
        key="all12_ours",
        label="Ours, trained off heldout",
        short_label="Ours",
        panel="12-map panel",
        csv_name="rishi12_wave10_newdata_clean_20260416_224144.csv",
        color="#E76F51",
        train_note="trained on non-heldout maps",
    ),
]

PANEL_ORDER = ["8 heldout maps", "12-map panel"]
BACKGROUND = "#FBFCFE"
INK = "#1F2933"
MUTED = "#65727F"
GRID = "#D7DEE8"


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def num(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_rows(root: Path) -> dict[str, list[dict[str, str]]]:
    data = {}
    for run in RUNS:
        path = root / run.csv_name
        if not path.exists():
            raise SystemExit(f"Missing expected CSV: {path}")
        with path.open(newline="") as handle:
            data[run.key] = list(csv.DictReader(handle))
    return data


def summarize(rows: list[dict[str, str]]) -> dict[str, float | int]:
    if not rows:
        return {
            "rows": 0,
            "maps": 0,
            "success_count": 0,
            "success_rate": 0.0,
            "at_goal_rate": 0.0,
            "runtime": 0.0,
            "cost_per_agent": 0.0,
        }
    success_count = sum(truthy(row["success"]) for row in rows)
    at_goal = [
        num(row, "num_agents_at_goal") / max(num(row, "agentNum"), 1.0) * 100.0
        for row in rows
    ]
    cost = [
        num(row, "total_cost_not_resting_at_goal") / max(num(row, "agentNum"), 1.0)
        for row in rows
    ]
    return {
        "rows": len(rows),
        "maps": len({row["mapName"] for row in rows}),
        "success_count": success_count,
        "success_rate": success_count / len(rows) * 100.0,
        "at_goal_rate": mean(at_goal),
        "runtime": mean(num(row, "runtime") for row in rows),
        "cost_per_agent": mean(cost),
    }


def grouped_summary(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
) -> dict[tuple[object, ...], dict[str, float | int]]:
    groups: dict[tuple[object, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key_parts: list[object] = []
        for key in keys:
            if key == "agentNum":
                key_parts.append(int(num(row, key)))
            else:
                key_parts.append(row[key])
        groups[tuple(key_parts)].append(row)
    return {key: summarize(value) for key, value in groups.items()}


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def write_markdown(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join("---" for _ in headers) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(cell) for cell in row) + " |\n")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_text(
    x: float,
    y: float,
    text: object,
    size: int = 18,
    fill: str = INK,
    weight: int = 500,
    anchor: str = "start",
    rotate: float | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{transform}>{esc(text)}</text>'
    )


def canvas(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
        svg_text(42, 48, title, size=28, weight=800),
    ]
    if subtitle:
        parts.append(svg_text(42, 78, subtitle, size=15, fill=MUTED, weight=500))
    return parts


def finish(parts: list[str]) -> str:
    return "\n".join(parts + ["</svg>\n"])


def nice_ticks(max_value: float, steps: int = 5) -> list[float]:
    if max_value <= 0:
        return [0, 1]
    raw_step = max_value / steps
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    top = math.ceil(max_value / step) * step
    return [i * step for i in range(int(top / step) + 1)]


def legend(parts: list[str], items: list[RunConfig], x: float, y: float) -> None:
    cursor = x
    for run in items:
        parts.append(
            f'<rect x="{cursor:.1f}" y="{y - 12:.1f}" width="16" height="16" rx="4" fill="{run.color}"/>'
        )
        parts.append(svg_text(cursor + 24, y + 1, run.short_label, size=15, fill=MUTED))
        cursor += 120 + len(run.short_label) * 4


def bar_chart(
    path: Path,
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[tuple[RunConfig, list[float | None]]],
    y_label: str = "Success rate",
    value_suffix: str = "%",
    y_max: float = 100.0,
    height: int = 620,
) -> None:
    width = max(980, 140 + len(categories) * 95)
    parts = canvas(width, height, title, subtitle)
    left, right, top, bottom = 78, 42, 115, 130
    plot_w, plot_h = width - left - right, height - top - bottom
    ticks = [0, 25, 50, 75, 100] if y_max == 100 else nice_ticks(y_max)

    for tick in ticks:
        y = top + plot_h - plot_h * tick / y_max
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(svg_text(left - 12, y + 5, f"{tick:g}", size=13, fill=MUTED, anchor="end"))
    parts.append(svg_text(18, top + plot_h / 2, y_label, size=14, fill=MUTED, anchor="middle", rotate=-90))

    group_w = plot_w / max(len(categories), 1)
    inner_gap = group_w * 0.18
    available = group_w - inner_gap
    bar_w = min(34, available / max(len(series), 1) * 0.78)
    for i, category in enumerate(categories):
        center = left + i * group_w + group_w / 2
        start = center - (len(series) * bar_w + (len(series) - 1) * 7) / 2
        for j, (run, values) in enumerate(series):
            value = values[i]
            if value is None:
                continue
            h = plot_h * value / y_max
            x = start + j * (bar_w + 7)
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="5" fill="{run.color}"/>')
            parts.append(svg_text(x + bar_w / 2, y - 8, f"{value:.0f}{value_suffix}", size=12, fill=INK, anchor="middle", weight=650))
        label = category.replace("warehouse-", "wh-").replace("random-", "rand-")
        parts.append(svg_text(center, top + plot_h + 23, label, size=13, fill=MUTED, anchor="end", rotate=-36))

    legend(parts, [run for run, _ in series], left, height - 34)
    path.write_text(finish(parts))


def line_chart(
    path: Path,
    title: str,
    subtitle: str,
    xs: list[int],
    series: list[tuple[RunConfig, list[float]]],
    y_label: str = "Success rate",
    value_suffix: str = "%",
) -> None:
    width, height = 1120, 630
    parts = canvas(width, height, title, subtitle)
    left, right, top, bottom = 84, 54, 116, 96
    plot_w, plot_h = width - left - right, height - top - bottom
    y_max = 100.0

    for tick in [0, 25, 50, 75, 100]:
        y = top + plot_h - plot_h * tick / y_max
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        parts.append(svg_text(left - 12, y + 5, f"{tick}", size=13, fill=MUTED, anchor="end"))
    parts.append(svg_text(24, top + plot_h / 2, y_label, size=14, fill=MUTED, anchor="middle", rotate=-90))

    min_x, max_x = min(xs), max(xs)
    def sx(value: int) -> float:
        return left + (value - min_x) / max(max_x - min_x, 1) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - value / y_max * plot_h

    for agent in xs:
        x = sx(agent)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}" stroke="{GRID}" stroke-width="1" opacity="0.55"/>')
        parts.append(svg_text(x, top + plot_h + 30, agent, size=14, fill=MUTED, anchor="middle"))
    parts.append(svg_text(left + plot_w / 2, height - 26, "Agents", size=15, fill=MUTED, anchor="middle"))

    for run, values in series:
        points = [(sx(x), sy(y)) for x, y in zip(xs, values)]
        point_string = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{point_string}" fill="none" stroke="{run.color}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{BACKGROUND}" stroke="{run.color}" stroke-width="4"/>')
        parts.append(svg_text(points[-1][0] + 10, points[-1][1] + 5, f"{values[-1]:.0f}{value_suffix}", size=13, fill=run.color, weight=800))

    legend(parts, [run for run, _ in series], left, 100)
    path.write_text(finish(parts))


def write_readme(path: Path, output_dir: Path, overall_rows: list[list[object]]) -> None:
    lines = [
        "# Final Eval Slide Outputs",
        "",
        "Generated from the five CSVs available in `evals/final_evals`.",
        "Parth's 12-map result was not available, so the 12-map plots compare only Rishi and Ours.",
        "",
        "## Key Figures",
        "",
        "- `figures/overall_success.svg`: high-level success-rate bars for available eval panels.",
        "- `figures/heldout_success_by_agents.svg`: heldout success vs. agent count.",
        "- `figures/heldout_success_by_map.svg`: heldout success by map.",
        "- `figures/all12_success_by_agents.svg`: 12-map success vs. agent count.",
        "- `figures/all12_success_by_map.svg`: 12-map success by map.",
        "- `tables/comparison_deltas.md`: compact deltas against Rishi for each panel.",
        "",
        "## Overall Table",
        "",
        "| Panel | Run | Training note | Rows | Maps | Success | Mean agents at goal | Avg runtime | Avg cost / agent | CSV |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in overall_rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    path.write_text("\n".join(lines) + "\n")


def generate(root: Path, output_dir: Path) -> None:
    data = load_rows(root)
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    overall_rows: list[list[object]] = []
    for run in RUNS:
        summary = summarize(data[run.key])
        overall_rows.append([
            run.panel,
            run.label,
            run.train_note,
            summary["rows"],
            summary["maps"],
            f'{summary["success_rate"]:.1f}% ({summary["success_count"]}/{summary["rows"]})',
            f'{summary["at_goal_rate"]:.1f}%',
            f'{summary["runtime"]:.2f}s',
            f'{summary["cost_per_agent"]:.1f}',
            run.csv_name,
        ])
    headers = [
        "Panel",
        "Run",
        "Training note",
        "Rows",
        "Maps",
        "Success",
        "Mean agents at goal",
        "Avg runtime",
        "Avg cost / agent",
        "CSV",
    ]
    write_csv(tables / "overall_summary.csv", headers, overall_rows)
    write_markdown(tables / "overall_summary.md", headers, overall_rows)

    delta_rows: list[list[object]] = []
    takeaway_lines = [
        "# Final Eval Takeaways",
        "",
        "Numbers below are computed from the five available CSVs in `evals/final_evals`.",
        "Parth's 12-map eval is still missing, so no 12-map Parth comparison is shown.",
        "",
    ]
    for panel in PANEL_ORDER:
        panel_runs = [run for run in RUNS if run.panel == panel]
        baseline = next(run for run in panel_runs if run.short_label == "Rishi")
        baseline_summary = summarize(data[baseline.key])
        takeaway_lines.append(f"## {panel}")
        for run in panel_runs:
            if run.key == baseline.key:
                continue
            summary = summarize(data[run.key])
            delta_rows.append([
                panel,
                run.label,
                baseline.label,
                f'{summary["success_rate"] - baseline_summary["success_rate"]:+.1f} pp',
                f'{summary["at_goal_rate"] - baseline_summary["at_goal_rate"]:+.1f} pp',
                f'{summary["runtime"] - baseline_summary["runtime"]:+.2f}s',
                f'{summary["cost_per_agent"] - baseline_summary["cost_per_agent"]:+.1f}',
            ])
            takeaway_lines.append(
                f"- {run.label}: {summary['success_rate']:.1f}% success "
                f"({summary['success_count']}/{summary['rows']}), "
                f"{summary['success_rate'] - baseline_summary['success_rate']:+.1f} pp vs. Rishi."
            )
        takeaway_lines.append(
            f"- Rishi classifier: {baseline_summary['success_rate']:.1f}% success "
            f"({baseline_summary['success_count']}/{baseline_summary['rows']})."
        )
        takeaway_lines.append("")
    delta_headers = [
        "Panel",
        "Run",
        "Baseline",
        "Success delta",
        "Mean agents at goal delta",
        "Avg runtime delta",
        "Avg cost / agent delta",
    ]
    write_csv(tables / "comparison_deltas.csv", delta_headers, delta_rows)
    write_markdown(tables / "comparison_deltas.md", delta_headers, delta_rows)
    (output_dir / "takeaways.md").write_text("\n".join(takeaway_lines) + "\n")

    map_rows: list[list[object]] = []
    agent_rows: list[list[object]] = []
    summaries_by_run = {run.key: summarize(data[run.key]) for run in RUNS}
    per_map = {run.key: grouped_summary(data[run.key], ("mapName",)) for run in RUNS}
    per_agent = {run.key: grouped_summary(data[run.key], ("agentNum",)) for run in RUNS}

    for run in RUNS:
        for (map_name,), summary in sorted(per_map[run.key].items()):
            map_rows.append([
                run.panel,
                run.label,
                map_name,
                f'{summary["success_rate"]:.1f}%',
                f'{summary["at_goal_rate"]:.1f}%',
                f'{summary["runtime"]:.2f}s',
                f'{summary["cost_per_agent"]:.1f}',
                summary["rows"],
            ])
        for (agent,), summary in sorted(per_agent[run.key].items()):
            agent_rows.append([
                run.panel,
                run.label,
                agent,
                f'{summary["success_rate"]:.1f}%',
                f'{summary["at_goal_rate"]:.1f}%',
                f'{summary["runtime"]:.2f}s',
                f'{summary["cost_per_agent"]:.1f}',
                summary["rows"],
            ])
    metric_headers = ["Panel", "Run", "Group", "Success", "Mean agents at goal", "Avg runtime", "Avg cost / agent", "Rows"]
    write_csv(tables / "per_map_summary.csv", metric_headers, map_rows)
    write_markdown(tables / "per_map_summary.md", metric_headers, map_rows)
    write_csv(tables / "per_agent_summary.csv", metric_headers, agent_rows)
    write_markdown(tables / "per_agent_summary.md", metric_headers, agent_rows)

    overall_categories = PANEL_ORDER
    representative_runs = [
        next(run for run in RUNS if run.key == key)
        for key in ["heldout_rishi", "heldout_ours", "heldout_parth"]
    ]
    overall_series = []
    for representative in representative_runs:
        values: list[float | None] = []
        for panel in PANEL_ORDER:
            matching = [
                run for run in RUNS
                if run.panel == panel and run.short_label == representative.short_label
            ]
            values.append(summaries_by_run[matching[0].key]["success_rate"] if matching else None)
        overall_series.append((representative, values))
    bar_chart(
        figures / "overall_success.svg",
        "Final Eval Success Rates",
        "Available grid-world evals; all-agent success averaged over rows in each CSV.",
        overall_categories,
        overall_series,
        height=610,
    )

    heldout_runs = [run for run in RUNS if run.panel == "8 heldout maps"]
    all12_runs = [run for run in RUNS if run.panel == "12-map panel"]

    for filename, title, runs in [
        ("heldout_success_by_agents.svg", "Heldout Success vs. Agent Count", heldout_runs),
        ("all12_success_by_agents.svg", "12-Map Success vs. Agent Count", all12_runs),
    ]:
        agents = sorted({key[0] for run in runs for key in per_agent[run.key]})
        series = [
            (run, [per_agent[run.key][(agent,)]["success_rate"] for agent in agents])
            for run in runs
        ]
        line_chart(
            figures / filename,
            title,
            "Success rate at each density bucket; each point aggregates all scenarios for that agent count.",
            agents,
            series,
        )

    for filename, title, runs in [
        ("heldout_success_by_map.svg", "Heldout Success by Map", heldout_runs),
        ("all12_success_by_map.svg", "12-Map Success by Map", all12_runs),
    ]:
        maps = sorted({key[0] for run in runs for key in per_map[run.key]})
        series = [
            (run, [per_map[run.key].get((map_name,), {"success_rate": 0.0})["success_rate"] for map_name in maps])
            for run in runs
        ]
        bar_chart(
            figures / filename,
            title,
            "All-agent success rate by map; missing runs are omitted from this eval panel.",
            maps,
            series,
            height=690 if len(maps) > 8 else 640,
        )

    write_readme(output_dir / "README.md", output_dir, overall_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Directory containing final eval CSVs")
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "slide_outputs"),
        help="Directory for generated figures and tables",
    )
    args = parser.parse_args()

    generate(Path(args.root), Path(args.out_dir))
    print(f"Wrote slide outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
