"""Summarize Rishi-protocol grid evaluation CSVs.

Example:
    python -m analysis_scripts.summarize_grid_eval \
        evals/rishi_full_wave8_heldout_best.csv \
        evals/rishi_full_ssil_classifier.csv \
        --labels flow_best ssil_classifier
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _summarize(rows: list[dict[str, str]]) -> dict[str, float | int]:
    successes = [_truthy(row["success"]) for row in rows]
    at_goal = [
        int(row["num_agents_at_goal"]) / int(row["agentNum"]) * 100.0
        for row in rows
    ]
    runtimes = [float(row["runtime"]) for row in rows]
    return {
        "runs": len(rows),
        "success_count": sum(successes),
        "success_rate": sum(successes) / len(rows) * 100.0 if rows else 0.0,
        "at_goal_rate": sum(at_goal) / len(at_goal) if at_goal else 0.0,
        "runtime": sum(runtimes) / len(runtimes) if runtimes else 0.0,
    }


def _map_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["mapName"]].append(row)
    return {map_name: _summarize(map_rows) for map_name, map_rows in grouped.items()}


def _print_markdown_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize grid eval CSVs")
    parser.add_argument("csvs", nargs="+", help="Simulator-format CSVs to summarize")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels for CSVs")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.csvs):
        parser.error("--labels must have the same length as csvs")

    labels = args.labels or [
        os.path.splitext(os.path.basename(path))[0] for path in args.csvs
    ]
    datasets = [(label, path, _load_rows(path)) for label, path in zip(labels, args.csvs)]

    print("\nOverall")
    overall_rows = []
    for label, path, rows in datasets:
        summary = _summarize(rows)
        overall_rows.append([
            label,
            str(summary["runs"]),
            f'{summary["success_rate"]:.2f}% ({summary["success_count"]}/{summary["runs"]})',
            f'{summary["at_goal_rate"]:.2f}%',
            f'{summary["runtime"]:.2f}s',
            path,
        ])
    _print_markdown_table(
        ["Run", "Rows", "All-agents success", "Mean agents at goal", "Avg runtime", "CSV"],
        overall_rows,
    )

    all_maps = sorted({
        row["mapName"]
        for _, _, rows in datasets
        for row in rows
    })
    per_map = [(label, _map_summary(rows)) for label, _, rows in datasets]

    print("\nPer Map Success")
    headers = ["Map"] + labels
    map_rows = []
    for map_name in all_maps:
        row = [map_name]
        for _, summary_by_map in per_map:
            summary = summary_by_map.get(map_name)
            row.append("N/A" if summary is None else f'{summary["success_rate"]:.2f}%')
        map_rows.append(row)
    _print_markdown_table(headers, map_rows)

    if len(per_map) >= 2:
        base_label, base_summary = per_map[0]
        print(f"\nDelta Vs {base_label}")
        delta_rows = []
        for label, summary_by_map in per_map[1:]:
            shared_maps = sorted(set(base_summary) & set(summary_by_map))
            for map_name in shared_maps:
                delta = summary_by_map[map_name]["success_rate"] - base_summary[map_name]["success_rate"]
                delta_rows.append([label, map_name, f"{delta:+.2f} pp"])
        _print_markdown_table(["Run", "Map", "Success Delta"], delta_rows)


if __name__ == "__main__":
    main()
