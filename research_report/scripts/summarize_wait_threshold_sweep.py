#!/usr/bin/env python3
"""Summarize FLOMAP wait-threshold sensitivity sweep CSVs.

This script is intentionally lightweight: it uses the Python standard library
for all table outputs and matplotlib only for the optional publication figure.
Inputs should be the combined simulator-format CSVs produced after each
threshold's four GPU shards are concatenated.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

sys.dont_write_bytecode = True

from report_data import (
    FIG_DIR,
    REPO_ROOT,
    TABLE_DIR,
    ensure_dirs,
    fmt_pct,
    latex_escape,
    parse_bool,
    write_csv,
    write_tex_table,
)


DEFAULT_THRESHOLD = 0.25
STRESS_MAPS = {"random-32-32-10", "maze-128-128-2"}
EXPECTED_ROWS = 1850


def parse_threshold(path: Path) -> float:
    """Infer a threshold from names such as wt_0p25_combined.csv."""

    text = str(path)
    match = re.search(r"wt[_-](\d+)p(\d+)", text)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    match = re.search(r"wt[_-](\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    raise ValueError(
        f"Could not infer wait threshold from {path}. "
        "Use --csv THRESHOLD=PATH for explicit inputs."
    )


def parse_csv_arg(value: str) -> tuple[float, Path]:
    if "=" not in value:
        path = Path(value)
        return parse_threshold(path), path
    threshold, path = value.split("=", 1)
    return float(threshold), Path(path)


def pct(count: float, rows: float) -> float:
    return 100.0 * count / rows if rows else 0.0


def summarize_rows(path: Path) -> dict:
    rows = 0
    successes = 0
    total_goal_fraction = 0.0
    total_runtime = 0.0
    slices: dict[str, MutableMapping[str, float]] = {
        "stress": {"rows": 0, "successes": 0, "goal_fraction": 0.0, "runtime": 0.0},
        "non_stress": {"rows": 0, "successes": 0, "goal_fraction": 0.0, "runtime": 0.0},
    }
    per_map: dict[str, MutableMapping[str, float]] = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"mapName", "agentNum", "success", "num_agents_at_goal", "runtime"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            rows += 1
            success = int(parse_bool(row["success"]))
            successes += success
            agent_num = float(row["agentNum"])
            goal_fraction = float(row["num_agents_at_goal"]) / agent_num
            runtime = float(row["runtime"])
            total_goal_fraction += goal_fraction
            total_runtime += runtime

            map_name = row["mapName"]
            map_cur = per_map.setdefault(
                map_name,
                {"rows": 0, "successes": 0, "goal_fraction": 0.0, "runtime": 0.0},
            )
            map_cur["rows"] += 1
            map_cur["successes"] += success
            map_cur["goal_fraction"] += goal_fraction
            map_cur["runtime"] += runtime

            slice_name = "stress" if map_name in STRESS_MAPS else "non_stress"
            slice_cur = slices[slice_name]
            slice_cur["rows"] += 1
            slice_cur["successes"] += success
            slice_cur["goal_fraction"] += goal_fraction
            slice_cur["runtime"] += runtime

    if rows == 0:
        raise ValueError(f"CSV has no data rows: {path}")

    return {
        "rows": rows,
        "successes": successes,
        "success_pct": pct(successes, rows),
        "mean_agents_at_goal_pct": pct(total_goal_fraction, rows),
        "avg_runtime_s": total_runtime / rows,
        "slices": {
            key: {
                "rows": int(value["rows"]),
                "successes": int(value["successes"]),
                "success_pct": pct(value["successes"], value["rows"]),
                "mean_agents_at_goal_pct": pct(value["goal_fraction"], value["rows"]),
                "avg_runtime_s": value["runtime"] / value["rows"] if value["rows"] else 0.0,
            }
            for key, value in slices.items()
        },
        "per_map": {
            key: {
                "rows": int(value["rows"]),
                "successes": int(value["successes"]),
                "success_pct": pct(value["successes"], value["rows"]),
                "mean_agents_at_goal_pct": pct(value["goal_fraction"], value["rows"]),
                "avg_runtime_s": value["runtime"] / value["rows"],
            }
            for key, value in sorted(per_map.items())
        },
    }


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def format_threshold(value: float) -> str:
    label = f"{value:.2f}"
    if abs(value - DEFAULT_THRESHOLD) < 1e-9:
        return r"\textbf{" + label + r"}"
    return label


def write_summary_outputs(results: Sequence[Mapping]) -> None:
    overall_rows = []
    per_map_rows = []
    for item in results:
        summary = item["summary"]
        threshold = float(item["threshold"])
        stress = summary["slices"]["stress"]
        non_stress = summary["slices"]["non_stress"]
        overall_rows.append(
            {
                "wait_threshold": f"{threshold:.2f}",
                "is_default": "yes" if abs(threshold - DEFAULT_THRESHOLD) < 1e-9 else "no",
                "rows": summary["rows"],
                "success": f"{summary['successes']}/{summary['rows']}",
                "success_pct": fmt_pct(summary["success_pct"]),
                "mean_agents_at_goal_pct": fmt_pct(summary["mean_agents_at_goal_pct"]),
                "avg_runtime_s": f"{summary['avg_runtime_s']:.2f}",
                "stress_success": f"{stress['successes']}/{stress['rows']}",
                "stress_success_pct": fmt_pct(stress["success_pct"]),
                "non_stress_success": f"{non_stress['successes']}/{non_stress['rows']}",
                "non_stress_success_pct": fmt_pct(non_stress["success_pct"]),
                "source": rel(Path(item["path"])),
            }
        )
        for map_name, map_summary in summary["per_map"].items():
            per_map_rows.append(
                {
                    "wait_threshold": f"{threshold:.2f}",
                    "map": map_name,
                    "rows": map_summary["rows"],
                    "success": f"{map_summary['successes']}/{map_summary['rows']}",
                    "success_pct": fmt_pct(map_summary["success_pct"]),
                    "mean_agents_at_goal_pct": fmt_pct(map_summary["mean_agents_at_goal_pct"]),
                    "avg_runtime_s": f"{map_summary['avg_runtime_s']:.2f}",
                }
            )

    write_csv(
        TABLE_DIR / "wait_threshold_sweep.csv",
        overall_rows,
        [
            "wait_threshold",
            "is_default",
            "rows",
            "success",
            "success_pct",
            "mean_agents_at_goal_pct",
            "avg_runtime_s",
            "stress_success",
            "stress_success_pct",
            "non_stress_success",
            "non_stress_success_pct",
            "source",
        ],
    )
    write_csv(
        TABLE_DIR / "wait_threshold_sweep_per_map.csv",
        per_map_rows,
        [
            "wait_threshold",
            "map",
            "rows",
            "success",
            "success_pct",
            "mean_agents_at_goal_pct",
            "avg_runtime_s",
        ],
    )

    write_tex_table(
        TABLE_DIR / "wait_threshold_sweep.tex",
        [
            [
                format_threshold(float(row["wait_threshold"])),
                row["success_pct"],
                row["stress_success_pct"],
                row["non_stress_success_pct"],
                row["mean_agents_at_goal_pct"],
                row["avg_runtime_s"],
            ]
            for row in overall_rows
        ],
        [
            "Wait thresh.",
            "Overall succ.",
            "Stress succ.",
            "Non-stress succ.",
            "Agents at goal",
            "Runtime",
        ],
        "rrrrrr",
    )

    per_map_headers = ["Map"] + [f"{float(row['wait_threshold']):.2f}" for row in overall_rows]
    maps = sorted({row["map"] for row in per_map_rows})
    per_map_by_key = {
        (row["map"], row["wait_threshold"]): row["success_pct"] for row in per_map_rows
    }
    write_tex_table(
        TABLE_DIR / "wait_threshold_sweep_per_map.tex",
        [
            [latex_escape(map_name)]
            + [per_map_by_key.get((map_name, row["wait_threshold"]), "--") for row in overall_rows]
            for map_name in maps
        ],
        per_map_headers,
        "l" + "r" * len(overall_rows),
    )


def write_plot(results: Sequence[Mapping]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib is not installed; skipped wait-threshold figure.")
        return

    thresholds = [float(item["threshold"]) for item in results]
    overall = [item["summary"]["success_pct"] for item in results]
    stress = [item["summary"]["slices"]["stress"]["success_pct"] for item in results]
    non_stress = [item["summary"]["slices"]["non_stress"]["success_pct"] for item in results]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.45, 2.35), constrained_layout=True)
    palette = {
        "overall": "#0072B2",
        "stress": "#D55E00",
        "non_stress": "#009E73",
    }
    ax.plot(thresholds, overall, marker="o", linewidth=1.8, markersize=4.2, color=palette["overall"], label="Overall")
    ax.plot(thresholds, stress, marker="s", linewidth=1.8, markersize=4.2, color=palette["stress"], label="Coordination stress")
    ax.plot(thresholds, non_stress, marker="^", linewidth=1.8, markersize=4.2, color=palette["non_stress"], label="Other maps")
    ax.axvline(DEFAULT_THRESHOLD, color="0.35", linewidth=0.9, linestyle="--")
    ax.text(DEFAULT_THRESHOLD, ax.get_ylim()[0], " default", color="0.35", va="bottom", ha="left", fontsize=7)
    ax.set_xlabel("Wait threshold")
    ax.set_ylabel("All-agent success (%)")
    ax.set_xticks(thresholds)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")

    png_path = FIG_DIR / "wait_threshold_sweep.png"
    pdf_path = FIG_DIR / "wait_threshold_sweep.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Wrote {rel(png_path)}")
    print(f"Wrote {rel(pdf_path)}")


def describe(results: Sequence[Mapping]) -> None:
    best = max(results, key=lambda item: item["summary"]["success_pct"])
    default = next(
        (item for item in results if abs(float(item["threshold"]) - DEFAULT_THRESHOLD) < 1e-9),
        None,
    )
    print("\nWait-threshold sweep summary:")
    for item in results:
        s = item["summary"]
        stress = s["slices"]["stress"]
        non_stress = s["slices"]["non_stress"]
        marker = " (default)" if abs(float(item["threshold"]) - DEFAULT_THRESHOLD) < 1e-9 else ""
        print(
            f"  wt={float(item['threshold']):.2f}{marker}: "
            f"overall {s['success_pct']:.2f}% ({s['successes']}/{s['rows']}), "
            f"stress {stress['success_pct']:.2f}%, "
            f"non-stress {non_stress['success_pct']:.2f}%, "
            f"agents-at-goal {s['mean_agents_at_goal_pct']:.2f}%, "
            f"runtime {s['avg_runtime_s']:.2f}s"
        )
    print(
        f"\nBest overall threshold: {float(best['threshold']):.2f} "
        f"at {best['summary']['success_pct']:.2f}% success."
    )
    if default is not None:
        delta = best["summary"]["success_pct"] - default["summary"]["success_pct"]
        print(
            f"Default wt={DEFAULT_THRESHOLD:.2f}: "
            f"{default['summary']['success_pct']:.2f}% success; "
            f"{delta:.2f} pp below the best observed threshold."
        )
    print(
        "\nReport wording cue: this sweep reports sensitivity of the fixed "
        "velocity-to-action wait interface. It supports or weakens an interface-"
        "bottleneck interpretation, but it does not by itself prove a single "
        "causal mechanism because training data, objective, architecture, and "
        "CS-PIBT interactions remain coupled."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csvs",
        nargs="+",
        help="Combined simulator CSVs, or THRESHOLD=CSV for explicit threshold labels.",
    )
    parser.add_argument(
        "--expect-rows",
        type=int,
        default=EXPECTED_ROWS,
        help="Expected rows per full threshold; use 0 to disable the check.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Write only CSV/LaTeX tables.",
    )
    args = parser.parse_args()

    ensure_dirs()
    inputs = [parse_csv_arg(value) for value in args.csvs]
    results = []
    for threshold, path in sorted(inputs, key=lambda item: item[0]):
        if not path.exists():
            raise FileNotFoundError(path)
        summary = summarize_rows(path)
        if args.expect_rows and summary["rows"] != args.expect_rows:
            raise ValueError(
                f"{path} has {summary['rows']} rows; expected {args.expect_rows}. "
                "Use --expect-rows 0 for smoke sweeps."
            )
        results.append({"threshold": threshold, "path": str(path), "summary": summary})

    write_summary_outputs(results)
    if not args.no_plot:
        write_plot(results)
    describe(results)
    print(f"\nWrote {rel(TABLE_DIR / 'wait_threshold_sweep.tex')}")
    print(f"Wrote {rel(TABLE_DIR / 'wait_threshold_sweep_per_map.tex')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
