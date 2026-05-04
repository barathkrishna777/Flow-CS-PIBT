#!/usr/bin/env python3
"""Generate compact CSV/LaTeX summary tables for the report."""

from __future__ import annotations

import argparse
import sys

sys.dont_write_bytecode = True

from report_data import (
    COORDINATION_SLICES,
    CSV_PATHS,
    LAMBDA_CSVS,
    REPO_ROOT,
    RISHI12_MAPS,
    RISHI8_MAPS,
    TABLE_DIR,
    ensure_dirs,
    fmt_pct,
    get_rate,
    get_summary,
    lambda_copy_commands,
    latex_escape,
    missing_lambda_csvs,
    write_csv,
    write_tex_table,
)


def gap_label(flow: float, ssil: float) -> str:
    delta = flow - ssil
    if abs(delta) < 0.005:
        return "0.00"
    if delta > 0:
        return f"FLOMAP +{delta:.2f}"
    return f"SSIL +{-delta:.2f}"


def per_map_rows(flow, ssil, maps):
    rows = []
    for map_name in maps:
        f = get_rate(flow, map_name)
        s = get_rate(ssil, map_name)
        rows.append(
            {
                "map": map_name,
                "flow_success_pct": fmt_pct(f),
                "ssil_success_pct": fmt_pct(s),
                "gap_pp": gap_label(f, s),
            }
        )
    return rows


def write_per_map_table(name: str, rows):
    write_csv(
        TABLE_DIR / f"{name}_per_map_success.csv",
        rows,
        ["map", "flow_success_pct", "ssil_success_pct", "gap_pp"],
    )
    tex_rows = [
        [
            latex_escape(row["map"]),
            row["flow_success_pct"],
            row["ssil_success_pct"],
            latex_escape(row["gap_pp"]),
        ]
        for row in rows
    ]
    write_tex_table(
        TABLE_DIR / f"{name}_per_map_success.tex",
        tex_rows,
        ["Map", "FLOMAP", "SSIL", "Gap (pp)"],
        "lrrr",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail when an expected CSV is absent instead of using prompt summaries.",
    )
    args = parser.parse_args()
    allow_fallback = not args.no_fallback

    ensure_dirs()

    summaries = {
        key: get_summary(key, allow_fallback=allow_fallback)
        for key in [
            "rishi8_pibt",
            "rishi8_flow",
            "rishi8_ssil",
            "rishi8_hybrid",
            "rishi12_pibt",
            "rishi12_flow",
            "rishi12_ssil",
        ]
    }

    overall_rows = []
    for key, summary in summaries.items():
        source_label = "remote summary" if summary["source"] == "prompt fallback" else "local CSV"
        overall_rows.append(
            {
                "dataset": key,
                "method": summary["label"],
                "rows": summary["rows"],
                "success": f"{summary['success_count']}/{summary['rows']}",
                "success_pct": fmt_pct(float(summary["success_rate"])),
                "mean_agents_at_goal_pct": fmt_pct(float(summary["mean_agents_at_goal"])),
                "avg_runtime_s": f"{float(summary['avg_runtime']):.2f}",
                "source": source_label,
            }
        )

    write_csv(
        TABLE_DIR / "overall_summary.csv",
        overall_rows,
        [
            "dataset",
            "method",
            "rows",
            "success",
            "success_pct",
            "mean_agents_at_goal_pct",
            "avg_runtime_s",
            "source",
        ],
    )
    write_tex_table(
        TABLE_DIR / "overall_summary.tex",
        [
            [
                latex_escape(row["method"]),
                row["rows"],
                row["success_pct"],
                row["mean_agents_at_goal_pct"],
                row["avg_runtime_s"],
            ]
            for row in overall_rows
        ],
        ["Method / Eval", "Rows", "Success", "Agents at goal", "Runtime"],
        "lrrrr",
    )

    write_per_map_table("primary8", per_map_rows(summaries["rishi8_flow"], summaries["rishi8_ssil"], RISHI8_MAPS))
    write_per_map_table("extended12", per_map_rows(summaries["rishi12_flow"], summaries["rishi12_ssil"], RISHI12_MAPS))

    write_csv(
        TABLE_DIR / "coordination_slices.csv",
        COORDINATION_SLICES,
        ["slice", "flow", "flow_count", "ssil", "ssil_count", "gap", "mcnemar_p"],
    )
    write_tex_table(
        TABLE_DIR / "coordination_slices.tex",
        [
            [
                latex_escape(row["slice"]),
                f"{row['flow']:.2f} ({row['flow_count']})",
                f"{row['ssil']:.2f} ({row['ssil_count']})",
                latex_escape(row["gap"]),
                latex_escape(row["mcnemar_p"]),
            ]
            for row in COORDINATION_SLICES
        ],
        ["Slice", "FLOMAP", "SSIL", "Gap (pp)", "McNemar p"],
        "p{0.34\\linewidth}p{0.16\\linewidth}p{0.16\\linewidth}p{0.14\\linewidth}p{0.12\\linewidth}",
    )

    source_rows = []
    for key, path in CSV_PATHS.items():
        source_rows.append(
            {
                "dataset": key,
                "relative_path": str(path.relative_to(REPO_ROOT)),
                "present": "yes" if path.exists() else "no",
                "lambda_required": "yes" if key in LAMBDA_CSVS else "no",
            }
        )
    write_csv(
        TABLE_DIR / "csv_source_status.csv",
        source_rows,
        ["dataset", "relative_path", "present", "lambda_required"],
    )

    missing = missing_lambda_csvs()
    note_path = TABLE_DIR / "missing_lambda_csvs.txt"
    with note_path.open("w") as f:
        if missing:
            f.write("Missing remote CSVs:\n")
            for path in missing:
                f.write(f"- {path.relative_to(REPO_ROOT)}\n")
            f.write("\nCopy command template:\n")
            f.write(lambda_copy_commands())
            f.write("\n")
        else:
            f.write("All expected remote CSVs are present locally.\n")

    print(f"Wrote summary tables to {TABLE_DIR}")
    if missing:
        print("Missing remote CSVs:")
        for path in missing:
            print(f"  - {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
