#!/usr/bin/env python3
"""Shared data utilities for the FLOMAP research report.

The scripts in this directory intentionally avoid pandas/matplotlib so they can
run in lightweight Python environments on the local Mac or remote evaluation
machines.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_DIR = SCRIPT_DIR.parent
REPO_ROOT = REPORT_DIR.parent
FIG_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"


RISHI8_MAPS = [
    "Paris_1_256",
    "den312d",
    "den520d",
    "empty-48-48",
    "maze-128-128-2",
    "random-32-32-10",
    "random-64-64-10",
    "warehouse-10-20-10-2-1",
]

RISHI12_MAPS = [
    "Berlin_1_256",
    "Paris_1_256",
    "den312d",
    "empty-32-32",
    "empty-48-48",
    "maze-128-128-2",
    "maze-32-32-4",
    "random-64-64-10",
    "random-64-64-20",
    "room-64-64-16",
    "warehouse-10-20-10-2-1",
    "warehouse-20-40-10-2-1",
]


CSV_PATHS = {
    "rishi8_pibt": REPO_ROOT
    / "evals/lambda_sharded_pibt_20260430_103517/rishi8/rishi8_pibt_combined.csv",
    "rishi8_flow": REPO_ROOT / "evals/rishi_full_wave9_compact_best.csv",
    "rishi8_ssil": REPO_ROOT / "evals/final_evals/rishi_full_ssil_classifier.csv",
    "rishi12_pibt": REPO_ROOT
    / "evals/lambda_sharded_pibt_20260430_103517/rishi12/rishi12_pibt_combined.csv",
    "rishi12_flow": REPO_ROOT
    / "evals/lambda_sharded_rishi_lambda_20260429_191606/rishi12_wave9/rishi12_wave9_flow_combined.csv",
    "rishi12_ssil": REPO_ROOT / "evals/final_evals/rishi12_ssil_classifier.csv",
    "rishi8_hybrid": REPO_ROOT
    / "evals/lambda_sharded_rishi_lambda_20260429_191606/rishi8_hybrid/rishi8_hybrid_action_head_combined.csv",
}


LAMBDA_CSVS = {
    "rishi8_pibt": CSV_PATHS["rishi8_pibt"],
    "rishi12_pibt": CSV_PATHS["rishi12_pibt"],
    "rishi12_flow": CSV_PATHS["rishi12_flow"],
    "rishi8_hybrid": CSV_PATHS["rishi8_hybrid"],
}

EXPECTED_ROWS = {
    "rishi8_pibt": 1850,
    "rishi8_flow": 1850,
    "rishi8_ssil": 1850,
    "rishi8_hybrid": 1850,
    "rishi12_pibt": 2700,
    "rishi12_flow": 2700,
    "rishi12_ssil": 2700,
}


FALLBACKS: Dict[str, Mapping] = {
    "rishi8_pibt": {
        "label": "BD-PIBT",
        "rows": 1850,
        "success_count": 1229,
        "success_rate": 66.43,
        "mean_agents_at_goal": 98.46,
        "avg_runtime": 6.47,
        "per_map": {},
    },
    "rishi8_flow": {
        "label": "FLOMAP",
        "rows": 1850,
        "success_count": 1195,
        "success_rate": 64.59,
        "mean_agents_at_goal": 91.93,
        "avg_runtime": 31.34,
        "per_map": {
            "Paris_1_256": 100.00,
            "den312d": 35.60,
            "den520d": 99.60,
            "empty-48-48": 95.20,
            "maze-128-128-2": 30.00,
            "random-32-32-10": 71.00,
            "random-64-64-10": 86.40,
            "warehouse-10-20-10-2-1": 2.80,
        },
    },
    "rishi8_ssil": {
        "label": "SSIL classifier",
        "rows": 1850,
        "success_count": 1229,
        "success_rate": 66.43,
        "mean_agents_at_goal": 95.69,
        "avg_runtime": 18.96,
        "per_map": {
            "Paris_1_256": 100.00,
            "den312d": 32.00,
            "den520d": 99.60,
            "empty-48-48": 100.00,
            "maze-128-128-2": 40.00,
            "random-32-32-10": 81.00,
            "random-64-64-10": 85.20,
            "warehouse-10-20-10-2-1": 2.40,
        },
    },
    "rishi12_pibt": {
        "label": "BD-PIBT",
        "rows": 2700,
        "success_count": 1606,
        "success_rate": 59.48,
        "mean_agents_at_goal": 98.57,
        "avg_runtime": 6.00,
        "per_map": {},
    },
    "rishi12_flow": {
        "label": "FLOMAP",
        "rows": 2700,
        "success_count": 1433,
        "success_rate": 53.07,
        "mean_agents_at_goal": 90.56,
        "avg_runtime": 32.16,
        "per_map": {
            "Berlin_1_256": 99.60,
            "Paris_1_256": 100.00,
            "den312d": 35.60,
            "empty-32-32": 80.80,
            "empty-48-48": 95.20,
            "maze-128-128-2": 30.00,
            "maze-32-32-4": 21.33,
            "random-64-64-10": 86.40,
            "random-64-64-20": 40.80,
            "room-64-64-16": 21.20,
            "warehouse-10-20-10-2-1": 2.80,
            "warehouse-20-40-10-2-1": 14.80,
        },
    },
    "rishi12_ssil": {
        "label": "SSIL classifier",
        "rows": 2700,
        "success_count": 1491,
        "success_rate": 55.22,
        "mean_agents_at_goal": 94.83,
        "avg_runtime": 18.62,
        "per_map": {
            "Berlin_1_256": 100.00,
            "Paris_1_256": 100.00,
            "den312d": 32.00,
            "empty-32-32": 100.00,
            "empty-48-48": 100.00,
            "maze-128-128-2": 42.00,
            "maze-32-32-4": 14.67,
            "random-64-64-10": 85.20,
            "random-64-64-20": 42.00,
            "room-64-64-16": 27.20,
            "warehouse-10-20-10-2-1": 2.40,
            "warehouse-20-40-10-2-1": 11.20,
        },
    },
    "rishi8_hybrid": {
        "label": "FLOMAP (action head)",
        "rows": 1850,
        "success_count": 1156,
        "success_rate": 62.49,
        "mean_agents_at_goal": 95.13,
        "avg_runtime": 21.24,
        "per_map": {
            "Paris_1_256": 100.00,
            "den312d": 32.80,
            "den520d": 99.60,
            "empty-48-48": 100.00,
            "maze-128-128-2": 35.20,
            "random-32-32-10": 58.00,
            "random-64-64-10": 69.20,
            "warehouse-10-20-10-2-1": 2.40,
        },
    },
}


COORDINATION_SLICES = [
    {
        "slice": "Full primary held-out evaluation",
        "flow": 64.59,
        "flow_count": "1195/1850",
        "ssil": 66.43,
        "ssil_count": "1229/1850",
        "gap": "SSIL +1.84",
        "mcnemar_p": "0.0039",
    },
    {
        "slice": "Coordination stress maps",
        "flow": 41.71,
        "flow_count": "146/350",
        "ssil": 51.71,
        "ssil_count": "181/350",
        "gap": "SSIL +10.00",
        "mcnemar_p": "<0.001",
    },
    {
        "slice": "Dense random map",
        "flow": 71.00,
        "flow_count": "71/100",
        "ssil": 81.00,
        "ssil_count": "81/100",
        "gap": "SSIL +10.00",
        "mcnemar_p": "0.064",
    },
    {
        "slice": "Maze bottleneck map",
        "flow": 30.00,
        "flow_count": "75/250",
        "ssil": 40.00,
        "ssil_count": "100/250",
        "gap": "SSIL +10.00",
        "mcnemar_p": "<0.001",
    },
    {
        "slice": "Excluding coordination stress maps",
        "flow": 69.93,
        "flow_count": "1049/1500",
        "ssil": 69.87,
        "ssil_count": "1048/1500",
        "gap": "FLOMAP +0.06",
        "mcnemar_p": "1.000",
    },
    {
        "slice": "Excluding stress and floor-case maps",
        "flow": 83.36,
        "flow_count": "1042/1250",
        "ssil": 83.36,
        "ssil_count": "1042/1250",
        "gap": "0.00",
        "mcnemar_p": "1.000",
    },
]


ABLATION_OBSERVATIONS = [
    {
        "factor": "Action-interface alignment",
        "evidence": (
            "Direct action prediction reached 0.00% in the initial diagnostic evaluation. "
            "Conditioning on the integrated flow state improved the same evaluation "
            "to 43.48%, and an action-state variant reached 62.70% under a "
            "larger diagnostic evaluation."
        ),
        "target_effect": "Improved with state alignment",
        "broad_effect": "Below BD-PIBT and FLOMAP",
        "inference": (
            "The action interface is sensitive to the state distribution seen at "
            "inference; adding an action head alone is insufficient."
        ),
    },
    {
        "factor": "Discrete-primary supervision",
        "evidence": (
            "A stronger action-loss objective reached 75% on the dense-random "
            "target in one diagnostic setting, while broad diagnostic success "
            "remained 60.00--61.08%."
        ),
        "target_effect": "Coordination target improved",
        "broad_effect": "Below BD-PIBT overall",
        "inference": (
            "Discrete action labels can help coordination-heavy behavior, but "
            "stronger discrete pressure can over-specialize."
        ),
    },
    {
        "factor": "Coordination-focused data",
        "evidence": (
            "A strict flow-only control trained on targeted coordination data "
            "reached 70% on the dense-random target, but broad diagnostic "
            "success stayed between 60.54% and 62.43% across selected model states."
        ),
        "target_effect": "Target topology improved",
        "broad_effect": "Below BD-PIBT overall",
        "inference": (
            "Topology-targeted data can move the desired slice; the data mixture "
            "and stopping rule still limit out-of-distribution transfer."
        ),
    },
    {
        "factor": "Hybrid action policy",
        "evidence": (
            "The final hybrid evaluation reached 62.49% overall and 58% on "
            "the dense-random map, below the 64.59% and 71% FLOMAP results."
        ),
        "target_effect": "No robust target gain",
        "broad_effect": "Below BD-PIBT and FLOMAP",
        "inference": (
            "The hybrid head did not replace the velocity policy as the most "
            "robust interface."
        ),
    },
    {
        "factor": "Reduced-capacity hybrid",
        "evidence": (
            "The smaller hybrid reached 60.87% in the initial diagnostic evaluation, below the "
            "primary FLOMAP policy."
        ),
        "target_effect": "No clear target advantage",
        "broad_effect": "Below BD-PIBT and FLOMAP",
        "inference": (
            "Capacity reduction did not explain the coordination-heavy gap."
        ),
    },
]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "<": r"$<$",
        ">": r"$>$",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def summarize_eval_csv(path: Path) -> Dict:
    rows = 0
    successes = 0
    total_goal_fraction = 0.0
    total_runtime = 0.0
    per_map: Dict[str, MutableMapping[str, float]] = {}

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            ok = parse_bool(row["success"])
            successes += int(ok)
            agent_num = float(row["agentNum"])
            total_goal_fraction += float(row["num_agents_at_goal"]) / agent_num
            total_runtime += float(row["runtime"])

            map_name = row["mapName"]
            cur = per_map.setdefault(map_name, {"rows": 0, "success_count": 0})
            cur["rows"] += 1
            cur["success_count"] += int(ok)

    if rows == 0:
        raise ValueError(f"CSV has no data rows: {path}")

    per_map_out = {}
    for map_name, cur in per_map.items():
        per_map_out[map_name] = 100.0 * cur["success_count"] / cur["rows"]

    return {
        "label": path.name,
        "rows": rows,
        "success_count": successes,
        "success_rate": 100.0 * successes / rows,
        "mean_agents_at_goal": 100.0 * total_goal_fraction / rows,
        "avg_runtime": total_runtime / rows,
        "per_map": per_map_out,
        "source": str(path.relative_to(REPO_ROOT)),
    }


def get_summary(key: str, allow_fallback: bool = True, prefer_csv: bool = True) -> Dict:
    path = CSV_PATHS[key]
    if prefer_csv and path.exists():
        summary = summarize_eval_csv(path)
        expected_rows = EXPECTED_ROWS.get(key)
        if expected_rows is not None and summary["rows"] != expected_rows:
            if allow_fallback:
                summary = dict(FALLBACKS[key])
                summary["source"] = "prompt fallback"
                return summary
            raise ValueError(
                f"CSV row count for {key} was {summary['rows']}, expected {expected_rows}: "
                f"{path.relative_to(REPO_ROOT)}"
            )
        summary["label"] = FALLBACKS[key]["label"]
        return summary
    if allow_fallback:
        summary = dict(FALLBACKS[key])
        summary["source"] = "prompt fallback"
        return summary
    raise FileNotFoundError(
        f"Missing CSV for {key}: {path.relative_to(REPO_ROOT)}. "
        "Run with fallback enabled or copy the remote CSV into the repo."
    )


def get_rate(summary: Mapping, map_name: str) -> float:
    return float(summary["per_map"][map_name])


def write_csv(path: Path, rows: Sequence[Mapping], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def fmt_pct(value: float) -> str:
    return f"{value:.2f}"


def write_tex_table(
    path: Path,
    rows: Sequence[Sequence[str]],
    headers: Sequence[str],
    align: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write(f"\\begin{{tabular}}{{{align}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(latex_escape(h) for h in headers) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(str(cell) for cell in row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def missing_lambda_csvs() -> List[Path]:
    return [path for path in LAMBDA_CSVS.values() if not path.exists()]


def lambda_copy_commands(lambda_host: str = "<lambda-host>") -> str:
    rel_root = "evals/lambda_sharded_rishi_lambda_20260429_191606/"
    return "\n".join(
        [
            "mkdir -p evals/lambda_sharded_rishi_lambda_20260429_191606",
            (
                f"rsync -av {lambda_host}:/home/anushree_mattlab/barath/Flow-CS-PIBT/"
                f"{rel_root} {rel_root}"
            ),
        ]
    )
