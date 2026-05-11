#!/usr/bin/env python3
"""Post-hoc kinematic adherence analysis for continuous MAPF evals.

The script expects eval_continuous.py CSV rows with velocity_history_path
columns produced by --save-velocity-history. Each velocity history is a
NumPy array with shape [T, N, 2].

Example:
    python scripts/analyze_kinematic_adherence.py \
        --csv evals/kinematics/*.csv \
        --maps random-32-32-10 \
        --agents 100 \
        --output-csv evals/kinematics/kinematic_summary.csv \
        --output-md docs/kinematic_adherence_results.md \
        --hist-dir evals/kinematics/histograms
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_csv_rows(paths: Iterable[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_csv_path"] = path
                rows.append(row)
    return rows


def resolve_history_path(row: Dict[str, str], column: str) -> Optional[str]:
    raw = (row.get(column) or "").strip()
    if not raw:
        return None
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(os.path.dirname(row["_csv_path"]), raw))


def default_method_label(row: Dict[str, str]) -> str:
    policy = (row.get("policy") or "").strip()
    shield = (row.get("shield_type") or "").strip()
    nav = (row.get("nav") or "").strip()
    kinematic = shield in {"epibt-kinematic", "cv-pibt-kinematic"}

    if policy in {"orca", "heuristic"} and shield == "orca":
        return "ORCA"

    if policy == "flow":
        prefix = "Flow"
    elif policy in {"orca", "heuristic"} and nav in {"", "straight", "goal"}:
        prefix = "Straight"
    elif policy in {"orca", "heuristic"}:
        prefix = nav.upper()
    elif policy == "discrete":
        prefix = "Discrete"
    else:
        prefix = policy or "Unknown"

    if shield in {"epibt", "cv-pibt", "epibt-kinematic", "cv-pibt-kinematic"}:
        suffix = "CV-PIBT-Kinematic" if kinematic else "CV-PIBT"
    elif shield:
        suffix = shield.upper()
    else:
        suffix = "No Shield"
    return f"{prefix}+{suffix}"


def method_label(row: Dict[str, str], method_column: Optional[str]) -> str:
    if method_column:
        value = (row.get(method_column) or "").strip()
        if value:
            return value
    return default_method_label(row)


def acceleration_magnitudes(velocities: np.ndarray, include_initial_rest: bool) -> np.ndarray:
    if velocities.ndim != 3 or velocities.shape[2] != 2:
        raise ValueError(f"Expected velocity history shape [T, N, 2], got {velocities.shape}")
    if include_initial_rest:
        initial = np.zeros((1, velocities.shape[1], 2), dtype=velocities.dtype)
        velocities = np.concatenate([initial, velocities], axis=0)
    if velocities.shape[0] < 2:
        return np.zeros((0, velocities.shape[1]), dtype=np.float32)
    return np.linalg.norm(np.diff(velocities, axis=0), axis=2)


def speed_variance(velocities: np.ndarray) -> float:
    if velocities.shape[0] == 0:
        return 0.0
    speeds = np.linalg.norm(velocities, axis=2)
    return float(np.mean(np.var(speeds, axis=0)))


def aligned_preferred_stats(
    committed: np.ndarray,
    preferred: np.ndarray,
    include_initial_rest: bool,
    spike_quantile: float,
) -> Tuple[int, float, float, float]:
    steps = min(committed.shape[0], preferred.shape[0])
    committed = committed[:steps]
    preferred = preferred[:steps]
    if include_initial_rest:
        initial = np.zeros((1, committed.shape[1], 2), dtype=committed.dtype)
        committed_for_delta = np.concatenate([initial, committed], axis=0)
        preferred_for_delta = np.concatenate([initial, preferred], axis=0)
        committed_accel = np.linalg.norm(np.diff(committed_for_delta, axis=0), axis=2)
        preferred_accel = np.linalg.norm(np.diff(preferred_for_delta, axis=0), axis=2)
        shield_gap = np.linalg.norm(committed - preferred, axis=2)
    else:
        if steps < 2:
            return 0, 0.0, 0.0, 0.0
        committed_accel = np.linalg.norm(np.diff(committed, axis=0), axis=2)
        preferred_accel = np.linalg.norm(np.diff(preferred, axis=0), axis=2)
        shield_gap = np.linalg.norm(committed[1:] - preferred[1:], axis=2)

    flat_committed = committed_accel.reshape(-1)
    if flat_committed.size == 0:
        return 0, 0.0, 0.0, 0.0
    threshold = float(np.quantile(flat_committed, spike_quantile))
    mask = committed_accel >= threshold
    count = int(mask.sum())
    if count == 0:
        return 0, 0.0, 0.0, 0.0
    return (
        count,
        float(committed_accel[mask].sum()),
        float(preferred_accel[mask].sum()),
        float(shield_gap[mask].sum()),
    )


def row_passes_filters(row: Dict[str, str], maps: Optional[set], agents: Optional[set]) -> bool:
    if maps is not None and row.get("map") not in maps:
        return False
    if agents is not None:
        agent_count = safe_int(row.get("agents"))
        if agent_count not in agents:
            return False
    return True


def empty_aggregate(amax_values: List[float]) -> Dict[str, object]:
    return {
        "episodes": 0,
        "pairs": 0,
        "accel_sum": 0.0,
        "violations": {amax: 0 for amax in amax_values},
        "speed_variance_sum": 0.0,
        "agents_at_goal": [],
        "agent_fraction_at_goal": [],
        "collisions": [],
        "spike_count": 0,
        "spike_committed_accel_sum": 0.0,
        "spike_preferred_accel_sum": 0.0,
        "spike_shield_gap_sum": 0.0,
    }


def aggregate_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, List[np.ndarray]], List[str]]:
    rows = load_csv_rows(args.csv)
    maps = set(args.maps) if args.maps else None
    agents = set(args.agents) if args.agents else None
    grouped = defaultdict(lambda: empty_aggregate(args.amax))
    hist_values: Dict[str, List[np.ndarray]] = defaultdict(list)
    warnings: List[str] = []

    for row in rows:
        if not row_passes_filters(row, maps, agents):
            continue
        history_path = resolve_history_path(row, args.velocity_history_column)
        if history_path is None:
            warnings.append(f"missing {args.velocity_history_column}: {row.get('_csv_path')}")
            continue
        if not os.path.exists(history_path):
            warnings.append(f"missing velocity history file: {history_path}")
            continue

        label = method_label(row, args.method_column)
        committed = np.load(history_path)
        accel = acceleration_magnitudes(committed, args.include_initial_rest)
        flat_accel = accel.reshape(-1)
        pairs = int(flat_accel.size)

        group = grouped[label]
        group["episodes"] = int(group["episodes"]) + 1
        group["pairs"] = int(group["pairs"]) + pairs
        group["accel_sum"] = float(group["accel_sum"]) + float(flat_accel.sum())
        group["speed_variance_sum"] = float(group["speed_variance_sum"]) + speed_variance(committed)
        hist_values[label].append(flat_accel.astype(np.float32, copy=False))

        violations = group["violations"]
        assert isinstance(violations, dict)
        for amax in args.amax:
            violations[amax] += int((flat_accel > amax).sum())

        for key in ("agents_at_goal", "agent_fraction_at_goal", "collisions"):
            value = safe_float(row.get(key))
            if value is not None:
                values = group[key]
                assert isinstance(values, list)
                values.append(value)

        preferred_path = resolve_history_path(row, args.preferred_history_column)
        if preferred_path and os.path.exists(preferred_path):
            preferred = np.load(preferred_path)
            spike_count, committed_sum, preferred_sum, shield_gap_sum = aligned_preferred_stats(
                committed,
                preferred,
                args.include_initial_rest,
                args.spike_quantile,
            )
            group["spike_count"] = int(group["spike_count"]) + spike_count
            group["spike_committed_accel_sum"] = float(group["spike_committed_accel_sum"]) + committed_sum
            group["spike_preferred_accel_sum"] = float(group["spike_preferred_accel_sum"]) + preferred_sum
            group["spike_shield_gap_sum"] = float(group["spike_shield_gap_sum"]) + shield_gap_sum

    output_rows: List[Dict[str, object]] = []
    for label in sorted(grouped):
        group = grouped[label]
        episodes = int(group["episodes"])
        pairs = int(group["pairs"])
        if episodes == 0 or pairs == 0:
            continue
        out: Dict[str, object] = {
            "method": label,
            "episodes": episodes,
            "pairs": pairs,
            "smoothness": float(group["accel_sum"]) / pairs,
            "speed_variance": float(group["speed_variance_sum"]) / episodes,
        }
        violations = group["violations"]
        assert isinstance(violations, dict)
        for amax in args.amax:
            out[f"accel_violation_rate@{amax:g}"] = violations[amax] / pairs
        for key in ("agents_at_goal", "agent_fraction_at_goal", "collisions"):
            values = group[key]
            assert isinstance(values, list)
            out[key] = float(np.mean(values)) if values else float("nan")
        spike_count = int(group["spike_count"])
        out["spike_count"] = spike_count
        if spike_count > 0:
            out["spike_committed_accel_mean"] = float(group["spike_committed_accel_sum"]) / spike_count
            out["spike_preferred_accel_mean"] = float(group["spike_preferred_accel_sum"]) / spike_count
            out["spike_shield_gap_mean"] = float(group["spike_shield_gap_sum"]) / spike_count
        else:
            out["spike_committed_accel_mean"] = float("nan")
            out["spike_preferred_accel_mean"] = float("nan")
            out["spike_shield_gap_mean"] = float("nan")
        output_rows.append(out)
    return output_rows, hist_values, warnings


def format_float(value: object, decimals: int = 3) -> str:
    if not isinstance(value, (float, int)) or math.isnan(float(value)):
        return ""
    return f"{float(value):.{decimals}f}"


def write_summary_csv(path: str, rows: List[Dict[str, object]], amax_values: List[float]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = (
        ["method", "episodes", "pairs"]
        + [f"accel_violation_rate@{amax:g}" for amax in amax_values]
        + [
            "smoothness",
            "speed_variance",
            "agents_at_goal",
            "agent_fraction_at_goal",
            "collisions",
            "spike_count",
            "spike_committed_accel_mean",
            "spike_preferred_accel_mean",
            "spike_shield_gap_mean",
        ]
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: List[Dict[str, object]], amax_values: List[float]) -> str:
    headers = (
        ["Method"]
        + [f"a_max={amax:g}" for amax in amax_values]
        + ["smoothness", "speed_var", "agents_at_goal", "collisions"]
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(row["method"])]
        cells += [format_float(row.get(f"accel_violation_rate@{amax:g}")) for amax in amax_values]
        cells += [
            format_float(row.get("smoothness")),
            format_float(row.get("speed_variance")),
            format_float(row.get("agents_at_goal"), decimals=1),
            format_float(row.get("collisions"), decimals=1),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary_markdown(path: str, rows: List[Dict[str, object]], amax_values: List[float]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# Kinematic Adherence Summary\n\n")
        f.write(markdown_table(rows, amax_values))
        f.write("\n\n")
        f.write(
            "Violation rates are computed from saved committed velocity histories as "
            "the fraction of agent-step acceleration magnitudes greater than a_max. "
            "By default, the initial rest-to-first-action transition is excluded to "
            "match compute_smoothness(). Use --include-initial-rest to include it.\n"
        )


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)


def write_histograms(hist_dir: str, hist_values: Dict[str, List[np.ndarray]], bins: int) -> None:
    os.makedirs(hist_dir, exist_ok=True)
    all_values = []
    for chunks in hist_values.values():
        if chunks:
            all_values.append(np.concatenate(chunks))
    if not all_values:
        return
    global_max = max(float(np.max(values)) for values in all_values if values.size)
    hist_range = (0.0, max(2.0, global_max))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None

    overlay_data = []
    for label, chunks in sorted(hist_values.items()):
        if not chunks:
            continue
        values = np.concatenate(chunks)
        counts, edges = np.histogram(values, bins=bins, range=hist_range)
        out_csv = os.path.join(hist_dir, f"{safe_filename(label)}_accel_hist.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["bin_left", "bin_right", "count"])
            for left, right, count in zip(edges[:-1], edges[1:], counts):
                writer.writerow([left, right, int(count)])
        overlay_data.append((label, values))

        if plt is not None:
            plt.figure(figsize=(7, 4))
            plt.hist(values, bins=bins, range=hist_range, density=True)
            plt.xlabel("||delta v||")
            plt.ylabel("density")
            plt.title(label)
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f"{safe_filename(label)}_accel_hist.png"), dpi=180)
            plt.close()

    if plt is not None and overlay_data:
        plt.figure(figsize=(8, 5))
        for label, values in overlay_data:
            plt.hist(values, bins=bins, range=hist_range, density=True, histtype="step", linewidth=1.8, label=label)
        plt.xlabel("||delta v||")
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(hist_dir, "accel_hist_overlay.png"), dpi=180)
        plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze acceleration adherence from saved velocity histories.")
    parser.add_argument("--csv", nargs="+", required=True, help="Evaluation CSV files from eval_continuous.py.")
    parser.add_argument("--maps", nargs="*", default=None, help="Optional map-name filter.")
    parser.add_argument("--agents", nargs="*", type=int, default=None, help="Optional agent-count filter.")
    parser.add_argument("--amax", nargs="+", type=float, default=[0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--method-column", default=None, help="Optional CSV column to use as the method label.")
    parser.add_argument("--velocity-history-column", default="velocity_history_path")
    parser.add_argument("--preferred-history-column", default="preferred_velocity_history_path")
    parser.add_argument("--include-initial-rest", action="store_true")
    parser.add_argument("--spike-quantile", type=float, default=0.95)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--hist-dir", default=None)
    parser.add_argument("--hist-bins", type=int, default=60)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not 0.0 < args.spike_quantile < 1.0:
        parser.error("--spike-quantile must be between 0 and 1")
    if any(amax <= 0.0 for amax in args.amax):
        parser.error("--amax values must be positive")

    rows, hist_values, warnings = aggregate_rows(args)
    if warnings:
        print("[kinematic] warnings:")
        for warning in warnings[:20]:
            print(f"  - {warning}")
        if len(warnings) > 20:
            print(f"  - ... {len(warnings) - 20} more")
    if not rows:
        raise SystemExit("No rows with usable velocity histories matched the inputs.")

    print(markdown_table(rows, args.amax))
    if args.output_csv:
        write_summary_csv(args.output_csv, rows, args.amax)
        print(f"[kinematic] wrote {args.output_csv}")
    if args.output_md:
        write_summary_markdown(args.output_md, rows, args.amax)
        print(f"[kinematic] wrote {args.output_md}")
    if args.hist_dir:
        write_histograms(args.hist_dir, hist_values, args.hist_bins)
        print(f"[kinematic] wrote histograms to {args.hist_dir}")


if __name__ == "__main__":
    main()
