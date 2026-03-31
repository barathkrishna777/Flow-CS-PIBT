import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent


def run_command(cmd: Sequence[str], dry_run: bool) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(list(cmd), check=True)


def scenario_split(
    scenario_id: int,
    train_range: Optional[Sequence[int]],
    val_range: Optional[Sequence[int]],
    test_range: Optional[Sequence[int]],
) -> str:
    if train_range and train_range[0] <= scenario_id <= train_range[1]:
        return "train"
    if val_range and val_range[0] <= scenario_id <= val_range[1]:
        return "val"
    if test_range and test_range[0] <= scenario_id <= test_range[1]:
        return "test"
    return "unassigned"


def load_scalar_str(data: Dict[str, np.ndarray], key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = data[key]
    if getattr(value, "ndim", 0) == 0:
        return str(value.item())
    return str(value[0])


def load_scalar_int(data: Dict[str, np.ndarray], key: str, default: int = 0) -> int:
    if key not in data:
        return default
    value = data[key]
    return int(value.item() if getattr(value, "ndim", 0) == 0 else value[0])


def load_scalar_float(data: Dict[str, np.ndarray], key: str, default: float = 0.0) -> float:
    if key not in data:
        return default
    value = data[key]
    return float(value.item() if getattr(value, "ndim", 0) == 0 else value[0])


def build_manifest(
    raw_dir: Path,
    manifests_dir: Path,
    train_range: Optional[Sequence[int]],
    val_range: Optional[Sequence[int]],
    test_range: Optional[Sequence[int]],
) -> None:
    raw_files = sorted(raw_dir.glob("*.npz"))
    if not raw_files:
        raise RuntimeError(f"No rollout .npz files found in {raw_dir}")

    rows: List[Dict[str, object]] = []
    source_counter: Counter = Counter()
    split_counter: Counter = Counter()
    for path in raw_files:
        with np.load(path, allow_pickle=True) as data:
            scenario_id = load_scalar_int(data, "scenario_id", -1)
            split = scenario_split(scenario_id, train_range, val_range, test_range)
            expert_source_used = load_scalar_str(data, "expert_source_used", load_scalar_str(data, "expert_source", ""))
            row = {
                "file": path.name,
                "map_name": load_scalar_str(data, "map_name"),
                "scenario_name": load_scalar_str(data, "scenario_name"),
                "scenario_id": scenario_id,
                "split": split,
                "agent_count": load_scalar_int(data, "agent_count", int(data["positions"].shape[1])),
                "expert_recipe_requested": load_scalar_str(data, "expert_recipe_requested", expert_source_used),
                "expert_source_used": expert_source_used,
                "fallback_reason": load_scalar_str(data, "fallback_reason", "none"),
                "rollout_length": load_scalar_int(data, "rollout_length", int(data["positions"].shape[0] - 1)),
                "fraction_moving": load_scalar_float(data, "fraction_moving", 0.0),
                "mean_nearest_neighbor_distance": load_scalar_float(data, "mean_nearest_neighbor_distance", 0.0),
                "dt": load_scalar_float(data, "dt", 0.0),
            }
            rows.append(row)
            source_counter[(row["expert_source_used"], row["agent_count"])] += 1
            split_counter[split] += 1

    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = manifests_dir / "manifest.csv"
    with manifest_csv.open("w", newline="") as handle:
        fieldnames = [
            "file",
            "map_name",
            "scenario_name",
            "scenario_id",
            "split",
            "agent_count",
            "expert_recipe_requested",
            "expert_source_used",
            "fallback_reason",
            "rollout_length",
            "fraction_moving",
            "mean_nearest_neighbor_distance",
            "dt",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    split_json = manifests_dir / "splits.json"
    split_json.write_text(json.dumps(dict(split_counter), indent=2, sort_keys=True))

    source_json = manifests_dir / "source_summary.json"
    source_json.write_text(
        json.dumps(
            {
                f"{source}|{agent_count}": count
                for (source, agent_count), count in sorted(source_counter.items())
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Wrote manifest: {manifest_csv}")


def parse_range(start: Optional[int], end: Optional[int]) -> Optional[List[int]]:
    if start is None or end is None:
        return None
    return [int(start), int(end)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the main continuous-space dataset")
    parser.add_argument("--dataset-root", default="data/continuous_main")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--scen-dir", required=True)
    parser.add_argument("--maps", nargs="+", default=["empty-48-48"])
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[16, 32, 64, 96, 128, 160])
    parser.add_argument("--expert-source", choices=["eecbs", "orca", "hybrid"], default="hybrid")
    parser.add_argument("--max-scenarios", type=int, default=80)
    parser.add_argument("--scenario-start", type=int, default=1)
    parser.add_argument("--scenario-end", type=int, default=80)
    parser.add_argument("--train-scenario-start", type=int, default=1)
    parser.add_argument("--train-scenario-end", type=int, default=50)
    parser.add_argument("--val-scenario-start", type=int, default=51)
    parser.add_argument("--val-scenario-end", type=int, default=60)
    parser.add_argument("--test-scenario-start", type=int, default=61)
    parser.add_argument("--test-scenario-end", type=int, default=80)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.3)
    parser.add_argument("--goal-tolerance", type=float, default=0.25)
    parser.add_argument("--wait-threshold", type=float, default=0.1)
    parser.add_argument("--num-directions", type=int, default=8)
    parser.add_argument("--suboptimality", type=float, default=1.2)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--eecbs-repo", default=str(REPO_ROOT.parent / "EECBS-flow"))
    parser.add_argument("--eecbs-binary", default=None)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    raw_dir = dataset_root / "raw"
    manifests_dir = dataset_root / "manifests"
    dataset_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_generate:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "generate_continuous_data.py"),
            "--map-dir",
            args.map_dir,
            "--scen-dir",
            args.scen_dir,
            "--maps",
            *args.maps,
            "--agent-counts",
            *[str(v) for v in args.agent_counts],
            "--output-dir",
            str(raw_dir),
            "--expert-source",
            args.expert_source,
            "--max-scenarios",
            str(args.max_scenarios),
            "--scenario-start",
            str(args.scenario_start),
            "--scenario-end",
            str(args.scenario_end),
            "--rollout-horizon",
            str(args.rollout_horizon),
            "--dt",
            str(args.dt),
            "--max-speed",
            str(args.max_speed),
            "--agent-radius",
            str(args.agent_radius),
            "--goal-tolerance",
            str(args.goal_tolerance),
            "--wait-threshold",
            str(args.wait_threshold),
            "--num-directions",
            str(args.num_directions),
            "--suboptimality",
            str(args.suboptimality),
            "--time-limit",
            str(args.time_limit),
        ]
        if args.eecbs_repo:
            cmd.extend(["--eecbs-repo", args.eecbs_repo])
        if args.eecbs_binary:
            cmd.extend(["--eecbs-binary", args.eecbs_binary])
        run_command(cmd, args.dry_run)

    generation_config = {
        "maps": args.maps,
        "agent_counts": args.agent_counts,
        "expert_source": args.expert_source,
        "scenario_start": args.scenario_start,
        "scenario_end": args.scenario_end,
        "train_range": parse_range(args.train_scenario_start, args.train_scenario_end),
        "val_range": parse_range(args.val_scenario_start, args.val_scenario_end),
        "test_range": parse_range(args.test_scenario_start, args.test_scenario_end),
        "rollout_horizon": args.rollout_horizon,
        "dt": args.dt,
        "max_speed": args.max_speed,
        "agent_radius": args.agent_radius,
        "goal_tolerance": args.goal_tolerance,
        "wait_threshold": args.wait_threshold,
        "num_directions": args.num_directions,
        "suboptimality": args.suboptimality,
        "time_limit": args.time_limit,
    }
    manifests_dir.mkdir(parents=True, exist_ok=True)
    (manifests_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=2, sort_keys=True))

    if not args.dry_run:
        build_manifest(
            raw_dir=raw_dir,
            manifests_dir=manifests_dir,
            train_range=parse_range(args.train_scenario_start, args.train_scenario_end),
            val_range=parse_range(args.val_scenario_start, args.val_scenario_end),
            test_range=parse_range(args.test_scenario_start, args.test_scenario_end),
        )
    else:
        print(f"Would write manifest and config under {manifests_dir}")


if __name__ == "__main__":
    main()
