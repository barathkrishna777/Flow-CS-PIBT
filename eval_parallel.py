"""
Parallel launcher for eval_continuous.py across multiple GPUs.

Enumerates all scenario IDs for the given maps, splits them round-robin
across N workers (one per GPU), launches N eval_continuous subprocesses,
then merges the per-worker CSV shards and prints a summary.

Usage:
    python eval_parallel.py \
        --map-dir data/mapf-map --scen-dir data/mapf-scen-random \
        --maps random-32-32-10 empty-48-48 \
        --agent-counts 50 100 --max-scenarios 25 \
        --policy flow \
        --model-path checkpoints/continuous_v4/continuous_flow_v4_best.pt \
        --shield-type epibt \
        --num-integration-steps 3 --max-steps 512 \
        --output-csv evals/v4/flow_epibt_512.csv \
        --num-gpus 4
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

from main_pys.continuous_scenarios import scenario_id_from_path, select_scenarios


def _find_scenario_ids(
    scen_dir: str,
    map_name: str,
    max_scenarios: int,
    scenario_ids: Optional[List[int]] = None,
    scenario_start: Optional[int] = None,
    scenario_end: Optional[int] = None,
) -> List[int]:
    files = sorted(glob.glob(os.path.join(scen_dir, f"{map_name}-random-*.scen")))
    selected = select_scenarios(
        files,
        max_scenarios=max_scenarios,
        scenario_ids=scenario_ids,
        scenario_start=scenario_start,
        scenario_end=scenario_end,
    )
    return sorted(scenario_id_from_path(f) for f in selected)


def _split_round_robin(ids: List[int], n: int) -> List[List[int]]:
    chunks: List[List[int]] = [[] for _ in range(n)]
    for i, sid in enumerate(ids):
        chunks[i % n].append(sid)
    return chunks


def _merge_csvs(shard_paths: List[str], output_csv: str) -> int:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    total_rows = 0
    header_written = False
    fieldnames = None
    with open(output_csv, "w", newline="") as out_f:
        writer = None
        for shard in shard_paths:
            if not os.path.exists(shard):
                continue
            with open(shard, newline="") as in_f:
                reader = csv.DictReader(in_f)
                if not header_written:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()
                    header_written = True
                for row in reader:
                    writer.writerow(row)
                    total_rows += 1
    return total_rows


def _print_summary(output_csv: str) -> None:
    if not os.path.exists(output_csv):
        print("[parallel] No output CSV to summarize")
        return
    rows = []
    with open(output_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[parallel] Output CSV is empty")
        return

    groups: Dict = defaultdict(list)
    for row in rows:
        groups[(row["map"], row["agents"])].append(row)

    print(f"\n[parallel] Summary — {len(rows)} total cases")
    print(f"{'Map':<22} {'N':>6} {'Scens':>6} {'Success':>9} {'AtGoal':>8} {'Collisions':>12}")
    print("-" * 68)
    for (map_name, agents), group in sorted(groups.items()):
        success = sum(float(r["success"]) for r in group) / len(group)
        at_goal = sum(float(r["agent_fraction_at_goal"]) for r in group) / len(group)
        collisions = sum(float(r["collisions"]) for r in group) / len(group)
        print(
            f"{map_name:<22} {agents:>6} {len(group):>6} "
            f"{success:>9.3f} {at_goal:>8.3f} {collisions:>12.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel launcher for eval_continuous.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--scen-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    # Task enumeration
    parser.add_argument("--maps", nargs="*", default=["empty-48-48", "random-32-32-10"])
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--max-scenarios", type=int, default=1)
    parser.add_argument("--scenario-ids", nargs="*", type=int, default=None)
    parser.add_argument("--scenario-start", type=int, default=None)
    parser.add_argument("--scenario-end", type=int, default=None)
    # Policy / model
    parser.add_argument("--policy", choices=["flow", "discrete", "orca"], default="orca")
    parser.add_argument("--nav", choices=["straight", "bd"], default="straight")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=0)
    # Shield / env
    parser.add_argument(
        "--shield-type",
        choices=["orca", "heuristic-orca", "po-orca", "epibt", "picbf-cs", "simple", "none"],
        default="orca",
    )
    parser.add_argument("--picbf-communication-radius", type=float, default=None)
    parser.add_argument("--num-integration-steps", type=int, default=3)
    parser.add_argument("--num-consensus-samples", type=int, default=1)
    parser.add_argument("--flow-aggregation", choices=["mean", "medoid", "best"], default="mean")
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.3)
    parser.add_argument("--goal-tolerance", type=float, default=0.25)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--viz-dir", default=None)
    # Parallelism
    parser.add_argument("--num-gpus", type=int, default=4)
    args = parser.parse_args()

    # Enumerate scenario IDs from the first map (IDs are consistent across maps)
    all_ids = _find_scenario_ids(
        args.scen_dir,
        args.maps[0],
        args.max_scenarios,
        scenario_ids=args.scenario_ids,
        scenario_start=args.scenario_start,
        scenario_end=args.scenario_end,
    )
    if not all_ids:
        print(f"[parallel] No scenarios found for {args.maps[0]} in {args.scen_dir}", file=sys.stderr)
        sys.exit(1)

    num_gpus = min(args.num_gpus, len(all_ids))
    chunks = _split_round_robin(all_ids, num_gpus)
    total_tasks = len(all_ids) * len(args.maps) * len(args.agent_counts)
    print(
        f"[parallel] {len(all_ids)} scenarios × {len(args.maps)} maps × "
        f"{len(args.agent_counts)} agent counts = {total_tasks} tasks"
    )
    print(f"[parallel] Splitting across {num_gpus} GPU worker(s)")

    # Shard output paths: output.0.csv, output.1.csv, ...
    base, ext = os.path.splitext(args.output_csv)
    if not ext:
        ext = ".csv"
    shard_paths = [f"{base}.{r}{ext}" for r in range(num_gpus)]

    # Build the base command forwarding all eval_continuous flags
    base_cmd: List[str] = [
        sys.executable, "-m", "eval_continuous",
        "--map-dir", args.map_dir,
        "--scen-dir", args.scen_dir,
        "--maps", *args.maps,
        "--agent-counts", *[str(a) for a in args.agent_counts],
        "--max-scenarios", str(args.max_scenarios),
        "--policy", args.policy,
        "--nav", args.nav,
        "--shield-type", args.shield_type,
        "--num-integration-steps", str(args.num_integration_steps),
        "--num-consensus-samples", str(args.num_consensus_samples),
        "--flow-aggregation", args.flow_aggregation,
        "--tau", str(args.tau),
        "--k", str(args.k),
        "--m", str(args.m),
        "--max-steps", str(args.max_steps),
        "--dt", str(args.dt),
        "--max-speed", str(args.max_speed),
        "--agent-radius", str(args.agent_radius),
        "--goal-tolerance", str(args.goal_tolerance),
        "--eval-seed", str(args.eval_seed),
        "--log-interval", str(args.log_interval),
    ]
    if args.model_path:
        base_cmd += ["--model-path", args.model_path]
    if args.run_name:
        base_cmd += ["--run-name", args.run_name]
    if args.train_seed is not None:
        base_cmd += ["--train-seed", str(args.train_seed)]
    if args.viz_dir:
        base_cmd += ["--viz-dir", args.viz_dir]
    if args.picbf_communication_radius is not None:
        base_cmd += ["--picbf-communication-radius", str(args.picbf_communication_radius)]

    # Launch one subprocess per GPU worker
    log_base = base
    procs = []
    launch_start = time.time()
    for rank, (ids_chunk, shard_path) in enumerate(zip(chunks, shard_paths)):
        if not ids_chunk:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        cmd = base_cmd + [
            "--output-csv", shard_path,
            "--scenario-ids", *[str(i) for i in ids_chunk],
        ]
        log_path = f"{log_base}.gpu{rank}.log"
        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        procs.append((rank, proc, log_f, shard_path, log_path))
        print(f"[parallel] GPU {rank}: scenarios={ids_chunk} -> {shard_path}  log={log_path}")

    print(f"[parallel] {len(procs)} worker(s) launched, waiting...")

    failed = []
    for rank, proc, log_f, shard_path, log_path in procs:
        retcode = proc.wait()
        log_f.close()
        status = "done" if retcode == 0 else f"FAILED (exit {retcode})"
        print(f"[parallel] GPU {rank}: {status}  log={log_path}")
        if retcode != 0:
            failed.append(rank)

    elapsed = time.time() - launch_start
    print(f"[parallel] All workers finished in {elapsed:.1f}s")
    if failed:
        print(f"[parallel] WARNING: workers {failed} exited with errors")

    # Merge shards
    existing = [p for _, _, _, p, _ in procs if os.path.exists(p)]
    if existing:
        n_rows = _merge_csvs(existing, args.output_csv)
        print(f"[parallel] Merged {len(existing)} shard(s) -> {args.output_csv} ({n_rows} rows)")
        _print_summary(args.output_csv)
    else:
        print("[parallel] No shard CSVs found — nothing to merge")
        sys.exit(1)


if __name__ == "__main__":
    main()
