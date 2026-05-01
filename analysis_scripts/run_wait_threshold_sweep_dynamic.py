#!/usr/bin/env python3
"""Dynamic multi-GPU wait-threshold sweep scheduler.

Unlike the shell sweep runners, this script does not run fixed synchronous
shards. It builds individual simulator jobs, keeps one process active per GPU,
and dispatches the next pending job to whichever GPU becomes free first. The
parent process is the only writer to the per-threshold combined CSVs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MAP_NPZ = "data/all_maps.npz"
BD_DIR = "data/bd_npzs/large_scale"
SCEN_DIR = "data/scen-random"

RISHI8_MAPS = [
    "Paris_1_256",
    "empty-48-48",
    "maze-128-128-2",
    "random-64-64-10",
    "random-32-32-10",
    "warehouse-10-20-10-2-1",
    "den312d",
    "den520d",
]

DEFAULT_THRESHOLDS = ["0.00", "0.10", "0.25", "0.40", "0.60"]
DEFAULT_FULL_AGENTS = list(range(100, 1001, 100))
DEFAULT_MINI_AGENTS = [100, 200, 400, 600, 800]
DEFAULT_QUICK_AGENTS = [100, 400, 800]


@dataclass(frozen=True)
class Task:
    threshold: str
    tag: str
    map_name: str
    scen_path: str
    bd_path: str
    agent_num: int
    key: tuple[str, str, int, str]


@dataclass
class Running:
    task: Task
    gpu: str
    proc: subprocess.Popen
    task_csv: Path
    log_path: Path
    started_at: float


def tag_for_threshold(value: str) -> str:
    return f"wt_{float(value):0.2f}".replace(".", "p")


def max_agents_in_scen(scen_path: str) -> int:
    with open(scen_path) as f:
        return max(0, len(f.readlines()) - 1)


def scenario_number(path: str) -> int:
    name = Path(path).stem
    try:
        return int(name.rsplit("-random-", 1)[1])
    except (IndexError, ValueError):
        return 10**9


def build_tasks(
    thresholds: list[str],
    maps: list[str],
    max_scenario: int,
    scenario_start: int,
    agents: list[int],
) -> list[Task]:
    tasks: list[Task] = []
    for threshold in thresholds:
        tag = tag_for_threshold(threshold)
        for map_name in maps:
            pattern = os.path.join(SCEN_DIR, f"{map_name}-random-*.scen")
            scens = sorted(glob.glob(pattern), key=scenario_number)
            scens = scens[scenario_start - 1: scenario_start - 1 + max_scenario]
            for scen_path in scens:
                bn = os.path.basename(scen_path).replace(".scen", "")
                bd_path = os.path.join(BD_DIR, f"{bn}_bds.npz")
                if not os.path.isfile(bd_path):
                    print(f"WARNING: missing BD {bd_path}, skip", file=sys.stderr)
                    continue
                max_avail = max_agents_in_scen(scen_path)
                for n in agents:
                    if n > max_avail:
                        continue
                    key = (map_name, scen_path, n, threshold)
                    tasks.append(Task(threshold, tag, map_name, scen_path, bd_path, n, key))
    return tasks


def read_done_keys(out_root: Path, thresholds: Iterable[str]) -> set[tuple[str, str, int, str]]:
    done: set[tuple[str, str, int, str]] = set()
    for threshold in thresholds:
        tag = tag_for_threshold(threshold)
        combined = out_root / tag / f"{tag}_combined.csv"
        if not combined.exists():
            continue
        with combined.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add((row["mapName"], row["scenFile"], int(row["agentNum"]), threshold))
    return done


def append_task_csv(task_csv: Path, combined_csv: Path) -> int:
    with task_csv.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0
        rows = [row for row in reader if row]

    if not rows:
        return 0

    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not combined_csv.exists() or combined_csv.stat().st_size == 0
    with combined_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


def launch_task(args: argparse.Namespace, task: Task, gpu: str) -> Running:
    task_dir = Path(args.out_root) / "task_csvs" / task.tag
    log_dir = Path(args.out_root) / "logs" / task.tag
    task_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    scen_name = Path(task.scen_path).stem
    stem = f"{task.tag}_{task.map_name}_{scen_name}_{task.agent_num}"
    task_csv = task_dir / f"{stem}.csv"
    log_path = log_dir / f"{stem}_gpu{gpu}.log"
    task_csv.unlink(missing_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "main_pys.simulator",
        f"--mapNpzFile={MAP_NPZ}",
        f"--mapName={task.map_name}",
        f"--scenFile={task.scen_path}",
        f"--bdNpzFile={task.bd_path}",
        f"--modelPath={args.model}",
        f"--outputCSVFile={task_csv}",
        f"--maxSteps={args.max_steps_multiplier}",
        f"--seed={args.seed}",
        "--useGPU=True",
        f"--agentNum={task.agent_num}",
        "--shieldType=CS-PIBT",
        f"--timeLimit={args.time_limit}",
        f"--numIntegrationSteps={args.num_integration_steps}",
        f"--tau={args.tau}",
        f"--waitThreshold={task.threshold}",
        f"--numConsensusSamples={args.consensus}",
        f"--policyType={args.policy_type}",
        "--useActionHead=False",
        "--actionHeadConditioning=integrated",
        f"--hiddenDim={args.hidden_dim}",
        f"--numLayers={args.num_layers}",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    log_f = log_path.open("w")
    print(
        f"launch gpu={gpu} wt={task.threshold} map={task.map_name} "
        f"scen={Path(task.scen_path).name} agents={task.agent_num}",
        flush=True,
    )
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
    return Running(task, gpu, proc, task_csv, log_path, time.time())


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--mode", choices=["quick", "mini", "full"], default="mini")
    parser.add_argument("--thresholds", nargs="*", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--maps", nargs="*", default=RISHI8_MAPS)
    parser.add_argument("--gpus", nargs="*", default=["0", "1", "2", "3"])
    parser.add_argument("--agents", nargs="*", type=int, default=None)
    parser.add_argument("--max-scenario", type=int, default=None)
    parser.add_argument("--scenario-start", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=120)
    parser.add_argument("--max-steps-multiplier", default="3x")
    parser.add_argument("--num-integration-steps", type=int, default=3)
    parser.add_argument("--consensus", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--policy-type", choices=["flow"], default="flow")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--stop-after-threshold", default=None)
    args = parser.parse_args()

    if args.mode == "quick":
        agents = args.agents or DEFAULT_QUICK_AGENTS
        max_scenario = args.max_scenario or 1
    elif args.mode == "mini":
        agents = args.agents or DEFAULT_MINI_AGENTS
        max_scenario = args.max_scenario or 5
    else:
        agents = args.agents or DEFAULT_FULL_AGENTS
        max_scenario = args.max_scenario or 25

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for threshold in args.thresholds:
        tag = tag_for_threshold(threshold)
        (out_root / tag).mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args.thresholds, args.maps, max_scenario, args.scenario_start, agents)
    done = read_done_keys(out_root, args.thresholds)
    queue = [task for task in tasks if task.key not in done]
    total = len(tasks)
    print(f"Total scheduled rows: {total}")
    print(f"Already complete rows: {len(done)}")
    print(f"Pending rows: {len(queue)}")

    running: dict[str, Running] = {}
    completed_now = 0
    failures = 0

    try:
        while queue or running:
            for gpu in args.gpus:
                if gpu in running or not queue:
                    continue
                task = queue.pop(0)
                running[gpu] = launch_task(args, task, gpu)

            time.sleep(args.poll_seconds)

            for gpu, run in list(running.items()):
                rc = run.proc.poll()
                if rc is None:
                    continue
                combined = out_root / run.task.tag / f"{run.task.tag}_combined.csv"
                rows = append_task_csv(run.task_csv, combined)
                elapsed = time.time() - run.started_at
                if rc != 0 or rows != 1:
                    failures += 1
                    print(
                        f"FAIL gpu={gpu} rc={rc} rows={rows} elapsed={elapsed:.1f}s "
                        f"task={run.task}",
                        file=sys.stderr,
                        flush=True,
                    )
                    print(f"  log: {run.log_path}", file=sys.stderr)
                else:
                    completed_now += 1
                    count = row_count(combined)
                    print(
                        f"done gpu={gpu} wt={run.task.threshold} rows={count} "
                        f"elapsed={elapsed:.1f}s map={run.task.map_name} "
                        f"agents={run.task.agent_num}",
                        flush=True,
                    )
                del running[gpu]

            if args.stop_after_threshold:
                tag = tag_for_threshold(args.stop_after_threshold)
                threshold_tasks = [task for task in tasks if task.tag == tag]
                combined = out_root / tag / f"{tag}_combined.csv"
                if row_count(combined) >= len(threshold_tasks):
                    print(f"stop-after-threshold reached: {args.stop_after_threshold}")
                    break
    finally:
        for run in running.values():
            run.proc.terminate()

    print(f"Completed in this run: {completed_now}")
    print(f"Failures: {failures}")
    print(f"OUT_ROOT={out_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
