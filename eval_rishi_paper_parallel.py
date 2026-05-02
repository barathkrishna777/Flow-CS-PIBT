"""
CPU-orchestrated parallel launcher for eval_rishi_paper-style simulator runs.

Each simulator subprocess gets a single visible GPU and writes to its own shard
CSV, then this script merges shards into the requested output CSV. This avoids
concurrent appends to the same file while keeping GPUs busy.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass

from eval_rishi_paper import (
    BD_DIR,
    DEFAULT_AGENT_COUNTS,
    MAP_NPZ,
    MAP_PRESETS,
    RISHI_HELD_OUT_MAPS,
    SCEN_DIR,
    _max_agents_in_scen,
    _print_preflight,
)


@dataclass(frozen=True)
class EvalTask:
    index: int
    map_name: str
    scen_path: str
    bd_path: str
    agent_count: int


def parse_args():
    p = argparse.ArgumentParser(description="Parallel Rishi-style grid benchmark launcher")
    p.add_argument("-m", "--model", default=None,
                   help="Checkpoint .pt path. Not required for --policy-type pibt.")
    p.add_argument("--output", "-o", required=True, help="Merged output CSV")
    p.add_argument("--map-set", choices=sorted(MAP_PRESETS), default="rishi8",
                   help="Named map preset (default: rishi8). Ignored when --maps is set.")
    p.add_argument("--maps", nargs="*", default=None,
                   help="Explicit map names. Overrides --map-set.")
    p.add_argument("--max-scenario", type=int, default=25,
                   help="Use random-1.scen .. random-N.scen per map (default 25)")
    p.add_argument("--scenario-start", type=int, default=1,
                   help="First 1-based random scenario index to include (default 1)")
    p.add_argument("--agents", nargs="*", type=int, default=DEFAULT_AGENT_COUNTS,
                   help="Agent counts (default: 100 200 ... 1000)")
    p.add_argument("--quick", action="store_true",
                   help="1 scenario, agents 100 400 800 only (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time-limit", type=int, default=120)
    p.add_argument("--max-steps-multiplier", type=str, default="3x")
    p.add_argument("--num-integration-steps", type=int, default=3)
    p.add_argument("--consensus", type=int, default=3)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--wait-thresh", type=float, default=0.25)
    p.add_argument("--wait-mode", choices=["threshold", "learned"], default="threshold")
    p.add_argument("--wait-logit-bias", type=float, default=0.0)
    p.add_argument("--wait-logit-scale", type=float, default=1.0)
    p.add_argument("--policy-type",
                   choices=["flow", "classifier", "flow_action_head", "local_classifier", "pibt"],
                   default="flow")
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--action-head-conditioning", choices=["integrated", "zero_t0", "zero_t05"],
                   default="integrated")
    p.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"],
                   help="GPU ids to schedule onto (default: 0 1 2 3)")
    p.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="Concurrent simulator subprocesses per GPU (default: 1)")
    p.add_argument("--shard-dir", default=None,
                   help="Directory for per-task CSV/log shards (default: <output stem>_shards)")
    p.add_argument("--resume-shards", action="store_true",
                   help="Skip tasks whose shard CSV already exists with at least one result row")
    p.add_argument("--merge-only", action="store_true",
                   help="Only merge existing shard CSVs into --output, then exit")
    p.add_argument("--torch-cpu-threads", type=int, default=None,
                   help="Set OMP/MKL/TORCH thread env vars for simulator subprocesses")
    return p.parse_args()


def shard_has_result(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader)
            return next(reader, None) is not None
    except OSError:
        return False


def build_tasks(args) -> tuple[list[EvalTask], list[tuple[str, int, int, int]], str, list[int]]:
    if args.policy_type == "pibt":
        model_path = "BD-PIBT"
    else:
        if args.model is None:
            print("ERROR: --model is required unless --policy-type pibt", file=sys.stderr)
            sys.exit(1)
        model_path = args.model
        if not os.path.isfile(model_path):
            print(f"ERROR: model not found: {model_path}", file=sys.stderr)
            sys.exit(1)

    maps = args.maps if args.maps else MAP_PRESETS[args.map_set]
    for map_name in maps:
        if map_name not in RISHI_HELD_OUT_MAPS:
            print(f"WARNING: {map_name} is not in the standard 8 held-out maps; continuing anyway.")

    if args.scenario_start < 1:
        print("ERROR: --scenario-start must be >= 1", file=sys.stderr)
        sys.exit(1)

    agent_counts = [100, 400, 800] if args.quick else list(args.agents)
    max_scen = 1 if args.quick else args.max_scenario
    scenario_start = 1 if args.quick else args.scenario_start

    tasks = []
    preflight = []
    for map_name in maps:
        pattern = os.path.join(SCEN_DIR, f"{map_name}-random-*.scen")
        scens = sorted(glob.glob(pattern))[scenario_start - 1: scenario_start - 1 + max_scen]
        if not scens:
            print(f"WARNING: no scenarios for {map_name}, skip")
            preflight.append((map_name, 0, 0, 0))
            continue

        available_bd = 0
        scheduled_runs = 0
        for scen_path in scens:
            basename = os.path.basename(scen_path).replace(".scen", "")
            bd_path = os.path.join(BD_DIR, f"{basename}_bds.npz")
            if not os.path.isfile(bd_path):
                print(f"WARNING: missing BD {bd_path}, skip")
                continue
            available_bd += 1
            max_avail = _max_agents_in_scen(scen_path)
            for agent_count in agent_counts:
                if agent_count > max_avail:
                    continue
                tasks.append(EvalTask(len(tasks) + 1, map_name, scen_path, bd_path, agent_count))
                scheduled_runs += 1
        preflight.append((map_name, len(scens), available_bd, scheduled_runs))

    return tasks, preflight, model_path, agent_counts


def simulator_cmd(args, task: EvalTask, model_path: str, shard_csv: str) -> list[str]:
    simulator_policy_type = "flow" if args.policy_type == "flow_action_head" else args.policy_type
    use_action_head = args.policy_type == "flow_action_head"
    return [
        sys.executable, "-m", "main_pys.simulator",
        f"--mapNpzFile={MAP_NPZ}",
        f"--mapName={task.map_name}",
        f"--scenFile={task.scen_path}",
        f"--bdNpzFile={task.bd_path}",
        f"--modelPath={model_path}",
        f"--outputCSVFile={shard_csv}",
        f"--maxSteps={args.max_steps_multiplier}",
        f"--seed={args.seed}",
        f"--useGPU={'False' if args.policy_type == 'pibt' else 'True'}",
        f"--agentNum={task.agent_count}",
        "--shieldType=CS-PIBT",
        f"--timeLimit={args.time_limit}",
        f"--numIntegrationSteps={args.num_integration_steps}",
        f"--tau={args.tau}",
        f"--waitThreshold={args.wait_thresh}",
        f"--waitMode={args.wait_mode}",
        f"--waitLogitBias={args.wait_logit_bias}",
        f"--waitLogitScale={args.wait_logit_scale}",
        f"--numConsensusSamples={args.consensus}",
        f"--policyType={simulator_policy_type}",
        f"--useActionHead={'True' if use_action_head else 'False'}",
        f"--actionHeadConditioning={args.action_head_conditioning}",
        f"--hiddenDim={args.hidden_dim}",
        f"--numLayers={args.num_layers}",
    ]


def worker_loop(worker_name, gpu_id, tasks, results, args, model_path, shard_dir, total):
    while True:
        try:
            task = tasks.get_nowait()
        except queue.Empty:
            return

        scen_tag = os.path.basename(task.scen_path)
        shard_csv = os.path.join(shard_dir, f"task_{task.index:05d}.csv")
        shard_log = os.path.join(shard_dir, f"task_{task.index:05d}_gpu{gpu_id}.log")
        print(
            f"[{task.index}/{total}] gpu={gpu_id} {task.map_name} | "
            f"{scen_tag} | {task.agent_count} ag",
            flush=True,
        )

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if args.torch_cpu_threads is not None:
            threads = str(args.torch_cpu_threads)
            env["OMP_NUM_THREADS"] = threads
            env["MKL_NUM_THREADS"] = threads
            env["OPENBLAS_NUM_THREADS"] = threads
            env["NUMEXPR_NUM_THREADS"] = threads
            env["TORCH_NUM_THREADS"] = threads
        cmd = simulator_cmd(args, task, model_path, shard_csv)
        with open(shard_log, "w") as log_file:
            try:
                completed = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                    timeout=args.time_limit + 120,
                )
                return_code = completed.returncode
            except subprocess.TimeoutExpired:
                return_code = -1
                log_file.write("\nSUBPROCESS TIMEOUT\n")

        results.append((task.index, return_code, shard_csv, shard_log, worker_name))
        tasks.task_done()


def merge_shards(output_csv: str, results: list[tuple[int, int, str, str, str]]) -> int:
    header = None
    rows_written = 0
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    with open(output_csv, "w", newline="") as out_file:
        writer = None
        for _, _, shard_csv, _, _ in sorted(results):
            if not os.path.isfile(shard_csv):
                continue
            with open(shard_csv, newline="") as in_file:
                reader = csv.reader(in_file)
                try:
                    shard_header = next(reader)
                except StopIteration:
                    continue
                if header is None:
                    header = shard_header
                    writer = csv.writer(out_file)
                    writer.writerow(header)
                for row in reader:
                    writer.writerow(row)
                    rows_written += 1
    return rows_written


def main():
    args = parse_args()
    tasks, preflight, model_path, agent_counts = build_tasks(args)
    if not tasks:
        _print_preflight(preflight)
        print("ERROR: no runs to execute.", file=sys.stderr)
        sys.exit(1)

    output_abs = os.path.abspath(args.output)
    shard_dir = args.shard_dir
    if shard_dir is None:
        stem, _ = os.path.splitext(output_abs)
        shard_dir = f"{stem}_shards"
    os.makedirs(shard_dir, exist_ok=True)
    if os.path.isfile(output_abs):
        os.remove(output_abs)

    completed_results = []
    pending_tasks = []
    for task in tasks:
        shard_csv = os.path.join(shard_dir, f"task_{task.index:05d}.csv")
        shard_log = os.path.join(shard_dir, f"task_{task.index:05d}_resume.log")
        if args.resume_shards and shard_has_result(shard_csv):
            completed_results.append((task.index, 0, shard_csv, shard_log, "resume"))
        else:
            pending_tasks.append(task)

    if args.merge_only:
        merge_results = []
        for task in tasks:
            shard_csv = os.path.join(shard_dir, f"task_{task.index:05d}.csv")
            if shard_has_result(shard_csv):
                merge_results.append((task.index, 0, shard_csv, "", "merge"))
        rows_written = merge_shards(output_abs, merge_results)
        print(f"Done. Merged {rows_written} existing shard rows to {output_abs}")
        return

    print("=" * 60)
    print("Parallel Rishi-style grid benchmark")
    print(f"Model: {model_path}")
    print(f"Maps: {len(args.maps) if args.maps else args.map_set}")
    print(f"Agents: {agent_counts}")
    print(f"waitMode={args.wait_mode} waitLogitScale={args.wait_logit_scale} "
          f"waitLogitBias={args.wait_logit_bias}")
    print(f"GPUs: {args.gpus} | jobs/gpu={args.jobs_per_gpu}")
    if args.resume_shards:
        print(f"Resume shards: skipped={len(completed_results)} pending={len(pending_tasks)}")
    if args.torch_cpu_threads is not None:
        print(f"Simulator CPU threads/process: {args.torch_cpu_threads}")
    _print_preflight(preflight)
    print(f"Total runs: {len(tasks)}")
    print(f"Output: {output_abs}")
    print(f"Shards/logs: {shard_dir}")
    print("=" * 60)

    task_queue = queue.Queue()
    for task in pending_tasks:
        task_queue.put(task)

    results = completed_results[:]
    threads = []
    for gpu_id in args.gpus:
        for slot in range(args.jobs_per_gpu):
            name = f"gpu{gpu_id}-slot{slot}"
            thread = threading.Thread(
                target=worker_loop,
                args=(name, gpu_id, task_queue, results, args, model_path, shard_dir, len(tasks)),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    rows_written = merge_shards(output_abs, results)
    failures = [item for item in results if item[1] != 0]
    print(f"\nDone. Wrote {rows_written} rows to {output_abs}")
    if failures:
        print(f"WARNING: {len(failures)} task(s) failed or timed out:")
        for task_index, return_code, _, shard_log, worker_name in sorted(failures):
            print(f"  task={task_index} returncode={return_code} worker={worker_name} log={shard_log}")
        sys.exit(1)


if __name__ == "__main__":
    main()
