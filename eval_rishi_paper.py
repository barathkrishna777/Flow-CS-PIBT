"""
Held-out benchmark aligned with Veerapaneni et al. (Work Smarter, Not Harder).

Evaluates named Rishi-style map presets, with the same scenario sweep and
density ladder used in prior project batch evals (e.g. batch_results_wave5_best.csv):
  - Default maps: HELD_OUT_TEST set (8 maps)
  - Optional maps: 12-map panel from Rishi's reported evals
  - Scenarios: random-1 .. random-N per map (default N=25, matching scen-random caps)
  - Agents: 100, 200, ..., 1000

Inference defaults match eval_full / paper: time limit 120s, maxSteps 3x,
tau=0.3, consensus=3, numIntegrationSteps=3 (override with flags).

Usage:
  python eval_rishi_paper.py -m large_scale_flow_wave8_base_best.pt \\
      --output checkpoints_and_evaluations/eval_rishi_wave8_base.csv

  python eval_rishi_paper.py -m model.pt --quick --output logs/rishi_quick.csv

  # Paper-scale run is large (8 maps x 25 scens x 10 agent levels = 2000 sims).
  # Rishi's 12-map panel is larger:
  python eval_rishi_paper.py -m model.pt --map-set rishi12 --output logs/rishi12.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

MAP_NPZ = "data/all_maps.npz"
BD_DIR = "data/bd_npzs/large_scale"
SCEN_DIR = "data/scen-random"

# Same 8 maps as generate_and_preprocess.HELD_OUT_TEST (paper protocol)
RISHI_HELD_OUT_MAPS = [
    "Paris_1_256",
    "empty-48-48",
    "maze-128-128-2",
    "random-64-64-10",
    "random-32-32-10",
    "warehouse-10-20-10-2-1",
    "den312d",
    "den520d",
]

RISHI_12_MAPS = [
    "Berlin_1_256",
    "empty-32-32",
    "maze-32-32-4",
    "random-64-64-20",
    "warehouse-20-40-10-2-1",
    "room-64-64-16",
    "Paris_1_256",
    "empty-48-48",
    "maze-128-128-2",
    "random-64-64-10",
    "warehouse-10-20-10-2-1",
    "den312d",
]

MAP_PRESETS = {
    "rishi8": RISHI_HELD_OUT_MAPS,
    "rishi12": RISHI_12_MAPS,
}

DEFAULT_AGENT_COUNTS = list(range(100, 1001, 100))


def _load_completed_runs(csv_path: str) -> set[tuple[str, str, int]]:
    """Keys (mapName, scenFile, agentNum) already in output — for --resume."""
    done: set[tuple[str, str, int]] = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return done
        for row in reader:
            try:
                done.add(
                    (row["mapName"], row["scenFile"], int(row["agentNum"]))
                )
            except (KeyError, ValueError):
                continue
    return done


def _run_sim(job):
    """Module-level so ProcessPoolExecutor can pickle it."""
    (idx, map_name, scen_path, bd_path, n, cfg) = job
    env = os.environ.copy()
    ng = int(cfg.get("num_gpus", 1))
    if ng > 1 and cfg.get("use_gpu") == "True":
        env["CUDA_VISIBLE_DEVICES"] = str((idx - 1) % ng)
    cmd = [
        cfg["python"], "-m", "main_pys.simulator",
        f"--mapNpzFile={cfg['map_npz']}",
        f"--mapName={map_name}",
        f"--scenFile={scen_path}",
        f"--bdNpzFile={bd_path}",
        f"--modelPath={cfg['model']}",
        f"--outputCSVFile={cfg['output']}",
        f"--maxSteps={cfg['max_steps_multiplier']}",
        f"--seed={cfg['seed']}",
        f"--useGPU={cfg['use_gpu']}",
        f"--agentNum={n}",
        "--shieldType=CS-PIBT",
        f"--timeLimit={cfg['time_limit']}",
        f"--numIntegrationSteps={cfg['num_integration_steps']}",
        f"--tau={cfg['tau']}",
        f"--waitThreshold={cfg['wait_thresh']}",
        f"--numConsensusSamples={cfg['consensus']}",
        f"--policyType={cfg['simulator_policy_type']}",
        f"--useActionHead={cfg['use_action_head']}",
        f"--hiddenDim={cfg['hidden_dim']}",
        f"--numLayers={cfg['num_layers']}",
    ]
    try:
        subprocess.run(
            cmd,
            check=False,
            timeout=cfg["time_limit"] + 120,
            env=env,
        )
        return idx, map_name, scen_path, n, "ok"
    except Exception:
        return idx, map_name, scen_path, n, "timeout"


def _max_agents_in_scen(scen_path: str) -> int:
    with open(scen_path) as f:
        return max(0, len(f.readlines()) - 1)


def _print_preflight(preflight: list[tuple[str, int, int, int]]) -> None:
    print("Preflight by map:")
    for map_name, scen_count, bd_count, run_count in preflight:
        status = "skip" if run_count == 0 else "ok"
        print(f"  {status:4} {map_name}: scenarios={scen_count} bds={bd_count} runs={run_count}")


def main():
    p = argparse.ArgumentParser(description="Rishi-style grid benchmark")
    p.add_argument("-m", "--model", required=True, help="Checkpoint .pt path")
    p.add_argument("--output", "-o", required=True, help="Output CSV (simulator format)")
    p.add_argument("--map-set", choices=sorted(MAP_PRESETS), default="rishi8",
                   help="Named map preset (default: rishi8). Ignored when --maps is set.")
    p.add_argument("--maps", nargs="*", default=None,
                   help="Explicit map names. Overrides --map-set.")
    p.add_argument("--max-scenario", type=int, default=25,
                   help="Use random-1.scen .. random-N.scen per map (default 25)")
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
    p.add_argument("--policy-type", choices=["flow", "classifier", "flow_action_head", "local_classifier"], default="flow",
                   help="Policy/model family to evaluate. flow_action_head loads a flow model and uses its action logits; local_classifier loads the repo's Rishi-like classifier.")
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--parallel", type=int, default=1,
                   help="Number of simulator subprocesses to run in parallel (default 1)")
    p.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("EVAL_NUM_GPUS", "1")),
        help="Round-robin runs across N GPUs (sets CUDA_VISIBLE_DEVICES per subprocess). Default: EVAL_NUM_GPUS or 1.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If output CSV exists, skip runs already recorded and append new rows (for time-limit restarts).",
    )
    args = p.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    maps = args.maps if args.maps else MAP_PRESETS[args.map_set]
    map_source = "custom --maps" if args.maps else args.map_set
    for m in maps:
        if m not in RISHI_HELD_OUT_MAPS:
            print(f"WARNING: {m} is not in the standard 8 held-out maps; continuing anyway.")

    agent_counts = [100, 400, 800] if args.quick else list(args.agents)
    max_scen = 1 if args.quick else args.max_scenario

    runs = []
    preflight = []
    for map_name in maps:
        pattern = os.path.join(SCEN_DIR, f"{map_name}-random-*.scen")
        scens = sorted(glob.glob(pattern))[:max_scen]
        if not scens:
            print(f"WARNING: no scenarios for {map_name}, skip")
            preflight.append((map_name, 0, 0, 0))
            continue
        available_bd = 0
        scheduled_runs = 0
        for scen_path in scens:
            bn = os.path.basename(scen_path).replace(".scen", "")
            bd_path = os.path.join(BD_DIR, f"{bn}_bds.npz")
            if not os.path.isfile(bd_path):
                print(f"WARNING: missing BD {bd_path}, skip")
                continue
            available_bd += 1
            max_avail = _max_agents_in_scen(scen_path)
            for n in agent_counts:
                if n > max_avail:
                    continue
                runs.append((map_name, scen_path, bd_path, n))
                scheduled_runs += 1
        preflight.append((map_name, len(scens), available_bd, scheduled_runs))

    if not runs:
        _print_preflight(preflight)
        print("ERROR: no runs to execute.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.resume and os.path.isfile(args.output):
        done = _load_completed_runs(args.output)
        before_n = len(runs)
        runs = [
            (mn, sp, bp, na)
            for mn, sp, bp, na in runs
            if (mn, sp, na) not in done
        ]
        print(
            f"Resume: {before_n - len(runs)} runs already in {args.output}, "
            f"{len(runs)} remaining.",
            flush=True,
        )
        if not runs:
            print("Nothing left to run.", flush=True)
            sys.exit(0)
    elif os.path.isfile(args.output):
        os.remove(args.output)

    total = len(runs)
    print("=" * 60)
    print("Rishi-style grid benchmark")
    print(f"Model: {args.model}")
    print(f"Map set: {map_source} | Requested maps: {len(maps)}")
    print(f"Scenarios/map: <= {max_scen} | Agents: {agent_counts[0]}..{agent_counts[-1]}")
    print(f"Policy: {args.policy_type}")
    print(f"steps={args.num_integration_steps} consensus={args.consensus} tau={args.tau}")
    print(f"timeLimit={args.time_limit}s maxSteps={args.max_steps_multiplier} | GPU: {use_gpu}")
    _print_preflight(preflight)
    print(f"Total runs: {total}")
    print(f"Output: {args.output}")
    print("=" * 60)

    simulator_policy_type = "flow" if args.policy_type == "flow_action_head" else args.policy_type
    use_action_head = args.policy_type == "flow_action_head"

    num_gpus = max(1, int(args.num_gpus))
    cfg = {
        "python": sys.executable,
        "map_npz": MAP_NPZ,
        "model": args.model,
        "output": args.output,
        "max_steps_multiplier": args.max_steps_multiplier,
        "seed": args.seed,
        "use_gpu": "True" if use_gpu else "False",
        "time_limit": args.time_limit,
        "num_integration_steps": args.num_integration_steps,
        "tau": args.tau,
        "wait_thresh": args.wait_thresh,
        "consensus": args.consensus,
        "simulator_policy_type": simulator_policy_type,
        "use_action_head": "True" if use_action_head else "False",
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_gpus": num_gpus,
    }

    jobs = [(i, map_name, scen_path, bd_path, n, cfg)
            for i, (map_name, scen_path, bd_path, n) in enumerate(runs, 1)]

    completed = 0
    if args.parallel <= 1:
        for job in jobs:
            idx, map_name, scen_path, bd_path, n, _ = job
            print(f"[{idx}/{total}] {map_name} | {os.path.basename(scen_path)} | {n} ag", flush=True)
            _, _, _, _, status = _run_sim(job)
            if status == "timeout":
                print(f"  -> subprocess timeout")
            completed += 1
    else:
        print(f"Running with --parallel {args.parallel}", flush=True)
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(_run_sim, job) for job in jobs]
            for fut in as_completed(futures):
                idx, map_name, scen_path, n, status = fut.result()
                completed += 1
                scen_tag = os.path.basename(scen_path)
                suffix = " [timeout]" if status == "timeout" else ""
                print(f"[{completed}/{total}] {map_name} | {scen_tag} | {n} ag{suffix}", flush=True)

    print(f"\nDone. Results appended to {args.output}")
    print("Tip: aggregate success with analysis_scripts/ or pandas:")
    print("  import pandas as pd; df=pd.read_csv('...'); "
          "print((df['success']==True).mean())")


if __name__ == "__main__":
    main()
