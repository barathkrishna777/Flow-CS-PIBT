"""
Held-out benchmark aligned with Veerapaneni et al. (Work Smarter, Not Harder).

Evaluates all 8 paper held-out maps, with the same scenario sweep and density
ladder used in prior project batch evals (e.g. batch_results_wave5_best.csv):
  - Maps: HELD_OUT_TEST set (8 maps)
  - Scenarios: random-1 .. random-N per map (default N=25, matching scen-random caps)
  - Agents: 100, 200, ..., 1000

Inference defaults match eval_full / paper: time limit 120s, maxSteps 3x,
tau=0.3, consensus=3, numIntegrationSteps=3 (override with flags).

Usage:
  python eval_rishi_paper.py -m large_scale_flow_wave8_base_best.pt \\
      --output checkpoints_and_evaluations/eval_rishi_wave8_base.csv

  python eval_rishi_paper.py -m model.pt --quick --output logs/rishi_quick.csv

  # Paper-scale run is large (8 maps x 25 scens x 10 agent levels = 2000 sims).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from collections import deque

import numpy as np

MAP_NPZ_CANDIDATES = [
    "data/all_maps.npz",
    "data/constant_npzs/all_maps.npz",
]
BD_DIR_CANDIDATES = [
    "data/bd_npzs/large_scale",
    "data/constant_npzs/bd_npzs",
    "data/bd_npzs",
]
SCEN_DIR_CANDIDATES = [
    "data/scen-random",
    "data/mapf-scen-random",
]

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

DEFAULT_AGENT_COUNTS = list(range(100, 1001, 100))


def _max_agents_in_scen(scen_path: str) -> int:
    with open(scen_path) as f:
        return max(0, len(f.readlines()) - 1)


def _first_existing_path(candidates: list[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def _bd_candidates(map_name: str, scenario_base_name: str) -> list[str]:
    paths = []
    for bd_dir in BD_DIR_CANDIDATES:
        paths.append(os.path.join(bd_dir, f"{scenario_base_name}_bds.npz"))
        paths.append(os.path.join(bd_dir, f"{map_name}_bds.npz"))
    return paths


def _parse_scenario_goals(scen_path: str, max_agents: int = 1000) -> np.ndarray:
    goals = []
    with open(scen_path) as f:
        f.readline()
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) >= 8:
                goals.append((int(parts[7]), int(parts[6])))
            if len(goals) >= max_agents:
                break
    return np.asarray(goals, dtype=np.int64)


def _compute_bd_for_goals(map_grid: np.ndarray, goals: np.ndarray) -> np.ndarray:
    height, width = map_grid.shape
    bd = np.full((len(goals), height, width), 10000, dtype=np.int16)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i, (goal_row, goal_col) in enumerate(goals):
        if map_grid[goal_row, goal_col] == 1:
            continue
        queue = deque([(goal_row, goal_col)])
        bd[i, goal_row, goal_col] = 0
        while queue:
            row, col = queue.popleft()
            next_dist = int(bd[i, row, col]) + 1
            for drow, dcol in moves:
                nrow, ncol = row + drow, col + dcol
                if 0 <= nrow < height and 0 <= ncol < width:
                    if map_grid[nrow, ncol] == 0 and bd[i, nrow, ncol] == 10000:
                        bd[i, nrow, ncol] = next_dist
                        queue.append((nrow, ncol))
    return bd


def _generate_bd(map_npz_path: str, map_name: str, scen_path: str) -> str:
    out_dir = "data/bd_npzs/large_scale"
    os.makedirs(out_dir, exist_ok=True)
    scenario_base = os.path.basename(scen_path).replace(".scen", "")
    out_path = os.path.join(out_dir, f"{scenario_base}_bds.npz")
    if os.path.exists(out_path):
        return out_path

    map_key = f"{map_name}.map"
    map_npz = np.load(map_npz_path)
    if map_key not in map_npz:
        raise KeyError(f"Map key {map_key} not found in {map_npz_path}")
    goals = _parse_scenario_goals(scen_path)
    print(f"Generating missing BD: {out_path} ({len(goals)} goals)")
    bd = _compute_bd_for_goals(map_npz[map_key], goals)
    scen_num = scen_path.split("-")[-1].split(".")[0]
    bd_key = f"{map_name}-random-{scen_num}"
    tmp_path = f"{out_path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp_path, **{bd_key: bd})
    os.replace(tmp_path, out_path)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Rishi paper held-out benchmark (8 maps)")
    p.add_argument("-m", "--model", required=True, help="Checkpoint .pt path")
    p.add_argument("--output", "-o", required=True, help="Output CSV (simulator format)")
    p.add_argument("--maps", nargs="*", default=None,
                   help="Subset of map names (default: all 8 held-out)")
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
    p.add_argument("--policy-type", choices=["flow", "classifier"], default="flow",
                   help="Policy/model family to pass to simulator (default: flow)")
    p.add_argument("--generate-missing-bd", action="store_true",
                   help="Generate missing BD heuristic npzs from map/scenario files")
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=6)
    args = p.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    map_npz = _first_existing_path(MAP_NPZ_CANDIDATES)
    scen_dir = _first_existing_path(SCEN_DIR_CANDIDATES)
    print(f"Map NPZ: {map_npz}")
    print(f"Scenario dir: {scen_dir}")

    maps = args.maps if args.maps else RISHI_HELD_OUT_MAPS
    for m in maps:
        if m not in RISHI_HELD_OUT_MAPS:
            print(f"WARNING: {m} is not in the standard 8 held-out maps; continuing anyway.")

    agent_counts = [100, 400, 800] if args.quick else list(args.agents)
    max_scen = 1 if args.quick else args.max_scenario

    runs = []
    for map_name in maps:
        pattern = os.path.join(scen_dir, f"{map_name}-random-*.scen")
        scens = sorted(glob.glob(pattern))[:max_scen]
        if not scens:
            print(f"WARNING: no scenarios for {map_name}, skip")
            continue
        for scen_path in scens:
            bn = os.path.basename(scen_path).replace(".scen", "")
            bd_path = _first_existing_path(_bd_candidates(map_name, bn))
            if not os.path.isfile(bd_path):
                if args.generate_missing_bd:
                    bd_path = _generate_bd(map_npz, map_name, scen_path)
                else:
                    print(f"WARNING: missing BD for {map_name} {bn}; tried:")
                    for candidate in _bd_candidates(map_name, bn):
                        print(f"  - {candidate}")
                    print("  Tip: rerun with --generate-missing-bd or download the BD bundle.")
                    continue
            max_avail = _max_agents_in_scen(scen_path)
            for n in agent_counts:
                if n > max_avail:
                    continue
                runs.append((map_name, scen_path, bd_path, n))

    if not runs:
        print("ERROR: no runs to execute.", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.isfile(args.output):
        os.remove(args.output)

    total = len(runs)
    print("=" * 60)
    print("Rishi held-out benchmark (8-map protocol)")
    print(f"Model: {args.model}")
    print(f"Maps: {len(maps)} | Scenarios/map: <= {max_scen} | Agents: {agent_counts[0]}..{agent_counts[-1]}")
    print(f"steps={args.num_integration_steps} consensus={args.consensus} tau={args.tau}")
    print(f"timeLimit={args.time_limit}s maxSteps={args.max_steps_multiplier} | GPU: {use_gpu}")
    print(f"Total runs: {total}")
    print(f"Output: {args.output}")
    print("=" * 60)

    for i, (map_name, scen_path, bd_path, n) in enumerate(runs, 1):
        scen_tag = os.path.basename(scen_path)
        print(f"[{i}/{total}] {map_name} | {scen_tag} | {n} ag", flush=True)
        cmd = [
            sys.executable, "-m", "main_pys.simulator",
            f"--mapNpzFile={map_npz}",
            f"--mapName={map_name}",
            f"--scenFile={scen_path}",
            f"--bdNpzFile={bd_path}",
            f"--modelPath={args.model}",
            f"--outputCSVFile={args.output}",
            f"--maxSteps={args.max_steps_multiplier}",
            f"--seed={args.seed}",
            f"--useGPU={'True' if use_gpu else 'False'}",
            f"--agentNum={n}",
            "--shieldType=CS-PIBT",
            f"--timeLimit={args.time_limit}",
            f"--numIntegrationSteps={args.num_integration_steps}",
            f"--tau={args.tau}",
            f"--waitThreshold={args.wait_thresh}",
            f"--numConsensusSamples={args.consensus}",
            f"--policyType={args.policy_type}",
            f"--hiddenDim={args.hidden_dim}",
            f"--numLayers={args.num_layers}",
        ]
        try:
            subprocess.run(cmd, check=False, timeout=args.time_limit + 120)
        except subprocess.TimeoutExpired:
            print(f"  -> subprocess timeout")

    print(f"\nDone. Results appended to {args.output}")
    print("Tip: aggregate success with analysis_scripts/ or pandas:")
    print("  import pandas as pd; df=pd.read_csv('...'); "
          "print((df['success']==True).mean())")


if __name__ == "__main__":
    main()
