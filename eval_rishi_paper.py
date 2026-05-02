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

  # Non-learned BD-guided PIBT baseline; no checkpoint required.
  python eval_rishi_paper.py --policy-type pibt --map-set rishi8 --output logs/rishi8_pibt.csv
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

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
    p.add_argument("-m", "--model", default=None,
                   help="Checkpoint .pt path. Not required for --policy-type pibt.")
    p.add_argument("--output", "-o", required=True, help="Output CSV (simulator format)")
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
    p.add_argument("--wait-mode", choices=["threshold", "learned", "learned_gate"], default="threshold",
                   help="Flow wait action mode: threshold uses --wait-thresh; learned uses wait as a fifth "
                        "competing logit; learned_gate uses binary P(wait) gating over move directions")
    p.add_argument("--wait-logit-bias", type=float, default=None,
                   help="Override learned wait-logit bias; omitted uses checkpoint calibration")
    p.add_argument("--wait-logit-scale", type=float, default=None,
                   help="Override learned wait-logit scale; omitted uses checkpoint calibration")
    p.add_argument("--policy-type", choices=["flow", "classifier", "flow_action_head", "local_classifier", "pibt"], default="flow",
                   help="Policy/model family to evaluate. pibt is the non-learned BD-guided PIBT baseline and does not load a checkpoint.")
    p.add_argument("--hidden-dim", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--action-head-conditioning", choices=["integrated", "zero_t0", "zero_t05"],
                   default="integrated",
                   help="Action head conditioning for flow_action_head policy (default: integrated)")
    args = p.parse_args()

    if args.policy_type == "pibt":
        model_path = "BD-PIBT"
    else:
        if args.model is None:
            print("ERROR: --model is required unless --policy-type pibt", file=sys.stderr)
            sys.exit(1)
        model_path = args.model

    if args.policy_type != "pibt" and not os.path.isfile(model_path):
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        use_gpu = torch.cuda.is_available() and args.policy_type != "pibt"
    except ImportError:
        use_gpu = False

    maps = args.maps if args.maps else MAP_PRESETS[args.map_set]
    map_source = "custom --maps" if args.maps else args.map_set
    for m in maps:
        if m not in RISHI_HELD_OUT_MAPS:
            print(f"WARNING: {m} is not in the standard 8 held-out maps; continuing anyway.")

    agent_counts = [100, 400, 800] if args.quick else list(args.agents)
    if args.scenario_start < 1:
        print("ERROR: --scenario-start must be >= 1", file=sys.stderr)
        sys.exit(1)
    max_scen = 1 if args.quick else args.max_scenario
    scenario_start = 1 if args.quick else args.scenario_start

    runs = []
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
    if os.path.isfile(args.output):
        os.remove(args.output)

    total = len(runs)
    print("=" * 60)
    print("Rishi-style grid benchmark")
    print(f"Model: {model_path}")
    print(f"Map set: {map_source} | Requested maps: {len(maps)}")
    print(f"Scenarios/map: <= {max_scen} | Agents: {agent_counts[0]}..{agent_counts[-1]}")
    print(f"Policy: {args.policy_type}")
    print(f"steps={args.num_integration_steps} consensus={args.consensus} tau={args.tau}")
    wait_scale_label = "model" if args.wait_logit_scale is None else args.wait_logit_scale
    wait_bias_label = "model" if args.wait_logit_bias is None else args.wait_logit_bias
    print(f"waitMode={args.wait_mode} waitThresh={args.wait_thresh} "
          f"waitLogitScale={wait_scale_label} waitLogitBias={wait_bias_label}")
    print(f"timeLimit={args.time_limit}s maxSteps={args.max_steps_multiplier} | GPU: {use_gpu}")
    _print_preflight(preflight)
    print(f"Total runs: {total}")
    print(f"Output: {args.output}")
    print("=" * 60)

    simulator_policy_type = "flow" if args.policy_type == "flow_action_head" else args.policy_type
    use_action_head = args.policy_type == "flow_action_head"

    for i, (map_name, scen_path, bd_path, n) in enumerate(runs, 1):
        scen_tag = os.path.basename(scen_path)
        print(f"[{i}/{total}] {map_name} | {scen_tag} | {n} ag", flush=True)
        cmd = [
            sys.executable, "-m", "main_pys.simulator",
            f"--mapNpzFile={MAP_NPZ}",
            f"--mapName={map_name}",
            f"--scenFile={scen_path}",
            f"--bdNpzFile={bd_path}",
            f"--modelPath={model_path}",
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
            f"--waitMode={args.wait_mode}",
            f"--numConsensusSamples={args.consensus}",
            f"--policyType={simulator_policy_type}",
            f"--useActionHead={'True' if use_action_head else 'False'}",
            f"--actionHeadConditioning={args.action_head_conditioning}",
            f"--hiddenDim={args.hidden_dim}",
            f"--numLayers={args.num_layers}",
        ]
        if args.wait_logit_bias is not None:
            cmd.append(f"--waitLogitBias={args.wait_logit_bias}")
        if args.wait_logit_scale is not None:
            cmd.append(f"--waitLogitScale={args.wait_logit_scale}")
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
