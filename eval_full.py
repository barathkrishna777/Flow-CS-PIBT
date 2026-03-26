"""Evaluate model across maps, agent densities, and Euler steps.

Runs 1 scenario each on 5 maps, for agents={100, 400, 800}, sweeping steps={1, 2, 3, 5}.
Use --extended to also test hard scenarios (high agent counts on den312d, Paris, empty).

Usage:
    python eval_full.py <model_path>
    python eval_full.py <model_path> --extended               # include hard scenarios
    python eval_full.py <model_path> --output logs/my_results.csv
    python eval_full.py <model_path> --steps 1 2 3           # custom steps
    python eval_full.py <model_path> --agents 100 200 400     # custom agent counts
    python eval_full.py <model_path> --maps den312d empty-48-48  # custom maps
"""
import os, sys, subprocess, argparse, csv

MAP_NPZ = "data/all_maps.npz"
BD_DIR = "data/bd_npzs/large_scale"
SCEN_DIR = "data/scen-random"

# 5 diverse maps: open, random obstacles, structured, large city, warehouse
DEFAULT_MAPS = [
    ("empty-48-48", "empty-48-48-random-1.scen"),
    ("random-32-32-10", "random-32-32-10-random-1.scen"),
    ("den312d", "den312d-random-1.scen"),
    ("Paris_1_256", "Paris_1_256-random-1.scen"),
    ("warehouse-10-20-10-2-1", "warehouse-10-20-10-2-1-random-1.scen"),
]

DEFAULT_AGENTS = [100, 400, 800]
DEFAULT_STEPS = [1, 2, 3, 5]
DEFAULT_CONSENSUS = 3
DEFAULT_TAU = 0.3
DEFAULT_WAIT_THRESH = 0.25
DEFAULT_TIME_LIMIT = 120  # 2 minutes, matching Veerapaneni et al.
DEFAULT_MAX_STEPS = "3x"  # 3x makespan, matching Veerapaneni et al.

# Hard scenarios: extra agent counts per map when --extended is used
EXTENDED_AGENTS = {
    "den312d": [700, 800, 900, 1000],
    "Paris_1_256": [900, 1000],
    "empty-48-48": [700, 800, 900, 1000],
}


def main():
    parser = argparse.ArgumentParser(description="Phase 1 ablation: steps vs quality")
    parser.add_argument("model", nargs="?", default=None,
                        help="Path to model checkpoint (or use --model)")
    parser.add_argument("--model", "-m", dest="model_opt", default=None,
                        help="Path to model checkpoint (alternative to positional)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--maps", nargs="*", default=None,
                        help="Map names to evaluate (default: 5 diverse maps)")
    parser.add_argument("--agents", nargs="*", type=int, default=DEFAULT_AGENTS,
                        help="Agent counts to test (default: 100 400 800)")
    parser.add_argument("--extended", action="store_true",
                        help="Include hard scenarios: den312d 700-1000, Paris 900-1000, empty 700-1000")
    parser.add_argument("--steps", nargs="*", type=int, default=DEFAULT_STEPS,
                        help="Euler integration steps to sweep (default: 1 2 3 5)")
    parser.add_argument("--consensus", type=int, default=DEFAULT_CONSENSUS,
                        help="Consensus samples (default: 3)")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument("--wait-thresh", type=float, default=DEFAULT_WAIT_THRESH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT,
                        help="Wall-clock time limit in seconds (default: 120, matching paper)")
    parser.add_argument("--max-steps-multiplier", type=str, default=DEFAULT_MAX_STEPS,
                        help="Max steps for simulator (default: 3x, matching paper)")
    parser.add_argument("--hidden-dim", type=int, default=1024,
                        help="Model hidden dimension (default: 1024)")
    parser.add_argument("--num-layers", type=int, default=6,
                        help="Number of GNN layers (default: 6)")
    args = parser.parse_args()

    args.model = args.model_opt or args.model
    if not args.model:
        parser.error("Provide model path as positional argument or --model / -m")

    if not os.path.exists(args.model):
        print(f"ERROR: {args.model} not found")
        sys.exit(1)

    # Resolve maps
    if args.maps:
        map_configs = []
        for m in args.maps:
            scen = f"{m}-random-1.scen"
            map_configs.append((m, scen))
    else:
        map_configs = DEFAULT_MAPS

    # Output path
    if args.output:
        csv_path = args.output
    else:
        model_tag = os.path.basename(args.model).replace(".pt", "")
        csv_path = f"logs/eval_ablation_{model_tag}.csv"

    os.makedirs("logs", exist_ok=True)

    import torch
    use_gpu = torch.cuda.is_available()

    # Build run list: maps x agents x steps
    runs = []
    for map_name, scen_file in map_configs:
        scen_path = os.path.join(SCEN_DIR, scen_file)
        bn = scen_file.replace(".scen", "")
        bd_path = os.path.join(BD_DIR, f"{bn}_bds.npz")

        if not os.path.exists(scen_path):
            print(f"WARNING: scenario {scen_path} not found, skipping {map_name}")
            continue
        if not os.path.exists(bd_path):
            print(f"WARNING: BD file {bd_path} not found, skipping {map_name}")
            continue

        # Check max agents available
        with open(scen_path) as f:
            max_avail = len(f.readlines()) - 1

        agent_counts = sorted(set(args.agents))
        if args.extended and map_name in EXTENDED_AGENTS:
            agent_counts = sorted(set(agent_counts) | set(EXTENDED_AGENTS[map_name]))

        for n_agents in agent_counts:
            if n_agents > max_avail:
                print(f"WARNING: {map_name} has only {max_avail} agents, skipping {n_agents}")
                continue
            for num_steps in args.steps:
                runs.append((map_name, scen_path, bd_path, n_agents, num_steps))

    # Write CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "map", "agents", "num_steps", "consensus", "tau", "wait_thresh",
            "success", "at_goal", "total", "at_goal_pct",
            "runtime", "total_cost", "cost_not_resting"
        ])

    total = len(runs)
    print(f"{'='*60}")
    print(f"Evaluation: Euler Steps vs Quality")
    print(f"Model: {args.model}")
    print(f"Architecture: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
    print(f"Maps: {len(map_configs)} | Agents: {args.agents} | Steps: {args.steps}")
    print(f"Extended: {args.extended} | Consensus: {args.consensus} | Tau: {args.tau} | GPU: {use_gpu}")
    print(f"Total runs: {total}")
    print(f"Output: {csv_path}")
    print(f"{'='*60}\n")

    for i, (map_name, scen_path, bd_path, n_agents, num_steps) in enumerate(runs, 1):
        print(f"[{i}/{total}] {map_name} | {n_agents} agents | {num_steps} steps", end="", flush=True)

        tmp_csv = "logs/_eval_tmp.csv"
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)

        cmd = [
            "python", "-m", "main_pys.simulator",
            f"--mapNpzFile={MAP_NPZ}", f"--mapName={map_name}",
            f"--scenFile={scen_path}", f"--bdNpzFile={bd_path}",
            f"--modelPath={args.model}", f"--outputCSVFile={tmp_csv}",
            f"--maxSteps={args.max_steps_multiplier}", f"--seed={args.seed}",
            f"--useGPU={'True' if use_gpu else 'False'}",
            f"--agentNum={n_agents}", "--shieldType=CS-PIBT",
            f"--numIntegrationSteps={num_steps}",
            f"--tau={args.tau}",
            f"--waitThreshold={args.wait_thresh}",
            f"--numConsensusSamples={args.consensus}",
            f"--timeLimit={args.time_limit}",
            f"--hiddenDim={args.hidden_dim}",
            f"--numLayers={args.num_layers}",
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=args.time_limit + 60)
        except subprocess.TimeoutExpired:
            print(" TIMEOUT")
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    map_name, n_agents, num_steps, args.consensus, args.tau, args.wait_thresh,
                    False, 0, n_agents, "0.0", args.time_limit, 0, 0
                ])
            continue

        if os.path.exists(tmp_csv):
            with open(tmp_csv) as rf:
                reader = csv.DictReader(rf)
                for row in reader:
                    at_goal = int(row.get("num_agents_at_goal", 0))
                    success = row.get("success", "False")
                    runtime = float(row.get("runtime", 0))
                    total_cost = row.get("total_cost_true", 0)
                    cost_nr = row.get("total_cost_not_resting_at_goal", 0)
                    pct = 100.0 * at_goal / n_agents

                    with open(csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            map_name, n_agents, num_steps, args.consensus,
                            args.tau, args.wait_thresh,
                            success, at_goal, n_agents, f"{pct:.1f}",
                            f"{runtime:.2f}", total_cost, cost_nr
                        ])

                    print(f" -> {at_goal}/{n_agents} ({pct:.1f}%) {'OK' if success == 'True' else 'FAIL'} [{runtime:.1f}s]")
        else:
            print(" NO OUTPUT")
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    map_name, n_agents, num_steps, args.consensus, args.tau, args.wait_thresh,
                    False, 0, n_agents, "0.0", 0, 0, 0
                ])

    # Print summary table
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        df["at_goal_pct"] = pd.to_numeric(df["at_goal_pct"], errors="coerce")
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")

        # Pivot: rows = (map, agents), cols = steps
        pivot = df.pivot_table(
            index=["map", "agents"],
            columns="num_steps",
            values=["at_goal_pct", "runtime"],
            aggfunc="first"
        )
        print("\nAgents at goal (%):")
        print(pivot["at_goal_pct"].to_string())
        print("\nRuntime (s):")
        print(pivot["runtime"].to_string())
    except Exception as e:
        print(f"  (install pandas for summary: {e})")

    print(f"\nDone! Full results: {csv_path}")


if __name__ == "__main__":
    main()
