"""Sweep inference hyperparameters on multiple maps and agent densities.

Sweeps: num_steps x tau x wait_threshold x consensus_samples (flow path)
        + action head baseline (single forward pass, no flow integration)

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m scripts.sweep_inference
    CUDA_VISIBLE_DEVICES=0 python -m scripts.sweep_inference --model path/to/model.pt
    CUDA_VISIBLE_DEVICES=0 python -m scripts.sweep_inference --quick   # small subset for testing
"""
import subprocess
import os
import csv
import argparse
import itertools

DEFAULT_MODEL = "large_scale_flow_wave6_best.pt"
MAP_NPZ = "data/all_maps.npz"
BD_DIR = "data/bd_npzs"
SCEN_DIR = "data/scen-random"

TEST_CASES = [
    ("empty-48-48", "empty-48-48-random-1.scen", 100),
    ("empty-48-48", "empty-48-48-random-1.scen", 400),
    ("den312d", "den312d-random-1.scen", 100),
    ("den312d", "den312d-random-1.scen", 400),
    ("den312d", "den312d-random-1.scen", 600),
    ("random-32-32-10", "random-32-32-10-random-1.scen", 100),
    ("Paris_1_256", "Paris_1_256-random-1.scen", 400),
    ("Paris_1_256", "Paris_1_256-random-1.scen", 800),
]

QUICK_TEST_CASES = [
    ("empty-48-48", "empty-48-48-random-1.scen", 100),
    ("den312d", "den312d-random-1.scen", 100),
]

NUM_STEPS_OPTIONS = [1, 2, 3, 5]
TAU_OPTIONS = [0.3, 0.5]
WAIT_THRESH_OPTIONS = [0.25]
CONSENSUS_OPTIONS = [1, 3]


def run_single(model, map_name, scen_file, n_agents, csv_path, use_gpu,
               num_steps=5, tau=0.3, wait_thresh=0.25, consensus=3, use_action_head=False,
               hidden_dim=1024, num_layers=6):
    scen_path = f"{SCEN_DIR}/{scen_file}"
    bn = scen_file.replace(".scen", "")
    bd_path = f"{BD_DIR}/large_scale/{bn}_bds.npz"

    tmp_csv = "logs/_sweep_tmp.csv"
    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

    cmd = [
        "python", "-m", "main_pys.simulator",
        f"--mapNpzFile={MAP_NPZ}", f"--mapName={map_name}",
        f"--scenFile={scen_path}", f"--bdNpzFile={bd_path}",
        f"--modelPath={model}", f"--outputCSVFile={tmp_csv}",
        "--maxSteps=3x", "--seed=0",
        f"--useGPU={'True' if use_gpu else 'False'}",
        f"--agentNum={n_agents}", "--shieldType=CS-PIBT",
        f"--numIntegrationSteps={num_steps}",
        f"--tau={tau}",
        f"--waitThreshold={wait_thresh}",
        f"--numConsensusSamples={consensus}",
        f"--useActionHead={'True' if use_action_head else 'False'}",
        f"--hiddenDim={hidden_dim}",
        f"--numLayers={num_layers}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if os.path.exists(tmp_csv):
        with open(tmp_csv) as rf:
            reader = csv.DictReader(rf)
            for row in reader:
                at_goal = int(row.get("num_agents_at_goal", 0))
                success = row.get("success", "False")
                runtime = float(row.get("runtime", 0))
                pct = 100.0 * at_goal / n_agents

                with open(csv_path, "a", newline="") as wf:
                    writer = csv.writer(wf)
                    writer.writerow([
                        num_steps, tau, wait_thresh, consensus, use_action_head,
                        map_name, n_agents, at_goal, n_agents, f"{pct:.1f}",
                        success, f"{runtime:.2f}"
                    ])

                return at_goal, n_agents, pct, success, runtime

    return None, n_agents, 0, "FAIL", 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint path")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer configs")
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Model hidden dimension")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of GNN layers")
    args = parser.parse_args()

    import torch
    use_gpu = torch.cuda.is_available()

    test_cases = QUICK_TEST_CASES if args.quick else TEST_CASES
    output = args.output or "logs/inference_sweep_extended.csv"
    os.makedirs("logs", exist_ok=True)

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "num_steps", "tau", "wait_thresh", "consensus", "action_head",
            "map", "agents", "at_goal", "total", "at_goal_pct",
            "success", "runtime"
        ])

    # Flow-based configs
    flow_configs = list(itertools.product(
        NUM_STEPS_OPTIONS, TAU_OPTIONS, WAIT_THRESH_OPTIONS, CONSENSUS_OPTIONS
    ))

    total_runs = len(flow_configs) * len(test_cases) + len(test_cases)  # +action head runs
    print(f"{'='*60}")
    print(f"Inference Sweep: {args.model}")
    print(f"Flow configs: {len(flow_configs)} | Action head: 1")
    print(f"Test cases: {len(test_cases)} | Total runs: {total_runs}")
    print(f"GPU: {use_gpu} | Output: {output}")
    print(f"{'='*60}\n")

    run_idx = 0

    # Flow-based sweep
    for i, (num_steps, tau, wait_thresh, consensus) in enumerate(flow_configs):
        print(f"\n[Config {i+1}/{len(flow_configs)}] steps={num_steps}, tau={tau}, "
              f"wait={wait_thresh}, consensus={consensus}")

        for map_name, scen_file, n_agents in test_cases:
            run_idx += 1
            at_goal, total, pct, success, runtime = run_single(
                args.model, map_name, scen_file, n_agents, output, use_gpu,
                num_steps=num_steps, tau=tau, wait_thresh=wait_thresh,
                consensus=consensus, use_action_head=False,
                hidden_dim=args.hidden_dim, num_layers=args.num_layers,
            )
            status = f"{at_goal}/{total} ({pct:.1f}%)" if at_goal is not None else "FAIL"
            print(f"  [{run_idx}/{total_runs}] {map_name} {n_agents}ag: {status}  [{runtime:.1f}s]")

    # Action head baseline (tau still matters for softmax temperature)
    print(f"\n[Action Head] tau=0.3")
    for map_name, scen_file, n_agents in test_cases:
        run_idx += 1
        at_goal, total, pct, success, runtime = run_single(
            args.model, map_name, scen_file, n_agents, output, use_gpu,
            num_steps=1, tau=0.3, wait_thresh=0.25,
            consensus=1, use_action_head=True,
            hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        )
        status = f"{at_goal}/{total} ({pct:.1f}%)" if at_goal is not None else "FAIL"
        print(f"  [{run_idx}/{total_runs}] {map_name} {n_agents}ag: {status}  [{runtime:.1f}s]")

    print(f"\nDone! Results in {output}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    try:
        import pandas as pd
        df = pd.read_csv(output)
        df["at_goal_pct"] = pd.to_numeric(df["at_goal_pct"], errors="coerce")

        # Best flow config per map/agent combo
        flow_df = df[df["action_head"] == "False"]
        if not flow_df.empty:
            print("\nBest flow config per test case:")
            for (m, a), sub in flow_df.groupby(["map", "agents"]):
                best = sub.loc[sub["at_goal_pct"].idxmax()]
                print(f"  {m} {a}ag: {best['at_goal_pct']}% @ "
                      f"steps={best['num_steps']}, tau={best['tau']}, "
                      f"consensus={best['consensus']}, runtime={best['runtime']}s")

        # Action head results
        ah_df = df[df["action_head"] == "True"]
        if not ah_df.empty:
            print("\nAction head results:")
            for _, row in ah_df.iterrows():
                print(f"  {row['map']} {row['agents']}ag: {row['at_goal_pct']}%, runtime={row['runtime']}s")
    except Exception as e:
        print(f"  (install pandas for summary: {e})")


if __name__ == "__main__":
    main()
