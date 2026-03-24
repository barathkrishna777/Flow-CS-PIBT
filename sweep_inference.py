"""Sweep inference hyperparameters (num_steps, tau, wait_threshold) on a single scenario.
No retraining needed — just tests different inference configs on the same checkpoint.

Usage:
    CUDA_VISIBLE_DEVICES=1 python sweep_inference.py
"""
import subprocess
import os
import csv
import itertools

MODEL = "large_scale_flow_wave4_epoch_4.pt"
MAP_NPZ = "data/all_maps.npz"
BD_DIR = "data/bd_npzs"
SCEN_DIR = "data/scen-random"

# Test on one scenario per map at 100 agents (the sweet spot where we're at ~90%)
TEST_CASES = [
    ("empty-48-48", "empty-48-48-random-1.scen", 100),
    ("random-32-32-10", "random-32-32-10-random-1.scen", 100),
    ("den312d", "den312d-random-1.scen", 100),
]

# Parameters to sweep
NUM_STEPS_OPTIONS = [5, 10, 20]
TAU_OPTIONS = [0.3, 0.5, 0.7, 1.0]
WAIT_THRESH_OPTIONS = [0.15, 0.25, 0.35]

os.makedirs("logs", exist_ok=True)
RESULTS_FILE = "logs/inference_sweep.csv"

with open(RESULTS_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["num_steps", "tau", "wait_thresh", "map", "agents", "at_goal", "total", "at_goal_pct"])

configs = list(itertools.product(NUM_STEPS_OPTIONS, TAU_OPTIONS, WAIT_THRESH_OPTIONS))
print(f"Sweeping {len(configs)} configs × {len(TEST_CASES)} test cases = {len(configs) * len(TEST_CASES)} runs")

for i, (num_steps, tau, wait_thresh) in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}] steps={num_steps}, tau={tau}, wait={wait_thresh}")

    for map_name, scen_file, n_agents in TEST_CASES:
        scen_path = f"{SCEN_DIR}/{scen_file}"
        bn = scen_file.replace(".scen", "")
        bd_path = f"{BD_DIR}/large_scale/{bn}_bds.npz"

        # Use a temp CSV for this single run
        tmp_csv = f"logs/_sweep_tmp.csv"
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)

        cmd = [
            "python", "-m", "main_pys.simulator",
            f"--mapNpzFile={MAP_NPZ}", f"--mapName={map_name}",
            f"--scenFile={scen_path}", f"--bdNpzFile={bd_path}",
            f"--modelPath={MODEL}", f"--outputCSVFile={tmp_csv}",
            "--maxSteps=3x", "--seed=0", "--useGPU=True",
            f"--agentNum={n_agents}", "--shieldType=CS-PIBT",
            f"--numIntegrationSteps={num_steps}",
            f"--tau={tau}",
            f"--waitThreshold={wait_thresh}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        # Parse result from tmp CSV
        if os.path.exists(tmp_csv):
            with open(tmp_csv) as rf:
                reader = csv.DictReader(rf)
                for row in reader:
                    at_goal = int(row.get("num_agents_at_goal", 0))
                    total = int(row.get("num_agents", n_agents))
                    pct = 100.0 * at_goal / total

                    with open(RESULTS_FILE, "a", newline="") as wf:
                        writer = csv.writer(wf)
                        writer.writerow([num_steps, tau, wait_thresh, map_name, n_agents, at_goal, total, f"{pct:.1f}"])

                    print(f"  {map_name} {n_agents}ag: {at_goal}/{total} ({pct:.1f}%)")
        else:
            print(f"  {map_name} {n_agents}ag: FAILED")
            with open(RESULTS_FILE, "a", newline="") as wf:
                writer = csv.writer(wf)
                writer.writerow([num_steps, tau, wait_thresh, map_name, n_agents, "FAIL", n_agents, "FAIL"])

print(f"\nDone! Results in {RESULTS_FILE}")

# Print summary: best config per map
print("\n" + "="*60)
print("  BEST CONFIGS PER MAP")
print("="*60)
import pandas as pd
try:
    df = pd.read_csv(RESULTS_FILE)
    for m in df["map"].unique():
        sub = df[df["map"] == m]
        best = sub.loc[sub["at_goal_pct"].astype(float).idxmax()]
        print(f"  {m}: {best['at_goal_pct']}% @ steps={best['num_steps']}, tau={best['tau']}, wait={best['wait_thresh']}")
except Exception as e:
    print(f"  (install pandas for summary, or check {RESULTS_FILE})")
