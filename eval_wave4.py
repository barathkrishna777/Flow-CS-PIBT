"""Evaluate wave 4 checkpoints.

Usage:
    python eval_wave4.py 1          # eval epoch 1
    python eval_wave4.py 3          # eval epoch 3
    python eval_wave4.py best       # eval best checkpoint
"""
import os, sys, subprocess, glob

if len(sys.argv) < 2:
    print("Usage: python eval_wave4.py <epoch|best>")
    sys.exit(1)

tag = sys.argv[1]
if tag == "best":
    MODEL = "large_scale_flow_wave4_best.pt"
else:
    MODEL = f"large_scale_flow_wave4_epoch_{tag}.pt"

if not os.path.exists(MODEL):
    print(f"ERROR: {MODEL} not found")
    sys.exit(1)

MAP_NPZ = "data/all_maps.npz"
BD_DIR = "data/bd_npzs"
SCEN_DIR = "data/scen-random"
CSV = f"logs/batch_results_wave4_ep{tag}.csv"
os.makedirs("logs", exist_ok=True)
if os.path.exists(CSV):
    os.remove(CSV)

import torch
USE_GPU = torch.cuda.is_available()

TEST_MAPS = ["empty-48-48", "random-32-32-10", "den312d"]
AGENT_COUNTS = [50, 100, 200]
SCENARIOS_PER_MAP = 3

total = len(TEST_MAPS) * SCENARIOS_PER_MAP * len(AGENT_COUNTS)
completed = 0

print(f"{'='*60}")
print(f"Evaluating Wave 4 ({tag}): {MODEL}")
print(f"GPU: {USE_GPU} | {total} total runs")
print(f"Output: {CSV}")
print(f"{'='*60}\n")

for map_name in TEST_MAPS:
    scens = sorted(glob.glob(f"{SCEN_DIR}/{map_name}-random-*.scen"))[:SCENARIOS_PER_MAP]
    if not scens:
        print(f"Skipping {map_name} - no scenario files")
        continue
    for scen in scens:
        bn = os.path.basename(scen).replace(".scen", "")
        bd = f"{BD_DIR}/large_scale/{bn}_bds.npz"
        if not os.path.exists(bd):
            print(f"  Skipping {bn}: BD file not found")
            continue
        for n in AGENT_COUNTS:
            completed += 1
            print(f"[{completed}/{total}] {map_name} | {n} agents")
            subprocess.run([
                "python", "-m", "main_pys.simulator",
                f"--mapNpzFile={MAP_NPZ}", f"--mapName={map_name}",
                f"--scenFile={scen}", f"--bdNpzFile={bd}",
                f"--modelPath={MODEL}", f"--outputCSVFile={CSV}",
                "--maxSteps=3x", "--seed=0",
                f"--useGPU={'True' if USE_GPU else 'False'}",
                f"--agentNum={n}", "--shieldType=CS-PIBT"
            ])

print(f"\nDone! Results: {CSV}")
