"""
Evaluate the overfit big model on the SAME scenarios it was trained on.
Tests empty-48-48-random-1 with 20 and 50 agents.
The wait fix in simulator.py is already applied.
"""
import subprocess
import os
import sys

MODEL_PATH = "checkpoints/overfit_big_best.pt"
OUTPUT_CSV = "logs/overfit_big_eval.csv"
MAP_NPZ = "data/all_maps.npz"

MAP_NAME = "empty-48-48"
SCEN_FILE = "data/scen-random/empty-48-48-random-1.scen"
BD_FILE = "data/bd_npzs/large_scale/empty-48-48-random-1_bds.npz"

# Test the agent counts we trained on
AGENT_COUNTS = [20, 50, 100]

SHIELD_TYPE = "CS-PIBT"
MAX_STEPS = "5x"
TIME_LIMIT = 120


def main():
    os.makedirs("logs", exist_ok=True)

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    print(f"Evaluating overfit BIG model on training scenarios...")
    print(f"  Model:  {MODEL_PATH}")
    print(f"  Map:    {MAP_NAME}")
    print(f"  Shield: {SHIELD_TYPE}")
    print()

    for agent_num in AGENT_COUNTS:
        print(f"\n{'='*60}")
        print(f"  Agents: {agent_num}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "-m", "main_pys.simulator",
            f"--mapNpzFile={MAP_NPZ}",
            f"--mapName={MAP_NAME}",
            f"--scenFile={SCEN_FILE}",
            f"--bdNpzFile={BD_FILE}",
            f"--modelPath={MODEL_PATH}",
            f"--outputCSVFile={OUTPUT_CSV}",
            f"--maxSteps={MAX_STEPS}",
            "--seed=0",
            "--useGPU=False",
            f"--agentNum={agent_num}",
            f"--shieldType={SHIELD_TYPE}",
            f"--timeLimit={TIME_LIMIT}",
        ]

        result = subprocess.run(cmd, text=True)

        if result.returncode != 0:
            print(f"  -> FAILED (exit code {result.returncode})")

    print(f"\n{'='*60}")
    print(f"Results written to {OUTPUT_CSV}")
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            for line in f:
                print(line.strip())


if __name__ == "__main__":
    main()
