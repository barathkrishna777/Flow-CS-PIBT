"""
Evaluate the overfit model on the SAME scenario it was trained on.
If overfit worked, this should achieve ~100% success on 20 agents, empty-32-32.
"""
import subprocess
import os
import sys

MODEL_PATH = "overfit_best.pt"
OUTPUT_CSV = "logs/overfit_eval.csv"
MAP_NPZ = "data/all_maps.npz"

# The exact scenario we trained on
MAP_NAME = "empty-32-32"
SCEN_FILE = "data/scen-random/empty-32-32-random-10.scen"
BD_FILE = "data/bd_npzs/large_scale/empty-32-32-random-10_bds.npz"
AGENT_NUM = 20  # matches training file: empty-32-32-random-10_20.npz

SHIELD_TYPE = "CS-PIBT"
MAX_STEPS = "5x"  # generous step budget
TIME_LIMIT = 120  # seconds


def main():
    os.makedirs("logs", exist_ok=True)

    # Remove old results
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    print(f"Evaluating overfit model on training scenario...")
    print(f"  Model:  {MODEL_PATH}")
    print(f"  Map:    {MAP_NAME}")
    print(f"  Agents: {AGENT_NUM}")
    print(f"  Shield: {SHIELD_TYPE}")
    print()

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
        f"--agentNum={AGENT_NUM}",
        f"--shieldType={SHIELD_TYPE}",
        f"--timeLimit={TIME_LIMIT}",
    ]

    print("Running:", " ".join(cmd))
    print("=" * 60)

    result = subprocess.run(cmd, text=True)

    print("=" * 60)
    if result.returncode != 0:
        print(f"Evaluation failed with return code {result.returncode}")
    else:
        print(f"\nResults written to {OUTPUT_CSV}")
        # Print the CSV
        if os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV) as f:
                for line in f:
                    print(line.strip())


if __name__ == "__main__":
    main()
