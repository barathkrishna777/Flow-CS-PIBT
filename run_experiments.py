import os
import subprocess
import glob

# ==========================================
# Configuration
# ==========================================
MODEL_PATH = "flow_model_epoch_50.pt" # Update to whichever epoch you are testing
OUTPUT_CSV = "logs/batch_results.csv"  # All stats will be aggregated here
MAP_NPZ = "data/all_maps.npz"
BD_DIR = "data/bd_npzs"
SCEN_DIR = "data/scen-random"

AGENT_NUM = 10
MAX_STEPS = "3x"
SHIELD_TYPE = "CS-PIBT"
SCENARIOS_PER_MAP = 5 # Number of random scenarios to evaluate per map

# Define the maps you want to test. 
# You can add more maps like "maze-32-32-4", "room-32-32-4", "den312d", etc.
TEST_MAPS = [
    "random-32-32-10",
    "random-32-32-20",
    "empty-32-32"
]

def run_batch():
    os.makedirs("logs", exist_ok=True)
    total_runs = len(TEST_MAPS) * SCENARIOS_PER_MAP
    current_run = 0

    print(f"Starting batch evaluation for {total_runs} total scenarios...")
    print(f"Model: {MODEL_PATH} | Shield: {SHIELD_TYPE} | Agents: {AGENT_NUM}\n")

    for map_name in TEST_MAPS:
        # Find the scenarios for this map
        scen_pattern = os.path.join(SCEN_DIR, f"{map_name}-random-*.scen")
        scen_files = glob.glob(scen_pattern)[:SCENARIOS_PER_MAP]
        
        bd_file = os.path.join(BD_DIR, f"{map_name}_bds.npz")
        
        if not os.path.exists(bd_file):
            print(f"Skipping {map_name}: BD file not found at {bd_file}")
            continue

        for scen_file in scen_files:
            current_run += 1
            scen_basename = os.path.basename(scen_file)
            print(f"[{current_run}/{total_runs}] Map: {map_name} | Scenario: {scen_basename}")
            
            # Construct the exact command line arguments
            cmd = [
                "python", "-m", "main_pys.simulator",
                f"--mapNpzFile={MAP_NPZ}",
                f"--mapName={map_name}",
                f"--scenFile={scen_file}",
                f"--bdNpzFile={bd_file}",
                f"--modelPath={MODEL_PATH}",
                f"--outputCSVFile={OUTPUT_CSV}",
                # Note: We omit --outputPathsFile here to avoid generating 
                # dozens of massive .npy files during a batch run!
                f"--maxSteps={MAX_STEPS}",
                "--seed=0",
                "--useGPU=True",
                f"--agentNum={AGENT_NUM}",
                f"--shieldType={SHIELD_TYPE}"
            ]
            
            try:
                # Run the simulator and capture its terminal output
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"  -> ERROR running {scen_basename}:")
                    print(f"  -> {result.stderr.strip().splitlines()[-1]}") # Print the last error line
                else:
                    # Parse the stdout to just grab the summary line to keep the terminal clean
                    for line in result.stdout.split('\n'):
                        if "Success:" in line:
                            print(f"  -> {line}")
            except Exception as e:
                print(f"  -> Failed to execute command: {e}")

if __name__ == "__main__":
    run_batch()
    print(f"\nBatch testing complete! Open {OUTPUT_CSV} to see your full statistics.")