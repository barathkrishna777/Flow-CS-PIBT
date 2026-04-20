import os
import subprocess
import glob

# ==========================================
# Configuration
# ==========================================
MODEL_PATH = "large_scale_flow_epoch_1.pt"
OUTPUT_CSV = "logs/batch_results.csv"  
MAP_NPZ = "data/all_maps.npz"
BD_DIR = "data/bd_npzs"
SCEN_DIR = "data/scen-random"

MAX_STEPS = "5x"
SHIELD_TYPE = "CS-PIBT"
SCENARIOS_PER_MAP = 3 

# Testing 3 distinct topologies
TEST_MAPS = ["empty-48-48", "random-32-32-10", "den312d"]
# Testing 3 distinct density levels
AGENT_COUNTS = [50, 100, 200]

def run_batch():
    os.makedirs("logs", exist_ok=True)
    total_runs = len(TEST_MAPS) * SCENARIOS_PER_MAP * len(AGENT_COUNTS)

    print(f"Starting Multi-Density Evaluation for {total_runs} total runs...")
    print(f"Model: {MODEL_PATH} | Shield: {SHIELD_TYPE}\n")

    for map_name in TEST_MAPS:
        scen_pattern = os.path.join(SCEN_DIR, f"{map_name}-random-*.scen")
        scen_files = sorted(glob.glob(scen_pattern))
        
        if not scen_files:
            print(f"Skipping {map_name} - no scenario files found.")
            continue

        for scen_file in scen_files[:SCENARIOS_PER_MAP]:
            scen_basename = os.path.basename(scen_file)
            scen_name_only = scen_basename.replace('.scen', '')
            bd_file = os.path.join(BD_DIR, "large_scale", f"{scen_name_only}_bds.npz")
            
            if not os.path.exists(bd_file):
                print(f"  -> Skipping {scen_basename}: BD file not found.")
                continue

            for agent_num in AGENT_COUNTS:
                print(f"Evaluating: Map: {map_name} | Scen: {scen_basename} | Agents: {agent_num}")
                
                cmd = [
                    "python", "-m", "main_pys.simulator", 
                    f"--mapNpzFile={MAP_NPZ}",
                    f"--mapName={map_name}",
                    f"--scenFile={scen_file}",
                    f"--bdNpzFile={bd_file}",
                    f"--modelPath={MODEL_PATH}",
                    f"--outputCSVFile={OUTPUT_CSV}",
                    f"--maxSteps={MAX_STEPS}",
                    "--seed=0",
                    "--useGPU=True",
                    f"--agentNum={agent_num}",
                    f"--shieldType={SHIELD_TYPE}"
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        print(f"  -> ERROR:")
                        print(f"  -> {result.stderr.strip().splitlines()[-1]}") 
                    else:
                        for line in result.stdout.split('\n'):
                            if "Success:" in line:
                                print(f"  -> {line}")
                except Exception as e:
                    print(f"  -> Failed to execute command: {e}")

if __name__ == "__main__":
    run_batch()
    print(f"\nBatch testing complete! Open {OUTPUT_CSV} to see your full statistics.")