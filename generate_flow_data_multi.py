import os
import subprocess
import re
import argparse
import numpy as np
from scipy.signal import savgol_filter
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque

from main_pys.grid_actions import get_actions, validate_action_mode, validate_diagonal_rule

# Configuration
DEFAULT_EECBS_BIN = "./build/eecbs"
DATA_DIR = "data"
MAP_DIR = os.path.join(DATA_DIR, "mapf-map")
SCEN_DIR = os.path.join(DATA_DIR, "scen-random")
OUTPUT_NPZ_DIR = os.path.join(DATA_DIR, "flow_training_data_multi")
BD_DIR = os.path.join(DATA_DIR, "bd_npzs", "large_scale")

# Rishi's Omitted and Held-out (Test) Maps
OMITTED_MAPS = {"brc202d", "orz900", "maze-128-128-1", "maze-128-128-10"}
HELD_OUT_TEST = {
    "Paris_1_256", "empty-48-48", "maze-128-128-2", "random-64-64-10",
    "random-32-32-10", "warehouse-10-20-10-2-1", "den312d", "den520d"
}

WINDOW_LENGTH = 3 
POLY_ORDER = 2    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate grid-world Flow-CS expert trajectories and BD heuristics."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes (default: all reported CPU cores).",
    )
    parser.add_argument(
        "--eecbs-bin",
        default=os.environ.get("EECBS_BIN", DEFAULT_EECBS_BIN),
        help="Path to the EECBS executable (default: ./build/eecbs or EECBS_BIN env var).",
    )
    parser.add_argument(
        "--output-npz-dir",
        default=OUTPUT_NPZ_DIR,
        help="Output directory for generated expert trajectory .npz files.",
    )
    parser.add_argument(
        "--bd-dir",
        default=BD_DIR,
        help="Output directory for generated BD .npz files.",
    )
    parser.add_argument(
        "--action-mode",
        choices=["grid4", "grid8"],
        default="grid4",
        help="Action mode used for BD heuristics. Trajectory connectivity still depends on the solver binary.",
    )
    parser.add_argument(
        "--diagonal-rule",
        choices=["blocked_pair", "both_clear", "allow"],
        default="blocked_pair",
        help="Diagonal obstacle rule for grid8 BD heuristics.",
    )
    return parser.parse_args()

def print_log(message):
    print(message, flush=True)

def parse_paths_txt(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    all_paths = []
    for line in lines:
        if not line.startswith("Agent"): continue
        coords = re.findall(r"\((\d+),(\d+)\)", line)
        path = np.array([[int(r), int(c)] for r, c in coords], dtype=np.float32)
        all_paths.append(path)
    if not all_paths: return np.array([])
    max_len = max(len(p) for p in all_paths)
    paths_array = np.zeros((len(all_paths), max_len, 2), dtype=np.float32)
    for i, p in enumerate(all_paths):
        paths_array[i, :len(p), :] = p
        paths_array[i, len(p):, :] = p[-1] 
    return paths_array

def smooth_and_extract_velocities(paths_array):
    if len(paths_array) == 0: return paths_array, paths_array
    N, T, D = paths_array.shape
    window = min(WINDOW_LENGTH, T)
    if window % 2 == 0: window -= 1
    if window < 3: return paths_array, np.gradient(paths_array, axis=1)
    smoothed_positions = savgol_filter(paths_array, window_length=window, polyorder=POLY_ORDER, axis=1, deriv=0)
    velocities = savgol_filter(paths_array, window_length=window, polyorder=POLY_ORDER, axis=1, deriv=1)
    return smoothed_positions, velocities

def read_map(map_file):
    with open(map_file, 'r') as f:
        f.readline(); h = int(f.readline().split()[1]); w = int(f.readline().split()[1]); f.readline()
        map_data = np.zeros((h, w), dtype=np.int8)
        for r in range(h):
            line = f.readline().strip()
            for c in range(w):
                if line[c] in ['@', 'T', 'O']: map_data[r, c] = 1
    return map_data

def parse_scenario_goals(scen_file, max_agents=1000):
    goals = []
    with open(scen_file, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 8:
                goals.append([int(parts[7]), int(parts[6])]) 
            if len(goals) == max_agents: break
    return np.array(goals)

def _diagonal_allowed(map_data, r, c, dr, dc, diagonal_rule):
    if abs(dr) != 1 or abs(dc) != 1 or diagonal_rule == "allow":
        return True
    side_a_blocked = map_data[r + dr, c] == 1
    side_b_blocked = map_data[r, c + dc] == 1
    if diagonal_rule == "both_clear":
        return not (side_a_blocked or side_b_blocked)
    return not (side_a_blocked and side_b_blocked)


def compute_bd_heuristic(map_data, goals, action_mode="grid4", diagonal_rule="blocked_pair"):
    H, W = map_data.shape
    N = len(goals)
    bd_array = np.full((N, H, W), 10000, dtype=np.int16)
    moves = [(a.dr, a.dc) for a in get_actions(action_mode) if not a.is_wait]
    
    for i, (gr, gc) in enumerate(goals):
        if map_data[gr, gc] == 1: continue
        queue = deque([(gr, gc, 0)])
        bd_array[i, gr, gc] = 0
        
        while queue:
            r, c, dist = queue.popleft()
            ndist = dist + 1
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (
                        map_data[nr, nc] == 0
                        and bd_array[i, nr, nc] == 10000
                        and _diagonal_allowed(map_data, r, c, dr, dc, diagonal_rule)
                    ):
                        bd_array[i, nr, nc] = ndist
                        queue.append((nr, nc, ndist))
    return bd_array

# ==========================================
# PHASE 1: Generate BD Heuristics Safely
# ==========================================
def generate_scenario_bd(map_path, scen_path, bd_dir, action_mode, diagonal_rule):
    scen_name = os.path.basename(scen_path).replace(".scen", "")
    map_name = os.path.basename(map_path).replace(".map", "")
    bd_key = f"{map_name}-random-{scen_name.split('-random-')[-1]}"
    out_bd_file = os.path.join(bd_dir, f"{scen_name}_bds.npz")
    
    if os.path.exists(out_bd_file):
        return f"BD exists: {scen_name}"
        
    print_log(f"--> Building {action_mode} Heuristic Grid: {scen_name} (Takes ~30s)")
    try:
        map_data = read_map(map_path)
        goals = parse_scenario_goals(scen_path, max_agents=1000)
        bd_array = compute_bd_heuristic(
            map_data,
            goals,
            action_mode=action_mode,
            diagonal_rule=diagonal_rule,
        )
        
        # Atomic write to prevent file corruption
        # FIX: Ensure the tmp file ends in .npz so numpy doesn't silently append it
        tmp_file = out_bd_file + f".{os.getpid()}.tmp.npz" 
        np.savez_compressed(tmp_file, **{bd_key: bd_array})
        os.rename(tmp_file, out_bd_file)
        return f"BD Generated: {scen_name}"
    except Exception as e:
        return f"BD Error: {scen_name} ({str(e)})"

# ==========================================
# PHASE 2: Generate EECBS Trajectories
# ==========================================
def generate_scenario_trajectory(map_path, scen_path, num_agents, eecbs_bin, output_npz_dir):
    scen_name = os.path.basename(scen_path).replace(".scen", "")
    out_traj_file = os.path.join(output_npz_dir, f"{scen_name}_{num_agents}.npz")
    tmp_path_file = f"tmp_{scen_name}_{num_agents}_{os.getpid()}.txt" 
    
    if os.path.exists(out_traj_file): 
        return f"Traj Exists: {scen_name} (N={num_agents})"
        
    cmd = [
        eecbs_bin, "-m", map_path, "-a", scen_path, "-k", str(num_agents),
        "--outputPaths", tmp_path_file, "--suboptimality", "2.0"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if os.path.exists(tmp_path_file):
            paths_discrete = parse_paths_txt(tmp_path_file)
            if len(paths_discrete) > 0:
                positions, velocities = smooth_and_extract_velocities(paths_discrete)
                np.savez_compressed(out_traj_file, discrete_positions=paths_discrete, 
                                   smoothed_positions=positions, expert_velocities=velocities)
                os.remove(tmp_path_file)
                return f"Traj Done: {scen_name} (N={num_agents})"
    except Exception as e:
        if os.path.exists(tmp_path_file): os.remove(tmp_path_file)
        return f"Traj Fail: {scen_name}_{num_agents} ({str(e)})"
    
    if os.path.exists(tmp_path_file): os.remove(tmp_path_file)
    return f"Traj Fail: {scen_name}_{num_agents}"

# ==========================================
# EXECUTION PIPELINE
# ==========================================
def process_benchmark_parallel(max_workers, eecbs_bin, output_npz_dir, bd_dir, action_mode, diagonal_rule):
    validate_action_mode(action_mode)
    validate_diagonal_rule(diagonal_rule)
    os.makedirs(output_npz_dir, exist_ok=True)
    os.makedirs(bd_dir, exist_ok=True)

    map_files = glob.glob(os.path.join(MAP_DIR, "*.map"))
    agent_counts = [20, 50, 100, 200, 400, 600, 800, 1000]
    
    bd_jobs = []
    traj_jobs = []
    
    for map_path in map_files:
        map_name = os.path.basename(map_path).replace(".map", "")
        if map_name in OMITTED_MAPS:
            continue
            
        scen_files = sorted(glob.glob(os.path.join(SCEN_DIR, f"{map_name}-random-*.scen")))
        
        # We need BD heuristics for the test set too!
        limit = 25 if map_name in HELD_OUT_TEST else 128
        
        for scen_path in scen_files[:limit]:
            # Queue exactly ONE heuristic calculation per scenario
            bd_jobs.append((map_path, scen_path))
            
            # Queue trajectory generations (skip for held-out test set)
            if map_name not in HELD_OUT_TEST:
                for n in agent_counts:
                    if n > 200 and "32-32" in map_name: continue
                    traj_jobs.append((map_path, scen_path, n))

    cpu_count = os.cpu_count() or 1
    print_log(
        f"Using {max_workers} workers (Python reports {cpu_count} CPU cores)."
    )
    print_log(f"Using EECBS binary: {eecbs_bin}")
    print_log(f"Action mode for BD heuristics: {action_mode} ({diagonal_rule})")
    print_log(f"Trajectory output: {output_npz_dir}")
    print_log(f"BD output: {bd_dir}")

    # Run Phase 1
    print_log(f"--- PHASE 1: Generating Heuristics ({len(bd_jobs)} files) ---")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(generate_scenario_bd, m, s, bd_dir, action_mode, diagonal_rule)
            for m, s in bd_jobs
        ]
        for future in as_completed(futures):
            # Print instantly when a job finishes
            print_log(future.result())

    # Run Phase 2
    print_log(f"\n--- PHASE 2: Generating Expert Trajectories ({len(traj_jobs)} files) ---")
    if traj_jobs and not os.path.isfile(eecbs_bin):
        raise FileNotFoundError(
            f"EECBS binary not found: {eecbs_bin}. Pass --eecbs-bin or set EECBS_BIN."
        )
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_scenario_trajectory, m, s, a, eecbs_bin, output_npz_dir): (m, s, a)
            for m, s, a in traj_jobs
        }
        for i, future in enumerate(as_completed(futures)):
            if i % 50 == 0: 
                print_log(f"Progress: {i}/{len(traj_jobs)} | {future.result()}")

if __name__ == "__main__":
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    process_benchmark_parallel(
        args.workers,
        args.eecbs_bin,
        args.output_npz_dir,
        args.bd_dir,
        args.action_mode,
        args.diagonal_rule,
    )
