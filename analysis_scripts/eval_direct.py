"""
Evaluate the direct velocity prediction model.
Bypasses Euler integration — uses model output directly as velocity.
"""
import subprocess
import os
import sys
import shutil

# First, create a patched simulator that skips Euler integration for direct models
PATCH = '''
import os
import argparse
import numpy as np
import torch
import csv
import time
from tqdm import tqdm
from collections import defaultdict

from main_pys.generative_model import FlowGNNModel
from main_pys.model_inputs import create_data_object, normalize_graph_data, get_bd_prefs
from main_pys.simulator import (
    parse_scene, getCosts, convertProbsToPreferences, LABEL_TO_MOVES,
    pibt, updatePriorities, str2bool, CustomTimer
)


def runNNOnState_direct(cur_locs, bd, grid_map, k, m, model, device, goal_locations, timer):
    """Direct prediction: no Euler integration, just one forward pass with v_t=0, t=0."""
    with torch.no_grad():
        timer.start("create_nn_data")
        data = create_data_object(cur_locs, bd, grid_map, k, m, goal_locations)
        data = normalize_graph_data(data, k)
        data = data.to(device)
        timer.stop("create_nn_data")

        num_nodes = cur_locs.shape[0]
        v_t = torch.zeros(num_nodes, 2, device=device)
        t = torch.zeros(num_nodes, 1, device=device)

        timer.start("forward_pass")
        predicted_velocity = model(v_t, t, data)
        timer.stop("forward_pass")

        predicted_velocity = predicted_velocity.cpu().numpy()

        # --- WAIT FIX: give wait a magnitude-based score ---
        WAIT_BIAS = 0.5
        magnitudes = np.linalg.norm(predicted_velocity, axis=1)

        action_vectors = np.array([[0,0], [0,1], [1,0], [-1,0], [0,-1]])
        scores = predicted_velocity @ action_vectors.T
        scores[:, 0] = WAIT_BIAS - magnitudes

        tau = 0.5
        scores = scores / tau
        scores = scores - np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    return probs


def simulate_direct(device, model, k, m, grid_map, bd, start_locations, goal_locations,
                    max_steps, args, timer):
    cur_locs = start_locations
    agent_priorities = bd[range(len(start_locations)), start_locations[:,0], start_locations[:,1]]
    agent_priorities = agent_priorities / agent_priorities.max()

    solution_path = [cur_locs.copy()]
    success = False
    start_time = time.time()

    for step in tqdm(range(max_steps)):
        agents_at_goal = np.all(np.equal(cur_locs, goal_locations), axis=1)
        agent_priorities = updatePriorities(agent_priorities, agents_at_goal)

        if time.time() - start_time > args.timeLimit and args.timeLimit > 0:
            print("time limit hit")
            break

        probs = runNNOnState_direct(cur_locs, bd, grid_map, k, m, model, device, goal_locations, timer)

        # Mask walls
        action_mask = grid_map[cur_locs[:, 0, None] + LABEL_TO_MOVES[:, 0],
                               cur_locs[:, 1, None] + LABEL_TO_MOVES[:, 1]] == 1
        probs[action_mask] = 1e-8
        probs = probs / probs.sum(axis=1, keepdims=True)

        action_preferences = convertProbsToPreferences(probs, "sampled")

        timer.start("cs-time")
        new_move, cspibt_worked = pibt(grid_map, action_preferences, cur_locs,
                                       agent_priorities, [], start_time, args.timeLimit)
        timer.stop("cs-time")

        if not cspibt_worked:
            if time.time() - start_time < args.timeLimit:
                print("ERROR: CS-PIBT failed")
            break

        cur_locs = cur_locs + new_move
        solution_path.append(cur_locs.copy())

        if np.all(np.equal(cur_locs, goal_locations)):
            success = True
            break

    solution_path = np.array(solution_path)
    total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal = getCosts(solution_path, goal_locations)
    return solution_path, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapNpzFile', type=str, required=True)
    parser.add_argument('--mapName', type=str, required=True)
    parser.add_argument('--scenFile', type=str, required=True)
    parser.add_argument('--agentNum', type=int, required=True)
    parser.add_argument('--bdNpzFile', type=str, required=True)
    parser.add_argument('--modelPath', type=str, required=True)
    parser.add_argument('--useGPU', type=lambda x: bool(str2bool(x)), required=True)
    parser.add_argument('--maxSteps', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeLimit', type=int, default=120)
    parser.add_argument('--outputCSVFile', type=str, required=True)
    args = parser.parse_args()

    k = 4
    m = 5

    map_npz = np.load(args.mapNpzFile)
    map_grid = map_npz[args.mapName + ".map"]
    map_grid = np.pad(map_grid, k, 'constant', constant_values=1)

    start_locations, goal_locations = parse_scene(args.scenFile)
    num_agents = args.agentNum
    start_locations = start_locations[:num_agents] + k
    goal_locations = goal_locations[:num_agents] + k

    scen_num = args.scenFile.split('-')[-1].split('.')[0]
    bd_key = f"{args.mapName}-random-{scen_num}"
    bd_npz = np.load(args.bdNpzFile)
    bd = bd_npz[bd_key][:num_agents].astype(np.float32)
    bd = np.pad(bd, ((0,0),(k,k),(k,k)), 'constant', constant_values=10000)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.useGPU else "cpu")
    model = FlowGNNModel(k=k).to(device)
    checkpoint = torch.load(args.modelPath, map_location=device, weights_only=True)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.maxSteps.endswith('x'):
        longest_single_path = bd[range(num_agents), start_locations[:,0], start_locations[:,1]].max()
        max_steps = int(int(args.maxSteps[:-1]) * longest_single_path)
    else:
        max_steps = int(args.maxSteps)

    timer = CustomTimer()
    timer.start("total_simulate")
    solution_path, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, success = \\
        simulate_direct(device, model, k, m, map_grid, bd, start_locations, goal_locations,
                        max_steps, args, timer)
    timer.stop("total_simulate")
    total_simulate_time = timer.getTimes("total_simulate")

    print(f"Success: {success}, Total cost true: {total_cost_true}, "
          f"Num agents at goal: {num_agents_at_goal}/{num_agents}, "
          f"Seconds: {total_simulate_time:.1f}")

    if not os.path.exists(args.outputCSVFile):
        with open(args.outputCSVFile, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['mapName','scenFile','agentNum','success',
                           'total_cost_true','num_agents_at_goal','runtime'])
    with open(args.outputCSVFile, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.mapName, args.scenFile, args.agentNum, success,
                        total_cost_true, num_agents_at_goal, total_simulate_time])


if __name__ == '__main__':
    main()
'''

def run():
    os.makedirs("logs", exist_ok=True)

    # Write the direct simulator as a standalone script
    with open("eval_direct_sim.py", "w") as f:
        f.write(PATCH)

    output_csv = "logs/direct_overfit_eval.csv"
    if os.path.exists(output_csv):
        os.remove(output_csv)

    print("=" * 60)
    print("Evaluating DIRECT velocity model (no flow matching)")
    print("=" * 60)

    cmd = [
        sys.executable, "eval_direct_sim.py",
        "--mapNpzFile=data/all_maps.npz",
        "--mapName=empty-32-32",
        "--scenFile=data/scen-random/empty-32-32-random-10.scen",
        "--bdNpzFile=data/bd_npzs/large_scale/empty-32-32-random-10_bds.npz",
        "--modelPath=direct_overfit_best.pt",
        f"--outputCSVFile={output_csv}",
        "--maxSteps=5x",
        "--seed=0",
        "--useGPU=False",
        "--agentNum=20",
        "--timeLimit=120",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, text=True)

    print("=" * 60)
    if os.path.exists(output_csv):
        with open(output_csv) as f:
            for line in f:
                print(line.strip())


if __name__ == "__main__":
    run()
