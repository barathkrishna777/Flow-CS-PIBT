"""Simulator with NO wait fix — for A/B comparison."""
import os, argparse, numpy as np, torch, csv, time
from tqdm import tqdm
from collections import defaultdict
from main_pys.generative_model import FlowGNNModel
from main_pys.model_inputs import create_data_object, normalize_graph_data
from main_pys.simulator import (
    parse_scene, getCosts, convertProbsToPreferences, LABEL_TO_MOVES,
    pibt, updatePriorities, str2bool, CustomTimer
)

def runNNOnState_nofix(cur_locs, bd, grid_map, k, m, model, device, goal_locations, timer):
    with torch.no_grad():
        timer.start("create_nn_data")
        data = create_data_object(cur_locs, bd, grid_map, k, m, goal_locations)
        data = normalize_graph_data(data, k)
        data = data.to(device)
        timer.stop("create_nn_data")
        v = torch.randn(cur_locs.shape[0], 2).to(device)
        num_steps = 5
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((cur_locs.shape[0], 1), step * dt, device=device)
            flow = model(v, t, data)
            v = v + flow * dt
        predicted_velocity = v.cpu().numpy()
        # ORIGINAL: no wait fix
        action_vectors = np.array([[0,0], [0,1], [1,0], [-1,0], [0,-1]])
        scores = predicted_velocity @ action_vectors.T
        tau = 0.5
        scores = scores / tau
        scores = scores - np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    return probs

def simulate(device, model, k, m, grid_map, bd, start_locations, goal_locations, max_steps, args, timer):
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
            print("time limit hit"); break
        probs = runNNOnState_nofix(cur_locs, bd, grid_map, k, m, model, device, goal_locations, timer)
        action_mask = grid_map[cur_locs[:, 0, None] + LABEL_TO_MOVES[:, 0],
                               cur_locs[:, 1, None] + LABEL_TO_MOVES[:, 1]] == 1
        probs[action_mask] = 1e-8
        probs = probs / probs.sum(axis=1, keepdims=True)
        action_preferences = convertProbsToPreferences(probs, "sampled")
        timer.start("cs-time")
        new_move, cspibt_worked = pibt(grid_map, action_preferences, cur_locs, agent_priorities, [], start_time, args.timeLimit)
        timer.stop("cs-time")
        if not cspibt_worked:
            if time.time() - start_time < args.timeLimit: print("ERROR: CS-PIBT failed")
            break
        cur_locs = cur_locs + new_move
        solution_path.append(cur_locs.copy())
        if np.all(np.equal(cur_locs, goal_locations)):
            success = True; break
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
    k, m = 4, 5
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
    if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else: model.load_state_dict(checkpoint, strict=False)
    model.eval()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.maxSteps.endswith('x'):
        longest = bd[range(num_agents), start_locations[:,0], start_locations[:,1]].max()
        max_steps = int(int(args.maxSteps[:-1]) * longest)
    else: max_steps = int(args.maxSteps)
    timer = CustomTimer()
    timer.start("total")
    _, total_cost_true, _, num_agents_at_goal, success = simulate(
        device, model, k, m, map_grid, bd, start_locations, goal_locations, max_steps, args, timer)
    timer.stop("total")
    t = timer.getTimes("total")
    print(f"Success: {success}, Agents at goal: {num_agents_at_goal}/{num_agents}, Cost: {total_cost_true}, Time: {t:.1f}s")
    if not os.path.exists(args.outputCSVFile):
        with open(args.outputCSVFile, 'w') as f:
            csv.writer(f).writerow(['mapName','scenFile','agentNum','success','total_cost_true','num_agents_at_goal','runtime'])
    with open(args.outputCSVFile, 'a') as f:
        csv.writer(f).writerow([args.mapName, args.scenFile, args.agentNum, success, total_cost_true, num_agents_at_goal, t])

if __name__ == '__main__':
    main()
