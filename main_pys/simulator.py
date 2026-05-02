import os
import argparse
import pdb
from typing import Any
import numpy as np
import torch 
import torch.nn as nn
import csv 
from collections import deque, defaultdict 
import cProfile 
import pstats 
from tqdm import tqdm 
import time

from main_pys.model import GNNStack, CustomConv 
from main_pys.model_inputs import create_data_object, normalize_graph_data, get_bd_prefs
from main_pys.custom_timer import CustomTimer
from main_pys.generative_model import FlowGNNModel, hybrid_action_logits_from_velocity
from main_pys.rishi_like_model import RishiLikeClassifier

def str2bool(v: str) -> bool:
    return v.lower() in ("yes", "true", "t", "1")

def parse_scene(scen_file):
    start_locations = []
    goal_locations = []
    with open(scen_file) as f:
        line = f.readline().strip()
        start_locations = list()
        goal_locations = list()
        for line in f:
            line = line.rstrip() 
            tokens = line.split("\t") 
            assert(len(tokens) == 9) 
            tokens = tokens[4:]  
            col = int(tokens[0])
            row = int(tokens[1])
            start_locations.append((row,col)) 
            col = int(tokens[2])
            row = int(tokens[3])
            goal_locations.append((row,col)) 
    return np.array(start_locations, dtype=int), np.array(goal_locations, dtype=int)

def createScenFile(locs, goal_locs, map_name, scenFilepath):
    assert(locs.min() >= 0 and goal_locs.min() >= 0)
    with open(scenFilepath, 'w') as f:
        f.write(f"version {len(locs)} \n")
        for i in range(locs.shape[0]):
            f.write(f"0\t{map_name}\t{0}\t{0}\t{locs[i,1]}\t{locs[i,0]}\t{goal_locs[i,1]}\t{goal_locs[i,0]}\t0 \n")

def getCosts(solution_path, goal_locs):
    at_goal = np.all(np.equal(solution_path, np.expand_dims(goal_locs, 0)), axis=2) 
    not_at_goal = 1 - at_goal 
    last_timestep_at_goal = len(at_goal) - np.argmax(not_at_goal[::-1], axis=0) 
    last_timestep_at_goal = np.minimum(last_timestep_at_goal, len(at_goal)-1) 
    total_cost_true = last_timestep_at_goal.sum()
    resting_at_goal = np.logical_and(at_goal[:-1], at_goal[1:]) 
    total_cost_not_resting_at_goal = (1-resting_at_goal).sum() 
    num_agents_at_goal = np.sum(at_goal[-1]) 
    assert(total_cost_true >= total_cost_not_resting_at_goal)
    assert(total_cost_true <= (solution_path.shape[0]-1)*solution_path.shape[1])
    return total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal


def normalize_probability_rows(probs, floor=1e-8):
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.maximum(probs, 0.0)
    probs = np.maximum(probs, floor)
    row_sums = probs.sum(axis=1, keepdims=True)
    bad_rows = row_sums.squeeze(1) <= 0
    if np.any(bad_rows):
        probs[bad_rows] = 1.0
        row_sums = probs.sum(axis=1, keepdims=True)
    return probs / row_sums


def convertProbsToPreferences(probs, conversion_type):
    probs = normalize_probability_rows(probs)
    if conversion_type == "sorted":
        preferences = np.argsort(-probs, axis=1)
    elif conversion_type == "sampled":
        probs = torch.tensor(probs, dtype=torch.float32)
        preferences = torch.zeros_like(probs, dtype=torch.int64)
        for i in range(5):
            cur_sample = torch.multinomial(probs, num_samples=1, replacement=False) 
            probs.scatter_(1, cur_sample, 0) 
            preferences[:,i] = cur_sample[:,0]
        preferences = preferences.numpy()
        assert(np.all(preferences.sum(axis=1) == 10)) 
    else:
        raise ValueError('Invalid conversion type: {}'.format(conversion_type))
    return preferences

LABEL_TO_MOVES = np.array([[0,0], [0,1], [1,0], [-1,0], [0,-1]]) 

def pibtRecursive(grid_map, agent_id, action_preferences, planned_agents, move_matrix, 
         occupied_nodes, occupied_edges, current_locs, current_locs_to_agent,
         constrained_agents_to_action, start_time, timeLimit):
    moves_ordered = LABEL_TO_MOVES[action_preferences[agent_id]]
    if agent_id in constrained_agents_to_action: 
        action_index = constrained_agents_to_action[agent_id]
        moves_ordered = moves_ordered[action_index:action_index+1] 
    
    cur_time=time.time()
    if cur_time-start_time>timeLimit:
        return False

    current_pos = current_locs[agent_id] 
    for aMove in moves_ordered:
        next_loc = current_pos + aMove 
        if next_loc[0] < 0 or next_loc[0] >= grid_map.shape[0] or next_loc[1] < 0 or next_loc[1] >= grid_map.shape[1]:
            continue
        if grid_map[next_loc[0], next_loc[1]] == 1:
            continue
        if occupied_nodes[next_loc[0], next_loc[1]]:
            continue
        rev_edge_key = tuple([*next_loc, *current_pos])
        if rev_edge_key in occupied_edges and occupied_edges[rev_edge_key]:
            continue
        
        move_matrix[agent_id] = aMove
        planned_agents[agent_id] = True
        occupied_nodes[next_loc[0], next_loc[1]] = True
        edge_key = tuple([*current_pos, *next_loc])
        occupied_edges[edge_key] = True

        conflicting_agent = current_locs_to_agent[next_loc[0], next_loc[1]]
        if conflicting_agent != -1 and conflicting_agent != agent_id and not planned_agents[conflicting_agent]:
            isvalid = pibtRecursive(grid_map, conflicting_agent, action_preferences, planned_agents,
                                move_matrix, occupied_nodes, occupied_edges, current_locs,
                                current_locs_to_agent, constrained_agents_to_action, start_time,timeLimit)
            if isvalid:
                return True
            else:
                planned_agents[agent_id] = False
                occupied_edges[edge_key] = False
                continue
        else:
            return True
    
    if agent_id not in constrained_agents_to_action:
        move_matrix[agent_id] = np.array([0,0])
        planned_agents[agent_id] = True
        occupied_nodes[current_pos[0], current_pos[1]] = True
    return False

def pibt(grid_map, action_preferences, current_locs, agent_priorities, agent_constraints, start_time, timeLimit):
    agent_order = np.argsort(-agent_priorities) 
    move_matrix = np.zeros((len(agent_priorities), 2), dtype=int) 
    occupied_nodes = np.zeros(grid_map.shape, dtype=bool) 
    occupied_edges = defaultdict(bool) 
    planned_agents = np.zeros(len(agent_priorities), dtype=bool) 

    current_locs_to_agent = np.zeros(grid_map.shape, dtype=int) - 1  
    current_locs_to_agent[current_locs[:,0], current_locs[:,1]] = np.arange(len(current_locs))  

    constrained_agents_to_action = dict()
    for agent_id, action_index in agent_constraints:
        which_agent = agent_order[agent_id]
        constrained_agents_to_action[which_agent] = action_preferences[which_agent, (action_index+1)%5]

    for agent_id in agent_order:
        if planned_agents[agent_id]:
            continue
        pibt_worked = pibtRecursive(grid_map, agent_id, action_preferences, planned_agents, 
                            move_matrix, occupied_nodes, occupied_edges, 
                            current_locs, current_locs_to_agent, constrained_agents_to_action, start_time, timeLimit)
        if pibt_worked is False:
            break
    return move_matrix, pibt_worked

def updatePriorities(prev_priorities, at_goal):
    agent_priorities = prev_priorities.copy()
    agent_priorities[(prev_priorities <= 0) & at_goal] -= 1
    agent_priorities[(prev_priorities > 0) & at_goal] = 0 
    agent_priorities[~at_goal] = np.maximum(prev_priorities[~at_goal], 0) + 1
    return agent_priorities

class LaCAMRunner:
    class HLNode:
        def __init__(self, state: np.ndarray, action_preferences: np.ndarray, 
                     parent: 'LaCAMRunner.HLNode', bd: np.ndarray, goal_locations: np.ndarray) -> None:
            self.state = state
            self.action_preferences = action_preferences
            self.parent = parent
            self.queue_of_constraints = deque() 
            self.queue_of_constraints.append([]) 

            if parent is None:
                self.depth = 0
                distance_to_goal = bd[range(len(state)), state[:,0], state[:,1]] 
                self.agent_priorities = distance_to_goal / distance_to_goal.max() 
            else:
                self.depth = parent.depth + 1
                at_goal = np.all(np.equal(state, goal_locations), axis=1) 
                self.agent_priorities = updatePriorities(self.parent.agent_priorities, at_goal)

        def getNextState(self, grid_map: np.ndarray, start_time, timeLimit):
            assert(len(self.queue_of_constraints) > 0)
            curConstraint = self.queue_of_constraints.popleft()
            
            if len(curConstraint) == 0: 
                for i in range(0,5):
                    self.queue_of_constraints.append([(0,i)])
            else:
                curAgent = curConstraint[-1][0] 
                if curAgent + 1 < len(self.state): 
                    for i in range(0,5): 
                        self.queue_of_constraints.append(curConstraint + [(curAgent+1,i)])

            new_move, pibt_worked = pibt(grid_map, self.action_preferences, self.state, self.agent_priorities, curConstraint, start_time, timeLimit)

            if not pibt_worked: 
                return None
            new_state = self.state + new_move
            return new_state
            
    def __init__(self, real_time=False) -> None:
        self.mainStack = deque() 
        self.stateToHLNodes = dict() 
        self.real_time = real_time
        if self.real_time:
            print("Real-Time LaCAM enabled")
        else:
            print("LaCAM enabled")
    
    def lacam(self, start_locations, goal_locations, bd, grid_map, getActionPrefsFromLocs, lacamLimit, start_time, timeLimit):
        if not self.real_time:
            self.mainStack.clear() 
            self.stateToHLNodes = dict() 

        if len(self.mainStack) == 0: 
            curNode = LaCAMRunner.HLNode(start_locations, getActionPrefsFromLocs(start_locations), None, bd, goal_locations)
            self.mainStack.appendleft(curNode) 
            self.stateToHLNodes[start_locations.tobytes()] = curNode

        success = False
        MAXGENERATED = lacamLimit
        numNodesExpanded = 0
        numGenerated = 1 
        while len(self.mainStack) > 0:
            curNode : LaCAMRunner.HLNode = self.mainStack.popleft()
            if len(curNode.queue_of_constraints) != 0: 
                self.mainStack.appendleft(curNode)
            new_locs = curNode.getNextState(grid_map, start_time, timeLimit)
            if time.time() - start_time > timeLimit: 
                break
            if new_locs is None:
                continue
            numNodesExpanded += 1

            if np.all(np.equal(new_locs, goal_locations)):
                print("Stopping as found goal in LaCAM, depth: {}, nodes expanded: {}".format(curNode.depth, numNodesExpanded))
                success = True
                break

            key = new_locs.tobytes()
            if key in self.stateToHLNodes.keys(): 
                curNode = self.stateToHLNodes[key] 
                self.mainStack.appendleft(curNode) 
            else:
                newHLNode = LaCAMRunner.HLNode(new_locs, getActionPrefsFromLocs(new_locs), curNode, bd, goal_locations)
                numGenerated += 1
                self.stateToHLNodes[key] = newHLNode
                self.mainStack.appendleft(newHLNode)
                
            if numGenerated >= MAXGENERATED: 
                break
            
        if self.real_time:
            entirePath = [start_locations, new_locs]
        else:    
            entirePath = [new_locs]
            while curNode is not None:
                entirePath.append(curNode.state)
                curNode = curNode.parent
            entirePath.reverse() 
        return entirePath, success, numNodesExpanded, numGenerated

class WrapperNNWithCache:
    def __init__(self, bd, grid_map, model, device, k, m, goal_locations, timer) -> None:
        self.bd = bd
        self.grid_map = grid_map
        self.model = model
        self.device = device
        self.k = k
        self.m = m
        self.saved_calls = dict()
        self.hits = 0
        self.goal_locations = goal_locations
        self.timer = timer

    def __call__(self, locs: np.ndarray):
        key = locs.tobytes()
        if key in self.saved_calls.keys():
            self.hits += 1
            return self.saved_calls[key]
        else:
            probs = runNNOnState(locs, self.bd, self.grid_map, self.k, self.m, self.model, self.device, self.goal_locations, self.timer)
            self.saved_calls[key] = probs
            return probs

def runNNOnState(cur_locs, bd, grid_map, k, m, model, device, goal_locations, timer):
    with torch.no_grad():
        timer.start("create_nn_data")
        data = create_data_object(cur_locs, bd, grid_map, k, m, goal_locations)
        data = normalize_graph_data(data, k)
        data = data.to(device)
        timer.stop("create_nn_data")

        n_agents = cur_locs.shape[0]

        if args.policyType in ("classifier", "local_classifier"):
            timer.start("forward_pass")
            _, predictions = model(data)
            probs = torch.softmax(predictions, dim=1).cpu().numpy()
            timer.stop("forward_pass")
        elif args.useActionHead or args.policyType == "flow_action_head":
            conditioning = getattr(args, 'actionHeadConditioning', 'integrated')
            if conditioning == "integrated":
                # Run flow ODE to get v ≈ x_1 before querying action head — matches training distribution at t→1
                num_steps = args.numIntegrationSteps
                dt = 1.0 / num_steps
                v = torch.randn(n_agents, 2, device=device)
                for step in range(num_steps):
                    t_step = torch.full((n_agents, 1), step * dt, device=device)
                    timer.start("forward_pass")
                    flow = model(v, t_step, data)
                    timer.stop("forward_pass")
                    v = v + flow * dt
                t_final = torch.full((n_agents, 1), 0.99, device=device)
                timer.start("forward_pass")
                _, action_logits = model(v, t_final, data, return_action_logits=True)
                timer.stop("forward_pass")
            elif conditioning == "zero_t0":
                v_dummy = torch.zeros(n_agents, 2, device=device)
                t_dummy = torch.full((n_agents, 1), 0.0, device=device)
                timer.start("forward_pass")
                _, action_logits = model(v_dummy, t_dummy, data, return_action_logits=True)
                timer.stop("forward_pass")
            else:  # zero_t05
                v_dummy = torch.zeros(n_agents, 2, device=device)
                t_dummy = torch.full((n_agents, 1), 0.5, device=device)
                timer.start("forward_pass")
                _, action_logits = model(v_dummy, t_dummy, data, return_action_logits=True)
                timer.stop("forward_pass")
            scores = action_logits.cpu().numpy()
            scores = scores / args.tau
            scores = scores - np.max(scores, axis=1, keepdims=True)
            probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
            # Clip to floor so convertProbsToPreferences multinomial never sees an all-zero row
            # (confident logits + low tau can underflow to 0 in float32 after the first scatter_)
            probs = np.clip(probs, 1e-8, 1.0)
            probs = probs / probs.sum(axis=1, keepdims=True)
        else:
            # Flow-based inference with multi-sample consensus
            num_steps = args.numIntegrationSteps
            dt = 1.0 / num_steps
            num_samples = args.numConsensusSamples

            all_velocities = torch.zeros(n_agents, 2, device=device)
            for _ in range(num_samples):
                v = torch.randn(n_agents, 2, device=device)
                for step in range(num_steps):
                    t = torch.full((n_agents, 1), step * dt, device=device)
                    timer.start("forward_pass")
                    flow = model(v, t, data)
                    timer.stop("forward_pass")
                    v = v + flow * dt
                all_velocities += v
            velocity_tensor = all_velocities / num_samples
            predicted_velocity = velocity_tensor.cpu().numpy()

            if args.waitMode == "learned":
                t_final = torch.full((n_agents, 1), 0.99, device=device)
                timer.start("forward_pass")
                _, wait_logit = model(velocity_tensor, t_final, data, return_wait_logit=True)
                timer.stop("forward_pass")
                scores = hybrid_action_logits_from_velocity(velocity_tensor, wait_logit).cpu().numpy()
            else:
                # --- WAIT FIX: magnitude threshold ---
                magnitudes = np.linalg.norm(predicted_velocity, axis=1)
                should_wait = magnitudes < args.waitThreshold

                action_vectors = np.array([[0,0], [0,1], [1,0], [-1,0], [0,-1]])
                scores = predicted_velocity @ action_vectors.T

            scores = scores / args.tau
            scores = scores - np.max(scores, axis=1, keepdims=True)
            probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

            if args.waitMode == "threshold":
                # For agents that should wait: set wait prob high, suppress others
                probs[should_wait] = 0.01
                probs[should_wait, 0] = 0.96  # action 0 = wait
            probs = normalize_probability_rows(probs)

    return probs

def load_flow_model(args, device, k):
    model = FlowGNNModel(k=k, hidden_dim=args.hiddenDim, num_layers=args.numLayers).to(device)

    checkpoint = torch.load(args.modelPath, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        incompatible = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        incompatible = model.load_state_dict(checkpoint, strict=False)
    if args.waitMode == "learned" and any(key.startswith("wait_head.") for key in incompatible.missing_keys):
        print("WARNING: checkpoint has no learned wait_head weights; --waitMode learned will use a randomly initialized wait head.")
    if incompatible.unexpected_keys:
        print(f"WARNING: ignored unexpected checkpoint keys: {incompatible.unexpected_keys}")
    return model

def load_classifier_model(args, device, k):
    checkpoint = torch.load(args.modelPath, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device)

    if isinstance(checkpoint, dict):
        for key in ("model", "net", "module"):
            maybe_model = checkpoint.get(key)
            if isinstance(maybe_model, nn.Module):
                return maybe_model.to(device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    linear_dim = args.classifierLinearDim
    if linear_dim <= 0:
        patch_width = 2 * k + 1
        linear_dim = (patch_width - 2) ** 2 * args.classifierInChannels + 5
    model = GNNStack(
        linear_dim,
        args.classifierInChannels,
        args.classifierHiddenDim,
        args.classifierOutputDim,
        args.classifierReluType,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    return model

def load_local_classifier_model(args, device, k):
    checkpoint = torch.load(args.modelPath, map_location=device, weights_only=False)
    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device)

    config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model = RishiLikeClassifier(
        k=config.get("k", k),
        hidden_dim=config.get("hidden_dim", args.localClassifierHiddenDim),
        num_layers=config.get("num_layers", args.localClassifierNumLayers),
        dropout=config.get("dropout", args.localClassifierDropout),
        in_channels=config.get("in_channels", args.localClassifierInChannels),
        aux_feature_dim=config.get("aux_feature_dim", 5),
        action_dim=config.get("action_dim", 5),
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    return model

class WrapperBDGetActionPrefs:
    def __init__(self, bd, grid_map, k, m, num_agents) -> None:
        self.bd = bd
        self.grid_map = grid_map
        self.k = k
        self.m = m
        self.range_num_agents = np.arange(num_agents)

    def __call__(self, locs):
        return get_bd_prefs(locs, self.bd, self.range_num_agents, add_noise=True)

def simulate(device, model, k, m, grid_map, bd, start_locations, goal_locations, 
             max_steps, shield_type, lacam_lookahead, args, timer: CustomTimer):
    if shield_type not in ["CS-PIBT", "CS-Freeze", "LaCAM", "Real-Time-LaCAM"]:
        raise KeyError('Invalid shield type: {}'.format(shield_type))
    
    wrapper_nn = WrapperNNWithCache(bd, grid_map, model, device, k, m, goal_locations, timer)
    wrapper_bd_prefs = WrapperBDGetActionPrefs(bd, grid_map, k, m, len(start_locations)) 
    def getActionPrefsFromLocs(locs):
        action_mask = grid_map[locs[:, 0, None] + LABEL_TO_MOVES[:, 0], locs[:, 1, None] + LABEL_TO_MOVES[:, 1]] == 1
        assert(not np.any(action_mask[:,0]))
        at_goal = np.all(np.equal(locs, goal_locations), axis=1)

        if args.policyType == "pibt":
            preferences = wrapper_bd_prefs(locs)
            preferences = preferences.copy()

            for agent_id in range(preferences.shape[0]):
                ordered = [int(a) for a in preferences[agent_id] if not action_mask[agent_id, a]]
                ordered.extend(int(a) for a in preferences[agent_id] if action_mask[agent_id, a])
                if at_goal[agent_id]:
                    ordered = [0] + [a for a in ordered if a != 0]
                preferences[agent_id] = ordered
            return preferences

        probs = runNNOnState(locs, bd, grid_map, k, m, model, device, goal_locations, timer)

        # Force at-goal agents to wait — prevents wandering away from goal
        probs[at_goal] = 1e-6  # small epsilon so multinomial can still rank all 5 actions
        probs[at_goal, 0] = 1.0  # action 0 = wait (dominant)

        probs[action_mask] = 1e-8
        probs = normalize_probability_rows(probs)
        return convertProbsToPreferences(probs, "sampled")
    
    cur_locs = start_locations 
    assert(grid_map[start_locations[:,0], start_locations[:,1]].sum() == 0)
    assert(grid_map[goal_locations[:,0], goal_locations[:,1]].sum() == 0)

    agent_priorities = bd[range(len(start_locations)), start_locations[:,0], start_locations[:,1]] 
    agent_priorities = agent_priorities / agent_priorities.max() 
    
    if shield_type in ["LaCAM", "Real-Time-LaCAM"]:
        lacamRunner = LaCAMRunner(real_time=(shield_type=="Real-Time-LaCAM"))

    solution_path = [cur_locs.copy()]
    success = False
    start_time = time.time()
    # Deadlock detection: track BD distances for stuck-agent boosting
    deadlock_window = 30
    num_agents = len(start_locations)
    range_num_agents = np.arange(num_agents)
    bd_dist_snapshot = bd[range_num_agents, cur_locs[:, 0], cur_locs[:, 1]].copy()
    for step in tqdm(range(max_steps)):
        agents_at_goal = np.all(np.equal(cur_locs, goal_locations), axis=1)
        agent_priorities = updatePriorities(agent_priorities, agents_at_goal)

        # Conservative deadlock detection: boost stuck agents every 30 steps
        if step > 0 and step % deadlock_window == 0:
            current_bd_dist = bd[range_num_agents, cur_locs[:, 0], cur_locs[:, 1]]
            stuck = (current_bd_dist >= bd_dist_snapshot) & (~agents_at_goal)
            agent_priorities[stuck] += 5
            bd_dist_snapshot = current_bd_dist.copy()

        if time.time()-start_time > args.timeLimit and args.timeLimit > 0:
            print("time limit hit")
            break
        
        if shield_type in ["CS-PIBT", "CS-Freeze"]:
            action_preferences = getActionPrefsFromLocs(cur_locs) 
            if shield_type == "CS-Freeze":
                action_preferences = action_preferences[:,:2] 
                action_preferences[:,1] = 0  
            timer.start("cs-time")
            new_move, cspibt_worked = pibt(grid_map, action_preferences, cur_locs, agent_priorities, [], start_time, args.timeLimit)
            timer.stop("cs-time")
            if not cspibt_worked:
                if (time.time() - start_time < args.timeLimit):
                    print("ERROR: CS-PIBT failed even though it did not time out! This should never happen!")
                break
        else:
            scaled_lookahead = lacam_lookahead
            next_locs, lacamFoundSolution, numNodesExpanded, numGenerated = lacamRunner.lacam(cur_locs, goal_locations, 
                                                    bd, grid_map, getActionPrefsFromLocs, scaled_lookahead, start_time, args.timeLimit)

            if lacamFoundSolution:
                for t in range(1, len(next_locs)):
                    assert(np.all(grid_map[next_locs[t][:,0], next_locs[t][:,1]] == 0)) 
                print("LaCAM found solution at step: {}".format(step))
                solution_path.extend(next_locs[1:]) 
                success = True
                break
            else:
                if time.time()-start_time > args.timeLimit and args.timeLimit > 0:
                    print("time limit hit")
                    break
                new_move = next_locs[1] - cur_locs 

        cur_locs = cur_locs + new_move 
        solution_path.append(cur_locs.copy())
        assert(np.all(grid_map[cur_locs[:,0], cur_locs[:,1]] == 0)) 
        if len(set(map(tuple, cur_locs))) != len(cur_locs):
            raise RuntimeError("Collision: Two or more agents are at the same location!")

        if np.all(np.equal(cur_locs, goal_locations)):
            success = True
            break
    
    solution_path = np.array(solution_path) 
    total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal = getCosts(solution_path, goal_locations)
    print("Total cache hits: {}, Total size: {}".format(wrapper_nn.hits, len(wrapper_nn.saved_calls)))

    return solution_path, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, success

def main(args: argparse.ArgumentParser):
    torch.set_num_threads(1) 
    k = 4 
    m = 5 
    
    if not os.path.exists(args.mapNpzFile):
        raise FileNotFoundError('Map file: {} not found.'.format(args.mapNpzFile))
    map_npz = np.load(args.mapNpzFile) 
    if args.mapName+".map" not in map_npz:
        raise ValueError('Map name not found in the map file.')
    map_grid = map_npz[args.mapName+".map"] 
    map_grid = np.pad(map_grid, k, 'constant', constant_values=1) 

    if not os.path.exists(args.scenFile):
        raise FileNotFoundError('Scen file: {} not found.'.format(args.scenFile))
    start_locations, goal_locations = parse_scene(args.scenFile) 
    num_agents = args.agentNum 
    if start_locations.shape[0] < num_agents:
        raise ValueError('Not enough agents in the scen file.')
    start_locations = start_locations[:num_agents] + k 
    goal_locations = goal_locations[:num_agents] + k 

    if not os.path.exists(args.bdNpzFile):
        raise FileNotFoundError('BD file: {} not found.'.format(args.bdNpzFile))
    scen_num = args.scenFile.split('-')[-1].split('.')[0]
    bd_key = f"{args.mapName}-random-{scen_num}"
    bd_npz = np.load(args.bdNpzFile)
    if bd_key not in bd_npz:
        raise ValueError('BD key {} not found in the bd file'.format(bd_key))
    
    bd = bd_npz[bd_key][:num_agents].astype(np.float32) 
    bd = np.pad(bd, ((0,0),(k,k),(k,k)), 'constant', constant_values=10000) 

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.useGPU else "cpu") 
    if args.policyType != "pibt" and not os.path.exists(args.modelPath):
        raise FileNotFoundError('Model file: {} not found.'.format(args.modelPath))
    
    if args.policyType in ("flow", "flow_action_head"):
        model = load_flow_model(args, device, k)
    elif args.policyType == "classifier":
        model = load_classifier_model(args, device, k)
    elif args.policyType == "local_classifier":
        model = load_local_classifier_model(args, device, k)
    elif args.policyType == "pibt":
        model = None
    else:
        raise ValueError(f"Unknown policyType: {args.policyType}")
        
    if model is not None:
        model.eval()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.maxSteps.endswith('x'):
        longest_single_path = bd[range(num_agents), start_locations[:,0], start_locations[:,1]].max()
        # --- THE FIX: Wrap the multiplication in int() to drop the float32 typing ---
        max_steps = int(int(args.maxSteps[:-1]) * longest_single_path)
    else:
        max_steps = int(args.maxSteps)
    # Floor: ensure enough steps for congested scenarios
    max_steps = max(max_steps, 400)
        
    if args.debug:
        profiler = cProfile.Profile()
        profiler.enable()
        
    timer = CustomTimer()
    timer.start("total_simulate")
    solution_path, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, success = simulate(device,
            model, k, m, map_grid, bd, start_locations, goal_locations, 
            max_steps, args.shieldType, args.lacamLookahead, args, timer)
    timer.stop("total_simulate")
    total_simulate_time = timer.getTimes("total_simulate")
    print("Success: {}, Total cost true: {}, Total cost not at goal: {}, Num agents at goal: {}/{}, Seconds spent: {}".format(success, 
                                    total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, num_agents, total_simulate_time))
    solution_path = solution_path - k 
    goal_locations = goal_locations - k 
    if args.debug:
        profiler.disable()
        profiler.dump_stats('profile.prof')
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(30) 
    
    if not os.path.exists(args.outputCSVFile):
        with open(args.outputCSVFile, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(['mapName', 'scenFile', 'agentNum', 'seed', 'shieldType', 'lacamLookahead',
                             'modelPath', 'useGPU', 'maxSteps', 
                             'success', 'total_cost_true', 'total_cost_not_resting_at_goal',
                             'num_agents_at_goal', 'runtime', 'create_nn_data', 'forward_pass', 'cs-time'])
            
    with open(args.outputCSVFile, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([args.mapName, args.scenFile, args.agentNum, args.seed, args.shieldType, args.lacamLookahead,
                         args.modelPath, args.useGPU, args.maxSteps,
                         success, total_cost_true, total_cost_not_resting_at_goal, num_agents_at_goal, total_simulate_time,
                         timer.getTimes("create_nn_data"), timer.getTimes("forward_pass"), timer.getTimes("cs-time")])

    if args.outputPathsFile is not None:
        assert(args.outputPathsFile.endswith('.npy'))
        np.save(args.outputPathsFile, solution_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapNpzFile', type=str, required=True)
    parser.add_argument('--mapName', type=str, help="Without .map", required=True)
    parser.add_argument('--scenFile', type=str, required=True)
    parser.add_argument('--agentNum', type=int, required=True)
    parser.add_argument('--bdNpzFile', type=str, required=True)
    parser.add_argument('--debug', type=lambda x: bool(str2bool(x)), help="Whether to enable debugging stats", default=False)
    parser.add_argument('--modelPath', type=str, default='BD-PIBT')
    parser.add_argument('--useGPU', type=lambda x: bool(str2bool(x)), default=False)
    parser.add_argument('--maxSteps', type=str, help="int or [int]x, e.g. 100 or 2x to denote multiplicative factor", required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shieldType', type=str, default='CS-PIBT', choices=['CS-PIBT', 'CS-Freeze', 'LaCAM', 'Real-Time-LaCAM'])
    parser.add_argument('--lacamLookahead', type=int, help="LaCAM node expansion limit", default=0)
    parser.add_argument('--timeLimit', type=int, help="Time limit in seconds (default: 120, matching paper)", default=120)
    parser.add_argument('--outputCSVFile', type=str, help="where to output statistics", required=True)
    parser.add_argument('--outputPathsFile', type=str, help="where to output path, ends with .npy", default=None)
    parser.add_argument('--numIntegrationSteps', type=int, help="Euler integration steps (default 5)", default=5)
    parser.add_argument('--tau', type=float, help="Softmax temperature (default 0.3)", default=0.3)
    parser.add_argument('--waitThreshold', type=float, help="Wait magnitude threshold (default 0.25)", default=0.25)
    parser.add_argument('--waitMode', '--wait-mode', dest='waitMode', type=str,
                        choices=['threshold', 'learned'], default='threshold',
                        help="Wait action scoring mode: threshold keeps the fixed velocity-magnitude wait rule; "
                             "learned uses FlowGNNModel.wait_head for action-0 logit")
    parser.add_argument('--numConsensusSamples', type=int, help="Number of flow samples to average (default 3)", default=3)
    parser.add_argument('--useActionHead', type=lambda x: bool(str2bool(x)), help="Use auxiliary action head instead of flow (default False)", default=False)
    parser.add_argument('--actionHeadConditioning', type=str,
                        choices=['integrated', 'zero_t0', 'zero_t05'], default='integrated',
                        help="How to condition action head: integrated=run flow ODE first (default), zero_t0=v=0 t=0, zero_t05=v=0 t=0.5")
    parser.add_argument('--policyType', '--policy-type', dest='policyType', type=str, choices=['flow', 'classifier', 'flow_action_head', 'local_classifier', 'pibt'], default='flow',
                        help="Policy/model family to load: pibt uses BD-guided preferences without a learned model; flow for FlowGNNModel, classifier for Rishi/SSIL GNNStack, flow_action_head for FlowGNNModel action logits, local_classifier for RishiLikeClassifier")
    parser.add_argument('--hiddenDim', type=int, help="Model hidden dimension (default 1024)", default=1024)
    parser.add_argument('--numLayers', type=int, help="Number of GNN layers (default 6)", default=6)
    parser.add_argument('--classifierLinearDim', type=int, default=-1,
                        help="Classifier GNN linear dimension; <=0 computes the standard SSIL value from k/channels")
    parser.add_argument('--classifierInChannels', type=int, default=3,
                        help="Classifier local patch channels (default: 3)")
    parser.add_argument('--classifierHiddenDim', type=int, default=64,
                        help="Classifier hidden dimension, only used for state_dict checkpoints")
    parser.add_argument('--classifierOutputDim', type=int, default=5,
                        help="Classifier output dimension (default: 5)")
    parser.add_argument('--classifierReluType', type=str, default='relu',
                        help="Classifier activation type, only used for state_dict checkpoints")
    parser.add_argument('--localClassifierHiddenDim', type=int, default=128,
                        help="Local Rishi-like classifier hidden dimension (default: 128)")
    parser.add_argument('--localClassifierNumLayers', type=int, default=3,
                        help="Local Rishi-like classifier SageConv layers (default: 3)")
    parser.add_argument('--localClassifierDropout', type=float, default=0.25,
                        help="Local Rishi-like classifier dropout (default: 0.25)")
    parser.add_argument('--localClassifierInChannels', type=int, default=3,
                        help="Local Rishi-like classifier input channels (default: 3)")
    args = parser.parse_args()

    if args.mapName.endswith('.map'): 
        args.mapName = args.mapName.removesuffix('.map')
    if args.policyType == "flow_action_head":
        args.useActionHead = True
    if args.policyType == "pibt":
        args.useGPU = False
        args.useActionHead = False
    if args.shieldType == "LaCAM" and args.lacamLookahead == 0:
        raise ValueError('LaCAM lookahead must be set when using LaCAM shield type.')
    if args.shieldType == "Real-Time-LaCAM":
        if args.lacamLookahead != 1:
            print("Warning: Real-Time-LaCAM only works with lookahad set to 1, ignoring input {} and setting to 1".format(args.lacamLookahead))
        args.lacamLookahead = 1
        
    main(args)
