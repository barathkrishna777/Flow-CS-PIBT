"""
Diagnostic: Trace the full pipeline from model velocity → discrete action.
For each agent at each step, log:
  1. Predicted velocity vector
  2. Dot-product scores against 5 action vectors
  3. Softmax probabilities (with tau=0.5)
  4. Which action CS-PIBT actually chose
  5. What the "correct" action would be (from expert velocity in training data)

This isolates exactly where signal is lost.
"""
import torch
import numpy as np
import os
from collections import Counter

from main_pys.generative_model import FlowGNNModel
from main_pys.model_inputs import create_data_object, normalize_graph_data
from main_pys.simulator import LABEL_TO_MOVES

ACTION_NAMES = ["wait", "right", "down", "up", "left"]
ACTION_VECTORS = np.array([[0,0], [0,1], [1,0], [-1,0], [0,-1]], dtype=float)


def velocity_to_best_action(velocity):
    """What discrete action best matches this velocity?"""
    scores = velocity @ ACTION_VECTORS.T
    return np.argmax(scores)


def expert_velocity_to_action(vel):
    """Convert expert continuous velocity to the best discrete action."""
    speed = np.linalg.norm(vel)
    if speed < 0.05:
        return 0  # wait
    return velocity_to_best_action(vel)


def analyze_tau_sweep(all_velocities, all_expert_actions):
    """Test different temperature values to see which gives best action accuracy."""
    print("\n" + "=" * 60)
    print("TEMPERATURE SWEEP: How does tau affect action accuracy?")
    print("=" * 60)

    taus = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]

    for tau in taus:
        correct = 0
        total = 0
        for vel, expert_act in zip(all_velocities, all_expert_actions):
            scores = vel @ ACTION_VECTORS.T
            scores = scores / tau
            scores = scores - np.max(scores)
            probs = np.exp(scores) / np.sum(np.exp(scores))
            predicted_act = np.argmax(probs)
            if predicted_act == expert_act:
                correct += 1
            total += 1
        acc = correct / total * 100
        print(f"  tau={tau:<5.2f} -> Accuracy: {correct}/{total} = {acc:.1f}%")


def main():
    device = torch.device("cpu")

    # Load model
    model = FlowGNNModel(k=4).to(device)
    model.load_state_dict(
        torch.load("direct_overfit_best.pt", map_location=device, weights_only=True),
        strict=False,
    )
    model.eval()

    # Load training data (the same file we overfit on)
    npz_path = os.path.join("data", "flow_training_data_multi", "empty-32-32-random-10_20.npz")
    with np.load(npz_path) as data:
        discrete_positions = data['discrete_positions']
        expert_velocities = data['expert_velocities']

    # Load map
    map_name = "empty-32-32"
    map_file = os.path.join("data", "mapf-map", f"{map_name}.map")
    k = 4
    m = 5
    with open(map_file, 'r') as f:
        f.readline()
        height = int(f.readline().split()[1])
        width = int(f.readline().split()[1])
        f.readline()
        grid_map = np.zeros((height, width), dtype=int)
        for r in range(height):
            line = f.readline().strip()
            for c in range(width):
                if line[c] in ['@', 'T', 'O']:
                    grid_map[r, c] = 1
    grid_map = np.pad(grid_map, k, 'constant', constant_values=1)

    # Load BD
    bd_file = os.path.join("data", "bd_npzs", "large_scale", "empty-32-32-random-10_bds.npz")
    bd_key = "empty-32-32-random-10"
    bd = np.load(bd_file)[bd_key][:discrete_positions.shape[0]].astype(np.float32)
    bd = np.pad(bd, ((0, 0), (k, k), (k, k)), 'constant', constant_values=10000)

    num_agents = discrete_positions.shape[0]
    num_timesteps = discrete_positions.shape[1]
    dummy_goals = np.zeros((num_agents, 2), dtype=int)

    print(f"Agents: {num_agents}, Timesteps: {num_timesteps}")
    print(f"Analyzing action mapping pipeline...\n")

    all_predicted_velocities = []
    all_expert_actions = []
    all_predicted_actions = []
    all_expert_velocities_flat = []

    dot_product_correct = 0
    softmax_correct = 0
    total = 0

    # Analyze a few timesteps in detail
    detail_steps = [0, 5, 10, 20, 30]

    for t_step in range(num_timesteps):
        cur_locs = discrete_positions[:, t_step, :].astype(float)
        cur_locs_discrete = (np.round(cur_locs) + k).astype(int)
        max_r = grid_map.shape[0] - k - 1
        max_c = grid_map.shape[1] - k - 1
        cur_locs_discrete[:, 0] = np.clip(cur_locs_discrete[:, 0], k, max_r)
        cur_locs_discrete[:, 1] = np.clip(cur_locs_discrete[:, 1], k, max_c)

        expert_vel = expert_velocities[:, t_step, :]

        # Forward pass
        with torch.no_grad():
            data = create_data_object(cur_locs_discrete, bd, grid_map, k, m, dummy_goals)
            data = normalize_graph_data(data, k)
            v_t = torch.zeros(num_agents, 2)
            t_input = torch.zeros(num_agents, 1)
            pred_vel = model(v_t, t_input, data).numpy()

        for agent_id in range(num_agents):
            pv = pred_vel[agent_id]
            ev = expert_vel[agent_id]

            expert_act = expert_velocity_to_action(ev)

            # Raw dot product → best action
            raw_scores = pv @ ACTION_VECTORS.T
            dot_best = np.argmax(raw_scores)

            # Softmax with tau=0.5
            tau = 0.5
            scaled = raw_scores / tau
            scaled = scaled - np.max(scaled)
            probs = np.exp(scaled) / np.sum(np.exp(scaled))
            softmax_best = np.argmax(probs)

            if dot_best == expert_act:
                dot_product_correct += 1
            if softmax_best == expert_act:
                softmax_correct += 1
            total += 1

            all_predicted_velocities.append(pv)
            all_expert_actions.append(expert_act)
            all_predicted_actions.append(softmax_best)
            all_expert_velocities_flat.append(ev)

        # Detailed log for selected timesteps
        if t_step in detail_steps:
            print(f"--- Timestep {t_step} ---")
            for agent_id in range(min(5, num_agents)):  # first 5 agents
                pv = pred_vel[agent_id]
                ev = expert_vel[agent_id]
                expert_act = expert_velocity_to_action(ev)
                raw_scores = pv @ ACTION_VECTORS.T
                tau = 0.5
                scaled = raw_scores / tau
                scaled = scaled - np.max(scaled)
                probs = np.exp(scaled) / np.sum(np.exp(scaled))

                print(f"  Agent {agent_id}:")
                print(f"    Expert vel:    ({ev[0]:+.3f}, {ev[1]:+.3f}) -> {ACTION_NAMES[expert_act]}")
                print(f"    Predicted vel: ({pv[0]:+.3f}, {pv[1]:+.3f})")
                print(f"    Raw scores:    {dict(zip(ACTION_NAMES, [f'{s:.3f}' for s in raw_scores]))}")
                print(f"    Softmax probs: {dict(zip(ACTION_NAMES, [f'{p:.3f}' for p in probs]))}")
                print(f"    Argmax action: {ACTION_NAMES[np.argmax(probs)]}")
                print(f"    Match: {'YES' if np.argmax(probs) == expert_act else 'NO'}")
            print()

    print("=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(f"Total agent-timestep pairs: {total}")
    print(f"Dot-product argmax accuracy: {dot_product_correct}/{total} = {dot_product_correct/total*100:.1f}%")
    print(f"Softmax(tau=0.5) argmax accuracy: {softmax_correct}/{total} = {softmax_correct/total*100:.1f}%")

    # Action distribution analysis
    expert_dist = Counter(all_expert_actions)
    pred_dist = Counter(all_predicted_actions)
    print(f"\nExpert action distribution:")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  {name}: {expert_dist.get(i, 0)} ({expert_dist.get(i, 0)/total*100:.1f}%)")
    print(f"Predicted action distribution:")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  {name}: {pred_dist.get(i, 0)} ({pred_dist.get(i, 0)/total*100:.1f}%)")

    # Velocity magnitude analysis
    pred_mags = [np.linalg.norm(v) for v in all_predicted_velocities]
    expert_mags = [np.linalg.norm(v) for v in all_expert_velocities_flat]
    print(f"\nVelocity magnitude stats:")
    print(f"  Expert:    mean={np.mean(expert_mags):.3f}, std={np.std(expert_mags):.3f}, "
          f"min={np.min(expert_mags):.3f}, max={np.max(expert_mags):.3f}")
    print(f"  Predicted: mean={np.mean(pred_mags):.3f}, std={np.std(pred_mags):.3f}, "
          f"min={np.min(pred_mags):.3f}, max={np.max(pred_mags):.3f}")

    # Wait action analysis
    expert_waits = [i for i, a in enumerate(all_expert_actions) if a == 0]
    pred_at_waits = [all_predicted_actions[i] for i in expert_waits]
    wait_correct = sum(1 for a in pred_at_waits if a == 0)
    print(f"\nWait action analysis:")
    print(f"  Expert says 'wait': {len(expert_waits)} times")
    print(f"  Model predicts 'wait' when expert says 'wait': {wait_correct}/{len(expert_waits)}")
    if expert_waits:
        wait_pred_dist = Counter(pred_at_waits)
        print(f"  What model predicts instead: {dict((ACTION_NAMES[k], v) for k, v in wait_pred_dist.items())}")

    # Tau sweep
    analyze_tau_sweep(all_predicted_velocities, all_expert_actions)


if __name__ == "__main__":
    main()
