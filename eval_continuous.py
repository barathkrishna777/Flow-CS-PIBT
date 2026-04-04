import argparse
import csv
import glob
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from main_pys.continuous_env import load_env_from_files
from main_pys.continuous_scenarios import scenario_id_from_path, select_scenarios
from main_pys.generative_model import FlowGNNModel
from main_pys.transformer_model import FlowTransformerModel
from main_pys.model_inputs import (
    create_continuous_data_object,
    labels_to_direction_vectors,
    normalize_continuous_graph_data,
)


DEFAULT_MAPS = ["empty-48-48", "random-32-32-10"]


def get_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_scenarios(
    scen_dir: str,
    map_name: str,
    max_scenarios: int,
    scenario_ids: Optional[List[int]] = None,
    scenario_start: Optional[int] = None,
    scenario_end: Optional[int] = None,
) -> List[str]:
    return select_scenarios(
        glob.glob(os.path.join(scen_dir, f"{map_name}-random-*.scen")),
        max_scenarios=max_scenarios,
        scenario_ids=scenario_ids,
        scenario_start=scenario_start,
        scenario_end=scenario_end,
    )


def score_candidate_velocities(env, velocities: np.ndarray) -> float:
    clipped = np.asarray(velocities, dtype=np.float32)
    norms = np.linalg.norm(clipped, axis=1)
    clipping_penalty = float(np.mean(np.maximum(norms - env.max_speed, 0.0)))
    clipped = clipped / np.maximum(norms[:, None] / max(env.max_speed, 1e-6), 1.0)

    current_dist = np.linalg.norm(env.goals - env.positions, axis=1)
    next_positions = env.positions + clipped * env.dt
    next_dist = np.linalg.norm(env.goals - next_positions, axis=1)
    progress = float(np.mean(current_dist - next_dist))

    conflict_penalty = 0.0
    min_dist = 2.0 * env.agent_radius
    for i in range(len(next_positions)):
        for j in range(i + 1, len(next_positions)):
            dist = np.linalg.norm(next_positions[i] - next_positions[j])
            if dist < min_dist:
                conflict_penalty += (min_dist - dist) / max(min_dist, 1e-6)
    conflict_penalty /= max(len(next_positions), 1)
    return progress - 0.25 * clipping_penalty - conflict_penalty


def aggregate_flow_samples(candidate_velocities: List[np.ndarray], aggregation: str, env) -> np.ndarray:
    if len(candidate_velocities) == 1:
        return candidate_velocities[0]
    if aggregation == "mean":
        return np.mean(np.stack(candidate_velocities, axis=0), axis=0)
    if aggregation == "medoid":
        flat = [vel.reshape(-1) for vel in candidate_velocities]
        total_dists = []
        for i, src in enumerate(flat):
            total = 0.0
            for j, dst in enumerate(flat):
                if i == j:
                    continue
                total += float(np.linalg.norm(src - dst))
            total_dists.append(total)
        return candidate_velocities[int(np.argmin(total_dists))]
    if aggregation == "best":
        scores = [score_candidate_velocities(env, vel) for vel in candidate_velocities]
        return candidate_velocities[int(np.argmax(scores))]
    raise ValueError(f"Unsupported flow aggregation: {aggregation}")


def load_model(model_path: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "gnn")
    config = checkpoint.get(
        "model_config",
        {
            "k": 4,
            "hidden_dim": 512,
            "num_layers": 4,
            "num_input_channels": 4,
            "aux_feature_dim": 5,
            "action_dim": 9,
            "velocity_dim": 2,
        },
    )
    if model_type == "transformer":
        # FlowTransformerModel accepts num_heads and chunk_horizon
        model = FlowTransformerModel(**config).to(device)
    else:
        # Strip transformer-only keys for GNN compatibility
        gnn_keys = {"k", "hidden_dim", "num_layers", "num_input_channels", "aux_feature_dim", "action_dim", "velocity_dim"}
        gnn_config = {k: v for k, v in config.items() if k in gnn_keys}
        model = FlowGNNModel(**gnn_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint, strict=False)
    model.eval()
    return model, checkpoint


def run_learned_policy(
    model,
    env,
    policy_type: str,
    positions: np.ndarray,
    goals: np.ndarray,
    k: int,
    m: int,
    max_steps: int,
    device: torch.device,
    shield_type: str,
    num_integration_steps: int,
    num_consensus_samples: int,
    tau: float,
    flow_aggregation: str,
) -> Dict[str, float]:
    start_time = time.time()
    env.reset(positions, goals)
    for _ in range(max_steps):
        data = create_continuous_data_object(env.positions, goals, env.obstacle_map, k=k, m=m, max_speed=env.max_speed)
        data = normalize_continuous_graph_data(data, k=k, max_speed=env.max_speed)
        data = data.to(device)
        n_agents = env.positions.shape[0]

        with torch.no_grad():
            if policy_type == "flow":
                dt = 1.0 / max(num_integration_steps, 1)
                candidate_velocities: List[np.ndarray] = []
                for _ in range(max(num_consensus_samples, 1)):
                    v = torch.randn(n_agents, 2, device=device)
                    for step in range(num_integration_steps):
                        t = torch.full((n_agents, 1), step * dt, device=device)
                        flow = model(v, t, data)
                        v = v + flow * dt
                    candidate_velocities.append((v.cpu().numpy() * env.max_speed).astype(np.float32))
                velocities = aggregate_flow_samples(candidate_velocities, aggregation=flow_aggregation, env=env)
            else:
                zero_v = torch.zeros(n_agents, 2, device=device)
                zero_t = torch.zeros(n_agents, 1, device=device)
                _, action_logits = model(zero_v, zero_t, data, return_action_logits=True)
                logits = (action_logits / max(tau, 1e-6)).cpu().numpy()
                labels = logits.argmax(axis=1)
                velocities = labels_to_direction_vectors(labels, num_directions=action_logits.shape[1] - 1) * env.max_speed

        env.step(velocities, shield_type=shield_type)
        if env.is_done():
            break

    metrics = env.current_metrics()
    metrics["runtime"] = time.time() - start_time
    return metrics


def run_orca_baseline(env, positions, goals, max_steps: int, shield_type: str = "orca") -> Dict[str, float]:
    start_time = time.time()
    env.reset(positions, goals)
    for _ in range(max_steps):
        env.step(env.goal_directed_velocities(), shield_type=shield_type)
        if env.is_done():
            break
    metrics = env.current_metrics()
    metrics["runtime"] = time.time() - start_time
    return metrics


def visualize_trajectory(env, output_path: str, title: str) -> None:
    plt = get_pyplot()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    positions = np.asarray(env.history_positions)
    plt.figure(figsize=(6, 6))
    plt.imshow(env.obstacle_map, cmap="Greys", origin="upper")
    for agent in range(positions.shape[1]):
        plt.plot(positions[:, agent, 1], positions[:, agent, 0], linewidth=1.2)
        plt.scatter(positions[0, agent, 1], positions[0, agent, 0], s=10, c="green")
        plt.scatter(positions[-1, agent, 1], positions[-1, agent, 0], s=10, c="red")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def write_rows(output_csv: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "map",
        "scenario",
        "scenario_id",
        "agents",
        "policy",
        "shield_type",
        "run_name",
        "model_name",
        "model_path",
        "train_seed",
        "eval_seed",
        "num_integration_steps",
        "num_consensus_samples",
        "tau",
        "flow_aggregation",
        "success",
        "agents_at_goal",
        "agent_fraction_at_goal",
        "path_length",
        "path_length_ratio",
        "smoothness",
        "collisions",
        "near_collisions",
        "obstacle_hits",
        "mean_arrival_step",
        "runtime",
    ]
    write_header = not os.path.exists(output_csv)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate continuous MAPF policies")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--scen-dir", required=True)
    parser.add_argument("--maps", nargs="*", default=DEFAULT_MAPS)
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--max-scenarios", type=int, default=1)
    parser.add_argument("--scenario-ids", nargs="*", type=int, default=None)
    parser.add_argument("--scenario-start", type=int, default=None)
    parser.add_argument("--scenario-end", type=int, default=None)
    parser.add_argument("--policy", choices=["flow", "discrete", "orca"], default="orca")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--viz-dir", default=None)
    parser.add_argument("--shield-type", choices=["orca", "heuristic-orca", "po-orca", "epibt", "simple", "none"], default="orca")
    parser.add_argument("--num-integration-steps", type=int, default=3)
    parser.add_argument("--num-consensus-samples", type=int, default=1)
    parser.add_argument("--flow-aggregation", choices=["mean", "medoid", "best"], default="mean")
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.3)
    parser.add_argument("--goal-tolerance", type=float, default=0.25)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = None
    if args.policy != "orca":
        if not args.model_path:
            raise ValueError("--model-path is required for learned policies")
        model, _ = load_model(args.model_path, device)

    rows = []
    for map_name in args.maps:
        map_file = os.path.join(args.map_dir, f"{map_name}.map")
        scenarios = find_scenarios(
            args.scen_dir,
            map_name,
            args.max_scenarios,
            scenario_ids=args.scenario_ids,
            scenario_start=args.scenario_start,
            scenario_end=args.scenario_end,
        )
        for scen_file in scenarios:
            scen_name = os.path.basename(scen_file).replace(".scen", "")
            scen_id = scenario_id_from_path(scen_file)
            for agent_num in args.agent_counts:
                env, starts, goals = load_env_from_files(
                    map_file,
                    scen_file,
                    agent_num=agent_num,
                    dt=args.dt,
                    max_speed=args.max_speed,
                    agent_radius=args.agent_radius,
                    goal_tolerance=args.goal_tolerance,
                )
                if args.policy == "orca":
                    metrics = run_orca_baseline(env, starts, goals, args.max_steps, shield_type=args.shield_type)
                else:
                    metrics = run_learned_policy(
                        model,
                        env,
                        args.policy,
                        starts,
                        goals,
                        args.k,
                        args.m,
                        args.max_steps,
                        device,
                        args.shield_type,
                        args.num_integration_steps,
                        args.num_consensus_samples,
                        args.tau,
                        args.flow_aggregation,
                    )
                row = {
                    "map": map_name,
                    "scenario": scen_name,
                    "scenario_id": scen_id,
                    "agents": agent_num,
                    "policy": args.policy,
                    "shield_type": args.shield_type,
                    "run_name": args.run_name,
                    "model_name": os.path.basename(args.model_path) if args.model_path else "",
                    "model_path": args.model_path or "",
                    "train_seed": "" if args.train_seed is None else args.train_seed,
                    "eval_seed": args.eval_seed,
                    "num_integration_steps": args.num_integration_steps,
                    "num_consensus_samples": args.num_consensus_samples,
                    "tau": args.tau,
                    "flow_aggregation": args.flow_aggregation if args.policy == "flow" else "",
                    **metrics,
                }
                rows.append(row)
                print(f"{map_name} {scen_name} N={agent_num} {args.policy}: success={metrics['success']:.0f} at_goal={metrics['agent_fraction_at_goal']:.3f}")

                if args.viz_dir:
                    out_png = os.path.join(args.viz_dir, f"{map_name}_{scen_name}_{agent_num}_{args.policy}.png")
                    visualize_trajectory(env, out_png, f"{map_name} | {scen_name} | N={agent_num} | {args.policy}")
    write_rows(args.output_csv, rows)


if __name__ == "__main__":
    main()
