#!/usr/bin/env python3
"""Live matplotlib visualization of continuous MAPF inference."""

import argparse
import glob
import math
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from main_pys.continuous_env import (
    ContinuousMAPFEnv,
    grid_starts_to_continuous,
    parse_scene_file,
)
from main_pys.continuous_scenarios import scenario_id_from_path, select_scenarios
from main_pys.generative_model import FlowGNNModel
from main_pys.transformer_model import FlowTransformerModel
from main_pys.model_inputs import (
    create_continuous_data_object,
    labels_to_direction_vectors,
    load_grid_map_from_file,
    normalize_continuous_graph_data,
)

DEFAULT_MAPS = ["empty-48-48", "random-32-32-10"]
TRAIL_LENGTH = 10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        model = FlowTransformerModel(**config).to(device)
    else:
        gnn_keys = {"k", "hidden_dim", "num_layers", "num_input_channels", "aux_feature_dim", "action_dim", "velocity_dim"}
        gnn_config = {k: v for k, v in config.items() if k in gnn_keys}
        model = FlowGNNModel(**gnn_config).to(device)
    model.load_state_dict(
        checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint,
        strict=False,
    )
    model.eval()
    return model, checkpoint


def aggregate_flow_samples(candidate_velocities: List[np.ndarray], env) -> np.ndarray:
    if len(candidate_velocities) == 1:
        return candidate_velocities[0]
    return np.mean(np.stack(candidate_velocities, axis=0), axis=0)


def find_scenarios(
    scen_dir: str,
    map_name: str,
    max_scenarios: int,
    scenario_ids: Optional[List[int]] = None,
) -> List[str]:
    return select_scenarios(
        glob.glob(os.path.join(scen_dir, f"{map_name}-random-*.scen")),
        max_scenarios=max_scenarios,
        scenario_ids=scenario_ids,
    )


def setup_plot(obstacle_map, map_name, scen_name, n_agents, save_frames):
    """Create figure and axes, draw obstacle background. Returns (fig, ax)."""
    h, w = obstacle_map.shape
    aspect = w / h
    fig_h = 8
    fig, ax = plt.subplots(1, 1, figsize=(fig_h * aspect, fig_h))
    ax.imshow(obstacle_map, cmap="Greys", origin="upper", extent=[0, w, h, 0], alpha=0.4)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    if not save_frames:
        plt.ion()
        plt.show(block=False)
    return fig, ax


def draw_frame(
    ax,
    fig,
    positions,
    goals,
    velocities,
    at_goal,
    colliding,
    trail_history,
    obstacle_map,
    step_idx,
    max_steps,
    frac_at_goal,
    collision_count,
    map_name,
    scen_name,
    n_agents,
    save_frames,
    frame_dir,
    agent_radius,
):
    ax.clear()
    h, w = obstacle_map.shape
    ax.imshow(obstacle_map, cmap="Greys", origin="upper", extent=[0, w, h, 0], alpha=0.4)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")

    n = len(positions)
    colors = np.full(n, "tab:blue", dtype=object)
    colors[at_goal] = "tab:green"
    colors[colliding] = "tab:red"

    # Trails
    if len(trail_history) > 1:
        trail = np.array(trail_history)  # (T, N, 2)
        for i in range(n):
            alpha_vals = np.linspace(0.1, 0.4, len(trail))
            for t in range(len(trail) - 1):
                ax.plot(
                    [trail[t, i, 1], trail[t + 1, i, 1]],
                    [trail[t, i, 0], trail[t + 1, i, 0]],
                    color=colors[i],
                    alpha=float(alpha_vals[t]),
                    linewidth=0.5,
                )

    # Goals
    ax.scatter(goals[:, 1], goals[:, 0], marker="x", s=20, c="grey", linewidths=0.8, zorder=2)

    # Agents
    ax.scatter(positions[:, 1], positions[:, 0], s=15, c=list(colors), zorder=4, edgecolors="k", linewidths=0.3)

    # Velocity arrows
    if velocities is not None:
        norms = np.linalg.norm(velocities, axis=1)
        moving = norms > 0.05
        if np.any(moving):
            ax.quiver(
                positions[moving, 1],
                positions[moving, 0],
                velocities[moving, 1],
                velocities[moving, 0],
                angles="xy",
                scale_units="xy",
                scale=3.0,
                width=0.003,
                color="tab:orange",
                alpha=0.7,
                zorder=3,
            )

    ax.set_title(
        f"{map_name} | {scen_name} | N={n_agents} | "
        f"step {step_idx}/{max_steps} | "
        f"at_goal={frac_at_goal:.1%} | "
        f"collisions={collision_count}",
        fontsize=10,
    )

    if save_frames:
        os.makedirs(frame_dir, exist_ok=True)
        fig.savefig(os.path.join(frame_dir, f"frame_{step_idx:04d}.png"), dpi=100, bbox_inches="tight")
    else:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.01)


def run_visualized(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = None
    if args.policy != "orca":
        if not args.model_path:
            raise ValueError("--model-path is required for learned policies")
        model, _ = load_model(args.model_path, device)

    max_agent_num = max(args.agent_counts)

    for map_name in args.maps:
        map_file = os.path.join(args.map_dir, f"{map_name}.map")
        obstacle_map = load_grid_map_from_file(map_file)
        scenarios = find_scenarios(args.scen_dir, map_name, args.max_scenarios, scenario_ids=args.scenario_ids)
        print(f"[viz] map={map_name} shape={obstacle_map.shape} scenarios={len(scenarios)}")

        for scen_file in scenarios:
            scen_name = os.path.basename(scen_file).replace(".scen", "")
            starts_all, goals_all = parse_scene_file(scen_file, agent_num=max_agent_num)
            starts_all = grid_starts_to_continuous(starts_all)
            goals_all = grid_starts_to_continuous(goals_all)

            for agent_num in args.agent_counts:
                starts = starts_all[:agent_num]
                goals = goals_all[:agent_num]

                env = ContinuousMAPFEnv(
                    obstacle_map=obstacle_map,
                    dt=args.dt,
                    max_speed=args.max_speed,
                    agent_radius=args.agent_radius,
                    goal_tolerance=args.goal_tolerance,
                )
                env.reset(starts, goals)

                frame_dir = ""
                if args.save_frames:
                    frame_dir = os.path.join(
                        args.save_frames, f"{map_name}_{scen_name}_{agent_num}"
                    )

                fig, ax = setup_plot(obstacle_map, map_name, scen_name, agent_num, args.save_frames)
                trail_history = [env.positions.copy()]

                print(
                    f"[viz] running {map_name} | {scen_name} | N={agent_num} | "
                    f"policy={args.policy} shield={args.shield_type}"
                )

                last_velocities = None
                for step_idx in range(args.max_steps):
                    # Compute velocities
                    if args.policy == "orca":
                        if args.nav == "bd":
                            velocities = env.bd_guided_velocities()
                        else:
                            velocities = env.goal_directed_velocities()
                    else:
                        data = create_continuous_data_object(
                            env.positions, goals, env.obstacle_map,
                            k=args.k, m=args.m, max_speed=env.max_speed,
                        )
                        data = normalize_continuous_graph_data(data, k=args.k, max_speed=env.max_speed)
                        data = data.to(device)
                        n_agents = env.positions.shape[0]

                        with torch.no_grad():
                            if args.policy == "flow":
                                dt_flow = 1.0 / max(args.num_integration_steps, 1)
                                candidate_velocities = []
                                for _ in range(max(args.num_consensus_samples, 1)):
                                    v = torch.randn(n_agents, 2, device=device)
                                    for s in range(args.num_integration_steps):
                                        t = torch.full((n_agents, 1), s * dt_flow, device=device)
                                        flow = model(v, t, data)
                                        v = v + flow * dt_flow
                                    candidate_velocities.append((v.cpu().numpy() * env.max_speed).astype(np.float32))
                                velocities = aggregate_flow_samples(candidate_velocities, env)
                            else:
                                zero_v = torch.zeros(n_agents, 2, device=device)
                                zero_t = torch.zeros(n_agents, 1, device=device)
                                _, action_logits = model(zero_v, zero_t, data, return_action_logits=True)
                                logits = (action_logits / max(args.tau, 1e-6)).cpu().numpy()
                                labels = logits.argmax(axis=1)
                                velocities = labels_to_direction_vectors(labels, num_directions=action_logits.shape[1] - 1) * env.max_speed

                    env.step(velocities, shield_type=args.shield_type)
                    last_velocities = env.history_velocities[-1] if env.history_velocities else None

                    # Update trail
                    trail_history.append(env.positions.copy())
                    if len(trail_history) > TRAIL_LENGTH:
                        trail_history.pop(0)

                    # Determine collision status for coloring
                    at_goal = env.agents_at_goal()
                    colliding = np.zeros(agent_num, dtype=bool)
                    min_dist = 2.0 * env.agent_radius
                    for i in range(agent_num):
                        for j in range(i + 1, agent_num):
                            if np.linalg.norm(env.positions[i] - env.positions[j]) < min_dist:
                                colliding[i] = True
                                colliding[j] = True

                    frac_at_goal = float(np.mean(at_goal))
                    draw_frame(
                        ax, fig,
                        env.positions, goals, last_velocities,
                        at_goal, colliding, trail_history,
                        obstacle_map,
                        step_idx + 1, args.max_steps,
                        frac_at_goal, env.metrics.collisions,
                        map_name, scen_name, agent_num,
                        args.save_frames, frame_dir,
                        env.agent_radius,
                    )

                    if env.is_done():
                        print(f"[viz] done at step {step_idx + 1}, all agents at goal")
                        break

                metrics = env.current_metrics()
                print(
                    f"[viz] finished: at_goal={metrics['agent_fraction_at_goal']:.3f} "
                    f"collisions={metrics['collisions']:.0f} "
                    f"obstacle_hits={metrics['obstacle_hits']:.0f}"
                )

                if not args.save_frames:
                    print("[viz] press any key in the plot window to continue (or close it)...")
                    try:
                        plt.waitforbuttonpress()
                    except Exception:
                        pass
                plt.close(fig)

    print("[viz] all scenarios complete")


def main():
    parser = argparse.ArgumentParser(description="Live visualization of continuous MAPF inference")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--scen-dir", required=True)
    parser.add_argument("--maps", nargs="*", default=DEFAULT_MAPS)
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[100])
    parser.add_argument("--max-scenarios", type=int, default=1)
    parser.add_argument("--scenario-ids", nargs="*", type=int, default=None)
    parser.add_argument("--policy", choices=["flow", "discrete", "orca"], default="orca")
    parser.add_argument("--nav", choices=["straight", "bd"], default="straight")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--shield-type",
                        choices=["orca", "heuristic-orca", "po-orca", "epibt", "picbf-cs", "simple", "none"],
                        default="orca")
    parser.add_argument("--num-integration-steps", type=int, default=3)
    parser.add_argument("--num-consensus-samples", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.3)
    parser.add_argument("--goal-tolerance", type=float, default=0.25)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--save-frames", default=None,
                        help="Directory to save PNG frames instead of live display")
    parser.add_argument("--backend", default=None,
                        help="Matplotlib backend (e.g. TkAgg, Qt5Agg). Auto-detected if omitted.")
    args = parser.parse_args()

    global plt
    import matplotlib
    if args.save_frames:
        matplotlib.use("Agg")
    elif args.backend:
        matplotlib.use(args.backend)
    import matplotlib.pyplot as plt_module
    plt = plt_module

    set_seed(args.eval_seed)
    run_visualized(args)


if __name__ == "__main__":
    main()
