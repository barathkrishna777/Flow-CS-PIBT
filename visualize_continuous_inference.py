#!/usr/bin/env python3
"""Live matplotlib visualization of continuous MAPF inference.

Dark-themed, publication-quality renderer with:
  - True-radius agent circles colored per-agent from a perceptual colormap
  - Status glow rings (green=at-goal, red=collision, amber=near-goal)
  - Gradient alpha trails via LineCollection
  - Faint goal connectors (dashed agent-to-goal lines)
  - Side panel: animated progress bar, live counters, speed histogram
  - Obstacle map with subtle styling
"""

import argparse
import glob
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

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
TRAIL_LENGTH = 15

# ── Color palette (dark theme) ──────────────────────────────────────────
BG_COLOR = "#0f0f1a"
PANEL_BG = "#161625"
OBSTACLE_COLOR = "#2a2a3d"
FREE_COLOR = "#12121f"
GRID_COLOR = "#1e1e30"
TEXT_COLOR = "#c8c8d8"
TEXT_DIM = "#6a6a80"
ACCENT = "#6c63ff"
ACCENT_BRIGHT = "#8b83ff"
GREEN_GLOW = "#00e676"
RED_GLOW = "#ff1744"
AMBER_GLOW = "#ffc400"
GOAL_COLOR = "#ffffff"
CONNECTOR_COLOR = "#ffffff"
ARROW_COLOR = "#ffab40"
PROGRESS_BG = "#1e1e30"
PROGRESS_FILL = "#6c63ff"
PROGRESS_FILL_DONE = "#00e676"


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
            "k": 4, "hidden_dim": 512, "num_layers": 4,
            "num_input_channels": 4, "aux_feature_dim": 5,
            "action_dim": 9, "velocity_dim": 2,
        },
    )
    if model_type == "transformer":
        model = FlowTransformerModel(**config).to(device)
    else:
        gnn_keys = {"k", "hidden_dim", "num_layers", "num_input_channels",
                     "aux_feature_dim", "action_dim", "velocity_dim"}
        gnn_config = {k: v for k, v in config.items() if k in gnn_keys}
        model = FlowGNNModel(**gnn_config).to(device)
    model.load_state_dict(
        checkpoint.get("model_state_dict", checkpoint), strict=False,
    )
    model.eval()
    return model, checkpoint


def aggregate_flow_samples(candidates: List[np.ndarray], env) -> np.ndarray:
    if len(candidates) == 1:
        return candidates[0]
    return np.mean(np.stack(candidates, axis=0), axis=0)


def find_scenarios(scen_dir, map_name, max_scenarios, scenario_ids=None):
    return select_scenarios(
        glob.glob(os.path.join(scen_dir, f"{map_name}-random-*.scen")),
        max_scenarios=max_scenarios, scenario_ids=scenario_ids,
    )


# ── Obstacle map rendering ──────────────────────────────────────────────

def render_obstacle_image(obstacle_map: np.ndarray) -> np.ndarray:
    """RGBA image: obstacles are dark raised blocks, free space is darker."""
    h, w = obstacle_map.shape
    img = np.zeros((h, w, 4), dtype=np.float32)
    free = obstacle_map == 0
    obs = ~free

    # Free space
    img[free] = _hex_to_rgba(FREE_COLOR, 1.0)
    # Obstacles with subtle highlight
    base = np.array(_hex_to_rgba(OBSTACLE_COLOR, 1.0))
    img[obs] = base

    # Top/left edge highlight on obstacle cells for a raised look
    for r in range(h):
        for c in range(w):
            if obstacle_map[r, c] == 0:
                continue
            above_free = r == 0 or obstacle_map[r - 1, c] == 0
            left_free = c == 0 or obstacle_map[r, c - 1] == 0
            if above_free or left_free:
                img[r, c, :3] = np.clip(base[:3] + 0.06, 0, 1)

    return img


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> Tuple[float, float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, alpha)


def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255)


# ── Figure setup ─────────────────────────────────────────────────────────

def setup_figure(obstacle_map, map_name, scen_name, n_agents, policy, shield, save_frames):
    from matplotlib.gridspec import GridSpec

    h, w = obstacle_map.shape
    map_aspect = w / h
    fig_h = 9
    map_w = fig_h * map_aspect
    panel_w = 3.0
    fig = plt.figure(figsize=(map_w + panel_w, fig_h), facecolor=BG_COLOR)

    gs = GridSpec(1, 2, width_ratios=[map_w, panel_w], wspace=0.02, figure=fig)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_panel = fig.add_subplot(gs[0, 1])

    # Map axes
    ax_map.set_facecolor(BG_COLOR)
    obs_img = render_obstacle_image(obstacle_map)
    ax_map.imshow(obs_img, origin="upper", extent=[0, w, h, 0], interpolation="nearest")
    ax_map.set_xlim(-0.2, w + 0.2)
    ax_map.set_ylim(h + 0.2, -0.2)
    ax_map.set_aspect("equal")
    ax_map.tick_params(colors=TEXT_DIM, labelsize=6)
    for spine in ax_map.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.5)

    # Panel axes
    ax_panel.set_facecolor(PANEL_BG)
    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)
    ax_panel.set_xticks([])
    ax_panel.set_yticks([])
    for spine in ax_panel.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.5)

    # Title
    fig.text(
        0.01, 0.97,
        f"{map_name}",
        fontsize=14, fontweight="bold", color=TEXT_COLOR,
        fontfamily="monospace", va="top",
    )
    fig.text(
        0.01, 0.935,
        f"{scen_name}  |  N={n_agents}  |  {policy}+{shield}",
        fontsize=9, color=TEXT_DIM, fontfamily="monospace", va="top",
    )

    if not save_frames:
        plt.ion()
        plt.show(block=False)

    return fig, ax_map, ax_panel


# ── Side panel drawing ───────────────────────────────────────────────────

def draw_panel(
    ax, step_idx, max_steps, frac_at_goal, n_at_goal, n_agents,
    collisions, near_collisions, obstacle_hits,
    speeds, elapsed, agent_radius,
):
    from matplotlib.patches import FancyBboxPatch
    ax.clear()
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.5)

    mx, mw = 0.08, 0.84  # margin x, bar width
    y = 0.94

    # ── Step counter ──
    ax.text(mx, y, "STEP", fontsize=7, color=TEXT_DIM, fontfamily="monospace",
            transform=ax.transAxes, va="top")
    ax.text(mx + mw, y, f"{step_idx}/{max_steps}", fontsize=11, color=TEXT_COLOR,
            fontfamily="monospace", transform=ax.transAxes, va="top", ha="right",
            fontweight="bold")
    y -= 0.04
    step_frac = min(step_idx / max(max_steps, 1), 1.0)
    _draw_bar(ax, mx, y, mw, 0.018, step_frac, ACCENT, PROGRESS_BG)

    # ── Time ──
    y -= 0.06
    ax.text(mx, y, "ELAPSED", fontsize=7, color=TEXT_DIM, fontfamily="monospace",
            transform=ax.transAxes, va="top")
    ax.text(mx + mw, y, f"{elapsed:.1f}s", fontsize=10, color=TEXT_COLOR,
            fontfamily="monospace", transform=ax.transAxes, va="top", ha="right")

    # ── Progress (at goal) ──
    y -= 0.08
    ax.text(mx, y, "AT GOAL", fontsize=7, color=TEXT_DIM, fontfamily="monospace",
            transform=ax.transAxes, va="top")
    pct_str = f"{frac_at_goal:.1%}"
    count_str = f"{n_at_goal}/{n_agents}"
    bar_color = PROGRESS_FILL_DONE if frac_at_goal >= 1.0 else GREEN_GLOW if frac_at_goal > 0.8 else ACCENT
    ax.text(mx + mw, y, f"{pct_str}  ({count_str})", fontsize=10,
            color=bar_color, fontfamily="monospace", transform=ax.transAxes,
            va="top", ha="right", fontweight="bold")
    y -= 0.04
    _draw_bar(ax, mx, y, mw, 0.022, frac_at_goal, bar_color, PROGRESS_BG)

    # ── Collision stats ──
    y -= 0.09
    ax.text(mx, y, "COLLISIONS", fontsize=7, color=TEXT_DIM, fontfamily="monospace",
            transform=ax.transAxes, va="top")
    col_color = RED_GLOW if collisions > 0 else TEXT_DIM
    ax.text(mx + mw, y, f"{collisions}", fontsize=13, color=col_color,
            fontfamily="monospace", transform=ax.transAxes, va="top", ha="right",
            fontweight="bold")

    y -= 0.065
    ax.text(mx, y, "NEAR MISS", fontsize=7, color=TEXT_DIM, fontfamily="monospace",
            transform=ax.transAxes, va="top")
    nm_color = AMBER_GLOW if near_collisions > 0 else TEXT_DIM
    ax.text(mx + mw, y, f"{near_collisions}", fontsize=11, color=nm_color,
            fontfamily="monospace", transform=ax.transAxes, va="top", ha="right")

    y -= 0.06
    ax.text(mx, y, "OBS HITS", fontsize=7, color=TEXT_DIM, fontfamily="monospace",
            transform=ax.transAxes, va="top")
    oh_color = RED_GLOW if obstacle_hits > 0 else TEXT_DIM
    ax.text(mx + mw, y, f"{obstacle_hits}", fontsize=11, color=oh_color,
            fontfamily="monospace", transform=ax.transAxes, va="top", ha="right")

    # ── Speed distribution mini-histogram ──
    y -= 0.08
    ax.text(mx, y, "SPEED DIST", fontsize=7, color=TEXT_DIM, fontfamily="monospace",
            transform=ax.transAxes, va="top")
    if speeds is not None and len(speeds) > 0:
        _draw_histogram(ax, mx, y - 0.04, mw, 0.12, speeds)

    # ── Legend ──
    y_legend = 0.13
    ax.text(mx, y_legend, "LEGEND", fontsize=7, color=TEXT_DIM, fontfamily="monospace",
            transform=ax.transAxes, va="top")
    y_legend -= 0.04
    _legend_dot(ax, mx + 0.02, y_legend, GREEN_GLOW, "At goal")
    y_legend -= 0.035
    _legend_dot(ax, mx + 0.02, y_legend, ACCENT_BRIGHT, "Moving")
    y_legend -= 0.035
    _legend_dot(ax, mx + 0.02, y_legend, RED_GLOW, "Collision")


def _draw_bar(ax, x, y, w, h, frac, fill_color, bg_color):
    from matplotlib.patches import FancyBboxPatch
    bg = FancyBboxPatch((x, y - h), w, h,
                        boxstyle="round,pad=0.003", facecolor=bg_color,
                        edgecolor="none", transform=ax.transAxes, zorder=2)
    ax.add_patch(bg)
    if frac > 0.005:
        bar = FancyBboxPatch((x, y - h), w * min(frac, 1.0), h,
                             boxstyle="round,pad=0.003", facecolor=fill_color,
                             edgecolor="none", alpha=0.85,
                             transform=ax.transAxes, zorder=3)
        ax.add_patch(bar)


def _draw_histogram(ax, x, y, w, h, speeds):
    max_speed_val = max(np.max(speeds), 0.01)
    n_bins = 12
    counts, edges = np.histogram(speeds, bins=n_bins, range=(0, max_speed_val))
    max_count = max(np.max(counts), 1)
    bar_w = w / n_bins * 0.85
    for i, count in enumerate(counts):
        bx = x + (w / n_bins) * i
        bh = h * (count / max_count)
        frac = edges[i] / max_speed_val
        color = _lerp_color(_hex_to_rgb(ACCENT), _hex_to_rgb(ARROW_COLOR), frac)
        from matplotlib.patches import Rectangle
        rect = Rectangle((bx, y - bh), bar_w, bh,
                          facecolor=color, edgecolor="none", alpha=0.8,
                          transform=ax.transAxes, zorder=3)
        ax.add_patch(rect)


def _legend_dot(ax, x, y, color, label):
    ax.plot(x, y, 'o', color=color, markersize=5, transform=ax.transAxes, zorder=4)
    ax.text(x + 0.06, y, label, fontsize=7, color=TEXT_COLOR,
            fontfamily="monospace", transform=ax.transAxes, va="center")


def _lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))


# ── Main map frame drawing ──────────────────────────────────────────────

def draw_map_frame(
    ax, fig, positions, goals, velocities, at_goal, colliding,
    trail_history, obstacle_map, agent_colors, agent_radius, step_idx,
):
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.patches import Circle

    ax.clear()
    h, w = obstacle_map.shape
    obs_img = render_obstacle_image(obstacle_map)
    ax.imshow(obs_img, origin="upper", extent=[0, w, h, 0], interpolation="nearest")
    ax.set_xlim(-0.2, w + 0.2)
    ax.set_ylim(h + 0.2, -0.2)
    ax.set_aspect("equal")
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_DIM, labelsize=6)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
        spine.set_linewidth(0.5)

    n = len(positions)

    # ── Goal connectors (faint dashed lines agent→goal) ──
    not_at_goal = ~at_goal
    if np.any(not_at_goal):
        idxs = np.where(not_at_goal)[0]
        segments = []
        for i in idxs:
            segments.append([(positions[i, 1], positions[i, 0]),
                             (goals[i, 1], goals[i, 0])])
        lc = LineCollection(segments, colors=CONNECTOR_COLOR, alpha=0.06,
                            linewidths=0.4, linestyles="dashed", zorder=1)
        ax.add_collection(lc)

    # ── Trails (gradient alpha LineCollection) ──
    if len(trail_history) > 1:
        trail = np.array(trail_history)
        T = len(trail)
        for i in range(n):
            segs = []
            alphas = []
            for t in range(T - 1):
                segs.append([(trail[t, i, 1], trail[t, i, 0]),
                             (trail[t + 1, i, 1], trail[t + 1, i, 0])])
                alphas.append(0.08 + 0.35 * (t / max(T - 1, 1)))
            base_c = agent_colors[i]
            colors_with_alpha = [(base_c[0], base_c[1], base_c[2], a) for a in alphas]
            lc = LineCollection(segs, colors=colors_with_alpha, linewidths=0.8, zorder=2)
            ax.add_collection(lc)

    # ── Goal markers ──
    goal_patches = []
    for i in range(n):
        if at_goal[i]:
            continue
        c = Circle((goals[i, 1], goals[i, 0]), agent_radius * 0.5,
                    fill=False, edgecolor=GOAL_COLOR, linewidth=0.5, alpha=0.25)
        goal_patches.append(c)
    # Cross markers for goals
    ax.scatter(goals[:, 1], goals[:, 0], marker="+", s=12, c=GOAL_COLOR,
               linewidths=0.5, alpha=0.3, zorder=3)

    for p in goal_patches:
        ax.add_patch(p)

    # ── Agent glow rings (status indication) ──
    glow_radius = agent_radius * 1.8
    for i in range(n):
        if colliding[i]:
            glow = Circle((positions[i, 1], positions[i, 0]), glow_radius,
                          fill=True, facecolor=RED_GLOW, alpha=0.18,
                          edgecolor="none", zorder=4)
            ax.add_patch(glow)
            ring = Circle((positions[i, 1], positions[i, 0]), agent_radius * 1.15,
                          fill=False, edgecolor=RED_GLOW, linewidth=1.2, alpha=0.7, zorder=6)
            ax.add_patch(ring)
        elif at_goal[i]:
            glow = Circle((positions[i, 1], positions[i, 0]), glow_radius * 0.9,
                          fill=True, facecolor=GREEN_GLOW, alpha=0.10,
                          edgecolor="none", zorder=4)
            ax.add_patch(glow)

    # ── Agent circles (true radius) ──
    green_rgb = _hex_to_rgb(GREEN_GLOW)
    red_rgb = _hex_to_rgb(RED_GLOW)
    for i in range(n):
        fc = agent_colors[i]
        if at_goal[i]:
            ec_rgba = (*green_rgb, 0.8)
            lw = 1.0
        elif colliding[i]:
            ec_rgba = (*red_rgb, 0.9)
            lw = 1.2
        else:
            ec_rgba = (1.0, 1.0, 1.0, 0.3)
            lw = 0.4

        circle = Circle(
            (positions[i, 1], positions[i, 0]), agent_radius,
            facecolor=(*fc[:3], 0.85), edgecolor=ec_rgba,
            linewidth=lw, zorder=7,
        )
        ax.add_patch(circle)

    # ── Velocity arrows ──
    if velocities is not None:
        norms = np.linalg.norm(velocities, axis=1)
        moving = norms > 0.05
        if np.any(moving):
            ax.quiver(
                positions[moving, 1], positions[moving, 0],
                velocities[moving, 1], velocities[moving, 0],
                angles="xy", scale_units="xy", scale=2.5,
                width=0.004, headwidth=3.5, headlength=3,
                color=ARROW_COLOR, alpha=0.65, zorder=8,
            )


# ── Agent color assignment ───────────────────────────────────────────────

def assign_agent_colors(n_agents: int) -> np.ndarray:
    import matplotlib
    try:
        cmap = matplotlib.colormaps["twilight_shifted"]
    except (AttributeError, KeyError):
        from matplotlib.cm import get_cmap
        cmap = get_cmap("twilight_shifted")
    hues = np.linspace(0.05, 0.95, n_agents, endpoint=False)
    np.random.shuffle(hues)
    return np.array([cmap(h)[:3] for h in hues])


# ── Main loop ────────────────────────────────────────────────────────────

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
        scenarios = find_scenarios(
            args.scen_dir, map_name, args.max_scenarios,
            scenario_ids=args.scenario_ids,
        )
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
                    obstacle_map=obstacle_map, dt=args.dt,
                    max_speed=args.max_speed, agent_radius=args.agent_radius,
                    goal_tolerance=args.goal_tolerance,
                )
                env.reset(starts, goals)

                frame_dir = ""
                if args.save_frames:
                    frame_dir = os.path.join(
                        args.save_frames, f"{map_name}_{scen_name}_{agent_num}"
                    )
                    os.makedirs(frame_dir, exist_ok=True)

                agent_colors = assign_agent_colors(agent_num)
                fig, ax_map, ax_panel = setup_figure(
                    obstacle_map, map_name, scen_name, agent_num,
                    args.policy, args.shield_type, args.save_frames,
                )
                trail_history = [env.positions.copy()]

                print(
                    f"[viz] running {map_name} | {scen_name} | N={agent_num} | "
                    f"policy={args.policy} shield={args.shield_type}"
                )

                t_start = time.time()
                last_velocities = None

                for step_idx in range(args.max_steps):
                    # Compute velocities
                    if args.policy == "orca":
                        velocities = (env.bd_guided_velocities() if args.nav == "bd"
                                      else env.goal_directed_velocities())
                    else:
                        data = create_continuous_data_object(
                            env.positions, goals, env.obstacle_map,
                            k=args.k, m=args.m, max_speed=env.max_speed,
                        )
                        data = normalize_continuous_graph_data(data, k=args.k, max_speed=env.max_speed)
                        data = data.to(device)
                        n_ag = env.positions.shape[0]

                        with torch.no_grad():
                            if args.policy == "flow":
                                dt_flow = 1.0 / max(args.num_integration_steps, 1)
                                cands = []
                                for _ in range(max(args.num_consensus_samples, 1)):
                                    v = torch.randn(n_ag, 2, device=device)
                                    for s in range(args.num_integration_steps):
                                        t = torch.full((n_ag, 1), s * dt_flow, device=device)
                                        flow = model(v, t, data)
                                        v = v + flow * dt_flow
                                    cands.append((v.cpu().numpy() * env.max_speed).astype(np.float32))
                                velocities = aggregate_flow_samples(cands, env)
                            else:
                                zero_v = torch.zeros(n_ag, 2, device=device)
                                zero_t = torch.zeros(n_ag, 1, device=device)
                                _, action_logits = model(zero_v, zero_t, data, return_action_logits=True)
                                logits = (action_logits / max(args.tau, 1e-6)).cpu().numpy()
                                labels = logits.argmax(axis=1)
                                velocities = (labels_to_direction_vectors(
                                    labels, num_directions=action_logits.shape[1] - 1
                                ) * env.max_speed)

                    env.step(velocities, shield_type=args.shield_type)
                    last_velocities = env.history_velocities[-1] if env.history_velocities else None

                    trail_history.append(env.positions.copy())
                    if len(trail_history) > TRAIL_LENGTH:
                        trail_history.pop(0)

                    at_goal = env.agents_at_goal()
                    colliding = np.zeros(agent_num, dtype=bool)
                    min_dist = 2.0 * env.agent_radius
                    for i in range(agent_num):
                        for j in range(i + 1, agent_num):
                            if np.linalg.norm(env.positions[i] - env.positions[j]) < min_dist:
                                colliding[i] = True
                                colliding[j] = True

                    frac_at_goal = float(np.mean(at_goal))
                    n_at_goal = int(np.sum(at_goal))
                    speeds = (np.linalg.norm(last_velocities, axis=1)
                              if last_velocities is not None else np.zeros(agent_num))

                    draw_map_frame(
                        ax_map, fig, env.positions, goals, last_velocities,
                        at_goal, colliding, trail_history, obstacle_map,
                        agent_colors, env.agent_radius, step_idx + 1,
                    )

                    draw_panel(
                        ax_panel, step_idx + 1, args.max_steps,
                        frac_at_goal, n_at_goal, agent_num,
                        env.metrics.collisions, env.metrics.near_collisions,
                        env.metrics.obstacle_hits, speeds,
                        time.time() - t_start, env.agent_radius,
                    )

                    if args.save_frames:
                        fig.savefig(
                            os.path.join(frame_dir, f"frame_{step_idx + 1:04d}.png"),
                            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(),
                        )
                    else:
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                        plt.pause(0.001)

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
                    print("[viz] press any key in the plot window to continue...")
                    try:
                        plt.waitforbuttonpress()
                    except Exception:
                        pass
                plt.close(fig)

    print("[viz] all scenarios complete")


# ── CLI ──────────────────────────────────────────────────────────────────

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
                        choices=["orca", "heuristic-orca", "po-orca", "epibt",
                                 "picbf-cs", "simple", "none"],
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
                        help="Directory to save PNG frames (headless)")
    parser.add_argument("--backend", default=None,
                        help="Matplotlib backend (TkAgg, Qt5Agg, ...)")
    args = parser.parse_args()

    global plt
    import matplotlib
    if args.save_frames:
        matplotlib.use("Agg")
    elif args.backend:
        matplotlib.use(args.backend)
    matplotlib.rcParams.update({
        "font.family": "monospace",
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_DIM,
        "ytick.color": TEXT_DIM,
    })
    import matplotlib.pyplot as plt_module
    plt = plt_module

    set_seed(args.eval_seed)
    run_visualized(args)


if __name__ == "__main__":
    main()
