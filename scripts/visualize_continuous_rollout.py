import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from PIL import Image

from main_pys.model_inputs import load_grid_map_from_file


def _load_scalar(data, key: str, default: Optional[str] = None) -> str:
    if key not in data:
        if default is None:
            raise KeyError(key)
        return default
    value = data[key]
    if getattr(value, "ndim", 0) == 0:
        return str(value.item())
    return str(value[0])


def save_overview_png(
    positions: np.ndarray,
    goals: np.ndarray,
    obstacle_map: np.ndarray,
    output_path: str,
    title: str,
    stride: int = 1,
) -> None:
    positions = np.asarray(positions, dtype=np.float32)
    goals = np.asarray(goals, dtype=np.float32)
    cmap = colormaps.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(obstacle_map, cmap="Greys", origin="upper")
    for agent_idx in range(positions.shape[1]):
        traj = positions[:: max(stride, 1), agent_idx]
        color = cmap((agent_idx % 20) / max(19, 1))
        ax.plot(traj[:, 1], traj[:, 0], linewidth=1.0, alpha=0.9, color=color)
        ax.scatter(positions[0, agent_idx, 1], positions[0, agent_idx, 0], s=12, color=color, marker="o")
        ax.scatter(goals[agent_idx, 1], goals[agent_idx, 0], s=18, color=color, marker="*")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_gif(
    positions: np.ndarray,
    goals: np.ndarray,
    obstacle_map: np.ndarray,
    output_path: str,
    title: str,
    frame_stride: int = 4,
    trail_length: int = 25,
) -> None:
    positions = np.asarray(positions, dtype=np.float32)
    goals = np.asarray(goals, dtype=np.float32)
    cmap = colormaps.get_cmap("tab20")
    frames = []
    render_indices = list(range(0, positions.shape[0], max(frame_stride, 1)))
    if render_indices[-1] != positions.shape[0] - 1:
        render_indices.append(positions.shape[0] - 1)

    for t in render_indices:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(obstacle_map, cmap="Greys", origin="upper")
        for agent_idx in range(positions.shape[1]):
            color = cmap((agent_idx % 20) / max(19, 1))
            trail_start = max(0, t - trail_length)
            trail = positions[trail_start : t + 1, agent_idx]
            ax.plot(trail[:, 1], trail[:, 0], linewidth=1.0, alpha=0.9, color=color)
            ax.scatter(goals[agent_idx, 1], goals[agent_idx, 0], s=18, color=color, marker="*")
            ax.scatter(positions[t, agent_idx, 1], positions[t, agent_idx, 0], s=12, color=color, marker="o")
        ax.set_title(f"{title} | t={t}/{positions.shape[0] - 1}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        frames.append(Image.fromarray(rgba[..., :3].copy()))
        plt.close(fig)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=[80] * max(len(frames) - 1, 0) + [1000],
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a saved continuous rollout .npz")
    parser.add_argument("--npz", required=True, help="Path to the rollout .npz file")
    parser.add_argument("--map-dir", required=True, help="Directory containing .map files")
    parser.add_argument("--output-png", required=True, help="Output overview PNG path")
    parser.add_argument("--output-gif", default=None, help="Optional animated GIF output path")
    parser.add_argument("--stride", type=int, default=1, help="Overview trajectory subsampling stride")
    parser.add_argument("--frame-stride", type=int, default=4, help="GIF frame stride")
    parser.add_argument("--trail-length", type=int, default=25, help="GIF trail length")
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=True) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        goals = np.asarray(data["goals"], dtype=np.float32)
        map_name = _load_scalar(data, "map_name")
        expert_source = _load_scalar(data, "expert_source_used", _load_scalar(data, "expert_source", "unknown"))
        scenario_name = _load_scalar(data, "scenario_name", Path(args.npz).stem)

    obstacle_map = load_grid_map_from_file(os.path.join(args.map_dir, f"{map_name}.map"))
    title = f"{scenario_name} | {map_name} | {positions.shape[1]} agents | {expert_source}"
    save_overview_png(
        positions=positions,
        goals=goals,
        obstacle_map=obstacle_map,
        output_path=args.output_png,
        title=title,
        stride=args.stride,
    )
    if args.output_gif:
        save_gif(
            positions=positions,
            goals=goals,
            obstacle_map=obstacle_map,
            output_path=args.output_gif,
            title=title,
            frame_stride=args.frame_stride,
            trail_length=args.trail_length,
        )


if __name__ == "__main__":
    main()
