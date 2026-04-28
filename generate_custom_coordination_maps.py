"""
Generate training-only MAPF maps/scenarios for dense coordination studies.

The defaults intentionally avoid the exact Rishi held-out map names while
creating nearby distributions: small dense random maps, larger random maps,
small dense maps, and bottleneck/corridor maps.

Typical use:
    python generate_custom_coordination_maps.py \
      --output-root data/custom_coordination \
      --scenarios 128

Then use the generated agent-count JSON with generate_flow_data_multi.py.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from dataclasses import dataclass

import numpy as np


HELD_OUT_TEST_MAPS = {
    "Paris_1_256",
    "empty-48-48",
    "maze-128-128-2",
    "random-64-64-10",
    "random-32-32-10",
    "warehouse-10-20-10-2-1",
    "den312d",
    "den520d",
}


@dataclass(frozen=True)
class MapSpec:
    name: str
    grid: np.ndarray
    max_agents: int
    agent_counts: list[int]
    scenario_style: str = "random"


def _free_component_size(grid: np.ndarray, start: tuple[int, int]) -> int:
    rows, cols = grid.shape
    queue: deque[tuple[int, int]] = deque([start])
    visited = np.zeros_like(grid, dtype=bool)
    visited[start] = True
    size = 0

    while queue:
        r, c = queue.popleft()
        size += 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] == 0 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
    return size


def _is_fully_connected(grid: np.ndarray) -> bool:
    free = np.argwhere(grid == 0)
    if len(free) == 0:
        return False
    return _free_component_size(grid, tuple(free[0])) == len(free)


def _random_connected_grid(
    height: int,
    width: int,
    obstacle_prob: float,
    rng: np.random.Generator,
    min_free: int,
) -> np.ndarray:
    for _ in range(10_000):
        grid = (rng.random((height, width)) < obstacle_prob).astype(np.int8)
        if int(np.count_nonzero(grid == 0)) < min_free:
            continue
        if _is_fully_connected(grid):
            return grid
    raise RuntimeError(
        f"Could not generate connected {height}x{width} map with "
        f"obstacle_prob={obstacle_prob} and min_free={min_free}"
    )


def _corridor_grid(size: int, rng: np.random.Generator, gap_width: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.int8)
    vertical = bool(rng.integers(0, 2))
    wall = int(rng.integers(size // 3, (2 * size) // 3 + 1))
    gap_center = int(rng.integers(size // 4, (3 * size) // 4 + 1))
    gap_start = max(1, gap_center - gap_width // 2)
    gap_end = min(size - 1, gap_start + gap_width)

    if vertical:
        grid[:, wall] = 1
        grid[gap_start:gap_end, wall] = 0
    else:
        grid[wall, :] = 1
        grid[wall, gap_start:gap_end] = 0

    # Add a little clutter away from the bottleneck while preserving connectivity.
    candidates = np.argwhere(grid == 0)
    rng.shuffle(candidates)
    target_extra = max(1, size // 4)
    added = 0
    for r, c in candidates:
        if added >= target_extra:
            break
        if abs(int(r) - wall) <= 1 or abs(int(c) - wall) <= 1:
            continue
        grid[r, c] = 1
        if _is_fully_connected(grid):
            added += 1
        else:
            grid[r, c] = 0

    return grid


def _map_to_text(grid: np.ndarray) -> str:
    rows = ["".join("@" if cell else "." for cell in row) for row in grid]
    height, width = grid.shape
    return f"type octile\nheight {height}\nwidth {width}\nmap\n" + "\n".join(rows) + "\n"


def _write_map(path: str, grid: np.ndarray, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        return
    with open(path, "w") as f:
        f.write(_map_to_text(grid))


def _random_pairs(
    free_cells: np.ndarray,
    num_agents: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    starts_idx = rng.choice(len(free_cells), num_agents, replace=False)
    goals_idx = rng.choice(len(free_cells), num_agents, replace=False)
    starts = free_cells[starts_idx]
    goals = free_cells[goals_idx]

    for _ in range(20):
        same = np.all(starts == goals, axis=1)
        if not np.any(same):
            break
        rng.shuffle(goals)
    return starts, goals


def _corridor_pairs(
    grid: np.ndarray,
    num_agents: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    free_cells = np.argwhere(grid == 0)
    rows, cols = grid.shape
    obstacle_rows = np.count_nonzero(grid == 1, axis=1)
    obstacle_cols = np.count_nonzero(grid == 1, axis=0)

    if obstacle_cols.max() >= obstacle_rows.max():
        wall = int(np.argmax(obstacle_cols))
        side_a = free_cells[free_cells[:, 1] < wall]
        side_b = free_cells[free_cells[:, 1] > wall]
    else:
        wall = int(np.argmax(obstacle_rows))
        side_a = free_cells[free_cells[:, 0] < wall]
        side_b = free_cells[free_cells[:, 0] > wall]

    if len(side_a) < num_agents // 2 or len(side_b) < num_agents // 2:
        return _random_pairs(free_cells, num_agents, rng)

    half = num_agents // 2
    starts_a = side_a[rng.choice(len(side_a), half, replace=False)]
    goals_a = side_b[rng.choice(len(side_b), half, replace=False)]
    starts_b = side_b[rng.choice(len(side_b), num_agents - half, replace=False)]
    goals_b = side_a[rng.choice(len(side_a), num_agents - half, replace=False)]
    starts = np.vstack([starts_a, starts_b])
    goals = np.vstack([goals_a, goals_b])

    order = rng.permutation(num_agents)
    return starts[order], goals[order]


def _write_scen(
    path: str,
    map_name: str,
    map_shape: tuple[int, int],
    starts: np.ndarray,
    goals: np.ndarray,
    overwrite: bool,
) -> None:
    if os.path.exists(path) and not overwrite:
        return
    with open(path, "w") as f:
        f.write(f"version {len(starts)}\n")
        height, width = map_shape
        for start, goal in zip(starts, goals):
            sr, sc = start
            gr, gc = goal
            distance = abs(int(sr) - int(gr)) + abs(int(sc) - int(gc))
            f.write(
                f"0\t{map_name}.map\t{width}\t{height}\t"
                f"{sc}\t{sr}\t{gc}\t{gr}\t{distance}\n"
            )


def _build_specs(args: argparse.Namespace, rng: np.random.Generator) -> list[MapSpec]:
    specs: list[MapSpec] = []

    for i in range(args.random32_count):
        max_agents = 400
        grid = _random_connected_grid(32, 32, 0.10, rng, min_free=max_agents)
        specs.append(
            MapSpec(
                name=f"random-32-32-10-custom-{i}",
                grid=grid,
                max_agents=max_agents,
                agent_counts=[100, 150, 200, 250, 300, 350, 400],
            )
        )

    for i in range(args.random64_count):
        max_agents = 800
        grid = _random_connected_grid(64, 64, 0.10, rng, min_free=max_agents)
        specs.append(
            MapSpec(
                name=f"random-64-64-10-custom-{i}",
                grid=grid,
                max_agents=max_agents,
                agent_counts=[100, 200, 300, 400, 500, 600, 800],
            )
        )

    for i in range(args.corridor_count):
        max_agents = 200
        grid = _corridor_grid(30, rng, gap_width=1 + int(i % 2))
        specs.append(
            MapSpec(
                name=f"corridor-30-30-custom-{i}",
                grid=grid,
                max_agents=max_agents,
                agent_counts=[50, 100, 150, 200],
                scenario_style="corridor",
            )
        )

    for i in range(args.dense_count):
        max_agents = 50
        grid = _random_connected_grid(15, 15, 0.15, rng, min_free=max_agents)
        specs.append(
            MapSpec(
                name=f"dense-15-15-custom-{i}",
                grid=grid,
                max_agents=max_agents,
                agent_counts=[20, 30, 40, 50],
            )
        )

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create custom training-only maps/scenarios for coordination ablations."
    )
    parser.add_argument("--output-root", default="data/custom_coordination")
    parser.add_argument("--scenarios", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random32-count", type=int, default=8)
    parser.add_argument("--random64-count", type=int, default=4)
    parser.add_argument("--corridor-count", type=int, default=4)
    parser.add_argument("--dense-count", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    map_dir = os.path.join(args.output_root, "maps")
    scen_dir = os.path.join(args.output_root, "scens")
    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(scen_dir, exist_ok=True)

    specs = _build_specs(args, rng)
    agent_counts: dict[str, list[int]] = {}

    for spec in specs:
        if spec.name in HELD_OUT_TEST_MAPS:
            raise ValueError(f"Generated map name collides with held-out map: {spec.name}")

        free_cells = np.argwhere(spec.grid == 0)
        max_agents = min(spec.max_agents, len(free_cells))
        counts = [n for n in spec.agent_counts if n <= max_agents]
        if not counts:
            raise ValueError(f"No usable agent counts for {spec.name}")
        agent_counts[spec.name] = counts

        _write_map(os.path.join(map_dir, f"{spec.name}.map"), spec.grid, args.overwrite)

        for scen_idx in range(1, args.scenarios + 1):
            if spec.scenario_style == "corridor":
                starts, goals = _corridor_pairs(spec.grid, max_agents, rng)
            else:
                starts, goals = _random_pairs(free_cells, max_agents, rng)
            _write_scen(
                os.path.join(scen_dir, f"{spec.name}-random-{scen_idx}.scen"),
                spec.name,
                spec.grid.shape,
                starts,
                goals,
                args.overwrite,
            )

    agent_counts_path = os.path.join(args.output_root, "agent_counts.json")
    manifest_path = os.path.join(args.output_root, "manifest.json")

    with open(agent_counts_path, "w") as f:
        json.dump(agent_counts, f, indent=2, sort_keys=True)

    manifest = {
        "map_dir": map_dir,
        "scen_dir": scen_dir,
        "agent_counts_json": agent_counts_path,
        "agent_counts": agent_counts,
        "scenarios_per_map": args.scenarios,
        "seed": args.seed,
        "held_out_map_names_avoided": sorted(HELD_OUT_TEST_MAPS),
        "maps": {
            spec.name: {
                "shape": list(spec.grid.shape),
                "free_cells": int(np.count_nonzero(spec.grid == 0)),
                "agent_counts": agent_counts[spec.name],
                "scenario_style": spec.scenario_style,
            }
            for spec in specs
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Wrote {len(specs)} maps to {map_dir}")
    print(f"Wrote {len(specs) * args.scenarios} scenarios to {scen_dir}")
    print(f"Wrote agent counts to {agent_counts_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
