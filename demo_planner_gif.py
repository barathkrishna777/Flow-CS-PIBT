#!/usr/bin/env python3
"""Self-contained MAPF planner demo that renders a presentation-ready GIF.

The full grid simulator in ``main_pys/simulator.py`` expects downloaded maps,
BD heuristic bundles, and a trained checkpoint.  This script keeps the demo
portable by generating a small grid scenario, computing backward-Dijkstra
distances on the fly, running a PIBT-style collision shield, and rendering the
result as an animated GIF.
"""

from __future__ import annotations

import argparse
import heapq
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


MOVE_LABELS = np.array(
    [
        [0, 0],   # wait
        [0, 1],   # right
        [1, 0],   # down
        [-1, 0],  # up
        [0, -1],  # left
    ],
    dtype=np.int32,
)


LOGGER: Optional[logging.Logger] = None


def setup_logging(log_file: Path, quiet: bool = False) -> None:
    global LOGGER
    LOGGER = logging.getLogger("planner_demo")
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not quiet:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(stream_handler)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def log_progress(message: str, enabled: bool = True) -> None:
    if not enabled:
        return
    if LOGGER is not None:
        LOGGER.info(message)
    else:
        print(f"[demo] {message}", flush=True)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "scenario"


@dataclass(frozen=True)
class DemoScenario:
    name: str
    grid: np.ndarray
    starts: np.ndarray
    goals: np.ndarray


def make_demo_scenario(
    name: str = "berlin",
    agents: int = 300,
    map_file: Optional[Path] = None,
    scen_file: Optional[Path] = None,
    seed: int = 7,
) -> DemoScenario:
    if agents < 2:
        raise ValueError("--agents must be at least 2")
    if name not in {"berlin", "warehouse", "crossing", "empty"}:
        raise ValueError(f"Unknown scenario {name!r}")

    if name == "berlin":
        resolved_map = find_berlin_map(map_file)
        if resolved_map is not None:
            grid = read_movingai_map(resolved_map)
            resolved_scen = find_berlin_scenario(scen_file)
            if resolved_scen is not None:
                starts, goals = read_movingai_scenario(resolved_scen, agents=agents)
                scenario_name = resolved_scen.stem
            else:
                starts, goals = choose_map_pairs(grid, agents=agents, seed=seed)
                scenario_name = resolved_map.stem
            assert_free(grid, starts, "start")
            assert_free(grid, goals, "goal")
            return DemoScenario(name=scenario_name, grid=grid, starts=starts, goals=goals)
        grid, starts, goals = make_embedded_berlin_scenario()
    elif name == "empty":
        grid = np.zeros((18, 28), dtype=np.int8)
        left_rows = [2, 4, 6, 8, 10, 12, 14, 15]
        right_rows = [15, 14, 12, 10, 8, 6, 4, 2]
        starts, goals = paired_side_swap(left_rows, right_rows, left_col=2, right_col=25)
    elif name == "crossing":
        grid = np.zeros((19, 27), dtype=np.int8)
        grid[7:12, 10:17] = 1
        grid[9, 10:17] = 0
        grid[7:12, 13] = 0
        horizontal_rows = [8, 9, 10, 11]
        vertical_cols = [11, 12, 14, 15]
        starts = [(r, 2) for r in horizontal_rows] + [(2, c) for c in vertical_cols]
        goals = [(r, 24) for r in reversed(horizontal_rows)] + [(16, c) for c in reversed(vertical_cols)]
        starts, goals = np.asarray(starts, dtype=np.int32), np.asarray(goals, dtype=np.int32)
    else:
        grid = np.zeros((22, 32), dtype=np.int8)
        # Shelf blocks leave horizontal aisles and two vertical cross-aisles.
        for r0 in (3, 7, 11, 15):
            for c0 in (5, 10, 20, 25):
                grid[r0 : r0 + 2, c0 : c0 + 3] = 1
        grid[:, 15:17] = 0
        grid[9:13, :] = 0
        left_rows = [2, 5, 8, 10, 12, 14, 17, 19]
        right_rows = [19, 17, 14, 12, 10, 8, 5, 2]
        starts, goals = paired_side_swap(left_rows, right_rows, left_col=2, right_col=29)

    starts, goals = tile_agents(starts, goals, agents)
    assert_free(grid, starts, "start")
    assert_free(grid, goals, "goal")
    return DemoScenario(name=name, grid=grid, starts=starts, goals=goals)


def find_berlin_map(map_file: Optional[Path]) -> Optional[Path]:
    candidates = []
    if map_file is not None:
        candidates.append(map_file)
    candidates.extend(
        [
            Path("data/mapf-map/Berlin_1_256.map"),
            Path("data/mapf-map/berlin_1_256.map"),
            Path("data/maps/Berlin_1_256.map"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def find_berlin_scenario(scen_file: Optional[Path]) -> Optional[Path]:
    candidates = []
    if scen_file is not None:
        candidates.append(scen_file)
    candidates.extend(
        [
            Path("data/mapf-scen-random/Berlin_1_256-random-1.scen"),
            Path("data/scen-random/Berlin_1_256-random-1.scen"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_movingai_map(path: Path) -> np.ndarray:
    with path.open("r") as handle:
        first = handle.readline().strip().lower()
        if not first.startswith("type"):
            raise ValueError(f"{path} does not look like a MovingAI .map file")
        height = int(handle.readline().split()[1])
        width = int(handle.readline().split()[1])
        marker = handle.readline().strip().lower()
        if marker != "map":
            raise ValueError(f"{path} has an unexpected map header")
        grid = np.zeros((height, width), dtype=np.int8)
        for row in range(height):
            line = handle.readline().rstrip("\n")
            if len(line) != width:
                raise ValueError(f"{path} row {row} has width {len(line)}, expected {width}")
            for col, char in enumerate(line):
                if char in {"@", "T", "O", "W"}:
                    grid[row, col] = 1
    return grid


def read_movingai_scenario(path: Path, agents: int) -> Tuple[np.ndarray, np.ndarray]:
    starts = []
    goals = []
    with path.open("r") as handle:
        first = handle.readline().strip().lower()
        if not first.startswith("version"):
            raise ValueError(f"{path} does not look like a MovingAI .scen file")
        for line in handle:
            tokens = line.rstrip().split("\t")
            if len(tokens) != 9:
                continue
            start_col = int(tokens[4])
            start_row = int(tokens[5])
            goal_col = int(tokens[6])
            goal_row = int(tokens[7])
            starts.append((start_row, start_col))
            goals.append((goal_row, goal_col))
            if len(starts) >= agents:
                break
    if len(starts) < agents:
        raise ValueError(f"{path} only contains {len(starts)} agents, need {agents}")
    return np.asarray(starts, dtype=np.int32), np.asarray(goals, dtype=np.int32)


def make_embedded_berlin_scenario() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.zeros((34, 48), dtype=np.int8)
    blocks = [
        (3, 4, 7, 7),
        (3, 14, 7, 9),
        (3, 28, 7, 7),
        (4, 38, 8, 6),
        (12, 5, 8, 8),
        (12, 18, 6, 6),
        (11, 30, 9, 10),
        (23, 4, 7, 11),
        (22, 20, 8, 8),
        (23, 34, 7, 8),
    ]
    for row, col, height, width in blocks:
        grid[row : row + height, col : col + width] = 1

    # Diagonal roads break up the rectilinear grid and make it read more like
    # a city map than a warehouse aisle.
    for row in range(4, 31):
        col = 8 + row
        if 0 <= col < grid.shape[1]:
            grid[max(0, row - 1) : min(grid.shape[0], row + 2), max(0, col - 1) : min(grid.shape[1], col + 2)] = 0
    for row in range(5, 29):
        col = 41 - row // 2
        grid[row, max(0, col - 1) : min(grid.shape[1], col + 2)] = 0

    starts = np.asarray(
        [
            (2, 2),
            (2, 11),
            (2, 25),
            (2, 45),
            (10, 2),
            (10, 16),
            (10, 27),
            (10, 45),
            (21, 2),
            (21, 17),
            (21, 31),
            (21, 45),
            (31, 2),
            (31, 14),
            (31, 29),
            (31, 45),
        ],
        dtype=np.int32,
    )
    goals = starts[::-1].copy()
    return grid, starts, goals


def choose_map_pairs(grid: np.ndarray, agents: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    free = np.argwhere(grid == 0)
    if len(free) < agents * 2:
        raise ValueError(f"Map only has {len(free)} free cells, not enough for {agents} agents")

    rng = np.random.default_rng(seed)
    h, w = grid.shape
    left = free[free[:, 1] < w * 0.38]
    right = free[free[:, 1] > w * 0.62]
    if len(left) < agents or len(right) < agents:
        left = free[free[:, 1] < w * 0.5]
        right = free[free[:, 1] >= w * 0.5]
    if len(left) < agents or len(right) < agents:
        raise ValueError("Could not choose enough start/goal pairs from the map")

    starts = left[rng.choice(len(left), size=agents, replace=False)]
    goals = right[rng.choice(len(right), size=agents, replace=False)]
    order = np.argsort(starts[:, 0])
    starts = starts[order]
    goals = goals[np.argsort(goals[:, 0])[::-1]]
    return starts.astype(np.int32), goals.astype(np.int32)


def paired_side_swap(
    left_rows: Iterable[int],
    right_rows: Iterable[int],
    left_col: int,
    right_col: int,
) -> Tuple[np.ndarray, np.ndarray]:
    left_rows = list(left_rows)
    right_rows = list(right_rows)
    starts = [(r, left_col) for r in left_rows] + [(r, right_col) for r in right_rows]
    goals = [(r, right_col) for r in right_rows] + [(r, left_col) for r in left_rows]
    return np.asarray(starts, dtype=np.int32), np.asarray(goals, dtype=np.int32)


def tile_agents(starts: np.ndarray, goals: np.ndarray, agents: int) -> Tuple[np.ndarray, np.ndarray]:
    if agents <= len(starts):
        return starts[:agents].copy(), goals[:agents].copy()
    raise ValueError(f"This demo scenario supports up to {len(starts)} agents; requested {agents}")


def assert_free(grid: np.ndarray, positions: np.ndarray, label: str) -> None:
    blocked = grid[positions[:, 0], positions[:, 1]] != 0
    if np.any(blocked):
        bad = positions[np.flatnonzero(blocked)[0]].tolist()
        raise ValueError(f"{label} position lands on an obstacle: {bad}")


def compute_distances_to_goal(grid: np.ndarray, goal: np.ndarray) -> np.ndarray:
    distances = np.full(grid.shape, 1_000_000, dtype=np.int32)
    distances[goal[0], goal[1]] = 0
    queue: List[Tuple[int, Tuple[int, int]]] = [(0, (int(goal[0]), int(goal[1])))]
    while queue:
        dist, (row, col) = heapq.heappop(queue)
        if dist != distances[row, col]:
            continue
        for drow, dcol in MOVE_LABELS[1:]:
            nr, nc = row + int(drow), col + int(dcol)
            if nr < 0 or nr >= grid.shape[0] or nc < 0 or nc >= grid.shape[1]:
                continue
            if grid[nr, nc] != 0:
                continue
            if dist + 1 < distances[nr, nc]:
                distances[nr, nc] = dist + 1
                heapq.heappush(queue, (dist + 1, (nr, nc)))
    return distances


def action_preferences(
    locations: np.ndarray,
    goals: np.ndarray,
    distances: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    prefs = np.zeros((len(locations), len(MOVE_LABELS)), dtype=np.int32)
    for agent, loc in enumerate(locations):
        if np.array_equal(loc, goals[agent]):
            prefs[agent] = np.array([0, 1, 2, 3, 4], dtype=np.int32)
            continue
        scores = np.full(len(MOVE_LABELS), 1_000_000.0, dtype=np.float64)
        for label, (drow, dcol) in enumerate(MOVE_LABELS):
            nr, nc = int(loc[0] + drow), int(loc[1] + dcol)
            if 0 <= nr < distances.shape[1] and 0 <= nc < distances.shape[2]:
                scores[label] = distances[agent, nr, nc]
        scores += rng.random(len(MOVE_LABELS)) * 1e-6
        prefs[agent] = np.argsort(scores)
    return prefs


def update_priorities(previous: np.ndarray, at_goal: np.ndarray) -> np.ndarray:
    priorities = previous.copy()
    priorities[(previous <= 0) & at_goal] -= 1
    priorities[(previous > 0) & at_goal] = 0
    priorities[~at_goal] = np.maximum(previous[~at_goal], 0) + 1
    return priorities


def pibt_recursive(
    grid: np.ndarray,
    agent_id: int,
    preferences: np.ndarray,
    planned: np.ndarray,
    moves: np.ndarray,
    occupied_nodes: np.ndarray,
    occupied_edges: Dict[Tuple[int, int, int, int], bool],
    current: np.ndarray,
    current_to_agent: np.ndarray,
    start_time: float,
    time_limit: float,
) -> bool:
    if time_limit > 0 and time.time() - start_time > time_limit:
        return False

    current_pos = current[agent_id]
    for move_label in preferences[agent_id]:
        move = MOVE_LABELS[move_label]
        next_pos = current_pos + move
        nr, nc = int(next_pos[0]), int(next_pos[1])
        if nr < 0 or nr >= grid.shape[0] or nc < 0 or nc >= grid.shape[1]:
            continue
        if grid[nr, nc] != 0 or occupied_nodes[nr, nc]:
            continue
        reverse_edge = (nr, nc, int(current_pos[0]), int(current_pos[1]))
        if occupied_edges[reverse_edge]:
            continue

        moves[agent_id] = move
        planned[agent_id] = True
        occupied_nodes[nr, nc] = True
        edge = (int(current_pos[0]), int(current_pos[1]), nr, nc)
        occupied_edges[edge] = True

        conflicting = current_to_agent[nr, nc]
        if conflicting != -1 and conflicting != agent_id and not planned[conflicting]:
            if pibt_recursive(
                grid,
                int(conflicting),
                preferences,
                planned,
                moves,
                occupied_nodes,
                occupied_edges,
                current,
                current_to_agent,
                start_time,
                time_limit,
            ):
                return True
            planned[agent_id] = False
            occupied_nodes[nr, nc] = False
            occupied_edges[edge] = False
            continue
        return True

    moves[agent_id] = np.array([0, 0], dtype=np.int32)
    planned[agent_id] = True
    occupied_nodes[int(current_pos[0]), int(current_pos[1])] = True
    return False


def pibt_step(
    grid: np.ndarray,
    preferences: np.ndarray,
    current: np.ndarray,
    priorities: np.ndarray,
    start_time: float,
    time_limit: float,
) -> np.ndarray:
    order = np.argsort(-priorities)
    moves = np.zeros_like(current)
    occupied_nodes = np.zeros(grid.shape, dtype=bool)
    occupied_edges: Dict[Tuple[int, int, int, int], bool] = defaultdict(bool)
    planned = np.zeros(len(current), dtype=bool)
    current_to_agent = np.full(grid.shape, -1, dtype=np.int32)
    current_to_agent[current[:, 0], current[:, 1]] = np.arange(len(current))

    for agent in order:
        if planned[agent]:
            continue
        pibt_recursive(
            grid,
            int(agent),
            preferences,
            planned,
            moves,
            occupied_nodes,
            occupied_edges,
            current,
            current_to_agent,
            start_time,
            time_limit,
        )
    return moves


def solve_scenario(
    scenario: DemoScenario,
    max_steps: int,
    seed: int,
    time_limit: float,
    progress: bool = True,
    progress_interval: int = 25,
    cache_dir: Optional[Path] = None,
    rebuild_cache: bool = False,
    use_cache: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    solve_start = time.time()
    distances = load_or_compute_distances(
        scenario,
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
        use_cache=use_cache,
        progress=progress,
        progress_interval=progress_interval,
    )
    current = scenario.starts.copy()
    priorities = distances[np.arange(len(current)), current[:, 0], current[:, 1]].astype(np.float64)
    priorities /= max(float(priorities.max()), 1.0)

    path = [current.copy()]
    start_time = time.time()
    success = False
    log_progress(f"rolling out planner for up to {max_steps} steps", progress)
    for step in range(max_steps):
        at_goal = np.all(current == scenario.goals, axis=1)
        if step == 0 or (step + 1) % max(progress_interval, 1) == 0:
            log_progress(
                f"step {step + 1}/{max_steps}: {int(np.sum(at_goal))}/{len(current)} at goal",
                progress,
            )
        priorities = update_priorities(priorities, at_goal)
        prefs = action_preferences(current, scenario.goals, distances, rng)
        moves = pibt_step(scenario.grid, prefs, current, priorities, start_time, time_limit)
        next_locations = current + moves
        check_step_is_valid(scenario.grid, current, next_locations)
        current = next_locations
        path.append(current.copy())
        if np.all(current == scenario.goals):
            success = True
            break
        if time_limit > 0 and time.time() - start_time > time_limit:
            break

    paths = np.asarray(path, dtype=np.int32)
    agents_at_goal = int(np.sum(np.all(paths[-1] == scenario.goals, axis=1)))
    metrics = {
        "success": float(success),
        "steps": float(len(paths) - 1),
        "agents_at_goal": float(agents_at_goal),
        "runtime": time.time() - start_time,
        "total_runtime": time.time() - solve_start,
    }
    metrics.update(
        compute_solution_stats(
            paths,
            scenario.goals,
            scenario.grid,
            planner_runtime=metrics["runtime"],
            total_solve_runtime=metrics["total_runtime"],
        )
    )
    log_progress(
        f"rollout complete: {agents_at_goal}/{len(current)} at goal in {len(paths) - 1} steps",
        progress,
    )
    return paths, metrics


def distance_cache_path(scenario: DemoScenario, cache_dir: Optional[Path]) -> Optional[Path]:
    if cache_dir is None:
        return None
    return cache_dir / f"{slugify(scenario.name)}_N{len(scenario.goals)}_bd_distances.npz"


def load_or_compute_distances(
    scenario: DemoScenario,
    cache_dir: Optional[Path],
    rebuild_cache: bool,
    use_cache: bool,
    progress: bool,
    progress_interval: int,
) -> np.ndarray:
    cache_path = distance_cache_path(scenario, cache_dir) if use_cache else None
    compute_start = time.time()
    if cache_path is not None and cache_path.exists() and not rebuild_cache:
        load_start = time.time()
        log_progress(f"loading cached BD fields from {cache_path}", progress)
        with np.load(cache_path) as cached:
            distances = cached["distances"]
            cached_goals = cached["goals"]
            cached_shape = tuple(int(v) for v in cached["map_shape"])
            cached_agents = int(cached["agents"][0])
            if (
                cached_agents != len(scenario.goals)
                or cached_shape != scenario.grid.shape
                or distances.shape != (len(scenario.goals), *scenario.grid.shape)
                or not np.array_equal(cached_goals, scenario.goals)
            ):
                log_progress("cached BD fields do not match this scenario; rebuilding", progress)
            else:
                loaded = distances.astype(np.int32, copy=True)
                log_progress(f"loaded cached BD fields in {time.time() - load_start:.2f}s", progress)
                return loaded

    log_progress(
        f"precomputing backward-Dijkstra fields for {len(scenario.goals)} goals "
        f"on {scenario.grid.shape[0]}x{scenario.grid.shape[1]} map",
        progress,
    )
    distance_fields = []
    for idx, goal in enumerate(scenario.goals, start=1):
        distance_fields.append(compute_distances_to_goal(scenario.grid, goal))
        if idx == 1 or idx == len(scenario.goals) or idx % max(progress_interval, 1) == 0:
            log_progress(
                f"distance fields {idx}/{len(scenario.goals)} "
                f"({100.0 * idx / len(scenario.goals):.0f}%)",
                progress,
            )
    distances = np.stack(distance_fields)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_start = time.time()
        log_progress(f"saving BD cache to {cache_path}", progress)
        np.savez(
            cache_path,
            distances=distances,
            goals=scenario.goals,
            starts=scenario.starts,
            map_shape=np.asarray(scenario.grid.shape, dtype=np.int32),
            agents=np.asarray([len(scenario.goals)], dtype=np.int32),
        )
        log_progress(f"saved BD cache in {time.time() - save_start:.2f}s", progress)

    log_progress(f"BD fields ready in {time.time() - compute_start:.2f}s", progress)
    return distances


def compute_solution_stats(
    paths: np.ndarray,
    goals: np.ndarray,
    grid: np.ndarray,
    planner_runtime: float,
    total_solve_runtime: float,
) -> Dict[str, float]:
    num_steps = max(0, paths.shape[0] - 1)
    num_agents = paths.shape[1]
    deltas = np.diff(paths, axis=0) if num_steps > 0 else np.zeros((0, num_agents, 2), dtype=np.int32)
    moved = np.any(deltas != 0, axis=2) if num_steps > 0 else np.zeros((0, num_agents), dtype=bool)
    at_goal = np.all(paths == goals[None, :, :], axis=2)

    arrival_steps = np.full(num_agents, np.nan, dtype=np.float64)
    for agent in range(num_agents):
        hits = np.flatnonzero(at_goal[:, agent])
        if hits.size > 0:
            arrival_steps[agent] = float(hits[0])

    final_at_goal = int(np.sum(at_goal[-1]))
    arrived_mask = ~np.isnan(arrival_steps)
    waits = int(np.sum(~moved)) if num_steps > 0 else 0
    moves = int(np.sum(moved)) if num_steps > 0 else 0
    total_path_length = float(np.sum(np.linalg.norm(deltas.astype(np.float64), axis=2)))
    edge_swaps = count_edge_swaps(paths)
    vertex_collisions = count_vertex_collisions(paths)
    obstacle_hits = int(np.sum(grid[paths[:, :, 0], paths[:, :, 1]] != 0))

    return {
        "agents": float(num_agents),
        "agents_at_goal": float(final_at_goal),
        "success_rate": float(final_at_goal / max(num_agents, 1)),
        "steps": float(num_steps),
        "planner_runtime": float(planner_runtime),
        "total_solve_runtime": float(total_solve_runtime),
        "mean_step_time_ms": float((planner_runtime / max(num_steps, 1)) * 1000.0),
        "mean_arrival_step": float(np.nanmean(arrival_steps)) if np.any(arrived_mask) else float("nan"),
        "max_arrival_step": float(np.nanmax(arrival_steps)) if np.any(arrived_mask) else float("nan"),
        "total_path_length": total_path_length,
        "mean_path_length": float(total_path_length / max(num_agents, 1)),
        "move_count": float(moves),
        "wait_count": float(waits),
        "wait_fraction": float(waits / max(waits + moves, 1)),
        "vertex_collisions": float(vertex_collisions),
        "edge_swaps": float(edge_swaps),
        "obstacle_hits": float(obstacle_hits),
    }


def count_vertex_collisions(paths: np.ndarray) -> int:
    collisions = 0
    for t in range(paths.shape[0]):
        positions = [tuple(pos) for pos in paths[t]]
        collisions += len(positions) - len(set(positions))
    return collisions


def count_edge_swaps(paths: np.ndarray) -> int:
    swaps = 0
    for t in range(paths.shape[0] - 1):
        edges = set()
        for before, after in zip(paths[t], paths[t + 1]):
            edge = (int(before[0]), int(before[1]), int(after[0]), int(after[1]))
            reverse = (edge[2], edge[3], edge[0], edge[1])
            if edge[:2] != edge[2:] and reverse in edges:
                swaps += 1
            edges.add(edge)
    return swaps


def check_step_is_valid(grid: np.ndarray, previous: np.ndarray, current: np.ndarray) -> None:
    if np.any(grid[current[:, 0], current[:, 1]] != 0):
        raise RuntimeError("Planner generated an obstacle collision")
    if len(set(map(tuple, current))) != len(current):
        raise RuntimeError("Planner generated a vertex collision")
    edges = set()
    for before, after in zip(previous, current):
        edge = (int(before[0]), int(before[1]), int(after[0]), int(after[1]))
        reverse = (edge[2], edge[3], edge[0], edge[1])
        if reverse in edges and edge[:2] != edge[2:]:
            raise RuntimeError("Planner generated an edge-swap collision")
        edges.add(edge)


def configure_plot_cache(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".plot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))


def draw_paths_panel(
    ax,
    scenario: DemoScenario,
    paths: np.ndarray,
    t: int,
    trail: Optional[int],
    title_prefix: str = "CS-PIBT demo",
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = plt.get_cmap("tab20", paths.shape[1])
    map_cmap = ListedColormap(["#f7f7f2", "#4b5563"])
    solved = np.all(paths[-1] == scenario.goals)
    t = min(max(int(t), 0), len(paths) - 1)
    trail_start = 0 if trail is None else max(0, t - trail)

    ax.clear()
    ax.set_facecolor("#f7f7f2")
    ax.imshow(scenario.grid, cmap=map_cmap, origin="upper", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, scenario.grid.shape[1] - 0.5)
    ax.set_ylim(scenario.grid.shape[0] - 0.5, -0.5)

    ax.scatter(
        scenario.starts[:, 1],
        scenario.starts[:, 0],
        marker="o",
        s=marker_size(paths.shape[1], base=52.0),
        c=[cmap(i) for i in range(paths.shape[1])],
        edgecolors="#111827",
        linewidths=edge_width(paths.shape[1], base=0.45),
        alpha=0.4,
        zorder=3,
    )
    ax.scatter(
        scenario.goals[:, 1],
        scenario.goals[:, 0],
        marker="*",
        s=marker_size(paths.shape[1], base=110.0),
        c=[cmap(i) for i in range(paths.shape[1])],
        edgecolors="#111827",
        linewidths=edge_width(paths.shape[1], base=0.5),
        alpha=0.85,
        zorder=3,
    )
    for agent in range(paths.shape[1]):
        color = cmap(agent)
        agent_path = paths[trail_start : t + 1, agent]
        ax.plot(
            agent_path[:, 1],
            agent_path[:, 0],
            c=color,
            linewidth=line_width(paths.shape[1]),
            alpha=0.62 if paths.shape[1] > 80 else 0.7,
            solid_capstyle="round",
            zorder=2,
        )
        at_goal = np.array_equal(paths[t, agent], scenario.goals[agent])
        ax.scatter(
            paths[t, agent, 1],
            paths[t, agent, 0],
            s=marker_size(paths.shape[1], base=82.0),
            c=["#e5e7eb" if at_goal else color],
            edgecolors="#111827",
            linewidths=edge_width(paths.shape[1], base=0.65),
            zorder=4,
        )
        if paths.shape[1] <= 40:
            ax.text(
                paths[t, agent, 1],
                paths[t, agent, 0],
                str(agent),
                ha="center",
                va="center",
                fontsize=6,
                color="#111827",
                zorder=5,
            )

    status = "solved" if solved else "partial"
    ax.set_title(
        f"{title_prefix}: {scenario.name} | {paths.shape[1]} agents | t={t}/{len(paths)-1} | {status}",
        fontsize=12,
        color="#166534" if solved else "#991b1b",
    )


def marker_size(num_agents: int, base: float) -> float:
    if num_agents <= 40:
        return base
    if num_agents <= 120:
        return base * 0.45
    return base * 0.18


def edge_width(num_agents: int, base: float) -> float:
    if num_agents <= 40:
        return base
    if num_agents <= 120:
        return base * 0.7
    return base * 0.35


def line_width(num_agents: int) -> float:
    if num_agents <= 40:
        return 2.2
    if num_agents <= 120:
        return 1.1
    return 0.45


def render_gif(
    scenario: DemoScenario,
    paths: np.ndarray,
    output: Path,
    frame_stride: int,
    trail: int,
    dpi: int,
    duration_ms: int,
    end_duration_ms: int,
    progress: bool = True,
    progress_interval: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = Path(tempfile.mkdtemp(prefix="planner_demo_frames_"))
    frames = list(range(0, len(paths), max(1, frame_stride)))
    if frames[-1] != len(paths) - 1:
        frames.append(len(paths) - 1)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#f7f7f2")

    render_start = time.time()
    log_progress(f"rendering {len(frames)} GIF frames to {output}", progress)
    for frame_number, t in enumerate(frames):
        draw_paths_panel(ax, scenario, paths, t=t, trail=trail)
        fig.tight_layout(pad=0.4)
        fig.savefig(frame_dir / f"{frame_number:04d}.png", dpi=dpi)
        rendered = frame_number + 1
        if rendered == 1 or rendered == len(frames) or rendered % max(progress_interval, 1) == 0:
            log_progress(
                f"rendered frames {rendered}/{len(frames)} "
                f"({100.0 * rendered / len(frames):.0f}%)",
                progress,
            )

    plt.close(fig)
    log_progress("encoding GIF", progress)
    images = [Image.open(frame) for frame in sorted(frame_dir.glob("*.png"))]
    if not images:
        raise RuntimeError("No frames were rendered")
    durations = [duration_ms] * (len(images) - 1) + [end_duration_ms]
    images[0].save(output, save_all=True, append_images=images[1:], duration=durations, loop=0)
    for image in images:
        image.close()
    shutil.rmtree(frame_dir, ignore_errors=True)
    log_progress(f"GIF saved in {time.time() - render_start:.1f}s", progress)


def format_stats_block(metrics: Dict[str, float], render_runtime: Optional[float]) -> str:
    lines = [
        "Planner demo statistics",
        f"  agents at goal:     {int(metrics['agents_at_goal'])}/{int(metrics['agents'])} ({100.0 * metrics['success_rate']:.1f}%)",
        f"  solved:             {bool(metrics['success'])}",
        f"  planner steps:      {int(metrics['steps'])}",
        f"  planner runtime:    {metrics['planner_runtime']:.3f}s",
        f"  total solve runtime: {metrics['total_solve_runtime']:.3f}s",
        f"  mean step time:     {metrics['mean_step_time_ms']:.2f} ms",
        f"  mean arrival step:  {metrics['mean_arrival_step']:.1f}",
        f"  max arrival step:   {metrics['max_arrival_step']:.0f}",
        f"  total path length:  {metrics['total_path_length']:.0f} grid moves",
        f"  mean path length:   {metrics['mean_path_length']:.1f} grid moves/agent",
        f"  wait fraction:      {100.0 * metrics['wait_fraction']:.1f}%",
        f"  vertex collisions:  {int(metrics['vertex_collisions'])}",
        f"  edge swaps:         {int(metrics['edge_swaps'])}",
        f"  obstacle hits:      {int(metrics['obstacle_hits'])}",
    ]
    if render_runtime is not None:
        lines.append(f"  GIF render runtime: {render_runtime:.1f}s")
    return "\n".join(lines)


def show_paths_panel(scenario: DemoScenario, paths: np.ndarray, trail: Optional[int]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#f7f7f2")
    draw_paths_panel(ax, scenario, paths, t=len(paths) - 1, trail=trail, title_prefix="Planner paths")
    fig.tight_layout(pad=0.5)
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a self-contained planner demo and render a GIF.")
    parser.add_argument("--scenario", choices=["berlin", "warehouse", "crossing", "empty"], default="berlin")
    parser.add_argument(
        "--map-file",
        type=Path,
        default=None,
        help="Optional MovingAI .map file. For --scenario berlin, defaults to data/mapf-map/Berlin_1_256.map if present.",
    )
    parser.add_argument(
        "--scen-file",
        type=Path,
        default=None,
        help="Optional MovingAI .scen file. For --scenario berlin, defaults to data/mapf-scen-random/Berlin_1_256-random-1.scen if present.",
    )
    parser.add_argument("--agents", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("logs/planner_demo.gif"))
    parser.add_argument("--paths-output", type=Path, default=Path("logs/planner_demo_paths.npy"))
    parser.add_argument("--frame-stride", type=int, default=6)
    parser.add_argument("--trail", type=int, default=32)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--duration-ms", type=int, default=80)
    parser.add_argument("--end-duration-ms", type=int, default=1800)
    parser.add_argument("--no-render", action="store_true", help="Only solve and save the paths .npy file.")
    parser.add_argument("--no-show", action="store_true", help="Do not open the final Matplotlib paths window.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/demo-cache"), help="Directory for cached BD distance fields.")
    parser.add_argument("--no-cache", action="store_true", help="Do not load or save cached BD distance fields.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Recompute and overwrite cached BD distance fields.")
    parser.add_argument("--precompute-only", action="store_true", help="Build/load the BD cache and exit before rollout/rendering.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging.")
    parser.add_argument("--log-file", type=Path, default=Path("logs/planner_demo.log"), help="Write timestamped progress logs here.")
    parser.add_argument("--progress-interval", type=int, default=25, help="Print solve progress every N goals/steps.")
    parser.add_argument("--render-progress-interval", type=int, default=10, help="Print render progress every N frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = not args.quiet
    setup_logging(args.log_file, quiet=args.quiet)
    total_start = time.time()
    log_progress(f"starting planner demo; log_file={args.log_file}", progress)
    configure_plot_cache(args.output.parent)
    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")
    log_progress("loading scenario", progress)
    scenario = make_demo_scenario(
        args.scenario,
        args.agents,
        map_file=args.map_file,
        scen_file=args.scen_file,
        seed=args.seed,
    )
    log_progress(
        f"scenario={scenario.name}, agents={len(scenario.starts)}, "
        f"map={scenario.grid.shape[0]}x{scenario.grid.shape[1]}",
        progress,
    )
    if args.precompute_only:
        load_or_compute_distances(
            scenario,
            cache_dir=args.cache_dir,
            rebuild_cache=args.rebuild_cache,
            use_cache=not args.no_cache,
            progress=progress,
            progress_interval=args.progress_interval,
        )
        cache_path = distance_cache_path(scenario, args.cache_dir)
        if cache_path is not None and not args.no_cache:
            log_progress(f"precompute complete; cache={os.path.abspath(cache_path)}", progress)
            print(f"cache: {os.path.abspath(cache_path)}", flush=True)
        return

    paths, metrics = solve_scenario(
        scenario,
        max_steps=args.max_steps,
        seed=args.seed,
        time_limit=args.time_limit,
        progress=progress,
        progress_interval=args.progress_interval,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        use_cache=not args.no_cache,
    )

    args.paths_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.paths_output, paths)
    log_progress(f"saved paths to {args.paths_output}", progress)

    render_runtime = None
    if not args.no_render:
        render_start = time.time()
        render_gif(
            scenario,
            paths,
            output=args.output,
            frame_stride=args.frame_stride,
            trail=args.trail,
            dpi=args.dpi,
            duration_ms=args.duration_ms,
            end_duration_ms=args.end_duration_ms,
            progress=progress,
            progress_interval=args.render_progress_interval,
        )
        render_runtime = time.time() - render_start

    status = "SUCCESS" if metrics["success"] else "PARTIAL"
    summary = (
        f"{status}: {int(metrics['agents_at_goal'])}/{args.agents} agents at goal "
        f"in {int(metrics['steps'])} steps, runtime={metrics['runtime']:.3f}s"
    )
    stats_block = format_stats_block(metrics, render_runtime)
    log_progress(summary, progress)
    for line in stats_block.splitlines():
        log_progress(line, progress)
    log_progress(f"paths={os.path.abspath(args.paths_output)}", progress)
    if not args.no_render:
        log_progress(f"gif={os.path.abspath(args.output)}", progress)
    log_progress(f"finished non-interactive work in {time.time() - total_start:.1f}s", progress)

    print(summary, flush=True)
    print(stats_block, flush=True)
    print(f"paths: {os.path.abspath(args.paths_output)}", flush=True)
    if not args.no_render:
        print(f"gif:   {os.path.abspath(args.output)}", flush=True)
    print(f"log:   {os.path.abspath(args.log_file)}", flush=True)

    if not args.no_show:
        log_progress("opening Matplotlib paths panel; close the window to end the script", progress)
        show_paths_panel(scenario, paths, trail=None)
        log_progress(f"Matplotlib panel closed; total elapsed={time.time() - total_start:.1f}s", progress)


if __name__ == "__main__":
    main()
