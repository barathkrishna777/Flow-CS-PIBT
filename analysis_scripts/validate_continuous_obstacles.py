import argparse
import csv
import os
from typing import Dict, List

import numpy as np

from main_pys.continuous_env import ContinuousMAPFEnv


def corridor_map() -> np.ndarray:
    grid = np.zeros((16, 16), dtype=np.int8)
    grid[4:12, 6] = 1
    grid[4:12, 9] = 1
    grid[7:9, 6:10] = 0
    return grid


def sparse_clutter_map() -> np.ndarray:
    grid = np.zeros((18, 18), dtype=np.int8)
    obstacles = [(4, 6), (5, 6), (10, 10), (11, 10), (8, 4), (13, 13), (7, 12)]
    for r, c in obstacles:
        grid[r, c] = 1
    return grid


def run_case(
    name: str,
    obstacle_map: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    shield_type: str,
    max_steps: int,
    dt: float,
    max_speed: float,
    agent_radius: float,
    goal_tolerance: float,
) -> Dict[str, float]:
    env = ContinuousMAPFEnv(
        obstacle_map=obstacle_map,
        dt=dt,
        max_speed=max_speed,
        agent_radius=agent_radius,
        goal_tolerance=goal_tolerance,
    )
    env.reset(starts.astype(np.float32), goals.astype(np.float32))
    for _ in range(max_steps):
        env.step(env.goal_directed_velocities(), shield_type=shield_type)
        if env.is_done():
            break
    return {"case": name, "shield_type": shield_type, **env.current_metrics()}


def write_rows(output_csv: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "case",
        "shield_type",
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
    ]
    with open(output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small continuous obstacle-shield sanity battery")
    parser.add_argument("--output-csv", default="evals/continuous_obstacle_battery.csv")
    parser.add_argument("--shield-types", nargs="+", default=["orca", "heuristic-orca"])
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.3)
    parser.add_argument("--goal-tolerance", type=float, default=0.25)
    args = parser.parse_args()

    cases = [
        (
            "pass_around_obstacle",
            np.pad(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int8), pad_width=6),
            np.array([[7.5, 5.5], [7.5, 9.5]], dtype=np.float32),
            np.array([[7.5, 9.5], [7.5, 5.5]], dtype=np.float32),
        ),
        (
            "corridor_crossing",
            corridor_map(),
            np.array([[8.5, 7.0], [8.5, 8.0]], dtype=np.float32),
            np.array([[8.5, 8.0], [8.5, 7.0]], dtype=np.float32),
        ),
        (
            "sparse_clutter",
            sparse_clutter_map(),
            np.array([[2.5, 2.5], [15.5, 2.5], [2.5, 15.5]], dtype=np.float32),
            np.array([[15.5, 15.5], [2.5, 15.5], [15.5, 2.5]], dtype=np.float32),
        ),
    ]

    rows: List[Dict[str, float]] = []
    for shield_type in args.shield_types:
        for name, obstacle_map, starts, goals in cases:
            row = run_case(
                name=name,
                obstacle_map=obstacle_map,
                starts=starts,
                goals=goals,
                shield_type=shield_type,
                max_steps=args.max_steps,
                dt=args.dt,
                max_speed=args.max_speed,
                agent_radius=args.agent_radius,
                goal_tolerance=args.goal_tolerance,
            )
            rows.append(row)
            print(
                f"{shield_type} | {name}: success={row['success']:.0f} "
                f"at_goal={row['agent_fraction_at_goal']:.3f} obstacle_hits={row['obstacle_hits']:.0f}"
            )

    write_rows(args.output_csv, rows)


if __name__ == "__main__":
    main()
