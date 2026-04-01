import argparse
import glob
import os
import subprocess
import tempfile
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from main_pys.continuous_env import (
    ContinuousMAPFEnv,
    grid_starts_to_continuous,
    parse_scene_file,
)
from main_pys.continuous_scenarios import scenario_id_from_path, select_scenarios
from main_pys.model_inputs import load_grid_map_from_file, velocity_to_direction_labels


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EECBS_REPO = os.environ.get("EECBS_FLOW_REPO", str(REPO_ROOT.parent / "EECBS-flow"))


def parse_paths_txt(file_path: str) -> np.ndarray:
    paths = []
    with open(file_path, "r") as f:
        for line in f:
            if not line.startswith("Agent"):
                continue
            coords = []
            current = ""
            reading = False
            for ch in line:
                if ch == "(":
                    current = ""
                    reading = True
                elif ch == ")":
                    reading = False
                    row, col = current.split(",")
                    coords.append([float(row), float(col)])
                elif reading:
                    current += ch
            if coords:
                paths.append(np.asarray(coords, dtype=np.float32))
    if not paths:
        return np.zeros((0, 0, 2), dtype=np.float32)
    max_len = max(len(p) for p in paths)
    padded = np.zeros((len(paths), max_len, 2), dtype=np.float32)
    for i, path in enumerate(paths):
        padded[i, : len(path)] = path
        padded[i, len(path) :] = path[-1]
    return padded


def discrete_paths_to_continuous(
    discrete_positions: np.ndarray,
    dt: float,
    max_speed: float,
) -> Tuple[np.ndarray, np.ndarray]:
    positions = discrete_positions.astype(np.float32) + 0.5
    velocities = []
    dense_positions = [positions[:, 0]]
    step_distance = max_speed * dt

    for t in range(positions.shape[1] - 1):
        start = positions[:, t]
        end = positions[:, t + 1]
        delta = end - start
        segment_len = np.linalg.norm(delta, axis=1)
        substeps = int(max(1, np.ceil(segment_len.max() / max(step_distance, 1e-6))))
        for s in range(1, substeps + 1):
            alpha = s / substeps
            interp = start + alpha * delta
            dense_positions.append(interp)
            velocities.append((interp - dense_positions[-2]) / dt)

    positions_arr = np.stack(dense_positions, axis=0)
    velocities_arr = np.stack(velocities, axis=0) if velocities else np.zeros((0, positions.shape[0], 2), dtype=np.float32)
    return positions_arr, velocities_arr


def rollout_orca_policy(
    obstacle_map: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    dt: float,
    max_speed: float,
    max_steps: int,
    agent_radius: float,
    goal_tolerance: float,
    shield_type: str = "orca",
) -> Tuple[np.ndarray, np.ndarray]:
    env = ContinuousMAPFEnv(
        obstacle_map=obstacle_map,
        dt=dt,
        max_speed=max_speed,
        agent_radius=agent_radius,
        goal_tolerance=goal_tolerance,
    )
    env.reset(starts, goals)
    for _ in range(max_steps):
        preferred = env.goal_directed_velocities()
        env.step(preferred, shield_type=shield_type)
        if env.is_done():
            break
    positions = np.asarray(env.history_positions, dtype=np.float32)
    velocities = np.asarray(env.history_velocities, dtype=np.float32)
    return positions, velocities


def validate_replay(
    obstacle_map: np.ndarray,
    starts: np.ndarray,
    goals: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    max_speed: float,
    agent_radius: float,
    goal_tolerance: float,
) -> bool:
    env = ContinuousMAPFEnv(
        obstacle_map=obstacle_map,
        dt=dt,
        max_speed=max_speed,
        agent_radius=agent_radius,
        goal_tolerance=goal_tolerance,
    )
    env.reset(starts, goals)
    for step in range(len(velocities)):
        env.step(velocities[step], shield_type="none")
        if np.max(np.abs(env.positions - positions[step + 1])) > 0.35:
            return False
    return env.metrics.collisions == 0 and env.metrics.obstacle_hits == 0


def resolve_eecbs_binary(explicit_binary: Optional[str], eecbs_repo: Optional[str]) -> str:
    candidates = []
    if explicit_binary:
        candidates.append(Path(explicit_binary).expanduser())

    if eecbs_repo:
        repo_path = Path(eecbs_repo).expanduser()
        candidates.extend([repo_path / "build" / "eecbs", repo_path / "eecbs"])

    candidates.extend(
        [
            REPO_ROOT / "build" / "eecbs",
            REPO_ROOT.parent / "EECBS-flow" / "build" / "eecbs",
            REPO_ROOT.parent / "EECBS-flow" / "eecbs",
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if os.path.exists(candidate):
            return str(candidate)

    checked = "\n".join(f"  - {c}" for c in seen)
    raise FileNotFoundError(
        "Could not find an EECBS binary. Checked:\n"
        f"{checked}\n"
        "Pass --eecbs-binary explicitly, set EECBS_FLOW_REPO, or place the solver in ../EECBS-flow/build/eecbs."
    )


def run_eecbs(
    map_file: str,
    scen_file: str,
    agent_num: int,
    suboptimality: float,
    time_limit: int,
    eecbs_binary: str,
) -> np.ndarray:
    if not os.path.exists(eecbs_binary):
        raise FileNotFoundError(f"EECBS binary not found: {eecbs_binary}")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_stats = os.path.join(tmpdir, "stats.csv")
        out_paths = os.path.join(tmpdir, "paths.txt")
        cmd = [
            eecbs_binary,
            "-m",
            map_file,
            "-a",
            scen_file,
            "-o",
            out_stats,
            "--outputPaths",
            out_paths,
            "-k",
            str(agent_num),
            "-t",
            str(time_limit),
            "--suboptimality",
            str(suboptimality),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if not os.path.exists(out_paths):
            raise RuntimeError(
                "EECBS finished without producing an output paths file. "
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        paths = parse_paths_txt(out_paths)
        if len(paths) == 0:
            raise RuntimeError(
                "EECBS produced an empty path file. "
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return paths


def derive_action_labels(velocities: np.ndarray, num_directions: int, wait_threshold: float) -> np.ndarray:
    action_labels = []
    for step in range(len(velocities)):
        action_labels.append(
            velocity_to_direction_labels(
                velocities[step],
                num_directions=num_directions,
                wait_threshold=wait_threshold,
            )
        )
    return np.asarray(action_labels, dtype=np.int64)


def _mean_nearest_neighbor_distance(positions: np.ndarray) -> float:
    if len(positions) <= 1:
        return 0.0
    deltas = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(deltas, axis=2)
    np.fill_diagonal(dists, np.inf)
    return float(np.mean(np.min(dists, axis=1)))


def save_rollout(
    output_path: str,
    map_name: str,
    scenario_name: str,
    scenario_id: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    goals: np.ndarray,
    dt: float,
    expert_recipe_requested: str,
    expert_source_used: str,
    fallback_reason: str,
    num_directions: int,
    wait_threshold: float,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        map_name=np.asarray(map_name),
        scenario_name=np.asarray(scenario_name),
        scenario_id=np.asarray(scenario_id, dtype=np.int32),
        positions=positions.astype(np.float32),
        velocities=velocities.astype(np.float32),
        goals=goals.astype(np.float32),
        dt=np.asarray(dt, dtype=np.float32),
        expert_recipe_requested=np.asarray(expert_recipe_requested),
        expert_source_used=np.asarray(expert_source_used),
        expert_source=np.asarray(expert_source_used),
        fallback_reason=np.asarray(fallback_reason),
        agent_count=np.asarray(positions.shape[1], dtype=np.int32),
        rollout_length=np.asarray(len(velocities), dtype=np.int32),
        fraction_moving=np.asarray(
            float(np.mean(np.linalg.norm(velocities, axis=2) >= wait_threshold)) if len(velocities) else 0.0,
            dtype=np.float32,
        ),
        mean_nearest_neighbor_distance=np.asarray(
            _mean_nearest_neighbor_distance(positions[0]) if len(positions) else 0.0,
            dtype=np.float32,
        ),
        action_labels=derive_action_labels(velocities, num_directions, wait_threshold),
    )


def build_map_scenario_pairs(
    map_dir: str,
    scen_dir: str,
    maps: Optional[List[str]],
    max_scenarios: int,
    scenario_ids: Optional[List[int]] = None,
    scenario_start: Optional[int] = None,
    scenario_end: Optional[int] = None,
) -> List[Tuple[str, str, str]]:
    pairs = []
    if maps:
        map_paths = [os.path.join(map_dir, f"{m}.map") for m in maps]
    else:
        map_paths = sorted(glob.glob(os.path.join(map_dir, "*.map")))
    for map_path in map_paths:
        map_name = os.path.basename(map_path).replace(".map", "")
        scen_paths = select_scenarios(
            glob.glob(os.path.join(scen_dir, f"{map_name}-random-*.scen")),
            max_scenarios=max_scenarios,
            scenario_ids=scenario_ids,
            scenario_start=scenario_start,
            scenario_end=scenario_end,
        )
        for scen_path in scen_paths:
            pairs.append((map_name, map_path, scen_path))
    return pairs


def generate_single_rollout(task, args, eecbs_binary: Optional[str]) -> str:
    map_name, map_path, scen_path, agent_num = task
    obstacle_map = load_grid_map_from_file(map_path)
    starts_grid, goals_grid = parse_scene_file(scen_path, agent_num=agent_num)
    starts = grid_starts_to_continuous(starts_grid)
    goals = grid_starts_to_continuous(goals_grid)

    positions = None
    velocities = None
    source_used = ""
    fallback_reason = "none"

    if args.expert_source in {"eecbs", "hybrid"}:
        try:
            discrete_paths = run_eecbs(
                map_path,
                scen_path,
                agent_num,
                args.suboptimality,
                args.time_limit,
                eecbs_binary,
            )
            positions, velocities = discrete_paths_to_continuous(discrete_paths, args.dt, args.max_speed)
            if not validate_replay(
                obstacle_map,
                starts,
                goals,
                positions,
                velocities,
                args.dt,
                args.max_speed,
                args.agent_radius,
                args.goal_tolerance,
            ):
                fallback_reason = "replay_validation_failed"
                positions, velocities = None, None
            else:
                source_used = "eecbs"
        except Exception as e:
            fallback_reason = "eecbs_failed"
            positions, velocities = None, None

    if positions is None and args.expert_source in {"orca", "po-orca", "hybrid"}:
        shield = "po-orca" if args.expert_source in {"po-orca", "hybrid"} else "orca"
        positions, velocities = rollout_orca_policy(
            obstacle_map,
            starts,
            goals,
            args.dt,
            args.max_speed,
            args.rollout_horizon,
            args.agent_radius,
            args.goal_tolerance,
            shield_type=shield,
        )
        source_used = shield
        if args.expert_source in {"orca", "po-orca"}:
            fallback_reason = "none"

    if positions is None or velocities is None:
        return f"[skip] {map_name} {os.path.basename(scen_path)} N={agent_num} (expert failed)"

    scenario_name = os.path.basename(scen_path).replace(".scen", "")
    scenario_id = scenario_id_from_path(scen_path)
    output_path = os.path.join(args.output_dir, f"{scenario_name}_{agent_num}.npz")
    save_rollout(
        output_path,
        map_name,
        scenario_name,
        scenario_id,
        positions,
        velocities,
        goals,
        args.dt,
        args.expert_source,
        source_used,
        fallback_reason,
        args.num_directions,
        args.wait_threshold,
    )
    return f"saved {output_path} [{source_used}]"


def main():
    parser = argparse.ArgumentParser(description="Generate continuous MAPF supervision")
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--scen-dir", required=True)
    parser.add_argument("--maps", nargs="*", default=None)
    parser.add_argument("--agent-counts", nargs="+", type=int, default=[100])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expert-source", choices=["eecbs", "orca", "po-orca", "hybrid"], default="hybrid")
    parser.add_argument("--max-scenarios", type=int, default=1)
    parser.add_argument("--scenario-ids", nargs="*", type=int, default=None)
    parser.add_argument("--scenario-start", type=int, default=None)
    parser.add_argument("--scenario-end", type=int, default=None)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.3)
    parser.add_argument("--goal-tolerance", type=float, default=0.25)
    parser.add_argument("--wait-threshold", type=float, default=0.1)
    parser.add_argument("--num-directions", type=int, default=8)
    parser.add_argument("--eecbs-repo", default=DEFAULT_EECBS_REPO)
    parser.add_argument("--eecbs-binary", default=None)
    parser.add_argument("--suboptimality", type=float, default=1.2)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    eecbs_binary = None
    if args.expert_source in {"eecbs", "hybrid"}:
        try:
            eecbs_binary = resolve_eecbs_binary(args.eecbs_binary, args.eecbs_repo)
            print(f"Using EECBS binary: {eecbs_binary}")
        except FileNotFoundError as e:
            if args.expert_source == "eecbs":
                raise
            print(f"[hybrid] EECBS unavailable, will fall back to ORCA:\n{e}")

    pairs = build_map_scenario_pairs(
        args.map_dir,
        args.scen_dir,
        args.maps,
        args.max_scenarios,
        scenario_ids=args.scenario_ids,
        scenario_start=args.scenario_start,
        scenario_end=args.scenario_end,
    )
    if not pairs:
        raise RuntimeError("No map/scenario pairs found")
    tasks = [
        (map_name, map_path, scen_path, agent_num)
        for map_name, map_path, scen_path in pairs
        for agent_num in args.agent_counts
    ]

    workers = max(1, args.workers)
    if workers == 1:
        for task in tasks:
            print(generate_single_rollout(task, args, eecbs_binary))
        return

    worker_count = min(workers, cpu_count(), len(tasks))
    print(f"Generating {len(tasks)} rollouts with {worker_count} workers")
    with Pool(worker_count) as pool:
        worker_fn = partial(generate_single_rollout, args=args, eecbs_binary=eecbs_binary)
        for message in pool.imap_unordered(worker_fn, tasks, chunksize=1):
            print(message)


if __name__ == "__main__":
    main()
