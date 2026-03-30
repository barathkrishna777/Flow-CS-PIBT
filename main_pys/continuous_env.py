import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from main_pys.model_inputs import load_grid_map_from_file

try:
    import rvo2  # type: ignore
except ImportError:
    rvo2 = None


def parse_scene_file(scen_file: str, agent_num: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    start_locations = []
    goal_locations = []
    with open(scen_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("version"):
                continue
            tokens = line.split("\t")
            if len(tokens) != 9:
                continue
            tokens = tokens[4:]
            start_locations.append((int(tokens[1]), int(tokens[0])))
            goal_locations.append((int(tokens[3]), int(tokens[2])))
            if agent_num is not None and len(start_locations) >= agent_num:
                break
    return np.asarray(start_locations, dtype=np.int32), np.asarray(goal_locations, dtype=np.int32)


def grid_starts_to_continuous(starts: np.ndarray) -> np.ndarray:
    return starts.astype(np.float32) + 0.5


def circle_intersects_rect(center: np.ndarray, radius: float, row: int, col: int) -> bool:
    nearest_r = np.clip(center[0], row, row + 1.0)
    nearest_c = np.clip(center[1], col, col + 1.0)
    diff = center - np.array([nearest_r, nearest_c], dtype=np.float32)
    return float(diff @ diff) < radius ** 2


def compute_smoothness(velocities: np.ndarray) -> float:
    if len(velocities) < 2:
        return 0.0
    accel = np.diff(velocities, axis=0)
    return float(np.mean(np.linalg.norm(accel, axis=2)))


def compute_path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    deltas = np.diff(positions, axis=0)
    return float(np.linalg.norm(deltas, axis=2).sum())


@dataclass
class StepMetrics:
    collisions: int = 0
    near_collisions: int = 0
    obstacle_hits: int = 0


class ORCAStyleShield:
    """ORCA-backed shield with heuristic fallback."""

    def __init__(
        self,
        agent_radius: float,
        max_speed: float,
        dt: float,
        time_horizon: float = 2.0,
        obstacle_horizon: float = 1.0,
        iterations: int = 3,
    ) -> None:
        self.agent_radius = agent_radius
        self.max_speed = max_speed
        self.dt = dt
        self.time_horizon = time_horizon
        self.obstacle_horizon = obstacle_horizon
        self.iterations = iterations
        self.neighbor_dist = max(4.0 * agent_radius, 2.0)
        self.max_neighbors = 16
        self._obstacle_cache = {}

    def project(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        obstacle_map: np.ndarray,
        use_true_orca: bool = True,
    ) -> np.ndarray:
        if use_true_orca and rvo2 is not None:
            try:
                return self._project_with_rvo2(positions, preferred_velocities, obstacle_map)
            except Exception:
                # Keep the heuristic path available as a robust fallback.
                pass
        return self._project_heuristic(positions, preferred_velocities, obstacle_map)

    def _project_with_rvo2(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        obstacle_map: np.ndarray,
    ) -> np.ndarray:
        safe = np.asarray(preferred_velocities, dtype=np.float32).copy()
        safe = self._clip_speeds(safe)

        sim = rvo2.PyRVOSimulator(
            self.dt,
            self.neighbor_dist,
            self.max_neighbors,
            self.time_horizon,
            self.obstacle_horizon,
            self.agent_radius,
            self.max_speed,
        )

        for polygon in self._obstacle_polygons(obstacle_map):
            sim.addObstacle(polygon)
        sim.processObstacles()

        for pos, vel in zip(np.asarray(positions, dtype=np.float32), safe):
            sim.addAgent(
                tuple(float(x) for x in pos),
                self.neighbor_dist,
                self.max_neighbors,
                self.time_horizon,
                self.obstacle_horizon,
                self.agent_radius,
                self.max_speed,
                tuple(float(x) for x in vel),
            )

        for agent_idx, vel in enumerate(safe):
            sim.setAgentPrefVelocity(agent_idx, tuple(float(x) for x in vel))

        sim.doStep()
        projected = np.zeros_like(safe)
        for agent_idx in range(len(safe)):
            projected[agent_idx] = np.asarray(sim.getAgentVelocity(agent_idx), dtype=np.float32)
        return self._clip_speeds(projected)

    def _project_heuristic(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        obstacle_map: np.ndarray,
    ) -> np.ndarray:
        safe = np.asarray(preferred_velocities, dtype=np.float32).copy()
        safe = self._clip_speeds(safe)

        for _ in range(self.iterations):
            safe = self._apply_pairwise_constraints(positions, safe)
            safe = self._apply_obstacle_constraints(positions, safe, obstacle_map)
            safe = self._clip_speeds(safe)
        return safe

    def _obstacle_polygons(self, obstacle_map: np.ndarray):
        key = (obstacle_map.shape, obstacle_map.tobytes())
        cached = self._obstacle_cache.get(key)
        if cached is not None:
            return cached

        polygons = []
        rows, cols = np.where(obstacle_map == 1)
        inflate = self.agent_radius
        for r, c in zip(rows.tolist(), cols.tolist()):
            polygons.append(
                [
                    (r - inflate, c - inflate),
                    (r - inflate, c + 1.0 + inflate),
                    (r + 1.0 + inflate, c + 1.0 + inflate),
                    (r + 1.0 + inflate, c - inflate),
                ]
            )
        self._obstacle_cache[key] = polygons
        return polygons

    def _clip_speeds(self, velocities: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        scale = np.maximum(norms / max(self.max_speed, 1e-6), 1.0)
        return velocities / scale

    def _apply_pairwise_constraints(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        adjusted = velocities.copy()
        min_dist = 2.0 * self.agent_radius
        min_dist_sq = min_dist ** 2
        n_agents = len(positions)
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                rel_pos = positions[j] - positions[i]
                dist_sq = float(rel_pos @ rel_pos)
                rel_vel = adjusted[i] - adjusted[j]
                next_rel = rel_pos + rel_vel * self.dt
                next_dist_sq = float(next_rel @ next_rel)
                if dist_sq > min_dist_sq and next_dist_sq > min_dist_sq:
                    continue

                if dist_sq < 1e-8:
                    normal = np.array([1.0, 0.0], dtype=np.float32)
                else:
                    normal = rel_pos / math.sqrt(dist_sq)

                overlap = max(min_dist - math.sqrt(max(dist_sq, 1e-8)), 0.0)
                closing = np.dot(rel_vel, normal)
                correction_mag = overlap / max(self.dt, 1e-6)
                if closing > 0:
                    correction_mag += closing
                correction = normal * 0.5 * correction_mag
                adjusted[i] -= correction
                adjusted[j] += correction
        return adjusted

    def _apply_obstacle_constraints(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        obstacle_map: np.ndarray,
    ) -> np.ndarray:
        adjusted = velocities.copy()
        height, width = obstacle_map.shape
        for i, pos in enumerate(positions):
            proposed = pos + adjusted[i] * self.dt
            row_min = max(int(math.floor(proposed[0] - self.agent_radius)) - 1, 0)
            row_max = min(int(math.ceil(proposed[0] + self.agent_radius)) + 1, height - 1)
            col_min = max(int(math.floor(proposed[1] - self.agent_radius)) - 1, 0)
            col_max = min(int(math.ceil(proposed[1] + self.agent_radius)) + 1, width - 1)
            push = np.zeros(2, dtype=np.float32)
            for r in range(row_min, row_max + 1):
                for c in range(col_min, col_max + 1):
                    if obstacle_map[r, c] == 0:
                        continue
                    if not circle_intersects_rect(proposed, self.agent_radius, r, c):
                        continue
                    nearest_r = np.clip(proposed[0], r, r + 1.0)
                    nearest_c = np.clip(proposed[1], c, c + 1.0)
                    diff = proposed - np.array([nearest_r, nearest_c], dtype=np.float32)
                    dist = np.linalg.norm(diff)
                    if dist < 1e-6:
                        cell_center = np.array([r + 0.5, c + 0.5], dtype=np.float32)
                        diff = proposed - cell_center
                        dist = np.linalg.norm(diff)
                    normal = diff / max(dist, 1e-6)
                    push += normal * (self.agent_radius - min(dist, self.agent_radius))
            if np.any(push):
                adjusted[i] += push / max(self.dt, 1e-6)
        return adjusted


class ContinuousMAPFEnv:
    def __init__(
        self,
        obstacle_map: np.ndarray,
        dt: float = 0.2,
        max_speed: float = 1.0,
        agent_radius: float = 0.3,
        goal_tolerance: float = 0.25,
    ) -> None:
        self.obstacle_map = np.asarray(obstacle_map, dtype=np.int8)
        self.dt = dt
        self.max_speed = max_speed
        self.agent_radius = agent_radius
        self.goal_tolerance = goal_tolerance
        self._shield = ORCAStyleShield(agent_radius=agent_radius, max_speed=max_speed, dt=dt)
        self.positions = None
        self.goals = None
        self.history_positions = []
        self.history_velocities = []
        self.metrics = StepMetrics()
        self.arrival_steps = None
        self.step_count = 0

    def reset(self, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
        self.positions = np.asarray(starts, dtype=np.float32).copy()
        self.goals = np.asarray(goals, dtype=np.float32).copy()
        self.history_positions = [self.positions.copy()]
        self.history_velocities = []
        self.metrics = StepMetrics()
        self.arrival_steps = np.full(len(self.positions), -1, dtype=np.int32)
        self.step_count = 0
        return self.positions.copy()

    def goal_directed_velocities(self) -> np.ndarray:
        delta = self.goals - self.positions
        norms = np.linalg.norm(delta, axis=1, keepdims=True)
        velocities = np.zeros_like(delta)
        moving = norms[:, 0] > self.goal_tolerance
        velocities[moving] = delta[moving] / np.maximum(norms[moving], 1e-6) * self.max_speed
        return velocities

    def apply_shield(self, preferred_velocities: np.ndarray, shield_type: str = "orca") -> np.ndarray:
        preferred_velocities = np.asarray(preferred_velocities, dtype=np.float32)
        if shield_type == "none":
            return self._clip_speeds(preferred_velocities)
        if shield_type == "simple":
            return self._simple_safety_filter(preferred_velocities)
        if shield_type == "heuristic-orca":
            return self._shield.project(
                self.positions,
                preferred_velocities,
                self.obstacle_map,
                use_true_orca=False,
            )
        if shield_type != "orca":
            raise ValueError(f"Unsupported shield type: {shield_type}")
        return self._shield.project(
            self.positions,
            preferred_velocities,
            self.obstacle_map,
            use_true_orca=True,
        )

    def step(self, velocities: np.ndarray, shield_type: str = "orca") -> Tuple[np.ndarray, bool, Dict[str, float]]:
        safe_velocities = self.apply_shield(velocities, shield_type=shield_type)
        proposed = self.positions + safe_velocities * self.dt

        obstacle_hits = 0
        for i, candidate in enumerate(proposed):
            if self._position_hits_obstacle(candidate):
                obstacle_hits += 1
                proposed[i] = self.positions[i]
                safe_velocities[i] = 0.0

        collisions, near_collisions = self._count_agent_interactions(proposed)
        self.metrics.collisions += collisions
        self.metrics.near_collisions += near_collisions
        self.metrics.obstacle_hits += obstacle_hits

        self.positions = proposed
        self.history_positions.append(self.positions.copy())
        self.history_velocities.append(safe_velocities.copy())
        self.step_count += 1

        done_mask = self.agents_at_goal()
        newly_done = (self.arrival_steps < 0) & done_mask
        self.arrival_steps[newly_done] = self.step_count
        return self.positions.copy(), self.is_done(), self.current_metrics()

    def current_metrics(self) -> Dict[str, float]:
        at_goal = self.agents_at_goal()
        positions = np.asarray(self.history_positions, dtype=np.float32)
        velocities = np.asarray(self.history_velocities, dtype=np.float32) if self.history_velocities else np.zeros((0, len(self.positions), 2), dtype=np.float32)
        direct = np.linalg.norm(self.goals - self.history_positions[0], axis=1).sum()
        path_length = compute_path_length(positions)
        return {
            "success": float(np.all(at_goal)),
            "agents_at_goal": float(at_goal.sum()),
            "agent_fraction_at_goal": float(np.mean(at_goal)),
            "path_length": path_length,
            "path_length_ratio": float(path_length / max(direct, 1e-6)),
            "smoothness": compute_smoothness(velocities),
            "collisions": float(self.metrics.collisions),
            "near_collisions": float(self.metrics.near_collisions),
            "obstacle_hits": float(self.metrics.obstacle_hits),
            "mean_arrival_step": float(np.mean(np.where(self.arrival_steps >= 0, self.arrival_steps, self.step_count))),
        }

    def is_done(self) -> bool:
        return bool(np.all(self.agents_at_goal()))

    def agents_at_goal(self) -> np.ndarray:
        dists = np.linalg.norm(self.positions - self.goals, axis=1)
        return dists <= self.goal_tolerance

    def _clip_speeds(self, velocities: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        scale = np.maximum(norms / max(self.max_speed, 1e-6), 1.0)
        return velocities / scale

    def _simple_safety_filter(self, velocities: np.ndarray) -> np.ndarray:
        safe = self._clip_speeds(velocities)
        proposed = self.positions + safe * self.dt
        for i, candidate in enumerate(proposed):
            if self._position_hits_obstacle(candidate):
                safe[i] = 0.0
                continue
            for j in range(len(proposed)):
                if i == j:
                    continue
                if np.linalg.norm(candidate - proposed[j]) < 2.0 * self.agent_radius:
                    safe[i] = 0.0
                    break
        return safe

    def _position_hits_obstacle(self, position: np.ndarray) -> bool:
        if (
            position[0] < self.agent_radius
            or position[1] < self.agent_radius
            or position[0] > self.obstacle_map.shape[0] - self.agent_radius
            or position[1] > self.obstacle_map.shape[1] - self.agent_radius
        ):
            return True

        row_min = max(int(math.floor(position[0] - self.agent_radius)) - 1, 0)
        row_max = min(int(math.ceil(position[0] + self.agent_radius)) + 1, self.obstacle_map.shape[0] - 1)
        col_min = max(int(math.floor(position[1] - self.agent_radius)) - 1, 0)
        col_max = min(int(math.ceil(position[1] + self.agent_radius)) + 1, self.obstacle_map.shape[1] - 1)
        for r in range(row_min, row_max + 1):
            for c in range(col_min, col_max + 1):
                if self.obstacle_map[r, c] == 1 and circle_intersects_rect(position, self.agent_radius, r, c):
                    return True
        return False

    def _count_agent_interactions(self, positions: np.ndarray) -> Tuple[int, int]:
        collisions = 0
        near_collisions = 0
        min_dist = 2.0 * self.agent_radius
        near_dist = 2.5 * self.agent_radius
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < min_dist:
                    collisions += 1
                elif dist < near_dist:
                    near_collisions += 1
        return collisions, near_collisions


def load_env_from_files(
    map_file: str,
    scen_file: str,
    agent_num: int,
    dt: float = 0.2,
    max_speed: float = 1.0,
    agent_radius: float = 0.3,
    goal_tolerance: float = 0.25,
) -> Tuple[ContinuousMAPFEnv, np.ndarray, np.ndarray]:
    obstacle_map = load_grid_map_from_file(map_file)
    starts, goals = parse_scene_file(scen_file, agent_num=agent_num)
    env = ContinuousMAPFEnv(
        obstacle_map=obstacle_map,
        dt=dt,
        max_speed=max_speed,
        agent_radius=agent_radius,
        goal_tolerance=goal_tolerance,
    )
    return env, grid_starts_to_continuous(starts), grid_starts_to_continuous(goals)
