import importlib
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    import rvo2  # type: ignore
except ImportError:
    rvo2 = None


def compute_sdf(obstacle_map: np.ndarray) -> np.ndarray:
    """Compute a signed distance field from a binary obstacle map.

    Returns an array where free cells have positive distance to the nearest
    obstacle and obstacle cells have negative distance to the nearest free cell.
    Units are in grid cells (1 cell = 1 unit length).
    """
    free_mask = obstacle_map == 0
    # Distance from each free cell to the nearest obstacle
    dist_to_obstacle = distance_transform_edt(free_mask)
    # Distance from each obstacle cell to the nearest free cell
    dist_to_free = distance_transform_edt(~free_mask)
    sdf = dist_to_obstacle - dist_to_free
    return sdf.astype(np.float32)


def sdf_gradient(sdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the gradient of the SDF using central differences.

    Returns (grad_row, grad_col) arrays of the same shape as sdf.
    """
    grad_row = np.zeros_like(sdf)
    grad_col = np.zeros_like(sdf)
    # Central differences, forward/backward at boundaries
    grad_row[1:-1] = (sdf[2:] - sdf[:-2]) / 2.0
    grad_row[0] = sdf[1] - sdf[0]
    grad_row[-1] = sdf[-1] - sdf[-2]
    grad_col[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / 2.0
    grad_col[:, 0] = sdf[:, 1] - sdf[:, 0]
    grad_col[:, -1] = sdf[:, -1] - sdf[:, -2]
    return grad_row, grad_col


def sample_sdf_bilinear(sdf: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Sample SDF values at continuous positions using bilinear interpolation.

    Args:
        sdf: (H, W) signed distance field.
        positions: (N, 2) continuous positions (row, col).

    Returns:
        (N,) interpolated SDF values.
    """
    h, w = sdf.shape
    r = np.clip(positions[:, 0], 0, h - 1.001)
    c = np.clip(positions[:, 1], 0, w - 1.001)
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    dr = r - r0
    dc = c - c0
    val = (
        sdf[r0, c0] * (1 - dr) * (1 - dc)
        + sdf[r1, c0] * dr * (1 - dc)
        + sdf[r0, c1] * (1 - dr) * dc
        + sdf[r1, c1] * dr * dc
    )
    return val.astype(np.float32)


def sample_sdf_gradient_bilinear(
    grad_row: np.ndarray, grad_col: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Sample SDF gradient at continuous positions using bilinear interpolation.

    Returns (N, 2) gradient vectors (row_grad, col_grad).
    """
    gr = sample_sdf_bilinear(grad_row, positions)
    gc = sample_sdf_bilinear(grad_col, positions)
    return np.stack([gr, gc], axis=1)


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


def default_picbf_communication_radius(
    agent_radius: float,
    max_speed: float,
    dt: float,
    safety_margin: float,
) -> float:
    return (
        2.0 * float(agent_radius)
        + float(safety_margin)
        + 2.0 * float(max_speed) * float(dt)
        + max(0.4, 4.0 * float(safety_margin))
    )


def _import_picbf_cs():
    """Import picbf-cs from the environment or PICBF_CS_PATH.

    ``picbf-cs`` is developed in a sibling repo during experiments, so the
    eval process can either install it normally or set PICBF_CS_PATH to the
    repo root (or directly to its ``src`` directory).
    """

    if sys.version_info < (3, 11):
        raise RuntimeError(
            "shield_type='picbf-cs' requires Python 3.11 or newer because the current "
            "picbf-cs package declares requires-python >=3.11. Run evals with python3.11 "
            "or apply a separate picbf-cs compatibility patch."
        )

    try:
        module = importlib.import_module("continuous_collision_shield")
    except ImportError as first_error:
        picbf_path = os.environ.get("PICBF_CS_PATH")
        if not picbf_path:
            raise ImportError(
                "shield_type='picbf-cs' requires the continuous-collision-shield package. "
                "Install picbf-cs or set PICBF_CS_PATH to the picbf-cs repo root/src directory."
            ) from first_error

        candidate = os.path.abspath(os.path.expanduser(picbf_path))
        candidate_src = os.path.join(candidate, "src")
        src_dir = candidate_src if os.path.isdir(candidate_src) else candidate
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        try:
            module = importlib.import_module("continuous_collision_shield")
        except ImportError as second_error:
            raise ImportError(
                f"Unable to import continuous_collision_shield from PICBF_CS_PATH={picbf_path!r}"
            ) from second_error

    required_names = (
        "LocalJointCBFShield",
        "LocalAgentState",
        "AgentState",
        "AABBObstacle",
        "ObstacleIndex",
        "ShieldConfig",
    )
    missing = [name for name in required_names if not hasattr(module, name)]
    if missing:
        raise ImportError(
            "Installed continuous_collision_shield is too old for shield_type='picbf-cs'; "
            f"missing {', '.join(missing)}. Pull/update picbf-cs so it exposes "
            "LocalJointCBFShield, LocalAgentState, ObstacleIndex, and ShieldConfig."
        )

    return (
        module.LocalJointCBFShield,
        module.LocalAgentState,
        module.AgentState,
        module.AABBObstacle,
        module.ObstacleIndex,
        module.ShieldConfig,
    )


class PICBFCSShield:
    """Adapter from this repo's row/col arrays to picbf-cs AgentState inputs."""

    def __init__(
        self,
        agent_radius: float,
        max_speed: float,
        dt: float,
        safety_margin: float = 0.05,
        alpha: float = 2.0,
        communication_radius: Optional[float] = None,
    ) -> None:
        (
            LocalJointCBFShield,
            LocalAgentState,
            AgentState,
            AABBObstacle,
            ObstacleIndex,
            ShieldConfig,
        ) = _import_picbf_cs()
        self.agent_radius = float(agent_radius)
        self.max_speed = float(max_speed)
        self.dt = float(dt)
        self.safety_margin = float(safety_margin)
        if communication_radius is None:
            communication_radius = default_picbf_communication_radius(
                self.agent_radius,
                self.max_speed,
                self.dt,
                self.safety_margin,
            )
        communication_radius = float(communication_radius)
        if communication_radius <= 0.0:
            raise ValueError("picbf communication_radius must be positive")
        self.communication_radius = communication_radius
        self._shield_cls = LocalJointCBFShield
        self._local_agent_state_cls = LocalAgentState
        self._agent_state_cls = AgentState
        self._aabb_obstacle_cls = AABBObstacle
        self._obstacle_index_cls = ObstacleIndex
        self._config = ShieldConfig(dt=self.dt, safety_margin=self.safety_margin, alpha=alpha)
        self._obstacle_cache: Dict[Tuple, Tuple] = {}
        self._shield_cache: Dict[Tuple, object] = {}
        self.last_debug_info: Dict[str, object] = {}

    def project(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        obstacle_map: np.ndarray,
        goals: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float32)
        preferred_velocities = self._clip_speeds(
            np.asarray(preferred_velocities, dtype=np.float32)
        )
        goals = None if goals is None else np.asarray(goals, dtype=np.float32)

        agents = []
        nominal_velocities = {}
        for idx, position in enumerate(positions):
            goal = None
            if goals is not None:
                goal = tuple(float(x) for x in goals[idx])
            agent_state = self._agent_state_cls(
                position=tuple(float(x) for x in position),
                radius=self.agent_radius,
                max_speed=self.max_speed,
                goal=goal,
            )
            agents.append(
                self._local_agent_state_cls(
                    agent_id=idx,
                    state=agent_state,
                )
            )
            nominal_velocities[idx] = tuple(float(x) for x in preferred_velocities[idx])

        result = self._get_shield(obstacle_map).step(agents, nominal_velocities)
        self.last_debug_info = self._debug_info_from_result(result)
        projected = np.asarray(
            [result.velocities_by_id[idx] for idx in range(len(positions))],
            dtype=np.float32,
        )
        return self._clip_speeds(projected)

    def _get_shield(self, obstacle_map: np.ndarray):
        key = self._obstacle_cache_key(obstacle_map)
        cached = self._shield_cache.get(key)
        if cached is not None:
            return cached
        obstacles, obstacle_index = self._get_obstacle_data(obstacle_map, key=key)
        shield = self._shield_cls(
            self._config,
            self.communication_radius,
            obstacles=obstacles,
            obstacle_index=obstacle_index,
        )
        self._shield_cache[key] = shield
        return shield

    def _get_obstacle_data(self, obstacle_map: np.ndarray, key: Optional[Tuple] = None) -> Tuple:
        obstacle_map = np.asarray(obstacle_map)
        key = self._obstacle_cache_key(obstacle_map) if key is None else key
        cached = self._obstacle_cache.get(key)
        if cached is not None:
            return cached

        obstacles = list(self._blocked_cells_to_aabbs(obstacle_map))
        rows, cols = obstacle_map.shape
        obstacles.extend(
            [
                self._aabb_obstacle_cls(
                    min_corner=(-1.0, 0.0),
                    max_corner=(0.0, float(cols)),
                ),
                self._aabb_obstacle_cls(
                    min_corner=(float(rows), 0.0),
                    max_corner=(float(rows + 1), float(cols)),
                ),
                self._aabb_obstacle_cls(
                    min_corner=(0.0, -1.0),
                    max_corner=(float(rows), 0.0),
                ),
                self._aabb_obstacle_cls(
                    min_corner=(0.0, float(cols)),
                    max_corner=(float(rows), float(cols + 1)),
                ),
            ]
        )
        obstacle_tuple = tuple(obstacles)
        cached = (obstacle_tuple, self._obstacle_index_cls(obstacle_tuple))
        self._obstacle_cache[key] = cached
        return cached

    @staticmethod
    def _obstacle_cache_key(obstacle_map: np.ndarray) -> Tuple:
        obstacle_map = np.asarray(obstacle_map)
        return (obstacle_map.shape, np.ascontiguousarray(obstacle_map).tobytes())

    @staticmethod
    def _debug_info_from_result(result) -> Dict[str, object]:
        components = tuple(getattr(result, "components", ()))
        component_sizes = [len(component.agent_ids) for component in components]
        status_counts: Dict[str, int] = {}
        total_solve_time = 0.0
        for component in components:
            component_result = getattr(component, "result", None)
            if component_result is None:
                continue
            total_solve_time += float(getattr(component_result, "solve_time_s", 0.0))
            status = getattr(component_result, "status", None)
            if status is not None:
                status_key = getattr(status, "value", str(status))
                status_counts[status_key] = status_counts.get(status_key, 0) + 1
        debug_info: Dict[str, object] = {
            "component_count": len(components),
            "max_component_size": max(component_sizes, default=0),
            "total_local_solver_time": total_solve_time,
        }
        if status_counts:
            debug_info["status_counts"] = status_counts
        return debug_info

    def _blocked_cells_to_aabbs(self, obstacle_map: np.ndarray) -> Tuple:
        # Scenario grid coordinates are converted to continuous cell centers via
        # ``grid + 0.5``, so blocked cell (r, c) spans [r, r+1] x [c, c+1].
        active_runs = {}
        finished = []
        for row_idx in range(obstacle_map.shape[0]):
            row_runs = self._blocked_runs(obstacle_map[row_idx])
            next_active = {}
            for col_start, col_end in row_runs:
                key = (col_start, col_end)
                if key in active_runs:
                    old_row_min, old_row_max = active_runs[key]
                    if abs(float(row_idx) - old_row_max) <= 1.0e-12:
                        next_active[key] = (old_row_min, float(row_idx + 1))
                        continue
                next_active[key] = (float(row_idx), float(row_idx + 1))
            for key, (row_min, row_max) in active_runs.items():
                if key not in next_active:
                    finished.append(
                        self._aabb_obstacle_cls(
                            min_corner=(row_min, float(key[0])),
                            max_corner=(row_max, float(key[1])),
                        )
                    )
            active_runs = next_active

        for key, (row_min, row_max) in active_runs.items():
            finished.append(
                self._aabb_obstacle_cls(
                    min_corner=(row_min, float(key[0])),
                    max_corner=(row_max, float(key[1])),
                )
            )
        return tuple(
            sorted(finished, key=lambda obstacle: (obstacle.min_corner, obstacle.max_corner))
        )

    @staticmethod
    def _blocked_runs(row: np.ndarray) -> Tuple[Tuple[int, int], ...]:
        runs = []
        start = None
        for col_idx, value in enumerate(row):
            if value != 0 and start is None:
                start = col_idx
            elif value == 0 and start is not None:
                runs.append((start, col_idx))
                start = None
        if start is not None:
            runs.append((start, len(row)))
        return tuple(runs)

    def _clip_speeds(self, velocities: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        scale = np.maximum(norms / max(self.max_speed, 1e-6), 1.0)
        return velocities / scale


class ORCAStyleShield:
    """ORCA-backed shield with SDF obstacle handling and optional priority ordering.

    Supports three modes via ``project()``:
    - ``use_priorities=False``: Standard symmetric ORCA (50/50 correction split).
    - ``use_priorities=True``: Priority-Ordered ORCA (PO-ORCA) where agents are
      processed sequentially in descending priority.  Higher-priority agents
      commit their velocities first; lower-priority agents treat them as fixed
      velocity obstacles and bear the full avoidance burden.

    Obstacle handling uses a precomputed Signed Distance Field (SDF) instead of
    per-cell polygon inflation, which correctly handles adjacent obstacle cells.
    """

    def __init__(
        self,
        agent_radius: float,
        max_speed: float,
        dt: float,
        time_horizon: float = 2.0,
        obstacle_horizon: float = 1.0,
        iterations: int = 3,
        obstacle_repulsion_gain: float = 2.0,
    ) -> None:
        self.agent_radius = agent_radius
        self.max_speed = max_speed
        self.dt = dt
        self.time_horizon = time_horizon
        self.obstacle_horizon = obstacle_horizon
        self.iterations = iterations
        self.obstacle_repulsion_gain = obstacle_repulsion_gain
        self.neighbor_dist = max(4.0 * agent_radius, 2.0)
        self.max_neighbors = 16
        self._sdf_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        obstacle_map: np.ndarray,
        use_true_orca: bool = True,
        priorities: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Project preferred velocities to collision-free velocities.

        Args:
            positions: (N, 2) agent positions.
            preferred_velocities: (N, 2) desired velocities.
            obstacle_map: (H, W) binary obstacle grid.
            use_true_orca: If True and rvo2 is available, use rvo2 for
                agent-agent avoidance (only when priorities is None).
            priorities: (N,) optional priority values.  When provided, enables
                Priority-Ordered ORCA (sequential processing, highest first).
        """
        use_priorities = priorities is not None
        # rvo2 does not support asymmetric priorities, so only use it for
        # the standard symmetric mode.
        if not use_priorities and use_true_orca and rvo2 is not None:
            try:
                return self._project_with_rvo2(positions, preferred_velocities, obstacle_map)
            except Exception:
                pass

        if use_priorities:
            return self._project_priority_ordered(
                positions, preferred_velocities, obstacle_map, priorities
            )
        return self._project_heuristic(positions, preferred_velocities, obstacle_map)

    # ------------------------------------------------------------------
    # SDF helpers (cached per obstacle map)
    # ------------------------------------------------------------------

    def _get_sdf(
        self, obstacle_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (obstacle_map.shape, obstacle_map.data.tobytes())
        cached = self._sdf_cache.get(key)
        if cached is not None:
            return cached
        sdf = compute_sdf(obstacle_map)
        grad_r, grad_c = sdf_gradient(sdf)
        self._sdf_cache[key] = (sdf, grad_r, grad_c)
        return sdf, grad_r, grad_c

    # ------------------------------------------------------------------
    # rvo2 path (symmetric, no priorities, kept for backwards compat)
    # ------------------------------------------------------------------

    def _project_with_rvo2(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        obstacle_map: np.ndarray,
    ) -> np.ndarray:
        safe = np.asarray(preferred_velocities, dtype=np.float32).copy()
        safe = self._clip_speeds(safe)

        # Apply SDF obstacle constraints first, then let rvo2 handle agents
        sdf, grad_r, grad_c = self._get_sdf(obstacle_map)
        safe = self._apply_sdf_obstacle_constraints(positions, safe, sdf, grad_r, grad_c)
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
        # No obstacle polygons passed to rvo2 -- SDF handles obstacles

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

    # ------------------------------------------------------------------
    # Heuristic path (symmetric, no priorities)
    # ------------------------------------------------------------------

    def _project_heuristic(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        obstacle_map: np.ndarray,
    ) -> np.ndarray:
        safe = np.asarray(preferred_velocities, dtype=np.float32).copy()
        safe = self._clip_speeds(safe)
        sdf, grad_r, grad_c = self._get_sdf(obstacle_map)

        for _ in range(self.iterations):
            safe = self._apply_pairwise_constraints(positions, safe)
            safe = self._apply_sdf_obstacle_constraints(positions, safe, sdf, grad_r, grad_c)
            safe = self._clip_speeds(safe)
        return safe

    # ------------------------------------------------------------------
    # Priority-Ordered ORCA (PO-ORCA) -- sequential processing
    # ------------------------------------------------------------------

    def _project_priority_ordered(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        obstacle_map: np.ndarray,
        priorities: np.ndarray,
    ) -> np.ndarray:
        """Process agents sequentially in descending priority order.

        Higher-priority agents commit their velocities first.  When processing
        a lower-priority agent, already-committed agents are treated as fixed
        velocity obstacles -- the lower-priority agent bears the full
        avoidance correction.

        After sequential processing, a symmetric cleanup pass resolves any
        remaining conflicts between agents of similar priority that were not
        handled during the sequential phase.
        """
        safe = np.asarray(preferred_velocities, dtype=np.float32).copy()
        safe = self._clip_speeds(safe)
        sdf, grad_r, grad_c = self._get_sdf(obstacle_map)

        n_agents = len(positions)
        agent_order = np.argsort(-np.asarray(priorities, dtype=np.float64))
        committed = np.zeros(n_agents, dtype=bool)
        min_dist = 2.0 * self.agent_radius
        min_dist_sq = min_dist ** 2

        for iteration in range(self.iterations):
            # --- Phase A: Sequential priority-ordered processing ---
            committed[:] = False
            for idx in agent_order:
                # Apply obstacle constraint for this agent
                safe[idx] = self._sdf_constrain_single(
                    positions[idx], safe[idx], sdf, grad_r, grad_c
                )

                # Apply pairwise constraints against already-committed agents
                for other in range(n_agents):
                    if other == idx or not committed[other]:
                        continue
                    rel_pos = positions[other] - positions[idx]
                    dist_sq = float(rel_pos @ rel_pos)
                    rel_vel = safe[idx] - safe[other]
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
                    # Full correction on the current (lower-priority) agent
                    safe[idx] -= normal * correction_mag

                safe[idx] = self._clip_single(safe[idx])
                committed[idx] = True

            # --- Phase B: Symmetric cleanup pass ---
            # Catches remaining conflicts between agents that were processed
            # close together in the priority order and didn't see each other.
            safe = self._apply_pairwise_constraints(positions, safe)
            safe = self._apply_sdf_obstacle_constraints(positions, safe, sdf, grad_r, grad_c)
            safe = self._clip_speeds(safe)

        return safe

    # ------------------------------------------------------------------
    # SDF-based obstacle constraints (replaces polygon inflation)
    # ------------------------------------------------------------------

    def _apply_sdf_obstacle_constraints(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        sdf: np.ndarray,
        grad_r: np.ndarray,
        grad_c: np.ndarray,
    ) -> np.ndarray:
        """Apply obstacle avoidance using the SDF.

        Checks both the **current** position and the **proposed** position
        (current + velocity * dt) to prevent overshooting into obstacles.

        Two mechanisms:
        1. **Velocity rejection**: remove the velocity component pointing
           toward the obstacle surface when the agent is within the safety zone.
        2. **Repulsive velocity**: push the agent away from the obstacle
           proportionally to how close it is (within the safety zone).
        """
        adjusted = velocities.copy()
        safety_margin = self.agent_radius * 1.0  # wider margin for early intervention

        for i in range(len(positions)):
            # Check at current position
            self._sdf_constrain_at_point(
                positions[i], adjusted, i, sdf, grad_r, grad_c, safety_margin
            )

            # Also check at proposed (forward-looking) position
            proposed = positions[i] + adjusted[i] * self.dt
            proposed_2d = proposed.reshape(1, 2)
            d_prop = float(sample_sdf_bilinear(sdf, proposed_2d)[0])

            if d_prop < self.agent_radius + safety_margin:
                grad_prop = sample_sdf_gradient_bilinear(grad_r, grad_c, proposed_2d)[0]
                grad_norm = np.linalg.norm(grad_prop)
                if grad_norm > 1e-6:
                    normal = grad_prop / grad_norm

                    # Velocity rejection at proposed position
                    vel_toward = np.dot(adjusted[i], -normal)
                    if vel_toward > 0:
                        adjusted[i] += normal * vel_toward

                    # Stronger repulsion if proposed position would penetrate
                    penetration = max(self.agent_radius - d_prop, 0.0)
                    if penetration > 0:
                        repulse_mag = self.obstacle_repulsion_gain * penetration / max(self.dt, 1e-6)
                        adjusted[i] += normal * repulse_mag

        return adjusted

    def _sdf_constrain_at_point(
        self,
        position: np.ndarray,
        velocities: np.ndarray,
        idx: int,
        sdf: np.ndarray,
        grad_r: np.ndarray,
        grad_c: np.ndarray,
        safety_margin: float,
    ) -> None:
        """Apply SDF constraint at a specific position, modifying velocities[idx] in place."""
        pos_2d = position.reshape(1, 2)
        d = float(sample_sdf_bilinear(sdf, pos_2d)[0])

        if d > self.agent_radius + safety_margin:
            return

        grad = sample_sdf_gradient_bilinear(grad_r, grad_c, pos_2d)[0]
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-6:
            return
        normal = grad / grad_norm  # points away from obstacle

        # Velocity rejection: remove component toward obstacle
        vel_toward = np.dot(velocities[idx], -normal)
        if vel_toward > 0:
            velocities[idx] += normal * vel_toward

        # Repulsive push when too close
        penetration = max(self.agent_radius - d, 0.0)
        if penetration > 0:
            repulse_mag = self.obstacle_repulsion_gain * penetration / max(self.dt, 1e-6)
            velocities[idx] += normal * repulse_mag

    def _sdf_constrain_single(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        sdf: np.ndarray,
        grad_r: np.ndarray,
        grad_c: np.ndarray,
    ) -> np.ndarray:
        """Apply SDF obstacle constraint to a single agent (current + proposed position)."""
        adjusted = velocity.copy()
        safety_margin = self.agent_radius * 1.0

        # --- Check at current position ---
        pos_2d = position.reshape(1, 2)
        d = float(sample_sdf_bilinear(sdf, pos_2d)[0])

        if d <= self.agent_radius + safety_margin:
            grad = sample_sdf_gradient_bilinear(grad_r, grad_c, pos_2d)[0]
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-6:
                normal = grad / grad_norm
                vel_toward = np.dot(adjusted, -normal)
                if vel_toward > 0:
                    adjusted += normal * vel_toward
                penetration = max(self.agent_radius - d, 0.0)
                if penetration > 0:
                    repulse_mag = self.obstacle_repulsion_gain * penetration / max(self.dt, 1e-6)
                    adjusted += normal * repulse_mag

        # --- Check at proposed position (forward-looking) ---
        proposed = position + adjusted * self.dt
        prop_2d = proposed.reshape(1, 2)
        d_prop = float(sample_sdf_bilinear(sdf, prop_2d)[0])

        if d_prop <= self.agent_radius + safety_margin:
            grad_prop = sample_sdf_gradient_bilinear(grad_r, grad_c, prop_2d)[0]
            grad_norm = np.linalg.norm(grad_prop)
            if grad_norm > 1e-6:
                normal = grad_prop / grad_norm
                vel_toward = np.dot(adjusted, -normal)
                if vel_toward > 0:
                    adjusted += normal * vel_toward
                penetration = max(self.agent_radius - d_prop, 0.0)
                if penetration > 0:
                    repulse_mag = self.obstacle_repulsion_gain * penetration / max(self.dt, 1e-6)
                    adjusted += normal * repulse_mag

        return adjusted

    # ------------------------------------------------------------------
    # Symmetric pairwise constraints (unchanged, used by heuristic path)
    # ------------------------------------------------------------------

    def _clip_speeds(self, velocities: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        scale = np.maximum(norms / max(self.max_speed, 1e-6), 1.0)
        return velocities / scale

    def _clip_single(self, velocity: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(velocity)
        if norm > self.max_speed:
            return velocity * (self.max_speed / norm)
        return velocity

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


class EPIBTShield:
    """Enhanced PIBT collision shield for continuous MAPF.

    Implements priority-based local coordination with backtracking in
    continuous space.  Agents are processed in descending priority order.
    Each agent selects the best candidate velocity that avoids collisions
    with already-committed higher-priority agents.

    Candidate velocities per agent:
      - The preferred (model or goal-directed) velocity
      - ``num_candidate_directions`` evenly-spaced directions at max_speed
      - Half-speed versions of the preferred velocity
      - Zero velocity (wait)

    Backtracking: if an agent cannot find a valid action it requests the
    blocking higher-priority agent to try an alternative, up to
    ``max_backtrack_depth`` levels deep.
    """

    def __init__(
        self,
        agent_radius: float,
        max_speed: float,
        dt: float,
        num_candidate_directions: int = 16,
        max_backtrack_depth: int = 3,
        obstacle_repulsion_gain: float = 2.0,
    ) -> None:
        self.agent_radius = agent_radius
        self.max_speed = max_speed
        self.dt = dt
        self.num_candidate_directions = num_candidate_directions
        self.max_backtrack_depth = max_backtrack_depth
        self.obstacle_repulsion_gain = obstacle_repulsion_gain
        self._sdf_cache: Dict[Tuple, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def project(
        self,
        positions: np.ndarray,
        preferred_velocities: np.ndarray,
        obstacle_map: np.ndarray,
        priorities: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return collision-free velocities using EPIBT coordination.

        Args:
            positions: (N, 2) agent positions.
            preferred_velocities: (N, 2) desired velocities.
            obstacle_map: (H, W) binary obstacle grid.
            priorities: (N,) priority values (higher = processed first).
                If None, agents are processed in index order.
        """
        n = len(positions)
        positions = np.asarray(positions, dtype=np.float32)
        preferred_velocities = np.asarray(preferred_velocities, dtype=np.float32)

        sdf, grad_r, grad_c = self._get_sdf(obstacle_map)

        # Agent processing order: highest priority first
        if priorities is not None:
            order = np.argsort(-np.asarray(priorities, dtype=np.float64))
        else:
            order = np.arange(n)

        # Pre-generate all candidate velocity sets
        all_candidates = [
            self._generate_candidates(preferred_velocities[i], positions[i], sdf, grad_r, grad_c)
            for i in range(n)
        ]

        # committed[i] = chosen velocity for agent i (-1 means not yet committed)
        committed = np.full(n, -1, dtype=np.int32)  # index into all_candidates[i]
        committed_vel = np.zeros((n, 2), dtype=np.float32)

        # Process agents in priority order with backtracking
        self._assign_actions(
            order,
            positions,
            preferred_velocities,
            all_candidates,
            committed,
            committed_vel,
            sdf,
            grad_r,
            grad_c,
            depth=0,
        )

        # Any agents still unassigned get their preferred velocity clipped
        for i in range(n):
            if committed[i] < 0:
                v = preferred_velocities[i].copy()
                committed_vel[i] = self._clip(v)

        return committed_vel

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assign_actions(
        self,
        order,
        positions,
        preferred,
        all_candidates,
        committed,
        committed_vel,
        sdf,
        grad_r,
        grad_c,
        depth: int,
    ) -> bool:
        """Recursively assign actions to agents in priority order.

        Returns True if all assignments in ``order`` succeeded.
        """
        if depth > self.max_backtrack_depth:
            return False

        for idx in order:
            if committed[idx] >= 0:
                continue  # already assigned (e.g. re-entry after backtrack)

            candidates = all_candidates[idx]
            placed = False
            for c_idx, cand in enumerate(candidates):
                # Check obstacle collision
                if self._hits_obstacle(positions[idx], cand, sdf):
                    continue
                # Check collision with all already-committed agents
                conflict = False
                for j in range(len(positions)):
                    if j == idx or committed[j] < 0:
                        continue
                    if self._agents_collide(positions[idx], cand, positions[j], committed_vel[j]):
                        conflict = True
                        break
                if not conflict:
                    committed[idx] = c_idx
                    committed_vel[idx] = cand
                    placed = True
                    break

            if not placed:
                # Backtrack: commit zero velocity for this agent
                committed[idx] = len(candidates) - 1  # last candidate is always zero
                committed_vel[idx] = np.zeros(2, dtype=np.float32)

        return True

    def _generate_candidates(
        self,
        preferred: np.ndarray,
        position: np.ndarray,
        sdf: np.ndarray,
        grad_r: np.ndarray,
        grad_c: np.ndarray,
    ) -> list:
        """Generate ordered list of candidate velocities for one agent.

        Candidates are sorted by score descending (best first).
        """
        candidates = []

        # Preferred velocity
        pref_norm = np.linalg.norm(preferred)
        if pref_norm > 1e-6:
            candidates.append(preferred.copy())
            # Half-speed preferred
            candidates.append(preferred * 0.5)

        # Evenly spaced directions at max_speed
        angles = np.linspace(0, 2 * math.pi, self.num_candidate_directions, endpoint=False)
        for angle in angles:
            v = np.array([math.sin(angle), math.cos(angle)], dtype=np.float32) * self.max_speed
            candidates.append(v)

        # Zero velocity (wait)
        candidates.append(np.zeros(2, dtype=np.float32))

        # Score and sort
        scores = [self._score_candidate(c, preferred) for c in candidates]
        sorted_pairs = sorted(zip(scores, candidates), key=lambda p: -p[0])
        return [c for _, c in sorted_pairs]

    def _score_candidate(self, candidate: np.ndarray, preferred: np.ndarray) -> float:
        """Score by alignment with preferred velocity."""
        pref_norm = np.linalg.norm(preferred)
        if pref_norm < 1e-6:
            # Prefer waiting when preferred is zero
            cand_norm = np.linalg.norm(candidate)
            return -float(cand_norm)
        return float(np.dot(candidate, preferred)) / max(pref_norm, 1e-6)

    def _agents_collide(
        self,
        pos_i: np.ndarray,
        vel_i: np.ndarray,
        pos_j: np.ndarray,
        vel_j: np.ndarray,
    ) -> bool:
        """Check if two agents would collide given their velocities."""
        min_dist = 2.0 * self.agent_radius
        # Check at proposed next positions
        next_i = pos_i + vel_i * self.dt
        next_j = pos_j + vel_j * self.dt
        dist = float(np.linalg.norm(next_i - next_j))
        if dist < min_dist:
            return True
        # Also check if they pass through each other (crossing check)
        rel_pos = pos_j - pos_i
        rel_vel = vel_j - vel_i
        # Time of closest approach
        denom = float(rel_vel @ rel_vel)
        if denom > 1e-8:
            t_closest = -float(rel_pos @ rel_vel) / denom
            t_closest = max(0.0, min(self.dt, t_closest))
            closest = pos_i + vel_i * t_closest - (pos_j + vel_j * t_closest)
            if float(np.linalg.norm(closest)) < min_dist:
                return True
        return False

    def _hits_obstacle(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        sdf: np.ndarray,
    ) -> bool:
        """Check if velocity would move agent into an obstacle."""
        proposed = position + velocity * self.dt
        proposed_2d = proposed.reshape(1, 2)
        d = float(sample_sdf_bilinear(sdf, proposed_2d)[0])
        return d < self.agent_radius

    def _clip(self, velocity: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(velocity)
        if norm > self.max_speed:
            return velocity * (self.max_speed / norm)
        return velocity

    def _get_sdf(
        self, obstacle_map: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (obstacle_map.shape, obstacle_map.data.tobytes())
        cached = self._sdf_cache.get(key)
        if cached is not None:
            return cached
        sdf = compute_sdf(obstacle_map)
        grad_r, grad_c = sdf_gradient(sdf)
        self._sdf_cache[key] = (sdf, grad_r, grad_c)
        return sdf, grad_r, grad_c


class ContinuousMAPFEnv:
    """Continuous-space MAPF environment with priority-ordered collision shielding.

    Supports shield types:
    - ``"none"``: No collision avoidance, just speed clipping.
    - ``"simple"``: Naive safety filter (stop on conflict).
    - ``"orca"``: Standard symmetric ORCA (heuristic or rvo2).
    - ``"heuristic-orca"``: Force heuristic path (no rvo2).
    - ``"po-orca"``: Priority-Ordered ORCA with sequential processing.
    - ``"epibt"``: Enhanced PIBT priority-based shield with backtracking.
    - ``"picbf-cs"``: CBF shield from the external picbf-cs package.
    """

    DEADLOCK_CHECK_INTERVAL = 30
    DEADLOCK_PRIORITY_BOOST = 10.0

    def __init__(
        self,
        obstacle_map: np.ndarray,
        dt: float = 0.2,
        max_speed: float = 1.0,
        agent_radius: float = 0.3,
        goal_tolerance: float = 0.25,
        picbf_communication_radius: Optional[float] = None,
    ) -> None:
        if picbf_communication_radius is not None and picbf_communication_radius <= 0.0:
            raise ValueError("picbf_communication_radius must be positive")
        self.obstacle_map = np.asarray(obstacle_map, dtype=np.int8)
        self.dt = dt
        self.max_speed = max_speed
        self.agent_radius = agent_radius
        self.goal_tolerance = goal_tolerance
        self.picbf_communication_radius = picbf_communication_radius
        self._shield = ORCAStyleShield(agent_radius=agent_radius, max_speed=max_speed, dt=dt)
        self._epibt_shield = EPIBTShield(agent_radius=agent_radius, max_speed=max_speed, dt=dt)
        self._picbf_shield = None
        self.last_shield_debug_info: Optional[Dict[str, object]] = None
        self.positions = None
        self.goals = None
        self.history_positions = []
        self.history_velocities = []
        self.metrics = StepMetrics()
        self.arrival_steps = None
        self.step_count = 0

        # Priority state
        self.priorities = None
        self._goal_dist_snapshot = None

        # Precompute SDF for obstacle checking
        self._sdf = compute_sdf(self.obstacle_map)

    def reset(self, starts: np.ndarray, goals: np.ndarray) -> np.ndarray:
        self.positions = np.asarray(starts, dtype=np.float32).copy()
        self.goals = np.asarray(goals, dtype=np.float32).copy()
        self.history_positions = [self.positions.copy()]
        self.history_velocities = []
        self.metrics = StepMetrics()
        self.arrival_steps = np.full(len(self.positions), -1, dtype=np.int32)
        self.step_count = 0
        self.last_shield_debug_info = None

        # Initialize priorities from Euclidean distance to goal
        # (farther agents get higher initial priority)
        goal_dists = np.linalg.norm(self.goals - self.positions, axis=1)
        self.priorities = goal_dists.astype(np.float64)
        self._goal_dist_snapshot = goal_dists.copy()

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
        self.last_shield_debug_info = None
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
        if shield_type == "po-orca":
            return self._shield.project(
                self.positions,
                preferred_velocities,
                self.obstacle_map,
                use_true_orca=False,
                priorities=self.priorities,
            )
        if shield_type == "orca":
            return self._shield.project(
                self.positions,
                preferred_velocities,
                self.obstacle_map,
                use_true_orca=True,
            )
        if shield_type == "epibt":
            return self._epibt_shield.project(
                self.positions,
                preferred_velocities,
                self.obstacle_map,
                priorities=self.priorities,
            )
        if shield_type in {"picbf-cs", "picbf"}:
            if self._picbf_shield is None:
                self._picbf_shield = PICBFCSShield(
                    agent_radius=self.agent_radius,
                    max_speed=self.max_speed,
                    dt=self.dt,
                    communication_radius=self.picbf_communication_radius,
                )
            safe_velocities = self._picbf_shield.project(
                self.positions,
                preferred_velocities,
                self.obstacle_map,
                goals=self.goals,
            )
            self.last_shield_debug_info = self._picbf_shield.last_debug_info
            return safe_velocities
        raise ValueError(f"Unsupported shield type: {shield_type}")

    def _update_priorities(self) -> None:
        """Update agent priorities each timestep (mirrors PIBT's updatePriorities).

        Agents not at goal get +1 priority per step (increasing urgency).
        Agents at goal get their priority reset to 0.
        Every ``DEADLOCK_CHECK_INTERVAL`` steps, agents that have made no
        progress toward their goal receive a priority boost.
        """
        at_goal = self.agents_at_goal()
        self.priorities[~at_goal] += 1.0
        self.priorities[at_goal] = 0.0

        # Deadlock detection: check progress every N steps
        if self.step_count > 0 and self.step_count % self.DEADLOCK_CHECK_INTERVAL == 0:
            current_dists = np.linalg.norm(self.goals - self.positions, axis=1)
            no_progress = current_dists >= self._goal_dist_snapshot - 1e-3
            stuck = no_progress & ~at_goal
            self.priorities[stuck] += self.DEADLOCK_PRIORITY_BOOST
            self._goal_dist_snapshot = current_dists.copy()

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

        # Update priorities after each step
        self._update_priorities()

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
        """Check exact circle-vs-blocked-cell obstacle collision."""
        if (
            position[0] < self.agent_radius
            or position[1] < self.agent_radius
            or position[0] > self.obstacle_map.shape[0] - self.agent_radius
            or position[1] > self.obstacle_map.shape[1] - self.agent_radius
        ):
            return True
        row_min = max(0, int(np.floor(position[0] - self.agent_radius)))
        row_max = min(self.obstacle_map.shape[0] - 1, int(np.floor(position[0] + self.agent_radius)))
        col_min = max(0, int(np.floor(position[1] - self.agent_radius)))
        col_max = min(self.obstacle_map.shape[1] - 1, int(np.floor(position[1] + self.agent_radius)))
        for row in range(row_min, row_max + 1):
            for col in range(col_min, col_max + 1):
                if self.obstacle_map[row, col] != 0 and circle_intersects_rect(
                    position, self.agent_radius, row, col
                ):
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
    from main_pys.model_inputs import load_grid_map_from_file

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
