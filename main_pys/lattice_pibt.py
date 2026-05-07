"""Lattice-PIBT: priority-based coordination over multi-step motion primitives.

Each round, every agent selects a 2-step primitive from a ranked preference
list.  PIBT processes agents in priority order, checking the full space-time
path of each candidate primitive against a reservation table.  Conflicts
trigger recursive backtracking, exactly mirroring classic PIBT but over
multi-cell, multi-timestep trajectories.
"""

import time
import numpy as np
from collections import defaultdict

from main_pys.lattice_primitives import (
    NUM_PRIMITIVES,
    PRIMITIVE_DURATION,
    PRIMITIVE_PATHS,
    PRIMITIVE_STEPS,
)


def _check_primitive_feasible(grid_map, agent_pos, prim_idx,
                              reserved_nodes, reserved_edges):
    """Check whether a primitive is feasible for an agent.

    Checks:
      1. Every cell along the path is in-bounds and obstacle-free.
      2. No (cell, timestep) is already reserved by another agent.
      3. No edge conflict: at each transition t→t+1, no other agent is
         moving in the opposite direction on the same edge at the same time.

    Parameters
    ----------
    grid_map : (H, W) int array — 1 = obstacle
    agent_pos : (2,) int array — current (row, col)
    prim_idx : int — index into PRIMITIVE_PATHS / PRIMITIVE_STEPS
    reserved_nodes : dict  (row, col, t) → agent_id
    reserved_edges : dict  (r_from, c_from, r_to, c_to, t) → agent_id

    Returns
    -------
    feasible : bool
    path : (PRIMITIVE_DURATION + 1, 2) int array — absolute positions if feasible, else None
    """
    path = PRIMITIVE_PATHS[prim_idx] + agent_pos  # (D+1, 2)
    H, W = grid_map.shape

    # Skip t=0: path[0] is always the agent's own current position, which is
    # pre-reserved for itself and can never conflict with another agent's path.
    # Checking it would block every primitive including WAIT.
    for t in range(1, PRIMITIVE_DURATION + 1):
        r, c = path[t]
        if r < 0 or r >= H or c < 0 or c >= W:
            return False, None
        if grid_map[r, c] == 1:
            return False, None
        if (r, c, t) in reserved_nodes:
            return False, None

    for t in range(PRIMITIVE_DURATION):
        r0, c0 = path[t]
        r1, c1 = path[t + 1]
        # Check reverse edge conflict (agent swaps)
        if (r1, c1, r0, c0, t) in reserved_edges:
            return False, None

    return True, path


def _commit_primitive(agent_id, path, reserved_nodes, reserved_edges):
    """Reserve space-time cells and edges for a committed primitive.

    t=0 (starting cell) is already pre-reserved during lattice_pibt init and
    is never modified by commit/uncommit — it persists for the full round.
    """
    for t in range(1, PRIMITIVE_DURATION + 1):
        r, c = path[t]
        reserved_nodes[(r, c, t)] = agent_id
    for t in range(PRIMITIVE_DURATION):
        r0, c0 = path[t]
        r1, c1 = path[t + 1]
        reserved_edges[(r0, c0, r1, c1, t)] = agent_id


def _uncommit_primitive(agent_id, path, reserved_nodes, reserved_edges):
    """Release reservations for a primitive (backtracking).

    Only releases t>=1 entries committed by this agent; the t=0 entry is
    left intact (it was set during init and never touched by commit).
    """
    for t in range(1, PRIMITIVE_DURATION + 1):
        r, c = path[t]
        key = (r, c, t)
        if reserved_nodes.get(key) == agent_id:
            del reserved_nodes[key]
    for t in range(PRIMITIVE_DURATION):
        r0, c0 = path[t]
        r1, c1 = path[t + 1]
        key = (r0, c0, r1, c1, t)
        if reserved_edges.get(key) == agent_id:
            del reserved_edges[key]


def _lattice_pibt_recursive(grid_map, agent_id, prim_preferences,
                            planned_agents, assigned_primitives,
                            assigned_paths, reserved_nodes, reserved_edges,
                            current_locs, current_locs_to_agent,
                            start_time, time_limit):
    """Recursive lattice-PIBT for a single agent.

    Tries primitives in preference order.  When a candidate primitive's
    endpoint cell is occupied by an unplanned agent, recursively plan
    that agent first (forcing it out of the way).
    """
    if time.time() - start_time > time_limit:
        return False

    for prim_idx in prim_preferences[agent_id]:
        feasible, path = _check_primitive_feasible(
            grid_map, current_locs[agent_id], prim_idx,
            reserved_nodes, reserved_edges,
        )
        if not feasible:
            continue

        # Commit tentatively
        planned_agents[agent_id] = True
        assigned_primitives[agent_id] = prim_idx
        assigned_paths[agent_id] = path
        _commit_primitive(agent_id, path, reserved_nodes, reserved_edges)

        # Check if any unplanned agent currently sits on a cell we need.
        # The key conflict point is the final position — an agent sitting
        # there at t=0 must move.  We also check intermediate cells.
        conflict_resolved = True
        for t in range(1, PRIMITIVE_DURATION + 1):
            r, c = path[t]
            conflicting = current_locs_to_agent[r, c]
            if conflicting != -1 and conflicting != agent_id and not planned_agents[conflicting]:
                ok = _lattice_pibt_recursive(
                    grid_map, conflicting, prim_preferences,
                    planned_agents, assigned_primitives, assigned_paths,
                    reserved_nodes, reserved_edges,
                    current_locs, current_locs_to_agent,
                    start_time, time_limit,
                )
                if not ok:
                    conflict_resolved = False
                    break

        if conflict_resolved:
            return True

        # Backtrack
        _uncommit_primitive(agent_id, path, reserved_nodes, reserved_edges)
        planned_agents[agent_id] = False
        assigned_primitives[agent_id] = -1
        assigned_paths[agent_id] = None

    # Fallback: force WAIT (primitive 0)
    fallback_feasible, fallback_path = _check_primitive_feasible(
        grid_map, current_locs[agent_id], 0,
        reserved_nodes, reserved_edges,
    )
    if fallback_feasible:
        planned_agents[agent_id] = True
        assigned_primitives[agent_id] = 0
        assigned_paths[agent_id] = fallback_path
        _commit_primitive(agent_id, fallback_path, reserved_nodes, reserved_edges)
        return True

    # Even WAIT failed (extremely rare — blocked by another agent's reservation)
    planned_agents[agent_id] = True
    assigned_primitives[agent_id] = 0
    wait_path = np.tile(current_locs[agent_id], (PRIMITIVE_DURATION + 1, 1))
    assigned_paths[agent_id] = wait_path
    return False


def lattice_pibt(grid_map, prim_preferences, current_locs, agent_priorities,
                 start_time, time_limit):
    """Run one round of lattice-PIBT.

    Parameters
    ----------
    grid_map : (H, W) int array — 1 = obstacle
    prim_preferences : (N, NUM_PRIMITIVES) int array
        Ranked primitive indices per agent (best first).
    current_locs : (N, 2) int array
        Current agent positions (row, col).
    agent_priorities : (N,) float array
        Higher = planned first.
    start_time : float
        time.time() at call start, for timeout.
    time_limit : float
        Max seconds before aborting.

    Returns
    -------
    assigned_primitives : (N,) int array — primitive index per agent
    move_sequences : (N, PRIMITIVE_DURATION, 2) int array — step deltas
    success : bool — True if all agents were planned without timeout
    """
    N = len(agent_priorities)
    agent_order = np.argsort(-agent_priorities)

    planned_agents = np.zeros(N, dtype=bool)
    assigned_primitives = np.full(N, -1, dtype=np.int32)
    assigned_paths = [None] * N

    reserved_nodes = {}
    reserved_edges = {}

    # Reserve t=0 positions for all agents
    current_locs_to_agent = np.full(grid_map.shape, -1, dtype=np.int32)
    for i in range(N):
        r, c = current_locs[i]
        current_locs_to_agent[r, c] = i
        reserved_nodes[(r, c, 0)] = i

    success = True
    for agent_id in agent_order:
        if planned_agents[agent_id]:
            continue
        ok = _lattice_pibt_recursive(
            grid_map, agent_id, prim_preferences,
            planned_agents, assigned_primitives, assigned_paths,
            reserved_nodes, reserved_edges,
            current_locs, current_locs_to_agent,
            start_time, time_limit,
        )
        if not ok:
            success = False

    # Build move sequences from assigned primitives
    move_sequences = np.zeros((N, PRIMITIVE_DURATION, 2), dtype=np.int32)
    for i in range(N):
        prim = assigned_primitives[i]
        if prim >= 0:
            move_sequences[i] = PRIMITIVE_STEPS[prim]

    return assigned_primitives, move_sequences, success
