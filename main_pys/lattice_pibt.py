"""Lattice-PIBT: priority-based coordination over multi-step motion primitives.

Each round, every agent selects a 2-step primitive from a ranked preference
list.  PIBT processes agents in priority order, checking the full space-time
path of each candidate primitive against a reservation table.  Conflicts
trigger recursive backtracking, exactly mirroring classic PIBT but over
multi-cell, multi-timestep trajectories.

Key invariant for WAIT safety:
  In classic 1-step PIBT, WAIT is always feasible because the agent's own
  cell is pre-reserved at t=0 and no other agent can reserve it at t=1
  (since that agent would have to recursively plan *this* agent first).

  In multi-step PIBT (PRIMITIVE_DURATION=2), a higher-priority agent B can
  commit a path that passes through agent A's cell at t>=1.  If A is then
  recursively planned and cannot find any valid primitive, A's WAIT also
  fails because B's reservation blocks (A_r, A_c, t>=1).

  Fix: when an agent fails during *recursive* planning (called by another
  agent), do NOT mark it as planned — just return False.  This lets the
  caller backtrack its primitive (removing the blocking reservation), after
  which the agent's WAIT becomes feasible again.  The unconditional WAIT
  fallback is only used at the *top level* where no caller can backtrack.
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
      3. No edge conflict: at each transition t->t+1, no other agent is
         moving in the opposite direction on the same edge at the same time.

    Parameters
    ----------
    grid_map : (H, W) int array — 1 = obstacle
    agent_pos : (2,) int array — current (row, col)
    prim_idx : int — index into PRIMITIVE_PATHS / PRIMITIVE_STEPS
    reserved_nodes : dict  (row, col, t) -> agent_id
    reserved_edges : dict  (r_from, c_from, r_to, c_to, t) -> agent_id

    Returns
    -------
    feasible : bool
    path : (PRIMITIVE_DURATION + 1, 2) int array — absolute positions if feasible, else None
    """
    path = PRIMITIVE_PATHS[prim_idx] + agent_pos  # (D+1, 2)
    H, W = grid_map.shape

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
        if (r1, c1, r0, c0, t) in reserved_edges:
            return False, None

    return True, path


def _commit_primitive(agent_id, path, reserved_nodes, reserved_edges):
    """Reserve space-time cells and edges for a committed primitive."""
    for t in range(1, PRIMITIVE_DURATION + 1):
        r, c = path[t]
        reserved_nodes[(r, c, t)] = agent_id
    for t in range(PRIMITIVE_DURATION):
        r0, c0 = path[t]
        r1, c1 = path[t + 1]
        reserved_edges[(r0, c0, r1, c1, t)] = agent_id


def _uncommit_primitive(agent_id, path, reserved_nodes, reserved_edges):
    """Release reservations for a primitive (backtracking)."""
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
                            start_time, time_limit, _depth=0):
    """Recursive lattice-PIBT for a single agent.

    Tries primitives in preference order.  When a candidate primitive's
    path cell is occupied by an unplanned agent, recursively plan that
    agent first (forcing it out of the way).

    When called recursively (_depth > 0) and all primitives including WAIT
    fail, returns False WITHOUT marking the agent as planned.  This lets the
    caller backtrack its own primitive, removing the reservation that blocked
    WAIT, so the agent can be successfully planned later.
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

        planned_agents[agent_id] = True
        assigned_primitives[agent_id] = prim_idx
        assigned_paths[agent_id] = path
        _commit_primitive(agent_id, path, reserved_nodes, reserved_edges)

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
                    _depth=_depth + 1,
                )
                if not ok:
                    conflict_resolved = False
                    break

        if conflict_resolved:
            return True

        # Backtrack this agent's primitive
        _uncommit_primitive(agent_id, path, reserved_nodes, reserved_edges)
        planned_agents[agent_id] = False
        assigned_primitives[agent_id] = -1
        assigned_paths[agent_id] = None

    # Fallback: WAIT (primitive 0)
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

    if _depth > 0:
        # Recursive call: WAIT blocked by the caller's reservation.
        # Return False WITHOUT marking as planned — the caller will
        # backtrack its primitive, unblocking our WAIT for later.
        return False

    # Top-level only: force WAIT even though it conflicts (last resort).
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
            _depth=0,
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
