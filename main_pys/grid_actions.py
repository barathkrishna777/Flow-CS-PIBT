from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GridAction:
    name: str
    dr: int
    dc: int
    cost: float = 1.0

    @property
    def is_wait(self) -> bool:
        return self.dr == 0 and self.dc == 0

    @property
    def is_diagonal(self) -> bool:
        return abs(self.dr) == 1 and abs(self.dc) == 1


# Keep the first five labels identical to the existing code:
# 0=wait, 1=right, 2=down, 3=up, 4=left.
GRID4_ACTIONS = (
    GridAction("wait", 0, 0, 0.0),
    GridAction("right", 0, 1),
    GridAction("down", 1, 0),
    GridAction("up", -1, 0),
    GridAction("left", 0, -1),
)

GRID8_ACTIONS = GRID4_ACTIONS + (
    GridAction("down_right", 1, 1, 2.0 ** 0.5),
    GridAction("down_left", 1, -1, 2.0 ** 0.5),
    GridAction("up_right", -1, 1, 2.0 ** 0.5),
    GridAction("up_left", -1, -1, 2.0 ** 0.5),
)

ACTION_MODES = {
    "grid4": GRID4_ACTIONS,
    "grid8": GRID8_ACTIONS,
}

DIAGONAL_RULES = ("blocked_pair", "both_clear", "allow")


def validate_action_mode(action_mode: str) -> str:
    if action_mode not in ACTION_MODES:
        valid = ", ".join(sorted(ACTION_MODES))
        raise ValueError(f"Unknown action_mode '{action_mode}'. Expected one of: {valid}")
    return action_mode


def validate_diagonal_rule(diagonal_rule: str) -> str:
    if diagonal_rule not in DIAGONAL_RULES:
        valid = ", ".join(DIAGONAL_RULES)
        raise ValueError(f"Unknown diagonal_rule '{diagonal_rule}'. Expected one of: {valid}")
    return diagonal_rule


def get_actions(action_mode: str = "grid4") -> tuple[GridAction, ...]:
    return ACTION_MODES[validate_action_mode(action_mode)]


def get_action_dim(action_mode: str = "grid4") -> int:
    return len(get_actions(action_mode))


@lru_cache(maxsize=None)
def get_label_to_moves(action_mode: str = "grid4") -> np.ndarray:
    actions = get_actions(action_mode)
    return np.asarray([(a.dr, a.dc) for a in actions], dtype=np.int64)


@lru_cache(maxsize=None)
def get_action_names(action_mode: str = "grid4") -> tuple[str, ...]:
    return tuple(a.name for a in get_actions(action_mode))


@lru_cache(maxsize=None)
def get_delta_to_label(action_mode: str = "grid4") -> dict[tuple[int, int], int]:
    return {(a.dr, a.dc): idx for idx, a in enumerate(get_actions(action_mode))}


@lru_cache(maxsize=None)
def get_bd_flatten_indices(action_mode: str = "grid4") -> tuple[int, ...]:
    """Return row-major 3x3 indices matching the action labels.

    The 3x3 local BD patch is flattened in row-major order:
    0=(-1,-1), 1=(-1,0), 2=(-1,1),
    3=(0,-1),  4=(0,0),  5=(0,1),
    6=(1,-1),  7=(1,0),  8=(1,1).
    """
    indices = []
    for action in get_actions(action_mode):
        indices.append((action.dr + 1) * 3 + (action.dc + 1))
    return tuple(indices)


def get_action_vectors(action_mode: str = "grid4", normalize: bool = False) -> np.ndarray:
    vectors = get_label_to_moves(action_mode).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        moving = norms.squeeze(1) > 0
        vectors[moving] = vectors[moving] / norms[moving]
    return vectors


def velocity_to_action_labels(
    velocities: np.ndarray,
    action_mode: str = "grid4",
    wait_threshold: float = 0.1,
) -> np.ndarray:
    velocities = np.asarray(velocities, dtype=np.float32)
    if velocities.ndim != 2 or velocities.shape[1] != 2:
        raise ValueError("velocities must have shape (num_agents, 2)")

    labels = np.zeros(velocities.shape[0], dtype=np.int64)
    norms = np.linalg.norm(velocities, axis=1)
    moving = norms >= wait_threshold
    if not np.any(moving):
        return labels

    vectors = get_action_vectors(action_mode, normalize=True)
    moving_vectors = vectors[1:]
    scores = velocities[moving] @ moving_vectors.T
    labels[moving] = np.argmax(scores, axis=1) + 1
    return labels


def action_mask_for_locs(
    grid_map: np.ndarray,
    locs: np.ndarray,
    action_mode: str = "grid4",
    diagonal_rule: str = "blocked_pair",
) -> np.ndarray:
    """Return True for actions that are illegal from each location."""
    validate_diagonal_rule(diagonal_rule)
    moves = get_label_to_moves(action_mode)
    locs = np.asarray(locs, dtype=np.int64)
    next_rows = locs[:, 0, None] + moves[None, :, 0]
    next_cols = locs[:, 1, None] + moves[None, :, 1]

    out_of_bounds = (
        (next_rows < 0)
        | (next_rows >= grid_map.shape[0])
        | (next_cols < 0)
        | (next_cols >= grid_map.shape[1])
    )
    safe_rows = np.clip(next_rows, 0, grid_map.shape[0] - 1)
    safe_cols = np.clip(next_cols, 0, grid_map.shape[1] - 1)
    blocked = grid_map[safe_rows, safe_cols] == 1
    blocked = blocked | out_of_bounds

    if action_mode == "grid8" and diagonal_rule != "allow":
        for action_idx, action in enumerate(get_actions(action_mode)):
            if not action.is_diagonal:
                continue
            side_a = grid_map[
                np.clip(locs[:, 0] + action.dr, 0, grid_map.shape[0] - 1),
                locs[:, 1],
            ] == 1
            side_b = grid_map[
                locs[:, 0],
                np.clip(locs[:, 1] + action.dc, 0, grid_map.shape[1] - 1),
            ] == 1
            if diagonal_rule == "both_clear":
                blocked[:, action_idx] |= side_a | side_b
            else:
                blocked[:, action_idx] |= side_a & side_b

    return blocked


def is_diagonal_move(move: Iterable[int]) -> bool:
    dr, dc = move
    return abs(int(dr)) == 1 and abs(int(dc)) == 1


def diagonal_clearance_ok(
    grid_map: np.ndarray,
    current_pos: np.ndarray,
    move: np.ndarray,
    diagonal_rule: str = "blocked_pair",
) -> bool:
    validate_diagonal_rule(diagonal_rule)
    if diagonal_rule == "allow" or not is_diagonal_move(move):
        return True

    row, col = int(current_pos[0]), int(current_pos[1])
    dr, dc = int(move[0]), int(move[1])
    side_a_blocked = grid_map[row + dr, col] == 1
    side_b_blocked = grid_map[row, col + dc] == 1
    if diagonal_rule == "both_clear":
        return not (side_a_blocked or side_b_blocked)
    return not (side_a_blocked and side_b_blocked)
