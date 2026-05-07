"""Lattice motion primitives for grid-world MAPF.

All primitives have a fixed duration of PRIMITIVE_DURATION timesteps on a
4-connected grid.  Shorter logical moves (single cardinal step, wait) are
padded with waits so every primitive occupies the same time window.  This
lets PIBT plan in lock-step rounds: one primitive per agent per round, with
full space-time collision checking over the primitive's swept path.

Primitive layout (17 total, PRIMITIVE_DURATION = 2):
    Index  Name          Steps            Net displacement   Characteristic velocity
    -----  ----          -----            ----------------   ----------------------
      0    WAIT          (0,0),(0,0)      (0, 0)             (0, 0)
      1    MOVE_R        (0,+1),(0,0)     (0,+1)             (0,+1)
      2    MOVE_D        (+1,0),(0,0)     (+1, 0)            (+1, 0)
      3    MOVE_U        (-1,0),(0,0)     (-1, 0)            (-1, 0)
      4    MOVE_L        (0,-1),(0,0)     (0,-1)             (0,-1)
      5    STRAIGHT_RR   (0,+1),(0,+1)    (0,+2)             (0,+1)  [normalized]
      6    STRAIGHT_DD   (+1,0),(+1,0)    (+2, 0)            (+1, 0)
      7    STRAIGHT_UU   (-1,0),(-1,0)    (-2, 0)            (-1, 0)
      8    STRAIGHT_LL   (0,-1),(0,-1)    (0,-2)             (0,-1)
      9    TURN_RU       (0,+1),(-1,0)    (-1,+1)            (-1,+1) / sqrt(2)
     10    TURN_RD       (0,+1),(+1,0)    (+1,+1)            (+1,+1) / sqrt(2)
     11    TURN_UR       (-1,0),(0,+1)    (-1,+1)            (-1,+1) / sqrt(2)
     12    TURN_UL       (-1,0),(0,-1)    (-1,-1)            (-1,-1) / sqrt(2)
     13    TURN_DR       (+1,0),(0,+1)    (+1,+1)            (+1,+1) / sqrt(2)
     14    TURN_DL       (+1,0),(0,-1)    (+1,-1)            (+1,-1) / sqrt(2)
     15    TURN_LU       (0,-1),(-1,0)    (-1,-1)            (-1,-1) / sqrt(2)
     16    TURN_LD       (0,-1),(+1,0)    (+1,-1)            (+1,-1) / sqrt(2)

Note: turn pairs (e.g. TURN_RU=9 and TURN_UR=11) share the same net
displacement but sweep different intermediate cells, giving them distinct
collision profiles — both are useful for coordination.
"""

import numpy as np
import torch

PRIMITIVE_DURATION = 2

PRIMITIVE_NAMES = [
    "WAIT",
    "MOVE_R", "MOVE_D", "MOVE_U", "MOVE_L",
    "STRAIGHT_RR", "STRAIGHT_DD", "STRAIGHT_UU", "STRAIGHT_LL",
    "TURN_RU", "TURN_RD", "TURN_UR", "TURN_UL",
    "TURN_DR", "TURN_DL", "TURN_LU", "TURN_LD",
]

NUM_PRIMITIVES = len(PRIMITIVE_NAMES)

# Each primitive is a tuple of (dr, dc) steps of length PRIMITIVE_DURATION.
PRIMITIVE_STEPS = np.array([
    # WAIT
    [[0, 0], [0, 0]],
    # MOVE_R, MOVE_D, MOVE_U, MOVE_L  (single cardinal + wait pad)
    [[0, 1], [0, 0]],
    [[1, 0], [0, 0]],
    [[-1, 0], [0, 0]],
    [[0, -1], [0, 0]],
    # STRAIGHT_RR, STRAIGHT_DD, STRAIGHT_UU, STRAIGHT_LL
    [[0, 1], [0, 1]],
    [[1, 0], [1, 0]],
    [[-1, 0], [-1, 0]],
    [[0, -1], [0, -1]],
    # TURN_RU, TURN_RD, TURN_UR, TURN_UL
    [[0, 1], [-1, 0]],
    [[0, 1], [1, 0]],
    [[-1, 0], [0, 1]],
    [[-1, 0], [0, -1]],
    # TURN_DR, TURN_DL, TURN_LU, TURN_LD
    [[1, 0], [0, 1]],
    [[1, 0], [0, -1]],
    [[0, -1], [-1, 0]],
    [[0, -1], [1, 0]],
], dtype=np.int32)  # shape (NUM_PRIMITIVES, PRIMITIVE_DURATION, 2)

assert PRIMITIVE_STEPS.shape == (NUM_PRIMITIVES, PRIMITIVE_DURATION, 2)

# Net displacement for each primitive: sum of steps.
PRIMITIVE_DISPLACEMENTS = PRIMITIVE_STEPS.sum(axis=1)  # (NUM_PRIMITIVES, 2)

# Characteristic velocity vectors for dot-product scoring with the flow
# model output.  We normalize to unit length (except WAIT which stays zero).
_disp_float = PRIMITIVE_DISPLACEMENTS.astype(np.float64)
_norms = np.linalg.norm(_disp_float, axis=1, keepdims=True)
_norms[_norms < 1e-8] = 1.0  # avoid division by zero for WAIT
PRIMITIVE_VELOCITY_VECTORS = (_disp_float / _norms).astype(np.float32)  # (NUM_PRIMITIVES, 2)

PRIMITIVE_VELOCITY_VECTORS_TORCH = torch.from_numpy(PRIMITIVE_VELOCITY_VECTORS)

# "Speed class" of each primitive: 0 = wait, 1 = single-step, 2 = double-step
PRIMITIVE_SPEED_CLASS = np.array([
    0,        # WAIT
    1, 1, 1, 1,  # MOVE_*
    2, 2, 2, 2,  # STRAIGHT_**
    2, 2, 2, 2,  # TURN_** (first group)
    2, 2, 2, 2,  # TURN_** (second group)
], dtype=np.int32)

# Precomputed full paths: for each primitive, the sequence of occupied
# cells relative to the start position.  path[t] = cumulative displacement
# after t steps (t=0 is the start cell).
# Shape: (NUM_PRIMITIVES, PRIMITIVE_DURATION + 1, 2)
PRIMITIVE_PATHS = np.zeros((NUM_PRIMITIVES, PRIMITIVE_DURATION + 1, 2), dtype=np.int32)
for i in range(NUM_PRIMITIVES):
    for t in range(PRIMITIVE_DURATION):
        PRIMITIVE_PATHS[i, t + 1] = PRIMITIVE_PATHS[i, t] + PRIMITIVE_STEPS[i, t]


# ---------------------------------------------------------------------------
# Mapping from the old 5-action cardinal system to lattice primitives
# ---------------------------------------------------------------------------
CARDINAL_TO_LATTICE = np.array([0, 1, 2, 3, 4], dtype=np.int32)
# 0=wait→WAIT, 1=right→MOVE_R, 2=down→MOVE_D, 3=up→MOVE_U, 4=left→MOVE_L


def primitive_scores_from_velocity(velocity, wait_logit=None,
                                   wait_logit_scale=1.0, wait_logit_bias=0.0,
                                   speed_bonus=0.0):
    """Score all primitives from a 2D velocity vector.

    Parameters
    ----------
    velocity : (N, 2) tensor
        Predicted velocity from the flow model.
    wait_logit : (N,) tensor or None
        Learned wait logit.  If None, wait is scored by velocity magnitude.
    wait_logit_scale, wait_logit_bias : float
        Calibration for the learned wait logit.
    speed_bonus : float
        Additive bonus for double-step primitives to encourage faster movement
        when the velocity magnitude is high.

    Returns
    -------
    scores : (N, NUM_PRIMITIVES) tensor
    """
    device = velocity.device
    dtype = velocity.dtype
    prim_vecs = PRIMITIVE_VELOCITY_VECTORS_TORCH.to(device=device, dtype=dtype)
    speed_cls = torch.from_numpy(PRIMITIVE_SPEED_CLASS).to(device=device, dtype=dtype)

    # Direction scores via dot product
    dir_scores = velocity @ prim_vecs.T  # (N, NUM_PRIMITIVES)

    # Speed modulation: agents with high velocity magnitude should prefer
    # double-step primitives; low magnitude → prefer single-step or wait.
    vel_mag = velocity.norm(dim=1, keepdim=True)  # (N, 1)
    speed_mod = speed_bonus * vel_mag * (speed_cls.unsqueeze(0) - 1.0)
    scores = dir_scores + speed_mod

    # Wait scoring
    if wait_logit is not None:
        calibrated = wait_logit_scale * wait_logit.view(-1, 1) + wait_logit_bias
        scores[:, 0] = calibrated.squeeze(1)
    else:
        scores[:, 0] = -vel_mag.squeeze(1)

    return scores


def lattice_action_labels_from_positions(discrete_positions, t_step):
    """Extract lattice primitive labels from expert integer trajectories.

    Looks at positions[t_step], positions[t_step+1], positions[t_step+2] to
    determine the 2-step primitive.  At the last two timesteps where a full
    primitive can't be extracted, falls back to single-step (duration-1)
    primitives padded with wait.

    Parameters
    ----------
    discrete_positions : (num_agents, timesteps, 2) int array
    t_step : int

    Returns
    -------
    labels : (num_agents,) int array  — indices into PRIMITIVE_NAMES
    """
    positions = np.asarray(discrete_positions)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError("discrete_positions must have shape (num_agents, timesteps, 2)")
    N, T, _ = positions.shape
    if t_step < 0 or t_step >= T:
        raise IndexError(f"t_step {t_step} outside trajectory length {T}")

    # Step 1 delta
    if t_step + 1 < T:
        step1 = np.rint(positions[:, t_step + 1] - positions[:, t_step]).astype(np.int32)
    else:
        step1 = np.zeros((N, 2), dtype=np.int32)

    # Step 2 delta
    if t_step + 2 < T:
        step2 = np.rint(positions[:, t_step + 2] - positions[:, t_step + 1]).astype(np.int32)
    else:
        step2 = np.zeros((N, 2), dtype=np.int32)

    # Match against primitive step sequences
    steps_pair = np.stack([step1, step2], axis=1)  # (N, 2, 2)
    labels = np.full(N, -1, dtype=np.int64)
    for prim_idx in range(NUM_PRIMITIVES):
        match = np.all(steps_pair == PRIMITIVE_STEPS[prim_idx], axis=(1, 2))
        labels[match] = prim_idx

    # Any unmatched agents get WAIT (defensive fallback)
    labels[labels < 0] = 0

    return labels


def expand_primitive_to_cardinal_actions(prim_labels):
    """Convert lattice primitive labels to a sequence of cardinal action labels.

    Useful for replaying lattice plans in the original 4-connected simulator.

    Parameters
    ----------
    prim_labels : (N,) int array — lattice primitive indices

    Returns
    -------
    actions : (N, PRIMITIVE_DURATION) int array — cardinal action labels per sub-step
        Uses the original encoding: 0=wait, 1=right, 2=down, 3=up, 4=left
    """
    STEP_TO_CARDINAL = {
        (0, 0): 0,   # wait
        (0, 1): 1,   # right
        (1, 0): 2,   # down
        (-1, 0): 3,  # up
        (0, -1): 4,  # left
    }
    N = len(prim_labels)
    actions = np.zeros((N, PRIMITIVE_DURATION), dtype=np.int32)
    for i in range(N):
        steps = PRIMITIVE_STEPS[prim_labels[i]]
        for t in range(PRIMITIVE_DURATION):
            actions[i, t] = STEP_TO_CARDINAL[tuple(steps[t])]
    return actions
