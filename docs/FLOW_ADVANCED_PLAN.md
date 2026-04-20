# Flow Matching Advanced Action and Representation Plan

## Motivation

The current grid pipeline primarily treats each agent decision as a wait action plus the four cardinal moves. That is a reliable execution model, but it also makes the learned flow field unnecessarily coarse. In open space, around obstacle corners, through bottlenecks, and in dense multi-agent traffic, the desired motion is often richer than "pick one of five cells." Flow matching can benefit from exposing more geometric intent during training while still keeping PIBT-style conflict handling as the safety layer at execution time.

The long-term goal is to separate three concerns:

1. Learn a rich local flow signal that can express direction, speed, uncertainty, congestion, and topology.
2. Project that signal onto the set of legal moves supported by the simulator.
3. Let a robust multi-agent resolver enforce occupancy, swaps, priorities, and feasibility.

This plan proposes an incremental path from the current 4-connected setup to richer action spaces and learned flow representations.

## Design Principles

- Keep execution legality explicit. The model may predict a rich flow signal, but the simulator or PIBT resolver should remain responsible for blocking illegal moves.
- Make the action space configurable instead of hardcoded. Four-connected, eight-connected, radius-based, and learned candidate sets should share one representation.
- Preserve backward compatibility. Existing checkpoints, datasets, and eval scripts should continue to work with the 5-action grid unless an expanded action mode is selected.
- Prefer structured action metadata. Each action should carry at least `dx`, `dy`, name, cost, and collision semantics instead of being inferred from a raw class index.
- Expand evaluation alongside training. Richer actions can look better by path length while hiding new corner conflicts or invalid diagonal crossings, so simulator metrics must be updated with the model.

## Phase 1: Generalize Grid Actions

Replace all hardcoded 5-action assumptions with a shared action specification.

Recommended action spec:

```text
ActionSpec(
  name: str,
  dx: int,
  dy: int,
  cost: float,
  is_wait: bool,
  requires_clearance: list[(int, int)] | None
)
```

Initial modes:

- `grid4`: wait plus north, south, east, west.
- `grid8`: wait plus cardinal and diagonal moves.
- `radius2`: wait plus offsets within Chebyshev radius 2, filtered by line-of-sight or decomposability.

Implementation targets:

- Dataset label encoding and decoding.
- Model output head size.
- Training loss.
- Simulator move generation.
- Eval action selection.
- Diagnostic scripts that assume five actions.
- Visualization labels and action histograms.

The first concrete implementation should be `grid8`, because it is the smallest useful expansion and naturally exposes diagonal flow.

## Phase 2: 8-Connected Grid Execution

Add the following action set:

```text
WAIT = (0, 0)
N    = (-1, 0)
S    = (1, 0)
W    = (0, -1)
E    = (0, 1)
NW   = (-1, -1)
NE   = (-1, 1)
SW   = (1, -1)
SE   = (1, 1)
```

Important simulator semantics:

- A diagonal move is valid only if the destination is in bounds and not blocked.
- Decide whether diagonal corner cutting is allowed. The safer default is to disallow it when both adjacent cardinal cells are blocked. A stricter option is to require both adjacent cardinal cells to be free.
- Detect vertex conflicts exactly as before.
- Detect swaps across identical edges.
- Add crossing checks for diagonal moves. Two agents moving on opposite diagonals through the same unit square can geometrically cross without sharing a destination cell.
- Keep wait as a first-class action.

Training and eval implications:

- The policy head must output 9 logits for `grid8`.
- Expert label generation must map diagonal next-step displacements to diagonal classes.
- If expert paths were generated with 4-connected search, the 8-connected model will still train but will not fully use diagonal actions. New expert generation should support 8-connected shortest paths or richer local targets.
- Checkpoint loading should fail clearly when the saved action dimension does not match the selected action mode.

Validation:

- Unit test action encode/decode round trips.
- Unit test diagonal obstacle rules.
- Unit test diagonal edge/crossing conflicts.
- Run a short training smoke test with `grid8`.
- Run direct simulator eval on a tiny map and compare action validity statistics.

## Phase 3: Candidate Ranking Instead of Single Action Prediction

The current model can be adapted from "choose action class" to "score candidate moves." This is especially natural for PIBT-like execution, where agents already need a preference order over neighboring cells.

Proposed interface:

```text
scores = model.score_candidates(agent_state, candidate_actions, map_context)
ordered_actions = sort_by_score(scores)
resolved_action = pibt_resolver.try_in_order(ordered_actions)
```

Benefits:

- The model can express a full preference ordering.
- The resolver can gracefully fall back when the top action is blocked by another agent.
- The same model structure can support variable-size candidate sets, such as visible cells or planner-generated path prefixes.

Training targets:

- Cross-entropy over expert action when one action is known.
- Pairwise ranking loss where expert action should score above alternatives.
- Soft target distribution over near-optimal moves, useful when multiple actions are equivalent.

## Phase 4: Continuous Velocity Targets With Discrete Projection

Flow matching is especially well suited to continuous vector fields. Instead of only learning a discrete action class, train the model to predict a desired local velocity:

```text
v = (vx, vy)
```

Execution then projects the vector onto legal actions:

```text
chosen_action = argmax_a dot(normalize(v), normalize((a.dx, a.dy))) - penalties(a)
```

This allows the model to learn diagonal intent, smooth obstacle avoidance, and crowd flow even if the executor is temporarily constrained to 4-connected or 8-connected actions.

Recommended outputs:

- Discrete action logits for compatibility.
- Continuous velocity head for geometric flow.
- Optional uncertainty or temperature head for ambiguous regions.

Training losses:

- Discrete cross-entropy.
- L2 or cosine loss against expert displacement.
- Flow matching loss against interpolated states.
- Consistency loss between discrete logits and projected velocity.

## Phase 5: Short-Horizon Flow Supervision

Immediate next actions can be noisy or arbitrary in symmetric situations. Add supervision for where the agent should be after `k` steps.

Suggested horizons:

- `k=2`: local bend and diagonal intent.
- `k=4`: corridor direction.
- `k=8`: obstacle-side and bottleneck choice.

Targets:

- Future displacement vector.
- Future waypoint cell.
- Local heatmap over reachable cells.
- Direction to path prefix endpoint.

This should improve learning in places where the next action alone hides the strategic flow, such as choosing the correct side of an obstacle or waiting before a bottleneck.

## Phase 6: Topological and Congestion-Aware Features

Richer actions are most useful when paired with richer input features. Add optional channels that describe the local traffic structure:

- Distance-to-goal gradient.
- Shortest-path direction field.
- Obstacle density.
- Corridor width.
- Bottleneck or articulation-point indicators.
- Current local occupancy.
- Predicted next occupancy.
- Opposing-flow pressure.
- Same-direction flow pressure.
- Agent priority or reservation state.

These features help the model distinguish open rooms, narrow corridors, corners, and contested bottlenecks even when local occupancy looks similar.

## Phase 7: Macro-Actions and Multi-Resolution Flow

Once the 8-connected path is stable, expand candidate actions beyond adjacent cells.

Possible action families:

- Chebyshev radius 2 or 3 offsets.
- Line-of-sight waypoint cells.
- Local path prefixes from A*.
- Top-k next cells from k-shortest path search.
- Region transitions such as "move toward corridor exit."

Execution should not teleport agents unless the simulator explicitly supports macro-steps. Instead, macro-actions can act as intermediate goals for a low-level resolver.

This gives the flow model a wider view of intent while preserving single-cell collision handling.

## Phase 8: Evaluation Matrix

Each representation should be evaluated across:

- Sparse open maps.
- Dense open maps.
- Narrow corridors.
- Random obstacle fields.
- Warehouse-style maps.
- Opposing-flow bottleneck scenarios.
- Mixed agent counts.

Metrics:

- Success rate.
- Sum of costs.
- Makespan.
- Invalid action rate.
- Conflict resolution fallbacks.
- Wait ratio.
- Diagonal usage ratio.
- Path stretch versus expert.
- Throughput through bottlenecks.
- Runtime per step.

Recommended ablations:

- 4-connected baseline.
- 8-connected execution with 4-connected-trained model.
- 8-connected training and execution.
- 8-connected plus candidate ranking.
- 8-connected plus short-horizon targets.
- Continuous velocity head with 4-connected projection.
- Continuous velocity head with 8-connected projection.

## Near-Term Implementation Checklist

1. Introduce one shared action utility module for grid action specs.
2. Replace hardcoded action-count constants with `len(action_specs)`.
3. Add a command-line/config option such as `--action-mode grid4|grid8`.
4. Update dataset label mapping to use action specs.
5. Update model initialization so output dimension follows action mode.
6. Update simulator legal move generation.
7. Add diagonal collision and corner-cutting checks.
8. Update eval scripts to select and decode actions through the shared utility.
9. Add smoke tests for action validity and decode behavior.
10. Document checkpoint compatibility and expected retraining behavior.

## Recommended First Experiment

Start with `grid8` and keep everything else unchanged.

Experiment:

- Generate or preprocess training data with `--action-mode grid8`.
- Train the existing model with a 9-way action head.
- Evaluate on the same scenarios as the 4-connected baseline.
- Track success rate, wait ratio, diagonal usage ratio, and invalid move rate.

Expected outcomes:

- Open maps should show shorter paths and smoother movement.
- Obstacle-heavy maps may improve if corner handling is correct.
- Dense maps may need better conflict detection because diagonal crossing conflicts become more common.
- If training data remains 4-connected, diagonal logits may be underused, so new expert generation is important.

## Longer-Term Direction

The strongest architecture is likely a hybrid:

```text
flow model predicts velocity, waypoint, or candidate ranking
PIBT-style resolver enforces legal multi-agent execution
simulator provides exact conflict and obstacle semantics
```

That keeps the learned representation rich enough for flow matching while preserving the deterministic safety properties that make PIBT useful.
