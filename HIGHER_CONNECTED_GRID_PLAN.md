# Higher-Connected Grid Expert Plan

This note records the current plan for moving the flow-matching pipeline beyond the
current 4-connected MAPF expert setup. The short-term decision is to pause any
solver-level changes until after the Monday project presentation, then revisit the
expert-generation stack carefully rather than forcing an ad hoc 8-connected patch
into EECBS.

## Motivation

The current data and simulator path operate on a 4-connected grid: wait, right,
down, up, and left. Flow matching may benefit from a richer action representation
because the model can learn smoother local transport fields, less axis-aligned
motion, and more informative local geometry around obstacles and other agents.

An 8-connected grid is the first useful step:

- wait
- 4 cardinal moves
- 4 diagonal moves

Longer term, the same framing could support richer local action sets, such as
radius-2 moves, motion primitives, continuous steering targets, or graph-native
actions over non-grid roadmaps.

## Current Constraint

The EECBS-flow expert generator currently appears to be 4-connected by design.
The README does not expose an 8-connected option, the binary help text has no
diagonal or connectivity flag, and the source comments explicitly mention
undirected unweighted 4-neighbor grids.

That matters because the expert trajectories are part of the supervision signal.
If the model is trained with 8-connected simulator actions but the expert paths
were generated only with 4-connected motion, then diagonal actions will be absent
or only artificially introduced. The model may still benefit from richer heuristic
features, but it will not learn true 8-connected expert behavior unless the
expert generator also supports that motion model.

## Why Not Patch EECBS Casually?

Changing the neighbor list from 4 moves to 8 moves is not enough. EECBS, CBS, and
their speedups rely on assumptions that are easy to violate when diagonal moves
are introduced.

Important solver-level considerations:

- Low-level search must expand valid diagonal moves and use an admissible
  heuristic for the new motion model.
- The cost model must be explicit. If diagonal moves cost 1, the heuristic should
  behave like Chebyshev distance. If diagonal moves cost sqrt(2), the solver is no
  longer an unweighted unit-cost grid solver unless the implementation is updated
  accordingly.
- Vertex conflicts still matter: two agents cannot occupy the same cell at the
  same timestep.
- Reverse edge conflicts still matter: two agents cannot swap cells over the same
  edge in opposite directions.
- Diagonal crossing conflicts must be added. Two diagonal moves can geometrically
  cross through the same square even if they do not share start or end cells.
- Corner-cutting rules must be defined. A diagonal move between two blocked
  cardinal-adjacent cells may be physically invalid depending on the intended
  robot footprint.
- MDD construction, focal search priorities, and dependency heuristics need to be
  checked under the new action set.
- Rectangle reasoning, corridor reasoning, target reasoning, and symmetry
  reasoning may encode 4-connected grid assumptions. These may need to be
  disabled initially, then reintroduced only after correctness checks.
- Performance will likely change because the branching factor increases from 5
  including wait to 9 including wait.

Because of these interactions, a quick patch could make the solver run but would
not automatically preserve correctness, optimality, bounded-suboptimality, or
performance.

## Preferred Direction After Presentation

After the presentation, the clean path is to evaluate solver options before
generating a large new dataset.

### Option A: Extend EECBS Carefully

This is attractive if we want to preserve the bounded-suboptimal EECBS workflow
and existing data scripts.

Work required:

1. Add an explicit connectivity or motion-model option, such as
   `--connectivity 4|8`.
2. Add a diagonal rule option, such as `none`, `blocked_pair`, or
   `no_corner_cutting`.
3. Update low-level neighbor expansion.
4. Update the low-level heuristic for the chosen diagonal cost model.
5. Add diagonal crossing conflict detection.
6. Audit MDD generation and conflict classification.
7. Disable 4-connected-specific symmetry reasoning during the first validation
   pass.
8. Add small regression tests on maps with known optimal 8-connected solutions.
9. Add collision validators that check every produced path under the grid8 rules.
10. Benchmark runtime and solution quality against the 4-connected baseline.

This route is more engineering work, but it keeps continuity with the current
pipeline.

### Option B: Use or Implement CBS for Optimal 8-Connected Experts

This may be the cleaner scientific choice if the goal is globally optimal
8-connected expert trajectories.

Work required:

1. Choose an existing CBS implementation that already supports 8-connected grids,
   or implement a minimal CBS variant for our dataset generation needs.
2. Make the cost model explicit:
   - unit-cost diagonals for discrete-time MAPF, or
   - weighted diagonals with sqrt(2) costs if matching continuous geometry.
3. Confirm the implementation supports vertex, edge-swap, and diagonal-crossing
   conflicts.
4. Confirm corner-cutting behavior matches the simulator.
5. Produce a small validation suite with hand-checkable maps.
6. Generate a small pilot dataset before committing to full-scale regeneration.
7. Compare path lengths, makespan, success rate, and generation time against the
   current EECBS data.

This route is slower if optimal CBS struggles on dense instances, but it gives a
clearer expert target.

### Option C: Hybrid Bootstrap

This is useful if solver work takes longer but we want early model experiments.

The idea is to use the current 4-connected expert paths and apply a validated
post-processing pass that replaces safe two-step cardinal corners with diagonal
moves while preserving synchronized timesteps.

This should be treated as augmented supervision, not a globally optimal
8-connected expert.

Required safeguards:

- Every transformed step must be valid in the grid8 simulator.
- The transform must preserve or revalidate vertex conflicts.
- The transform must preserve or revalidate edge-swap conflicts.
- The transform must reject diagonal crossing conflicts.
- The transform must obey the same corner-cutting rule used by training and eval.
- The output should be marked separately from true solver-generated grid8 data.

This can help the model see diagonal labels, but it should not replace real
8-connected expert generation for final claims.

## Dataset Regeneration Plan

Before deleting or replacing large data directories, run the transition as a
staged experiment.

Recommended stages:

1. Keep the current 4-connected data as the baseline whenever disk allows.
2. Generate or derive a tiny grid8 pilot dataset on a few empty and obstacle maps.
3. Preprocess with `--action-mode grid8` and a separate BD directory.
4. Train a short smoke model and verify that diagonal labels are present.
5. Run simulator evals with `--action-mode grid8`.
6. Inspect failure cases visually.
7. Only then generate the larger dataset.

Suggested dataset names:

- `data/flow_training_data_multi_grid8_pilot`
- `data/bd_npzs_grid8/large_scale`
- `data/preprocessed_grid8_pilot`
- `data/preprocessed_grid8_full`

On the Lambda machine, disk is tight because `data/preprocessed` is much larger
than the raw trajectories and BD files. If space is needed, prefer deleting or
moving preprocessed artifacts first, because they can be regenerated from the raw
data. Avoid deleting raw expert trajectories until the replacement dataset has
been validated.

## Training and Eval Alignment Checklist

All of these must agree for a clean grid8 experiment:

- Expert generator connectivity
- Expert generator diagonal cost model
- Expert generator corner-cutting rule
- Raw trajectory validator
- BD generation action set
- Preprocessing action labels
- Model action head dimension
- Auxiliary action dimension
- Simulator action mask
- PIBT or planner action ordering
- Eval scripts and checkpoint metadata

Any mismatch can create silent bugs where the model learns one action semantics
but evaluation executes another.

## Recommended Near-Term Position

For the presentation, keep the current 4-connected expert setup as the stable
baseline. Present higher-connected grids as a motivated next step with clear
solver and validation requirements.

After the presentation, prioritize this order:

1. Decide whether the expert target should be optimal CBS grid8 or
   bounded-suboptimal EECBS grid8.
2. Define the diagonal cost and corner-cutting rule.
3. Build a small solver validation suite.
4. Generate a tiny grid8 pilot dataset.
5. Run short training and eval smoke tests.
6. Scale only after the pilot behaves as expected.

This keeps the idea scientifically clean: richer flow matching should be tested
against trajectories that are genuinely generated under the richer motion model,
not just against a model-side action-space expansion.
