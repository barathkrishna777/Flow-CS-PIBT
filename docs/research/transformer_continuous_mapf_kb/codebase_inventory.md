# Codebase Inventory

Updated: 2026-04-20 19:28 EDT

## High-Level Repo State

- Project already has two stacks:
  - grid-world Flow-CS-PIBT stack: EECBS/BD data, `FlowGNNModel`, grid simulator and CS-PIBT/LaCAM evaluation;
  - continuous-space stack: continuous env, ORCA-style shields, continuous data generation, continuous training/eval, benchmark runner.
- New direction should preserve the current continuous stack and make transformer continuous policy first-class instead of extending grid action variants.

## Docs

### `docs/FLOW_ADVANCED_PLAN.md`

- Purpose: long-term plan for richer grid action/representation spaces.
- Key content:
  - generalize hardcoded 5-action grid actions to structured `ActionSpec`;
  - add `grid8`, diagonal legality, crossing conflicts, candidate ranking;
  - add continuous velocity targets with discrete projection;
  - add short-horizon flow targets, congestion/topology features, macro-actions.
- Assumptions/limitations:
  - primary executor is still grid simulator / PIBT resolver;
  - collision semantics are grid vertex/swap/diagonal crossing;
  - continuous prediction is framed as projected back to legal grid actions.
- Integration relevance:
  - keep: structured action metadata, short-horizon supervision, continuous velocity head, explicit safety projection;
  - demote: grid8/radius2/macro-grid action phases if continuous MAPF becomes main technical direction.

### `README.md`

- States main grid model is continuous flow matching policy converted to action preferences for CS-PIBT/LaCAM.
- Continuous stack section lists:
  - `main_pys/continuous_env.py`: continuous dynamics, obstacles, ORCA-style shield;
  - `scripts/generate_and_preprocess_continuous.py`: dataset root/manifest;
  - `scripts/generate_continuous_data.py`: EECBS-flow-to-continuous + ORCA fallback;
  - `main_pys/train_continuous.py`: continuous flow or 8-direction discrete baseline;
  - `scripts/eval_continuous.py`: ORCA/flow/discrete eval and trajectory plots.
- PICBF-CS usage is documented via `PICBF_CS_PATH` and `--shield-type picbf-cs`.

### `docs/implementation_details.md`

- Current continuous comparison: ORCA vs continuous flow vs discretized continuous-action baseline.
- Current claim scope:
  - open-space `empty-48-48` is primary continuous benchmark;
  - `agent_fraction_at_goal` is primary metric;
  - `success` is strict secondary metric;
  - obstacle-map continuous claims are not ready.
- Current defaults:
  - flow inference: integration steps 3, consensus samples 3;
  - discrete baseline: 8-direction continuous-action classifier;
  - shield: ORCA;
  - agent counts: 32/64/96/128.
- Reported current behavior:
  - ORCA strongest;
  - flow below ORCA but clearly above discrete continuous-action baseline;
  - flow `success` remains 0 in logged runs, so fraction-at-goal is more informative.
- Historical issue: prior `--max-scenarios 5` selected lexicographic scenarios `1,10,11,12,13`.
- Current status: `main_pys/continuous_scenarios.py::select_scenarios` uses `sort_scenarios_numerically`, so current code appears to fix this. Still include a smoke check before official runs.

### `docs/PAPER_PLAN.md`

- Older paper plan. Continuous phase originally described as days 11-17 with ORCA expert, continuous env, flow model, discrete baseline.
- Many items are now implemented. It still frames grid-world comparison as priority and continuous as possible preliminary result.
- For new direction, treat as historical context rather than active roadmap.

## Models

### `main_pys/generative_model.py`

- Class: `FlowGNNModel`.
- Architecture:
  - local visual patch CNN over `data.x`;
  - concatenate auxiliary features (`aux_features` or `bd_pred`);
  - concatenate flow variables `v_t` and `t`;
  - residual stack of PyG `SAGEConv(hidden_dim, hidden_dim)`;
  - velocity flow head `post_mp`;
  - auxiliary action head.
- Config defaults:
  - grid default `num_input_channels=3`, `action_dim=5`;
  - continuous callers use `num_input_channels=4`, `action_dim=num_directions+1`.
- Assumptions/limitations:
  - message passing only along precomputed neighbor graph;
  - edge attributes are unused by `SAGEConv`;
  - relation geometry influences graph construction but not messages after adjacency;
  - no explicit permutation issue because graph convolution is equivariant over nodes for fixed graph.
- Integration point:
  - current strong baseline for continuous; transformer must beat or explain tradeoff vs this model.

### `main_pys/transformer_model.py`

- Classes:
  - `SinusoidalTimeEmbedding`: defined but not used by `FlowTransformerModel` forward path;
  - `RelativePosEncoding`: linear map from 2D `edge_attr` to per-head additive bias;
  - `LocalMaskedAttention`: masked self-attention using `edge_index`, self attention, and relative bias;
  - `TransformerBlock`: pre-norm attention + FFN;
  - `FlowTransformerModel`: same forward signature as `FlowGNNModel`.
- Current design:
  - local patch CNN + aux features;
  - flow/time conditioning via concatenation of `v_t` and scalar `t`;
  - local masked attention per PyG graph; self + graph neighbors;
  - optional action chunking via `chunk_horizon`;
  - bias-free linear projections and FFN.
- Limitations:
  - attention implementation loops over graphs and builds dense `(n, n, heads)` logits before masking; memory is effectively `O(sum n_i^2 * heads)`, not true sparse `O(E * heads)`;
  - relative bias is symmetric reused for both directions even though `edge_attr` has direction; can lose directional semantics unless graph contains both directed edges with correct attrs;
  - no use of current velocities except noisy target `v_t`; no observed velocity/state history tokenization;
  - no global/map token path;
  - no benchmark-runner pass-through for transformer model type.
- Assessment: good prototype/foundation for interface compatibility; needs architectural and systems upgrade before major technical direction.

## Continuous Data And Inputs

### `main_pys/model_inputs.py`

- Grid functions:
  - `discrete_action_labels_from_positions`;
  - `create_data_object`;
  - `get_bd_prefs`;
  - `normalize_graph_data`.
- Continuous functions:
  - `extract_continuous_patches(grid, positions, k)`: local obstacle patch from float positions using floor sampling;
  - `build_continuous_neighbor_graph(pos_list, m, neighbor_radius=None)`: kNN graph, optional radius cutoff, directed source-to-neighbor edges;
  - `velocity_to_direction_labels`, `labels_to_direction_vectors`: wait + angular bins;
  - `create_continuous_data_object`: builds PyG `Data` with 4-channel patches `[map, goal_dx, goal_dy, agent_patch]`, kNN `edge_index/edge_attr`, `aux_features=[rel_goal/k, goal_dist/k, at_goal, max_speed]`, `y=velocity`, `action_label`;
  - `normalize_continuous_graph_data`: normalizes edge attrs by `k`, `y` by `max_speed`, clamps channels.
- Limitations:
  - local map representation is grid-derived patch, not SDF patch;
  - `agent_patch` is Gaussian occupancy built by nested Python loops over agents;
  - no explicit neighbor velocity/history in graph data;
  - `neighbor_radius` exists but dataset calls currently do not expose it.

### `main_pys/dataset_continuous.py`

- Class: `ContinuousFlowDataset`.
- Loads `.npz` rollouts and treats each timestep as one graph sample.
- Filters by expert source and scenario ids/ranges.
- Builds sample via `create_continuous_data_object`.
- Adds node weights to rebalance moving vs waiting.
- Splits train/val by rollout id.
- Sampler: `build_continuous_weighted_sampler` balances agent-count buckets and expert sources, optional difficulty oversampling.
- Limitations:
  - graph construction is per-sample and can be CPU-heavy;
  - uses only instantaneous position/goals/target velocities; no multi-step target chunks despite transformer support.

### `main_pys/dataset_continuous_preprocessed.py`

- Class: `PreprocessedContinuousShardDataset`.
- Loads compact sharded PyG samples from `manifest.pt` and `shard_*.pt`.
- Compact dtype storage: `float16`, `int32`, `uint8` action labels.
- Supports filters and weighted sampler.
- Limitation: no map reconstruction needed because compact sample stores full `x`; maps are loaded but not used in `_build_sample_from_compact`.

### `scripts/preprocess_continuous_shards.py`

- Converts raw continuous dataset into compact shards.
- Existing resume mode has placeholder metadata for skipped samples, which can distort weighted sampling if resumed from partial shards.
- Useful for future transformer benchmarks because raw graph construction will bottleneck data loading.

## Continuous Environment And Shields

### `main_pys/continuous_env.py`

- Utilities:
  - `compute_sdf`, `sdf_gradient`, bilinear sampling helpers;
  - `parse_scene_file`, `grid_starts_to_continuous`;
  - `compute_smoothness`, `compute_path_length`.
- Default dynamics:
  - `dt=0.2`;
  - `max_speed=1.0`;
  - `agent_radius=0.3`;
  - `goal_tolerance=0.25`;
  - positions are row/col continuous coordinates with grid cell centers at `+0.5`.
- `ContinuousMAPFEnv`:
  - maintains positions/goals/history/metrics/priorities;
  - `goal_directed_velocities()` creates direct-to-goal preferred velocities;
  - `step()` applies selected shield, integrates position, blocks obstacle hits, counts collisions/near collisions/obstacle hits, updates priorities.
- Metrics:
  - success, agents_at_goal, agent_fraction_at_goal, path_length, path_length_ratio, smoothness, collisions, near_collisions, obstacle_hits, mean_arrival_step.
- Shield modes:
  - `none`: clip speed only;
  - `simple`: stops agents predicted to collide/hit obstacle;
  - `orca`: `ORCAStyleShield.project(... use_true_orca=True)`; uses `rvo2` if import available and succeeds, otherwise heuristic fallback;
  - `heuristic-orca`: heuristic pairwise + SDF constraints, no rvo2;
  - `po-orca`: priority-ordered heuristic ORCA;
  - `epibt`: continuous candidate/backtracking priority heuristic;
  - `picbf-cs`: adapter to external `continuous_collision_shield` package.
- Important limitation:
  - ORCA obstacle handling is not pure RVO2 obstacle polygons; repo first applies SDF constraints, then uses rvo2 only for agent-agent. This is pragmatic but should be reported accurately.

### `ORCAStyleShield`

- Uses SDF obstacle constraints and pairwise velocity corrections.
- `rvo2` path:
  - applies SDF obstacle constraints;
  - initializes `rvo2.PyRVOSimulator`;
  - adds agents and preferred velocities;
  - does one step and returns velocities.
- Heuristic path:
  - iterates pairwise constraint + SDF obstacle constraints + speed clipping.
- PO-ORCA path:
  - processes agents by priority;
  - lower-priority agents avoid committed higher-priority velocities;
  - symmetric cleanup pass.
- Likely failure modes:
  - reactive local minima/deadlocks;
  - obstacle bottlenecks without global routing;
  - dense crowds where heuristic pairwise corrections accumulate poorly;
  - silent fallback from rvo2 makes `orca` ambiguous unless logged.

### `EPIBTShield`

- Continuous analogue of priority candidate selection/backtracking.
- Candidate velocities: preferred, half-speed preferred, 16 compass directions, wait.
- Checks next-position and continuous crossing conflicts.
- Current code documents backtracking but implementation mostly assigns zero when no candidate found; `_assign_actions` does not visibly perform recursive reassignment. Treat as heuristic ablation, not core guarantee.

### `PICBFCSShield`

- External CBF adapter.
- Converts grid obstacles to merged AABB obstacles and constructs an obstacle index.
- Exposes debug component counts/status/solve time.
- Better long-term shield candidate if external package is maintained and performance is acceptable.

## Continuous Training/Eval

### `main_pys/train_continuous.py`

- Supports `--policy-type flow|discrete` and `--model-type gnn|transformer`.
- `model_type=transformer` constructs `FlowTransformerModel`.
- Flow loss:
  - rectified-flow style interpolation `x_t = t*x_1 + (1-t)*x_0`;
  - predicts target `x_1 - x_0`;
  - optional shield-aware loss projecting predicted velocity through ORCAStyleShield;
  - auxiliary action CE loss.
- Discrete loss:
  - zero flow/time input;
  - action CE on wait + directions.
- Checkpoints include `model_type`, `model_config`, `dataset_config`, `loss_config`.
- Limitations:
  - shield-aware train shield choices only `orca|heuristic-orca`;
  - `_shield_project_batch` instantiates `ORCAStyleShield` per graph during loss, likely expensive;
  - chunk horizon is passed into model but losses still assume `(N, 2)` targets, so chunking is not actually trained yet.

### `scripts/eval_continuous.py`

- Evaluates `policy=orca|flow|discrete`.
- `load_model()` reconstructs `FlowTransformerModel` if checkpoint `model_type == "transformer"`.
- Flow inference:
  - Euler integration with `num_integration_steps`;
  - consensus samples; aggregation = mean/medoid/best.
- Discrete inference:
  - action logits + temperature + argmax + direction vectors.
- Records metrics and can save trajectory plots.
- Limitations:
  - no shield intervention metric;
  - no learned-policy preferred-vs-shielded delta metric;
  - no direct deadlock/stall metric beyond fraction-at-goal/mean arrival.

### `scripts/run_continuous_benchmark.py`

- Orchestrates generate/train/eval/summarize.
- Summaries cover main metrics and consensus sweeps.
- Current CLI lacks `--model-type`, `--num-heads`, `--chunk-horizon`, so transformer path is not first-class in packaged benchmark.
- Expert sources in benchmark parser are `eecbs|orca|hybrid`; data generator also supports `lacam3|po-orca`.

### `scripts/generate_continuous_data.py`

- Expert generation sources:
  - EECBS discrete paths converted to continuous dense trajectories;
  - LaCAM3 paths;
  - ORCA/PO-ORCA rollouts;
  - hybrid fallback.
- `discrete_paths_to_continuous`: adds `+0.5` cell centers, densifies by `dt` and `max_speed`.
- `rollout_orca_policy`: direct-to-goal preferred velocity shielded by selected ORCA mode.
- `validate_replay` exists but EECBS validation is skipped because SDF margins caused false-positive obstacle hits.
- Saved `.npz`: positions, velocities, goals, expert source used/requested, fallback reason, action labels, difficulty-ish metadata.
- Limitation:
  - ORCA expert is reactive and may not solve MAPF-like obstacle routing;
  - hybrid source consistency can be mixed across scenarios/densities and should be logged/sliced in eval.

### `scripts/generate_and_preprocess_continuous.py`

- Dataset root wrapper, manifest writer, split metadata, SDF precomputation.
- Uses `compute_sdf` for maps.
- Does not create compact PyG shards; that is separate `scripts/preprocess_continuous_shards.py`.

### `tests/test_po_orca.py`

- Lightweight synthetic validation for SDF, PO-ORCA, obstacle handling, scaling, PICBF adapter.
- Not a formal unit-test style suite; runnable script with PASS/FAIL counters.
- Useful smoke checks before shield changes.

## Immediate Integration Points For New Direction

- Make transformer first-class:
  - expose model type/heads/chunking in `scripts/run_continuous_benchmark.py`;
  - add smoke tests for transformer forward/train/eval;
  - replace dense local attention implementation with true sparse/local attention.
- Upgrade inputs:
  - include current velocity, previous velocity/history, neighbor velocity deltas, SDF samples, obstacle gradients;
  - expose `neighbor_radius` in datasets/eval and align with shield communication radius.
- Upgrade losses:
  - train chunked velocity targets;
  - log shield projection delta;
  - add collision/near-collision surrogate and goal progress/smoothness terms carefully.
- Evaluate shields separately:
  - ORCA true vs heuristic should be explicit;
  - PO-ORCA and EPIBT are ablations;
  - PICBF-CS is long-term research path, not near-term default until dependency/performance are stable.
