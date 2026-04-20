# Continuous MAPF Notes

Updated: 2026-04-20 20:02 EDT

## Recommended Problem Formulation

Continuous-space MAPF in this repo should be defined as:

- workspace: 2D continuous coordinates over grid-derived maps;
- obstacles: blocked grid cells treated as axis-aligned occupied unit squares; optionally represented by SDF/AABBs;
- agents: discs/circles with radius `r`;
- state per agent:
  - position `p_i(t) in R^2`;
  - current velocity `v_i(t) in R^2`;
  - goal `g_i in R^2`;
  - optional priority/deadlock state;
- action:
  - nominal preferred velocity `u_i(t) in R^2`;
  - shielded executable velocity `u_safe_i(t)`;
- dynamics:
  - first-order single integrator: `p(t+dt) = p(t) + u_safe(t) * dt`;
  - speed bound `||u_safe|| <= max_speed`;
- success:
  - all agents within `goal_tolerance` before horizon;
- safety:
  - pairwise distance >= `2*agent_radius`;
  - disc does not intersect obstacle cells/boundaries;
  - count near-collisions at configurable margin.

Repo defaults already present:

- `dt=0.2`;
- `max_speed=1.0`;
- `agent_radius=0.3`;
- `goal_tolerance=0.25`;
- max rollout/eval steps often `256`;
- local patch radius `k=4`;
- neighbor count `m=5`;
- directions for discrete baseline `8 + wait`.

## State Representation

Current:

- local raster patch channels:
  - map obstacle patch;
  - goal_dx patch;
  - goal_dy patch;
  - Gaussian occupancy patch of other agents;
- aux:
  - rel_goal/k;
  - goal_dist/k;
  - at_goal;
  - max_speed;
- graph:
  - kNN edges from positions;
  - edge_attr = relative position/k.

Recommended additions:

- current velocity and previous executed velocity;
- SDF local patch and SDF gradient patch;
- neighbor relative velocity and predicted closest approach;
- previous shield projection delta: `u_safe - u_pref`;
- priority and/or stalled-for steps;
- optional global guidance feature:
  - next waypoint from EECBS/LaCAM path;
  - vector to path prefix endpoint;
  - distance-to-goal or distance-to-path field.

## Output / Action Choice

Near-term primary:

- preferred velocity vector.

Rationale:

- aligns with RVO2/ORCA API;
- aligns with CBF nominal-control projection;
- keeps learned policy differentiable and shield-agnostic.

Secondary heads:

- action logits over wait + 8/16 directions for auxiliary CE and baseline comparison;
- future waypoint vector for obstacle maps;
- shield-intervention prediction or confidence head.

Chunking:

- horizon `H=3` is the first useful target because it covers ~0.6s at `dt=0.2`;
- `H=5` covers 1.0s and may be better for smoothness but harder.

## Expert Data

Current sources:

- EECBS discrete paths converted to continuous dense trajectories;
- LaCAM3 discrete paths converted to continuous trajectories;
- ORCA/PO-ORCA rollouts;
- hybrid fallback.

Recommended expert recipe by map type:

- open-space:
  - ORCA/PO-ORCA is acceptable expert and strong baseline;
  - useful for learning smooth reciprocal behavior.
- obstacle maps:
  - prefer EECBS/LaCAM-derived path guidance where available;
  - ORCA-only is likely poor because direct-to-goal preferred velocities plus local avoidance does not plan around walls/bottlenecks.
- dense traffic:
  - hybrid expert with source labels preserved;
  - evaluate source-stratified performance.

EECBS-to-continuous caveat:

- Densification creates piecewise-linear movement along grid path centers.
- It can be collision-free in discrete cells but not necessarily smooth or clearance-optimal for discs.
- Continuous replay validation was skipped for EECBS in current code due SDF false-positive obstacle margins; roadmap should revisit with robust clearance-aware validation.

LaCAM-derived path role:

- useful scalable discrete global guidance for hard MAPF instances;
- can provide waypoint/path-prefix supervision rather than direct velocity only.

ORCA role:

- baseline: yes.
- shield: yes, near-term primary.
- expert: yes for open maps and fallback, but not sole expert for obstacle maps.
- training label for shield-aware loss: yes, but log/source-stratify.

## Training Losses

Current:

- flow matching MSE on velocity targets;
- auxiliary action cross-entropy;
- optional shield-aware MSE after ORCA projection.

Recommended near-term:

```text
L = w_flow * L_flow
  + w_action * L_action
  + w_goal * L_progress_aux
  + w_smooth * L_chunk_smooth
  + w_shield * L_shield_aware
```

Phase order:

1. preserve current flow + action loss for comparability;
2. add logging-only shield intervention metrics before adding shield loss;
3. add shield-aware loss as ablation;
4. add chunk smoothness/progress only after chunk targets exist.

Loss details:

- Flow loss:
  - target expert velocity or velocity chunk.
- Velocity imitation:
  - direct MSE/cosine to expert velocity can be auxiliary for stability.
- Collision/near-collision penalty:
  - use differentiable one-step surrogate only during training;
  - do not claim safety guarantee; shield remains guarantee mechanism.
- Goal progress:
  - penalize predicted next distance if it increases, with masking for agents at goal.
- Smoothness:
  - penalize acceleration between current velocity and predicted chunk, and between chunk steps.
- Shield-aware:
  - project predicted velocity through shield;
  - penalize post-shield mismatch to expert or large projection deltas;
  - careful because shield projection is non-differentiable; current straight-through style is reasonable for ablation.

## ORCA Mismatch With MAPF

Direct issue:

- ORCA is reactive collision avoidance given preferred velocities.
- MAPF requires reaching goals, often through bottlenecks and obstacle topology.

Mitigations:

- policy must provide global/local intent, not just direct-to-goal vector;
- add path-prefix/global guidance features for obstacle maps;
- use priority/deadlock state to break symmetry;
- evaluate deadlock/stall explicitly;
- use ORCA only as safety projector, not as sole planning story.

## Obstacle Representation In ORCA/RVO2

RVO2 direct obstacle semantics:

- static obstacles are polygonal line-segment loops added before simulation and processed.
- RVO2 docs note global navigation/roadmaps may be needed for obstacles.

Current repo semantics:

- obstacle map is grid cells;
- repo computes SDF over occupied/free cells;
- ORCAStyleShield applies SDF obstacle constraints before rvo2 agent-agent step;
- rvo2 obstacle polygons are not passed in current `_project_with_rvo2`.

Implications:

- Label results as "ORCA-style with SDF obstacle constraints" unless true RVO2 obstacles are implemented.
- Obstacle-map failures may come from SDF constraints, lack of global guidance, or direct-to-goal preferred velocities.

## Shield Modes In `ContinuousMAPFEnv`

### `none`

- Provides: speed clipping only.
- Likely failure: collisions and obstacle hits.
- Use: ablation/debug; not safety evaluation.
- Learned interaction: reveals raw policy intent and collision rate.

### `simple`

- Provides: naive stop-on-conflict/obstacle filter.
- Likely failure: over-conservative freezing, deadlocks, no reciprocal avoidance.
- Use: ablation/sanity baseline.
- Learned interaction: good to measure whether learned policy can mostly avoid conflicts itself.

### `heuristic-orca`

- Provides: repo heuristic pairwise velocity projection + SDF obstacle constraints; deterministic fallback.
- Likely failure: dense pairwise correction artifacts, deadlocks, no true ORCA LP guarantee, obstacle bottlenecks.
- Use: ablation and fallback baseline.
- Learned interaction: suitable for fast shield-aware training if true rvo2 unavailable/slow, but label accurately.

### `orca`

- Provides: if `rvo2` import/path works, true rvo2 agent-agent ORCA after SDF obstacle preconstraint; else heuristic fallback.
- Likely failure: local minima/deadlocks, direct-to-goal obstacle traps, ambiguity if fallback not logged.
- Use: near-term primary baseline and shield.
- Learned interaction: transformer outputs preferred velocity; shield produces safe velocity; log projection delta/intervention.

### `po-orca`

- Provides: priority-ordered heuristic ORCA; higher-priority agents deviate less, lower-priority agents bear more avoidance.
- Likely failure: priority bias/starvation, heuristic artifacts, less symmetric than ORCA, not official ORCA guarantee.
- Use: ablation/research for deadlock breaking; maybe expert source for priority-aware data.
- Learned interaction: include priority/stall features; compare against ORCA on bottlenecks.

### `epibt`

- Provides: continuous candidate velocity selection inspired by PIBT with priorities/backtracking concept.
- Likely failure: implementation appears mostly greedy with wait fallback; candidate discretization can lose smoothness; no formal continuous guarantee.
- Use: ablation, not primary.
- Learned interaction: useful as bridge from grid CS-PIBT heritage; transformer candidate-ranking head could eventually feed it.

### `picbf-cs`

- Provides: adapter to external local joint CBF/QP-style shield with AABB obstacles and component debug info.
- Likely failure: dependency/setup friction, solver runtime, infeasibility/status handling, parameter sensitivity.
- Use: secondary research strategy and eventual stronger safety layer.
- Learned interaction: ideal nominal-control projection interface; keep policy output as nominal velocity to be compatible.

## Primary Shield Strategy

Near-term primary:

- `orca` with explicit runtime logging:
  - whether rvo2 available;
  - whether fallback occurred;
  - shield projection norm;
  - stopped/clipped fraction.

Use `heuristic-orca` as deterministic fallback/ablation.

Why:

- already implemented;
- aligns with preferred velocity output;
- strong baseline on open maps;
- low integration risk.

Secondary research:

- `picbf-cs` for formal control-barrier projection;
- `po-orca` for priority/deadlock ablations;
- `epibt` only after implementation correctness is clarified.

## Metrics

Keep existing:

- success;
- agents_at_goal;
- agent_fraction_at_goal;
- path_length;
- path_length_ratio;
- smoothness;
- collisions;
- near_collisions;
- obstacle_hits;
- mean_arrival_step;
- runtime.

Add:

- shield intervention rate:
  - fraction agents/steps where `||u_safe - u_pref|| > eps`;
- projection magnitude mean/p95;
- stopped-by-shield rate;
- clipped-speed rate;
- deadlock/stall rate:
  - no progress for K steps;
- minimum clearance p5/p1;
- time-to-first-collision for no/simple shield ablations;
- per-step runtime breakdown: model, graph construction, shield.

## Failure Modes To Track

- local ORCA deadlock in head-on/bottleneck traffic;
- learned policy depends on shield too much: high intervention, low raw safety;
- obstacle-map direct-to-goal trap;
- noisy flow samples create jitter/smoothness regressions;
- transformer global attention overfits to N/order/map coordinates;
- mixed expert sources produce inconsistent supervision;
- chunk predictions optimize imitation but first-step execution with shield drifts from chunk plan;
- shield-aware loss collapses to conservative/stationary outputs if weighted too high.
