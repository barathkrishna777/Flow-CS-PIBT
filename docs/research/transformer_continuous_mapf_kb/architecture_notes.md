# Architecture Notes

Updated: 2026-04-20 20:02 EDT

## Recommendation

Build a continuous MAPF policy as a permutation-equivariant local/sparse transformer over agent tokens, with optional map/obstacle tokens later. Keep the policy/shield contract simple:

```text
state + noisy velocity/chunk + t
  -> transformer
  -> nominal preferred velocity or H-step velocity chunk
  -> ORCA/PO-ORCA/PICBF shield
  -> safe executable velocity
```

Do not use a fully global dense transformer as the main architecture for target N beyond small smoke tests. Use radius/kNN local attention with explicit relative geometric features and make global summaries optional.

## Current GNN Baseline

### `FlowGNNModel`

- Inputs:
  - local CNN patch (`data.x`);
  - aux features (`aux_features` or `bd_pred`);
  - noisy velocity `v_t`;
  - interpolation time `t`.
- Interaction:
  - PyG `SAGEConv` over `edge_index`.
- Strengths:
  - proven path in repo;
  - simple, memory-efficient `O(E * hidden_dim)`;
  - permutation equivariant;
  - good baseline for continuous flow vs discrete baseline.
- Weaknesses:
  - ignores `edge_attr` after graph construction;
  - aggregation is relatively blunt for multi-neighbor coordination;
  - no head-specific relation reasoning;
  - no natural map-token or trajectory-token extension.

## Current Transformer Prototype

### `FlowTransformerModel`

- Strengths:
  - same forward interface as `FlowGNNModel`;
  - local masked attention over kNN graph;
  - additive per-head relative position bias from `edge_attr`;
  - bias-free projections;
  - action chunking parameter exists;
  - checkpoint load path in `eval_continuous.py`.
- Prototype limitations:
  - dense per-graph `(N, N, heads)` attention matrix with mask; memory/time quadratic per graph despite local mask;
  - Python graph loop limits GPU utilization;
  - `SinusoidalTimeEmbedding` defined but unused;
  - chunk horizon not supported by current loss/eval target shaping;
  - no explicit current velocity/history inputs except flow noise `v_t`;
  - no map tokens or global summary token;
  - benchmark runner does not expose transformer model arguments.

## Architecture V1: First-Class Local Agent Transformer

### Agent Token Features

Use one token per agent per timestep.

Required features:

- local obstacle/context patch:
  - keep existing 4-channel patch initially: map, goal_dx, goal_dy, agent occupancy;
  - add SDF and SDF gradient channels in Phase 4;
- scalar/vector features:
  - normalized relative goal `(goal - pos) / scale`;
  - distance to goal;
  - at-goal flag;
  - max speed, radius, dt;
  - current velocity and previous velocity if available;
  - priority value if using PO-ORCA/EPIBT-style shields;
  - optional shield status from previous step: projected velocity delta, was_clipped, was_stopped.
- flow-matching conditioning:
  - noisy target velocity/chunk `v_t`;
  - interpolation time `t`;
  - use sinusoidal or MLP time embedding, not raw scalar only.

Implementation inference:

- Existing `create_continuous_data_object` should eventually emit `positions`, `velocities/current_velocity`, `radii`, `priorities`, and SDF patch channels; do this after transformer smoke path is first-class.

### Attention Graph

Use local attention over:

- self edge always;
- `m` nearest neighbors;
- optional communication radius cutoff;
- optional include all agents inside safety horizon:
  - radius approx `2*agent_radius + margin + 2*max_speed*dt + buffer`;
  - align with `default_picbf_communication_radius`.

Target default:

- N range: start 32/64/96/128 to match current Phase A;
- local neighbor count: `m=8` or `m=16` for transformer, larger than current GNN `m=5`;
- radius: expose CLI `--neighbor-radius`; default no cutoff for backward compatibility, but benchmark radius ablations should include finite values.

### Sparse Attention Implementation

Do not keep dense masked attention for serious N.

Preferred implementation options:

1. Edge-list attention:
   - compute Q/K/V per node;
   - compute logits only for directed `edge_index` plus self edges;
   - softmax by source node over outgoing neighbors using PyG/torch scatter;
   - aggregate weighted V over destination neighbors.
2. PyG attention primitive:
   - inspect whether `TransformerConv` or `GATv2Conv` with edge features is sufficient;
   - caution: may reintroduce GNN-like parameterization but gives efficient sparse attention.
3. Block sparse attention:
   - only if grouping by spatial cells or batches becomes necessary.

Scaling:

- GNN: `O(E * hidden_dim)`.
- Current prototype: `O(sum_graphs N_g^2 * heads + E * heads)` memory/time.
- Sparse local transformer: `O(E * heads * head_dim)` plus FFN `O(N * hidden_dim^2)`.
- For `N=128`, dense attention is manageable for smoke but wasteful.
- For `N=256-1000`, use sparse local attention; global attention only via low-cardinality summary tokens or periodic layers.

### Relative Position / Edge Features

Recommended edge feature vector:

```text
edge = [
  delta_pos / radius_or_k,
  distance,
  unit_bearing,
  delta_velocity / max_speed,
  time_to_closest_approach,
  predicted_clearance,
  same_goal_or_crossing_indicator optional,
  line_of_sight_obstacle_indicator optional
]
```

Use edge features in two ways:

- additive per-head attention bias;
- value/message modulation MLP or FiLM.

Position encoding choice:

- Primary: learned edge-feature attention bias, because agent sets live in 2D metric space and are not sequences.
- Add optional radial ALiBi-style monotone distance penalty as regularizer.
- Avoid pure RoPE/absolute position as primary because map coordinates should not break translation generalization.

### Permutation Equivariance

Required:

- no learned agent-id embeddings;
- no order-dependent global attention pooling unless sorted by stable physical key and used only for training diagnostics;
- local attention over edge sets should be equivariant if edge construction is deterministic from positions.

For centralized autoregressive variants:

- only use if explicitly modeling priority order;
- otherwise it risks arbitrary order dependence.

### Variable Agent Counts

- PyG batching already handles variable N through `Batch.ptr`.
- Sparse attention should use `batch`/`ptr` to avoid cross-scenario edges.
- Model outputs one row per agent; no fixed max-agent head.
- Checkpoint configs should store `m`, `neighbor_radius`, patch channels, and edge feature dim.

### Map / Obstacle Representation Options

Phase order:

1. Keep local CNN patch, add SDF channels:
   - lowest code risk;
   - directly supports obstacle clearance gradients.
2. Add obstacle feature tokens:
   - tokens for nearby obstacle AABBs, wall segments, or SDF contour samples;
   - cross-attention from agent tokens to local map tokens.
3. Add global map tokens:
   - useful for maze/warehouse routing;
   - potentially expensive; need pooling/hierarchy.
4. Vectorized obstacle representation:
   - inspired by VectorNet; useful long term for continuous maps.

Recommended near-term:

- Do not build full global map transformer yet.
- Add SDF/local obstacle-gradient channels first; combine with global guidance from EECBS/LaCAM path waypoints if available.

### Flow Matching Conditioning

Current loss:

```text
x0 ~ N(0, I)
x1 = expert velocity / max_speed
t ~ sigmoid(N(0,1))
xt = t*x1 + (1-t)*x0
target = x1 - x0
model(xt, t, data) -> target
```

Keep this first for comparability.

Upgrade:

- use learned/sinusoidal time embedding;
- support velocity chunks:
  - `x1` shape `(N, H, 2)` from future expert velocities;
  - mask padded horizon after episode end;
  - output `(N, H*2)`;
  - action auxiliary head optional per horizon step.
- condition on current executed velocity and previous shielded velocity to improve smoothness/control continuity.

### Action / Output Representation

Recommended V1 output:

- preferred velocity `v_pref` for next step, normalized by max speed.

Recommended V2 output:

- short-horizon velocity chunk `H=3` or `H=5`;
- execution uses first element now, optionally receding-horizon averaging;
- training includes smoothness and goal progress over chunk.

Other outputs:

- acceleration:
  - better if dynamics become second-order; not now.
- waypoint:
  - useful for global planning; can be auxiliary head.
- distribution:
  - flow matching already models stochasticity via noise; uncertainty head can be later.

### Centralized vs Decentralized

Near-term:

- centralized training and evaluation implementation;
- decentralized-compatible information: local attention radius/kNN, local map patch, local goal, local neighbors.

Reason:

- repo currently constructs all-agent graphs centrally;
- shield is centralized in env but local/radius-compatible;
- decentralized execution claims require strict observation and communication constraints and should wait.

Long-term:

- CTDE framing: train with extra global/expert features; evaluate with local-only features.
- Keep a config flag separating train-only features from deployable features.

## Recommended Transformer Configs

Smoke:

- hidden `128`;
- layers `2`;
- heads `4`;
- m `5`;
- N `16/32`;
- no chunking.

First serious continuous transformer:

- hidden `256`;
- layers `4`;
- heads `8`;
- m `8` or `16`;
- patch radius `k=4` or `6`;
- dropout `0.1`;
- sparse attention;
- action_loss_weight `0.1`;
- shield-aware loss off for initial comparability, then on.

Scale-up:

- hidden `384/512`;
- layers `6`;
- m `16`;
- optional global summary/inducing tokens;
- chunk horizon `3`.

## Comparison Recommendation

Do not replace the GNN baseline until the transformer path is first-class and profiled. Run:

- GNN flow current baseline;
- current dense prototype transformer smoke;
- sparse local transformer;
- ORCA-only;
- discrete continuous-action baseline.

Success criterion for transformer direction:

- matches or beats GNN flow at N=32/64/96/128 on `empty-48-48`;
- reduces deadlock/stall or improves fraction-at-goal on obstacle maps after SDF/global guidance;
- comparable runtime to GNN within 1.5-2x at N<=128, with better scaling after sparse attention.
