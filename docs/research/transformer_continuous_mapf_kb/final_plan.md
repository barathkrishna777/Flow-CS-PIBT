# Final Plan: Transformer-Based Continuous MAPF With ORCA/CBF Shielding

Updated: 2026-04-20 20:28 EDT

## 1. Executive Summary

Flow-CS-PIBT should pivot its next major technical direction from "richer grid actions around a GNN policy" to "continuous-space MAPF with a transformer nominal policy and explicit collision shielding."

The repo already has a real continuous stack:

- continuous environment/dynamics;
- ORCA-style, PO-ORCA, EPIBT, simple, none, and PICBF-CS shield paths;
- continuous data generation from EECBS/LaCAM3/ORCA/hybrid experts;
- raw and compact sharded datasets;
- continuous flow/discrete training;
- continuous eval and benchmark orchestration.

The current transformer is a useful prototype, not yet the architecture to bet the project on. `FlowTransformerModel` has the right interface, local masked attention, relative position bias, and checkpoint compatibility, but it still uses dense per-graph attention matrices, lacks benchmark-runner support, has no documented results, and exposes chunking without loss/eval support.

Recommended direction:

- Keep `FlowGNNModel` as the strong baseline.
- Make transformer continuous training/eval first-class.
- Replace dense masked attention with true sparse/local attention over kNN/radius edges.
- Treat policy output as nominal preferred velocity or short-horizon velocity chunk.
- Use ORCA/ORCA-style shielding as the near-term safety projector and baseline.
- Keep PICBF/CBF-compatible nominal-control interface as the longer-term safety strategy.
- Add metrics for shield intervention, stall/deadlock, near-collision, obstacle hit, runtime breakdown, and smoothness.

## 2. What The Existing Plan Says

`docs/FLOW_ADVANCED_PLAN.md` is primarily a grid-action expansion plan:

- replace hardcoded five actions with structured action specs;
- add `grid8`, diagonal legality, diagonal crossing conflicts;
- add candidate ranking for PIBT-style execution;
- train continuous velocity targets and project them back to legal grid actions;
- add short-horizon flow supervision;
- add topological/congestion-aware features and macro-actions.

The plan's durable ideas:

- keep execution legality explicit;
- model can predict richer intent than the executor accepts;
- safety layer/resolver remains responsible for feasibility;
- short-horizon supervision can reduce next-action ambiguity;
- richer features for bottlenecks/congestion are important.

The plan's now-obsolete emphasis:

- grid4/grid8/radius2 should not be the main research path if continuous MAPF is the priority;
- diagonal grid conflict semantics should be demoted;
- projection from continuous vector to grid action should become a compatibility path, not the main architecture;
- PIBT grid resolver should be historical baseline/analogy, while continuous shields become primary.

## 3. What Should Change

Primary shift:

- From: GNN local patch policy with grid action preferences and CS-PIBT shielding.
- To: transformer local/sparse agent-token policy predicting continuous nominal control, shielded by ORCA/PICBF-style safety layers.

Concrete changes in direction:

- Make continuous-space benchmark the main technical proving ground.
- Make transformer a first-class `model_type` in data/training/eval/benchmark scripts.
- Replace dense local masked attention with edge-list sparse attention.
- Add SDF/velocity/history/shield-intervention features.
- Elevate expert-source consistency:
  - ORCA/PO-ORCA for open-space/reactive behavior;
  - EECBS/LaCAM-derived guidance for obstacle maps;
  - source-stratified reporting.
- Report safety and shield dependence, not only success/fraction-at-goal.
- Defer removing grid-world assumptions until continuous transformer results justify it.

## 4. Research Findings With Sources

### ORCA/RVO2

Sources:

- ORCA project: https://gamma-web.iacs.umd.edu/ORCA/
- RVO2 docs: https://gamma-web.iacs.umd.edu/RVO2/documentation/2.0/
- RVO2 using guide: https://gamma-web.iacs.umd.edu/RVO2/documentation/2.0/using.html
- RVO2 GitHub: https://github.com/snape/RVO2

Direct claims:

- ORCA computes collision-avoiding velocities from preferred velocities.
- Responsibility is reciprocal between pairs.
- RVO2 exposes an API where the caller supplies preferred velocities, agents, and obstacles.
- RVO2 docs explicitly state global navigation/roadmap guidance is outside the library.

Implications:

- Transformer should produce preferred velocities/intent, not replace the shield.
- ORCA is a strong baseline and near-term shield.
- ORCA alone is not enough for obstacle-map/global-routing claims.
- Repo results must distinguish true `rvo2` from heuristic fallback.

### CBF/PICBF

Sources:

- Safety barrier certificates: https://authors.library.caltech.edu/records/tshzw-g4v69
- Neural graph CBFs: https://proceedings.mlr.press/v229/zhang23h.html

Direct claims:

- CBF-style shields project nominal controls into safe controls using optimization constraints.
- Distributed/local CBF variants can use nearby agents and graph structure.

Implications:

- Keep learned policy output as nominal control so ORCA can later be swapped for PICBF/CBF.
- `picbf-cs` is the right secondary safety research track.

### MAPF / Learned Local Policies

Sources:

- MAPF definitions/benchmarks: https://cris.biu.ac.il/en/publications/multi-agent-pathfinding-definitions-variants-and-benchmarks/
- MovingAI MAPF benchmarks: https://www.movingai.com/benchmarks/mapf.html
- Work Smarter, Not Harder / SSIL + CS-PIBT: https://arthurjakobsson.github.io/ssil_mapf/
- PIBT: https://www.ijcai.org/Proceedings/2019/76
- LaCAM*: https://www.ijcai.org/proceedings/2023/28
- SILLM/lifelong MAPF: https://dblp.org/rec/conf/icra/JiangWVDSL25

Direct claims:

- MAPF requires concurrent collision-free paths, with assumptions/objectives needing precise definition.
- SSIL + CS-PIBT shows smart one-step collision shielding can dominate learned policy quality alone.
- PIBT/LaCAM show scalable priority/lazy-search machinery.
- Lifelong MAPF learning systems combine policy, communication, collision resolution, and global guidance.

Implications:

- Continuous MAPF section must define dynamics, geometry, and collision semantics explicitly.
- Shielded-learning is not a crutch; it is a core design.
- Transformer should focus on coordination, intent, and deadlock avoidance beyond one-step safety.
- LaCAM/EECBS-derived path-prefix features are likely important for obstacle maps.

### Continuous MAPF / Large Agents

Sources:

- Continuous-time MAPF: https://www.sciencedirect.com/science/article/pii/S0004370222000029
- Large-agent MAPF: https://publications.ri.cmu.edu/multi-agent-path-finding-for-large-agents
- Continuous MAPF with projected diffusion: https://womapf.github.io/aaai-25/pdf/Submission_38.pdf

Direct claims:

- Continuous-time/continuous-space variants expose issues hidden by discrete unit-time grids.
- Large/geometric agents need explicit collision geometry.
- Generative continuous MAPF methods need projection/constrained optimization to handle feasibility.

Implications:

- `agent_radius`, `dt`, collision between timesteps, and obstacle clearance are central.
- Flow matching should be coupled to shielding/projection and not treated as inherently safe.
- Longer-horizon trajectory/chunk generation is promising but should come after one-step transformer stability.

### Transformer / Agent Attention

Sources:

- Attention Is All You Need: https://huggingface.co/papers/1706.03762
- Set Transformer: https://proceedings.mlr.press/v97/lee19d
- Relative position attention: https://aclanthology.org/N18-2074/
- Graphormer: https://www.microsoft.com/en-us/research/publication/do-transformers-really-perform-badly-for-graph-representation/
- Multi-Agent Transformer: https://papers.nips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html
- AgentFormer: https://www.ri.cmu.edu/publications/agentformer-agent-aware-transformers-for-socio-temporal-multi-agent-forecasting/
- VectorNet: https://waymo.com/research/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation/

Direct claims:

- Self-attention models interactions but naive form is quadratic.
- Set Transformer addresses permutation-invariant/equivariant set processing and inducing-point scaling.
- Relative position/graph structural encodings are important for graph-like data.
- Multi-agent transformer/forecasting work supports attention over agents and time.

Implications:

- Agent tokens should be permutation equivariant.
- Use local/sparse attention with edge geometry, not full dense attention as the main path.
- Add relative position/velocity/clearance features as attention biases or edge modulation.
- Map representation can start raster-local but should eventually support vector/SDF/map-token features.

### Flow Matching / Rectified Flow

Sources:

- Flow Matching: https://openreview.net/forum?id=PqvMRDCJT9t
- Rectified Flow: https://huggingface.co/papers/2209.03003

Direct claims:

- Flow matching trains vector fields by simulation-free regression along probability paths.
- Rectified flow learns straight flows that can sometimes be simulated with few Euler steps.

Implications:

- Current flow loss is conceptually sound.
- Low integration-step inference is plausible, but must be benchmarked with shielded control.
- Velocity chunks are a natural next target.

## 5. Current Codebase Inventory

Most relevant files:

- `docs/FLOW_ADVANCED_PLAN.md`: grid-action future plan; keep only continuous velocity/short-horizon/safety projection ideas.
- `README.md`: documents continuous stack and PICBF-CS usage.
- `docs/implementation_details.md`: current continuous benchmark story; says ORCA strongest, flow above discrete baseline, obstacle maps not ready.
- `docs/PAPER_PLAN.md`: historical plan; continuous phase now mostly implemented.
- `main_pys/generative_model.py`: `FlowGNNModel`, current benchmark policy; local CNN + SAGEConv + velocity/action heads.
- `main_pys/transformer_model.py`: `FlowTransformerModel`, prototype local masked transformer; interface-compatible but dense attention and under-wired.
- `main_pys/model_inputs.py`: continuous PyG sample creation; local 4-channel patches, kNN graph, action labels, normalization.
- `main_pys/dataset_continuous.py`: raw rollout dataset; per-timestep graph samples.
- `main_pys/dataset_continuous_preprocessed.py`: compact sharded continuous dataset.
- `scripts/preprocess_continuous_shards.py`: raw-to-shard preprocessing.
- `main_pys/continuous_env.py`: dynamics, metrics, shields.
- `main_pys/train_continuous.py`: continuous flow/discrete training; `--model-type gnn|transformer` exists.
- `scripts/eval_continuous.py`: ORCA/flow/discrete eval; can load transformer checkpoint.
- `scripts/run_continuous_benchmark.py`: packaged benchmark; lacks transformer args.
- `scripts/generate_continuous_data.py`: EECBS/LaCAM/ORCA/hybrid expert data.
- `scripts/generate_and_preprocess_continuous.py`: dataset root, manifest, SDF precompute.
- `tests/test_po_orca.py`: shield/SDF smoke tests.

Key conclusion:

- The repo is ready for an incremental transformer-continuous roadmap.
- It is not ready for immediate large transformer experiments until benchmark wiring, sparse attention, metrics, and smoke tests are added.

## 6. Recommended Architecture

### Model Contract

Inputs:

- agent state tokens;
- local map/SDF/goal/occupancy patch;
- current velocity/history;
- neighbor relative geometry;
- noisy velocity/chunk `x_t`;
- flow time `t`.

Outputs:

- primary: preferred velocity `u_pref`;
- optional V2: short-horizon velocity chunk `(u_1, ..., u_H)`;
- auxiliary: wait + K-direction logits; waypoint/progress/confidence heads later.

Execution:

- transformer predicts nominal command;
- shield projects command;
- environment integrates shielded velocity.

### Tokenization

Agent token:

- CNN patch embedding;
- scalar features: rel goal, distance, at-goal, max speed, radius, dt;
- velocity features: current velocity, previous safe velocity, previous preferred velocity;
- optional priority/stall features.

Map representation:

- Phase 1: current local raster patch.
- Phase 2: add SDF value + SDF gradient local channels.
- Phase 3: optional obstacle/map tokens with cross-attention.
- Phase 4: vectorized obstacle/waypoint features for obstacle-heavy maps.

### Attention

Use local sparse attention over:

- self;
- kNN neighbors;
- communication-radius neighbors.

Edge features:

- relative position;
- distance and bearing;
- relative velocity;
- predicted closest approach;
- obstacle line-of-sight/clearance;
- priority relation if used.

Encoding:

- learned edge-feature attention bias as primary;
- optional radial distance penalty;
- avoid absolute coordinate embeddings as primary.

Permutation:

- no learned agent IDs;
- no arbitrary order-dependent decoding;
- batch by PyG graph boundaries.

### Scaling Targets

Near-term:

- N = 32/64/96/128, current Phase A range.

Medium:

- N = 160/256 with local sparse attention.

Longer term:

- N >= 512 only after graph construction, sparse attention, and shield runtime are profiled.

Compute expectation:

- GNN baseline: `O(E * d)`.
- Current transformer: `O(N^2 * heads)` per graph because dense masked matrix.
- Target transformer: `O(E * heads * head_dim) + O(N * FFN)`.

## 7. Continuous-Space Formulation

### State

Per agent:

- position `p_i`;
- current velocity `v_i`;
- goal `g_i`;
- radius/max speed;
- local obstacles/SDF;
- local occupancy/neighbors;
- optional priority/stall/shield-history state.

### Action

Primary:

- preferred velocity.

Alternatives:

- acceleration: defer until second-order dynamics;
- waypoint: auxiliary or obstacle-map head;
- short-horizon chunk: V2 after one-step path is stable;
- distribution: later uncertainty head or flow sampling analysis.

### Dynamics

Current:

- `p_{t+1} = p_t + shield(u_pref) * dt`;
- `dt=0.2`;
- `max_speed=1.0`;
- `agent_radius=0.3`;
- `goal_tolerance=0.25`.

Keep defaults initially for comparability; later ablate radius/dt/speed.

### Expert Data

Use three explicit expert regimes:

- ORCA/PO-ORCA:
  - open-space and reactive collision behavior;
  - baseline and fallback expert.
- EECBS/LaCAM continuous conversion:
  - global route/path-prefix supervision for obstacle maps;
  - piecewise-linear and less smooth, so use auxiliary waypoint/path-prefix targets.
- Hybrid:
  - useful but report source distribution and source-stratified metrics.

### Losses

Start:

- existing flow matching loss;
- auxiliary action CE;
- same normalization and weights for GNN comparability.

Then add:

- shield intervention logging;
- shield-aware loss as ablation;
- smoothness loss for chunks;
- goal progress auxiliary loss;
- near-collision surrogate loss;
- source-aware weighting if hybrid experts conflict.

## 8. Collision Shield Strategy

### Shield Mode Assessment

`none`:

- provides speed clipping only;
- use for raw-policy ablation;
- expect collisions.

`simple`:

- naive stop-on-conflict filter;
- useful sanity baseline;
- likely freezes/deadlocks.

`heuristic-orca`:

- pairwise heuristic + SDF obstacle constraints;
- useful deterministic fallback/training ablation;
- not official ORCA guarantee.

`orca`:

- uses true `rvo2` for agent-agent if available, with SDF obstacle preconstraint;
- silently falls back today;
- primary near-term shield/baseline, but must log implementation path.

`po-orca`:

- priority-ordered heuristic ORCA;
- useful for deadlock/priority research;
- not primary until more validation.

`epibt`:

- continuous candidate priority heuristic;
- implementation appears more greedy/wait-fallback than true recursive PIBT;
- ablation only for now.

`picbf-cs`:

- external CBF/QP-style projection adapter;
- promising long-term shield;
- dependency and runtime risk.

### Primary Strategy

Near-term:

- use `orca` as main eval shield and ORCA-only baseline;
- add explicit true-vs-fallback logging;
- include `heuristic-orca` ablation.

Secondary:

- use `picbf-cs` for research-grade safety projection once environment and dependency are stable.

Do not:

- train/evaluate only with shielded success and ignore raw policy/shield intervention.

## 9. Phased Implementation Plan

### Phase 1. Audit And Stabilize Current Continuous Pipeline

Goal:

- make the existing continuous stack reproducible and measurable before architecture changes.

Likely files:

- `main_pys/continuous_env.py`;
- `scripts/eval_continuous.py`;
- `scripts/run_continuous_benchmark.py`;
- `scripts/generate_continuous_data.py`;
- `main_pys/continuous_scenarios.py`;
- `tests/test_po_orca.py`.

New abstractions/functions:

- shield debug info for true ORCA vs fallback;
- shield intervention metrics;
- stall/deadlock metric;
- runtime breakdown hooks.

Backward compatibility:

- existing CSV columns retained; new columns appended.
- existing shield names preserved.

Tests/smoke:

- run `tests/test_po_orca.py`;
- one ORCA eval on `empty-48-48`, N=16/32;
- verify numeric scenario selection picks 1..5.

Acceptance:

- reproducible ORCA-only and GNN-flow eval command;
- CSV records shield implementation and intervention metrics;
- no obstacle-free regression.

Failure modes:

- `rvo2` unavailable;
- metrics slow down eval;
- old CSV summarizer breaks if fields change.

Difficulty/risk:

- low/medium.

### Phase 2. Make Transformer Path First-Class In Continuous Training/Eval

Goal:

- train/eval current `FlowTransformerModel` end-to-end as a baseline prototype.

Likely files:

- `main_pys/train_continuous.py`;
- `scripts/eval_continuous.py`;
- `scripts/run_continuous_benchmark.py`;
- `README.md` later, not in this research run;
- tests to add later.

New abstractions/functions:

- benchmark args: `--model-type`, `--num-heads`, `--chunk-horizon`;
- checkpoint/model config validation;
- transformer smoke script/test.

Backward compatibility:

- default remains `gnn`;
- old checkpoints load unchanged.

Tests/smoke:

- tiny batch forward for GNN and transformer;
- one epoch/one batch smoke on tiny continuous dataset;
- load checkpoint in eval and run N=8/16 rollout.

Acceptance:

- current transformer can be benchmarked against GNN on same data.

Failure modes:

- dense attention memory blowup;
- shape mismatch if chunk horizon >1;
- eval silently loads strict=False despite mismatch.

Difficulty/risk:

- medium.

### Phase 3. Replace Dense Masked Attention With Scalable Sparse/Local Attention

Goal:

- make transformer architecture appropriate for N=128+ and future scale.

Likely files:

- `main_pys/transformer_model.py`;
- `main_pys/model_inputs.py`;
- optional new attention module file later.

New abstractions/classes:

- `SparseLocalAttention`;
- `EdgeFeatureBias`;
- edge softmax by source node;
- self-edge construction helper.

Backward compatibility:

- preserve `FlowTransformerModel.forward(v_t, t, data, return_action_logits=False)`;
- allow loading old prototype checkpoint only with explicit compatibility flag or fail clearly.

Tests/smoke:

- permutation equivariance test;
- dense-vs-sparse equivalence on tiny fully connected graph;
- memory/runtime profile at N=32/64/128;
- no cross-graph attention in PyG batches.

Acceptance:

- memory scales with edge count;
- transformer forward at N=128 comparable enough to GNN for benchmark;
- permutation test passes.

Failure modes:

- incorrect softmax grouping;
- missing self edges;
- asymmetric edge features mishandled.

Difficulty/risk:

- medium/high.

### Phase 4. Improve Continuous Data Generation And Expert/Shield Consistency

Goal:

- make expert source reliable for both open and obstacle maps.

Likely files:

- `scripts/generate_continuous_data.py`;
- `scripts/generate_and_preprocess_continuous.py`;
- `scripts/preprocess_continuous_shards.py`;
- `main_pys/dataset_continuous.py`;
- `main_pys/model_inputs.py`.

New abstractions/functions:

- expert source policy config;
- source-stratified manifest;
- EECBS/LaCAM path-prefix/waypoint target extraction;
- optional continuous replay validator with robust clearance tolerance.

Backward compatibility:

- existing `.npz` keys still load;
- new keys optional.

Tests/smoke:

- generate 1 scenario with ORCA, EECBS/LaCAM if available;
- verify action labels and waypoint targets;
- verify source counts in manifest.

Acceptance:

- benchmark can train/eval by expert source;
- obstacle-map data has global guidance targets.

Failure modes:

- external solvers unavailable;
- EECBS-to-continuous trajectories clip walls for large radius;
- hybrid fallback hides poor expert quality.

Difficulty/risk:

- medium.

### Phase 5. Add Richer Losses And Shield-Aware Training

Goal:

- teach transformer policy to produce commands that remain useful after shielding.

Likely files:

- `main_pys/train_continuous.py`;
- `main_pys/dataset_continuous.py`;
- `main_pys/dataset_continuous_preprocessed.py`;
- `scripts/eval_continuous.py`.

New abstractions/functions:

- chunk target builder;
- loss component logging;
- shield-aware loss wrapper with cached shield;
- progress/smoothness/collision surrogate losses.

Backward compatibility:

- existing one-step loss remains default;
- chunk horizon >1 opt-in.

Tests/smoke:

- loss shape tests for H=1 and H=3;
- shield-aware loss does not crash on batch;
- gradients finite.

Acceptance:

- H=1 reproduces baseline behavior;
- H=3 train/eval works;
- shield-aware ablation reports intervention delta.

Failure modes:

- shield-aware loss too slow;
- conservative collapse to wait;
- non-differentiable projection creates noisy optimization.

Difficulty/risk:

- high.

### Phase 6. Run Benchmark Matrix

Goal:

- decide whether transformer+continuous is worth major investment.

Likely files:

- `scripts/run_continuous_benchmark.py`;
- `analysis_scripts` or new summary scripts later.

Baselines:

- ORCA-only;
- current GNN flow;
- current transformer prototype;
- sparse local transformer;
- discrete continuous-action baseline;
- optional no/simple shield ablations.

Acceptance:

- transformer matches/beats GNN on open map fraction-at-goal or improves deadlock/stall metrics;
- transformer gives clear advantage on obstacle maps after guidance/SDF features;
- runtime within acceptable envelope.

Difficulty/risk:

- medium/high due training cost.

### Phase 7. Optional Grid-World Demotion

Goal:

- simplify future docs/code focus only after continuous transformer evidence.

Likely files:

- docs and CLI defaults later.

Do only if:

- transformer continuous path is demonstrably valuable.

Difficulty/risk:

- low code risk, high project-positioning risk.

## 10. Test And Evaluation Plan

### Minimum Smoke Tests Before Expensive Runs

- import and instantiate GNN/transformer;
- forward pass on synthetic PyG batch with variable N;
- one tiny training batch for `flow` and `discrete`;
- one eval rollout for ORCA-only and learned policy;
- shield mode smoke: `none`, `simple`, `heuristic-orca`, `orca`, `po-orca`; `picbf-cs` skip if dependency absent;
- scenario selection numeric check;
- permutation equivariance test for transformer.

### Benchmark Matrix

Maps:

- open: `empty-48-48`;
- random obstacles: `random-32-32-10`, `random-64-64-10`;
- maze: `maze-128-128-2`;
- warehouse: warehouse map from existing benchmark set;
- dense/den maps: `den312d`, `den520d` later.

Agent counts:

- smoke: 8/16;
- phase A: 32/64/96/128;
- scale: 160/256;
- long-term: 512+ only after profiling.

Policies:

- ORCA-only;
- GNN flow;
- GNN discrete;
- transformer dense prototype;
- transformer sparse local;
- transformer + SDF;
- transformer + chunk;
- transformer + shield-aware loss.

Shields:

- `orca`;
- `heuristic-orca`;
- `po-orca`;
- `simple`;
- `none`;
- `picbf-cs` where available.

Metrics:

- success;
- agent_fraction_at_goal;
- collisions;
- near_collisions;
- obstacle_hits;
- path_length_ratio;
- makespan/mean_arrival_step;
- smoothness;
- runtime;
- model runtime;
- graph construction runtime;
- shield runtime;
- shield intervention rate;
- projection magnitude mean/p95;
- stall/deadlock rate;
- min clearance distribution.

Ablations:

- attention radius / neighbor count `m`;
- heads/layers/hidden dim;
- local patch radius `k`;
- SDF channels on/off;
- expert source;
- shield type;
- flow integration steps;
- consensus samples and aggregation;
- flow loss vs action loss vs shield-aware loss;
- chunk horizon.

Worth-pursuing threshold:

- On `empty-48-48`, sparse transformer matches or beats GNN flow at N=32/64/96/128 with acceptable runtime.
- On at least one obstacle map, transformer+SDF/guidance reduces stalls or improves fraction-at-goal vs GNN at same shield.
- Transformer has lower shield intervention rate or better post-shield progress than GNN.
- Runtime does not exceed GNN by more than about 2x at N<=128 after sparse attention.

## 11. Risks And Mitigations

Risk: transformer does not beat GNN.

- Mitigation: publish/use as ablation; keep GNN baseline; focus transformer on obstacle/global guidance where attention should help.

Risk: ORCA solves open maps better than learned policy.

- Mitigation: position learned policy as planner/intent component for harder settings, not replacement for ORCA safety; compare against ORCA-only honestly.

Risk: shield hides policy failures.

- Mitigation: add shield intervention, raw no-shield ablations, projection magnitude, and stall metrics.

Risk: obstacle maps remain poor.

- Mitigation: add LaCAM/EECBS path-prefix guidance, SDF gradients, map tokens; do not claim obstacle results prematurely.

Risk: dense attention blows up.

- Mitigation: sparse attention Phase 3 before large runs.

Risk: CBF dependency/runtimes are unstable.

- Mitigation: keep ORCA near-term; use PICBF-CS only as secondary research path.

Risk: hybrid expert data is inconsistent.

- Mitigation: source-stratified manifests, training filters, and per-source metrics.

Risk: chunk outputs add complexity without benefit.

- Mitigation: make chunking Phase 5 opt-in; H=1 remains baseline.

## 12. Open Questions

- Has current `FlowTransformerModel` ever trained/evaluated end-to-end?
- Is true `rvo2` installed in the target environment, and how often does fallback occur?
- What is the dominant obstacle-map failure: expert, shield, missing guidance, or local model?
- Which map representation gives best benefit/cost: SDF patch, map tokens, vector obstacles, or path-prefix fields?
- Should transformer execution claim decentralized behavior or remain centralized/local-compatible?
- What N range is the real target for the next paper/result: 128, 256, or 1000+?
- Can PICBF-CS run fast enough for benchmark-scale eval?

## 13. First 3 Concrete Next Steps

1. Add continuous benchmark observability before changing models:
   - true ORCA vs fallback logging;
   - shield intervention metrics;
   - deadlock/stall metrics;
   - numeric scenario selection smoke.

2. Make current transformer path first-class and smoke-test it:
   - expose `--model-type transformer`, `--num-heads`, `--chunk-horizon` in benchmark runner;
   - run one tiny train/eval smoke;
   - compare current dense transformer vs GNN at N=16/32.

3. Implement sparse local attention as the real transformer architecture:
   - edge-list attention with relative edge-feature bias;
   - permutation equivariance tests;
   - runtime/memory profiling at N=32/64/128 before any expensive training.
