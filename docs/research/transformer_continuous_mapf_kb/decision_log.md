# Decision Log

## 2026-04-20 19:13 EDT

- Decision: Create a persistent KB under `docs/research/transformer_continuous_mapf_kb/` before repo/source inspection.
- Rationale: User explicitly requested continuous KB writes for context recovery and restricted all writes to this folder.
- Alternatives considered: keep notes in context only; rejected because compaction would lose reasoning.
- Confidence: high.
- Unresolved risks: none for scaffold; source gathering must avoid local artifacts outside this folder.

## 2026-04-20 19:28 EDT

- Decision: Treat the existing continuous stack as the technical base for the new roadmap, not as a future stub to be created.
- Rationale: `continuous_env.py`, continuous data generation, datasets, training, evaluation, benchmark orchestration, and shield modes are all implemented.
- Alternatives considered: restart continuous MAPF from scratch; rejected because existing env/data/eval machinery already covers the minimum loop.
- Confidence: high.
- Unresolved risks: obstacle-map reliability and ORCA fallback behavior need more evaluation.

## 2026-04-20 19:28 EDT

- Decision: Treat `FlowTransformerModel` as a prototype foundation requiring architectural upgrades before large experiments.
- Rationale: It has the right forward interface and local attention concept, but uses dense per-graph attention matrices, lacks benchmark runner support, lacks chunk loss support, and has no documented results.
- Alternatives considered: use as-is for the main direction; keep GNN only; build transformer from scratch. Best path is incremental upgrade while preserving the existing interface.
- Confidence: high.
- Unresolved risks: actual transformer training stability has not been verified in this run.

## 2026-04-20 19:45 EDT

- Decision: Make the near-term policy/shield contract "transformer predicts nominal preferred velocity or short-horizon velocity chunk; safety layer projects to executable safe velocity."
- Rationale: ORCA/RVO2 docs define preferred velocity as the caller-supplied global/local intent and compute actual velocities subject to collision avoidance; learned MAPF literature emphasizes smart shields; CBF literature uses nominal-control projection.
- Alternatives considered: train transformer to output already-safe velocities with no shield; use ORCA as expert only. Rejected for near-term because safety should remain explicit and measurable.
- Confidence: high.
- Unresolved risks: projection can erase learned intent in dense deadlocks; need shield intervention metrics and shield-aware training.

## 2026-04-20 20:02 EDT

- Decision: Prioritize sparse/local attention over dense global attention for the main transformer architecture.
- Rationale: Agent sets are variable-size and current target counts are already 32-128; local safety/communication structure is spatial; Set Transformer/Graphormer sources support attention with explicit structure; current dense-masked prototype wastes memory.
- Alternatives considered: full global transformer; pure GNN; inducing-only global transformer. Full global is acceptable for smoke but not scaling; pure GNN remains baseline; inducing tokens may be added after local path is stable.
- Confidence: high.
- Unresolved risks: local attention may miss global bottleneck coordination without path/waypoint features.

## 2026-04-20 20:02 EDT

- Decision: Treat obstacle-map work as a separate milestone requiring SDF/global-guidance features and source-stratified expert data.
- Rationale: Existing docs already say obstacle-map claims are not ready; RVO2 docs note global navigation/roadmaps are caller responsibility; ORCA-only direct-to-goal rollouts are not enough for obstacle topology.
- Alternatives considered: include obstacle maps immediately in main transformer benchmark; rejected as likely confounded.
- Confidence: medium-high.
- Unresolved risks: transformer may still fail obstacle maps even with local SDF unless global path guidance is strong.
