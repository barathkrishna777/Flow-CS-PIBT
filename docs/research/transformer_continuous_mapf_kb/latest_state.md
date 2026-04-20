# Latest State

Updated: 2026-04-20 20:33 EDT

## Current Objective

Produce a deeply researched implementation plan for shifting Flow-CS-PIBT from a GNN/grid-centric future direction to transformer-based continuous-space MAPF with ORCA/ORCA-style shielding. No implementation code changes. Only this KB folder may be edited.

## What Has Been Read

- Skill instructions: `firecrawl-search/SKILL.md`; normal Firecrawl CLI persists JSON, so web research will be summarized directly into this KB to honor the write constraint.
- Docs: `docs/FLOW_ADVANCED_PLAN.md`, `README.md`, `docs/implementation_details.md`, `docs/PAPER_PLAN.md`.
- Model/training/data/eval code: `main_pys/generative_model.py`, `main_pys/transformer_model.py`, `main_pys/train_flow.py`, `main_pys/train_continuous.py`, `main_pys/model_inputs.py`, `main_pys/dataset_continuous.py`, `main_pys/dataset_continuous_preprocessed.py`, `main_pys/continuous_env.py`, `scripts/generate_continuous_data.py`, `scripts/generate_and_preprocess_continuous.py`, `scripts/eval_continuous.py`, `scripts/run_continuous_benchmark.py`, `scripts/preprocess_continuous_shards.py`, `tests/test_po_orca.py`.
- External sources gathered and summarized in `research_sources.md`: ORCA/RVO2, CBF/GCBF, MAPF/PIBT/LaCAM/SILLM, continuous-time/large-agent/continuous-space diffusion MAPF, Transformer/Set Transformer/Graphormer/MAT/AgentFormer/VectorNet, Flow Matching/Rectified Flow.

## Strongest Conclusions So Far

- Current continuous stack is implemented and separate from grid simulator: continuous env, ORCA-style/PO-ORCA/EPIBT/PICBF shields, data generation, raw/sharded datasets, continuous training, eval, and benchmark orchestration exist.
- Current learned continuous models still share a local patch + neighbor graph representation. `FlowGNNModel` is the benchmark default; `FlowTransformerModel` exists and is selectable in `train_continuous.py`, but benchmark runner does not expose `--model-type transformer`, and no docs/results show it as first-class.
- `FlowTransformerModel` is a useful prototype/foundation, not final architecture: local masked attention is implemented with dense per-graph `n x n x heads` logits inside a Python loop, so compute/memory still scale like local-masked dense attention per batch graph rather than true sparse attention.
- `FLOW_ADVANCED_PLAN.md` is increasingly obsolete for the new priority because it centers richer grid action specs, diagonal conflict semantics, candidate ranking for PIBT, and discrete projection. The relevant pieces to keep are continuous velocity heads, short-horizon flow supervision, structured action metadata, and explicit safety projection.
- Existing continuous formulation defaults: `dt=0.2`, `max_speed=1.0`, `agent_radius=0.3`, `goal_tolerance=0.25`, local patch radius `k=4`, neighbor count `m=5`, action labels = wait + 8 directions.
- External research strongly supports an explicit shielded-learning design: learned policy proposes nominal preferred velocity/trajectory intent; ORCA/RVO2/CBF-like shield projects to safe controls; global guidance cannot be delegated to ORCA alone.
- Recommended architecture: sparse/local permutation-equivariant agent transformer with edge-feature attention bias; start with preferred velocity output, then short-horizon chunks; add SDF/local obstacle features before map-token global attention.
- Recommended shield strategy: near-term primary `orca` with true-vs-fallback logging and intervention metrics; secondary research `picbf-cs`; `po-orca`/`epibt` as ablations.
- Durable final plan written to `final_plan.md` with 13 requested sections, phased roadmap, test/eval matrix, and first 3 next steps.

## Unresolved Questions

- Whether `FlowTransformerModel` has ever been smoke-tested/evaluated end-to-end; code path exists but benchmark script omits model-type pass-through.
- Whether true `rvo2` is installed in target environments; `orca` silently falls back to heuristic if import fails or rvo2 projection raises.
- Whether obstacle-map continuous performance is primarily limited by data, SDF obstacle shield behavior, missing global planning, or local model features.
- How best to encode maps for transformer policy: local SDF patch only, CNN patch, map tokens, or cross-attention to obstacle features.
- How far to push target N in the first transformer benchmark before sparse attention and shield runtime become bottlenecks.

## Next Actions

1. Await user direction before implementation.
2. If implementation begins, start with observability and transformer first-class wiring, not sparse attention immediately.
