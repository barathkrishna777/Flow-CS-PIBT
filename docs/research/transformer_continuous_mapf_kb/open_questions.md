# Open Questions

Updated: 2026-04-20 19:28 EDT

## Q1. Has `FlowTransformerModel` been trained/evaluated successfully end-to-end?

- Why it matters: determines whether Phase 2 is wiring/stabilization or deeper debugging.
- Evidence needed: quick CPU/GPU smoke training on tiny continuous dataset; one eval rollout; checkpoint load through `scripts/eval_continuous.py`.

## Q2. Is true `rvo2` available in the intended experiment environment?

- Why it matters: shield type `orca` silently falls back to heuristic behavior if `rvo2` import/projection fails. Results labeled ORCA may not be comparable to official ORCA/RVO2.
- Evidence needed: explicit runtime logging/version check; test comparing `orca` with/without `rvo2`; document install requirement.

## Q3. What is the main obstacle-map failure driver?

- Why it matters: roadmap differs if failures come from local learned policy, expert source, SDF obstacle shield, or missing global planner guidance.
- Evidence needed: obstacle-map matrix with ORCA-only, EECBS-continuous replay, PO-ORCA, learned flow, and learned+shield intervention metrics.

## Q4. Which map representation is best for transformer continuous MAPF?

- Why it matters: local CNN patches may be enough for empty/open maps but not obstacle routing; map tokens or SDF gradients may be needed.
- Evidence needed: ablation of local obstacle patch vs SDF patch/gradient vs map-token cross-attention on random/maze/warehouse maps.

## Q5. What agent count should the first transformer target?

- Why it matters: full dense attention is untenable at high N; sparse/local attention should be designed around target N and communication radius.
- Evidence needed: runtime/memory profiling for GNN and transformer at N=32/64/128/256 on existing continuous batches.

## Q6. Should ORCA be expert, shield, baseline, or all three?

- Why it matters: using ORCA as both expert and shield can hide policy weakness and create a reactive imitation ceiling.
- Evidence needed: source-stratified training/eval with EECBS-continuous, ORCA, PO-ORCA/hybrid experts and shield on/off/intervention metrics.
