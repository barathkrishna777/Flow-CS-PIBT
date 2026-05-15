# Continuous-Space MAPF Research Handoff

**Date:** 2026-05-03  
**Branch:** `barath/learned-wait-logit` (current work; new continuous research should start a fresh branch)  
**Repo (Mac):** `/Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT`  
**Repo (Lambda):** `~/barath/Flow-CS-PIBT`  
**Lambda setup:** `cd ~/barath/Flow-CS-PIBT && conda activate 3dposehsx_env`

---

## Why We Are Here: The Discrete Interface Bottleneck

The project is Flow-CS-PIBT / FLOMAP: a flow-matching model that predicts continuous 2D velocity for each agent, which is then converted to a discrete action preference ranking (wait, right, down, up, left) for the CS-PIBT collision shield.

After extensive work, we hit a hard ceiling. The root cause is the **discrete interface bottleneck**:

- The model outputs a continuous velocity vector
- This is dot-producted against 4 cardinal direction vectors to produce 4 movement scores
- A learned wait head produces a 5th logit
- These 5 logits are ranked (stochastically) and fed to CS-PIBT as action preferences

This interface loses information. Specifically, it cannot encode asymmetric yielding in corridors: when two agents face each other, both get "go forward" as their top velocity. The discrete ranking inherits this ambiguity. SSIL (a discrete classifier trained end-to-end) resolves it because it learns sharp wait/move decisions directly.

**The new hypothesis:** eliminating the discrete interface entirely — staying in continuous (x, y) space throughout — removes the information bottleneck and allows the collision shield to operate on richer velocity geometry.

---

## Key Baselines (Discrete Grid, Primary8 / Rishi12)

These are the targets. Any continuous-space system needs to be compared against these.

| Method | Primary8 Overall | Primary8 Stress | Rishi12 Overall |
|---|---|---|---|
| BD-PIBT (heuristic, no model) | 66.43% (1229/1850) | — | 59.48% (1606/2700) |
| SSIL (discrete classifier, ICRA baseline) | 66.43% (1229/1850) | 51.71% (181/350) | 55.22% (1491/2700) |
| FLOMAP best (learned wait, b=-5) | 66.05% (1222/1850) | 46.00% (161/350) | ~53% |

**Stress maps** (hardest coordination cases): `random-32-32-10`, `maze-128-128-2`

FLOMAP ties SSIL everywhere except stress maps, where SSIL wins by ~6pp. All inference-only attempts to close this gap failed (see section below).

---

## What Was Tried and Failed (This Session)

All of these were inference-only (no retraining). All tested on the current best checkpoint: `large_scale_flow_learned_wait_head_ddp_best.pt`.

| Intervention | Medium Stress | 540-case Overall | 540-case Stress | Verdict |
|---|---|---|---|---|
| Sorted (deterministic) ranking | 31.2% | — | — | Reject: -9pp vs baseline |
| BD-Flow hybrid (t=0.50) | 43.8% | — | — | Neutral |
| Aggressive deadlock (w=10, b=15) | 31.2% | — | — | Reject |
| Near-goal priority boost (t12, b5) | 56.2% | 52.8% | 28.0% | **Reject: destroys warehouse/room** |
| Near-goal priority boost (t12, b10) | 50.0% | 51.9% | 24.0% | Reject |

**Critical finding from paired failure analysis** (full FLOMAP vs full SSIL, row-by-row):
- FLOMAP's stress failures are **deadlock/livelock**, not direction errors
- On maze-128-128-2 @ 300 agents: FLOMAP gets 298/300 home but hits 120s timeout; SSIL gets 300/300 in 17s
- FLOMAP is 4-7x slower than SSIL even when both succeed on stress maps
- On non-stress maps: FLOMAP and SSIL are **exactly tied** (83.36% each on the non-stress, non-warehouse slice)
- FLOMAP **wins** on den312d (+9 net cases) — complex city topology favors continuous guidance

The medium eval used rishi8 maps (no warehouse). Near-goal boost looked promising on medium (56.2% stress) but catastrophically failed on rishi12 540-case (28% stress, 0% warehouse). Never use rishi8-only medium results to gate warehouse-sensitive interventions.

**Conclusion:** The inference-only space is saturated. The discrete interface is the structural problem.

---

## Existing Continuous-Space Infrastructure

There is **already a significant continuous-space codebase** in this repo. You are not starting from zero.

### Key files

| File | Purpose |
|---|---|
| `main_pys/continuous_env.py` | Core continuous environment + all collision shields |
| `main_pys/train_continuous.py` | Training loop for continuous-space flow model |
| `main_pys/dataset_continuous.py` | Dataset for continuous trajectories |
| `main_pys/dataset_continuous_preprocessed.py` | Preprocessed shard dataset |
| `eval_continuous.py` | Evaluation script for continuous runs |
| `generate_continuous_data.py` | Data generation script |
| `run_continuous_benchmark.py` | Benchmark runner |
| `main_pys/transformer_model.py` | Transformer-based flow model (alternative to GNN) |
| `evals/continuous_evals/` | Prior continuous eval CSVs |

### Existing collision shields in `continuous_env.py`

Three shields are already implemented:

1. **`ORCAStyleShield`** — ORCA-based (uses `rvo2` library), priority-ordered variant available via `_project_priority_ordered()`. Known issue: rvo2 polygon inflation creates invalid geometry on obstacle maps (adjacent cells create overlapping polygons for rvo2). Empty maps work; obstacle maps are broken.

2. **`PICBFCSShield`** — CBF-based shield (requires external `continuous_collision_shield` package). Most principled but has external dependency.

3. **`CV-PIBT`** — **Priority-ordered PIBT-style shield in continuous space.** This is the most relevant one. It:
   - Takes continuous positions `(N, 2)` and preferred velocities `(N, 2)`
   - Processes agents in descending priority order (same principle as discrete PIBT)
   - Each agent picks the best candidate velocity that avoids collisions with already-committed higher-priority agents
   - Candidates: preferred velocity, N evenly-spaced directions at max_speed, half-speed preferred, zero velocity (wait)
   - Backtracking: if agent can't find valid action, it requests the blocking higher-priority agent to try an alternative (up to `max_backtrack_depth` levels)
   - Uses SDF (signed distance field) for obstacle avoidance
   - **Does NOT have the rvo2 polygon bug** — uses SDF, not polygon primitives

### Prior continuous eval results (from memory, ~33 days ago)

- Flow + ORCA on `empty-48-48`: ~85% agent completion, 0 obstacle hits, 28–78 collisions at 96–128 agents
- ORCA baseline on `empty-48-48`: ~98% completion, 24–77 collisions at high density
- On `random-32-32-10` (obstacle map): **all methods catastrophically fail** (~3% completion, thousands of obstacle hits) — due to the rvo2 polygon inflation bug
- The CV-PIBT (SDF-based) was not fully evaluated yet at last check; it was being built

---

## The New Research Direction

**Hypothesis:** A flow-matching model that predicts continuous velocities, shielded by a PIBT-style priority-ordered collision shield in continuous (x, y) space, can resolve the coordination problems that broke the discrete interface — because:

1. The shield sees full velocity geometry, not just 5 discretized actions
2. Asymmetric yielding emerges naturally from priority ordering on continuous candidates
3. The model never has to commit to a discrete action; it just suggests a preferred velocity
4. Corridor bottleneck behavior (yield vs. advance) can be learned as a continuous control policy

The target shield is **`CV-PIBT`** with priority updates matching the discrete simulator's CS-PIBT logic.

### Key design questions to answer

1. **Priority scheme:** Discrete PIBT uses `priority += 1` per step for non-goal agents, reset to 0 on arrival. Does this transfer directly to continuous time? Or does BD distance (as a continuous surrogate) work better?

2. **Agent radius and max_speed:** Currently hardcoded. What values match the discrete grid semantics (1 cell = 1 unit, max 1 cell/step)?

3. **Candidate generation:** CV-PIBT currently samples N directions at max_speed + half-speed + zero. Should the preferred velocity be included as a candidate with higher weight? Should the BD-gradient direction be a candidate?

4. **Obstacle map:** The SDF-based approach should work for obstacle maps (unlike rvo2). Verify CV-PIBT correctly rejects wall-intersecting velocities on obstacle maps before training.

5. **Training data:** `generate_continuous_data.py` generates continuous trajectories. How are they generated — ORCA expert? Smoothed discrete expert? This determines what the model learns to imitate.

6. **Goal definition:** In continuous space, "at goal" = within radius ε of goal center. What ε?

---

## Evaluation Protocol

### Discrete reference (for comparison)
```bash
# Already have: evals/rishi_full_wave9_compact_best.csv (FLOMAP)
# Already have: evals/final_evals/rishi_full_ssil_classifier.csv (SSIL)
# Already have: evals/pibt_baseline_rishi12_40374462.csv (BD-PIBT)
```

### Continuous eval command structure
```bash
python eval_continuous.py \
  --map empty-48-48 \
  --policy flow \
  --shield cv-pibt \
  --model CHECKPOINT.pt \
  --agents 50 100 200 \
  --scenarios 5 \
  --output evals/continuous_evals/NAME.csv
```
(verify exact args against `eval_continuous.py --help` first — the interface may have evolved)

### Success metrics to track
- `success` (all agents reach goal within timeout)
- `num_agents_at_goal` (partial success / goal fraction)
- `collision_count` (agent-agent collisions)
- `obstacle_hits` (agent-obstacle collisions)
- `runtime`

### Map progression
1. **Start with `empty-48-48`** (no obstacles, pure coordination)
2. **Then `random-32-32-10`** (obstacle map, stress test — verify SDF shield works)
3. **Then `maze-128-128-2`** (the hardest discrete case)

---

## Lambda Commands

### Verify CV-PIBT works on obstacle maps (first thing to do)
```bash
cd ~/barath/Flow-CS-PIBT
conda activate 3dposehsx_env

# Smoke test: ORCA baseline (goal-directed velocities) with CV-PIBT on both map types
python eval_continuous.py \
  --map empty-48-48 \
  --policy goal_directed \
  --shield cv-pibt \
  --agents 50 100 \
  --scenarios 3 \
  --output evals/continuous_evals/epibt_smoke_empty.csv

python eval_continuous.py \
  --map random-32-32-10 \
  --policy goal_directed \
  --shield cv-pibt \
  --agents 20 50 \
  --scenarios 3 \
  --output evals/continuous_evals/epibt_smoke_obstacles.csv
```

Check: zero obstacle hits on obstacle map? If not, fix SDF-velocity projection in `CV-PIBT._hits_obstacle()`.

### Training continuous model (once data is verified)
```bash
python main_pys/train_continuous.py \
  --data-dir data/continuous_training_data/ \
  --output-dir checkpoints/continuous_epibt_v1/ \
  --shield-type cv-pibt \
  --epochs 100 \
  --batch-size 256 \
  --gpus 0 1 2 3 \
  --hidden-dim 1024 \
  --num-layers 6
```
(verify exact args against `train_continuous.py --help`)

### Medium continuous eval
```bash
python eval_continuous.py \
  --model checkpoints/continuous_epibt_v1/best.pt \
  --map empty-48-48 random-32-32-10 maze-128-128-2 \
  --policy flow \
  --shield cv-pibt \
  --agents 50 100 200 400 \
  --scenarios 5 \
  --output evals/continuous_evals/epibt_v1_medium.csv
```

---

## Key Code Locations

### `CV-PIBT.project()` — main entry point
`main_pys/continuous_env.py:901`

Input: `positions (N,2)`, `preferred_velocities (N,2)`, `obstacle_map (H,W)`, `priorities (N,)`  
Output: `collision-free velocities (N,2)`

### `ContinuousMAPFEnv.step()` — simulation step
`main_pys/continuous_env.py:1119+`

Calls the shield's `project()` and advances agent positions.

### Priority update logic (discrete reference)
`main_pys/simulator.py:176-181`
```python
def updatePriorities(prev_priorities, at_goal):
    agent_priorities[(prev_priorities <= 0) & at_goal] -= 1
    agent_priorities[(prev_priorities > 0) & at_goal] = 0
    agent_priorities[~at_goal] = np.maximum(prev_priorities[~at_goal], 0) + 1
```
Port this logic to the continuous simulator.

### Model: `FlowGNNModel`
`main_pys/generative_model.py` — same model used for discrete; already supports continuous inputs when `create_continuous_data_object` is used.

---

## What NOT to Do

- Do not attempt rvo2-based obstacle maps without fixing the polygon inflation bug first
- Do not run heavy evals (full primary8 / rishi12 scale) locally on Mac — Lambda only
- Do not commit with Co-Authored-By lines (git identity should be user's only)
- Do not run 540-case or full evals before medium gate passes
- Do not trust rishi8-only medium results for warehouse-sensitive interventions (learned from near-goal boost failure)

---

## Paper Framing

The contribution is: **flow matching for continuous MAPF with a PIBT-style priority-ordered collision shield**. The discrete system showed that:
- Flow matching is competitive with discrete classifiers on non-stress maps
- The bottleneck is the continuous-to-discrete interface, not the model quality

The continuous system removes that bottleneck. If it works, the result is a MAPF system that operates natively in (x,y) space with formal collision safety via priority ordering. This is directly comparable to ORCA but with learned velocity guidance. Target venue: ICRA or IROS.

Key baselines to beat:
1. ORCA (reactive, no learning) on continuous maps
2. Goal-directed CV-PIBT (greedy, no learning) on continuous maps
3. If possible: the discrete FLOMAP/SSIL results translated to the continuous benchmark
