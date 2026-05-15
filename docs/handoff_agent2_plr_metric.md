# Agent 2 Handoff: Arrived-Agent PLR Metric + Integration Step Ablation

**Branch:** `barath/continuous_space_epibt`  
**Constraint:** No code runs locally. Push all code changes, then provide Lambda shell commands for any eval runs.

---

## Context

The current `path_length_ratio` (PLR) metric is computed over **all agents**, including those that never reach their goal. Non-arrived agents contribute their full wandering distance to the numerator, inflating PLR for any method that doesn't achieve 100% completion. On cluttered maps, the flow model reaches 86.7% (not 100%), so its PLR of 5.9 is partially inflated by the ~13% of agents still wandering at episode end.

A reviewer will flag PLR=3.33 on the empty-48-48 map (where 98.6% of agents arrive) as evidence that the model hasn't learned efficient navigation. We need both:
1. A fair **arrived-agent PLR** metric to clarify the true path efficiency
2. An **integration step ablation** (3 vs 5 vs 10 vs 20 Euler steps) to test if more steps produce straighter paths

---

## Task 1: Add `arrived_path_length_ratio` to `continuous_env.py`

**File:** `main_pys/continuous_env.py`

**Location:** The `current_metrics()` method, around line 1382. It currently computes:

```python
direct = np.linalg.norm(self.goals - self.history_positions[0], axis=1).sum()
path_length = compute_path_length(positions)
return {
    ...
    "path_length_ratio": float(path_length / max(direct, 1e-6)),
    ...
}
```

**Change needed:** Add a new metric `arrived_path_length_ratio` that computes PLR only over agents that have arrived (`self.arrival_steps >= 0`). If no agents have arrived, return `float("nan")`.

The `positions` variable is already `np.asarray(self.history_positions, dtype=np.float32)` with shape `(T, N, 2)`. The `compute_path_length` function currently sums over all N agents — you will need to call it on a subset.

**Implementation approach:**

```python
# Arrived-agent PLR
arrived_mask = self.arrival_steps >= 0
if arrived_mask.any():
    # Slice positions to arrived agents only: shape (T, n_arrived, 2)
    positions_arrived = positions[:, arrived_mask, :]
    direct_arrived = np.linalg.norm(
        self.goals[arrived_mask] - self.history_positions[0][arrived_mask], axis=1
    ).sum()
    path_length_arrived = compute_path_length(positions_arrived)
    arrived_plr = float(path_length_arrived / max(direct_arrived, 1e-6))
else:
    arrived_plr = float("nan")
```

Add `"arrived_path_length_ratio": arrived_plr` to the returned dict.

Also check what `compute_path_length` does — it's defined around line 128. It takes `positions` of shape `(T, N, 2)` and sums step-to-step distances across all agents and all timesteps. The slice `positions[:, arrived_mask, :]` should work directly if arrived_mask is a boolean array of length N.

**Important:** For arrived agents, their path ends at their arrival step (they stop moving once at goal). This means even their "total" path includes any time they spent waiting at the goal. Check whether `history_positions` continues to record positions after arrival — if agents freeze at goal position after arrival, the path length is already correct. If not, you may want to truncate each arrived agent's trajectory at its arrival step. Look at how the env handles agent positions post-arrival to decide.

---

## Task 2: Add `arrived_path_length_ratio` to `analyze_evals.py`

**File:** `analyze_evals.py`

**Change 1:** Add `"arrived_path_length_ratio"` to the `METRICS` list (around line 42).

**Change 2:** In the per-map detail printing section, add `arr_PLR` to the per-row display alongside the existing `PLR` column. Format it as `arrPLR=X.XXX` immediately after `PLR=X.XXX`.

The display loop aggregates metrics by `(label, map_name, agents, max_steps)`. The arrived PLR should appear in the same row. Look at how `PLR` is currently formatted in the output and add `arrPLR` in the same style.

---

## Task 3: Integration Step Ablation (Lambda eval commands)

Add new entries to `analyze_evals.py`'s CATALOG for integration step variants:

```python
# ── Integration step ablation (Set A, 512 steps) ──────────────────────────
("evals/ablations/flow_epibt_512_setA_steps5.csv",  "Flow v4b + CV-PIBT [5 steps]",  "A_steps", 512),
("evals/ablations/flow_epibt_512_setA_steps10.csv", "Flow v4b + CV-PIBT [10 steps]", "A_steps", 512),
("evals/ablations/flow_epibt_512_setA_steps20.csv", "Flow v4b + CV-PIBT [20 steps]", "A_steps", 512),
```

Also add `"A_steps"` to `SET_NAMES`:
```python
"A_steps": "Set A — Integration steps ablation (512 steps)",
```

Create `scripts/run_agent2_evals.sh` committed to the repo:

```bash
#!/usr/bin/env bash
# Agent 2 eval commands — run on Lambda from repo root
set -e

MODEL=checkpoints/continuous_v4b/large_scale_flow_v4b_best.pt
MAPDIR=data/mapf-map
SCENDIR=data/mapf-scen-random
N=25

# ── Integration step ablation ────────────────────────────────────────────────
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type cv-pibt \
  --max-steps 512 --num-integration-steps 5 \
  --output-csv evals/ablations/flow_epibt_512_setA_steps5.csv

python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type cv-pibt \
  --max-steps 512 --num-integration-steps 10 \
  --output-csv evals/ablations/flow_epibt_512_setA_steps10.csv

python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type cv-pibt \
  --max-steps 512 --num-integration-steps 20 \
  --output-csv evals/ablations/flow_epibt_512_setA_steps20.csv

echo "Agent 2 evals complete."
```

---

## What the Results Will Tell Us

- **Arrived-agent PLR < all-agent PLR**: This is expected and important. If arrived-PLR on empty-48-48 is ~1.3–1.8 (vs all-agent PLR of 3.33), the path efficiency concern is mostly a presentation artifact. If it's still 2.5+, the model genuinely takes long paths.

- **Integration step ablation**: If PLR decreases as steps increase (3 → 5 → 10 → 20), it confirms the rectified flow needs more integration steps to produce straighter velocity fields. If PLR doesn't change with steps, the wandering is a training data artifact (grid-aligned EECBS paths), not a convergence issue.

- **Combined**: Arrived-agent PLR + integration step analysis gives the paper a complete, honest discussion of path efficiency that preempts reviewer concerns.
