# Paper Plan: Flow Matching for Multi-Agent Path Finding

## Context

Flow-CS-PIBT replaces the discrete 5-class action classifier from Veerapaneni et al. ("Work Smarter, Not Harder", ICRA 2025) with a Rectified Flow model that predicts continuous 2D velocity fields. We have evaluation results on the paper's 8 held-out maps and access to the original paper's discrete classifier numbers for comparison.

**Goal**: Produce a paper combining (1) rigorous grid-world comparison (flow vs discrete) and (2) continuous-space MAPF where flow matching is a natural fit.

**Resources**: 3x A6000 GPUs, 2-3 weeks, existing codebase + Rishi paper baselines.

---

## Current Results Snapshot

**Best model (heldout)**: 65.5% overall success (1,212/1,850 scenarios, all-agents-at-goal)

```
Map                          100  200  300  400  500  600  700  800  900 1000
Paris_1_256                 100% 100% 100% 100% 100% 100% 100%  96% 100% 100%
den520d                     100% 100% 100% 100% 100% 100% 100% 100% 100%  96%
empty-48-48                 100% 100% 100% 100% 100% 100% 100% 100% 100%  80%
random-64-64-10             100% 100%  92%  96%  92%  92%  92%  80%  68%  72%
random-32-32-10              96%  84%  68%  56%  N/A  N/A  N/A  N/A  N/A  N/A
den312d                     100%  84%  88%  72%   0%   0%   0%   0%   0%   0%
maze-128-128-2               96%  84%  64%  56%  12%   0%   0%   0%   0%   0%
warehouse-10-20-10-2-1       28%   4%   0%   0%   0%   0%   0%   0%   0%   0%
```

**Key issues**: den312d/maze cliff at 500 agents, warehouse near-total failure, median 88% agent completion on failures (deadlock, not model quality).

---

## Phase 1: Data Pipeline Optimization (Days 1-3)

### Problem
Current preprocessed dataset is 600+ GB of ~384K `.pt` files. Only 50-100 GB of disk space remains. On-the-fly graph construction is too slow (BD heuristic BFS is the CPU bottleneck). Need to shrink existing data in-place.

### Strategy: In-Place Re-encoding + Downsampling

**Target: 600 GB → ~60-70 GB** via three combined optimizations:

| Optimization | Multiplier | Details |
|-------------|-----------|---------|
| Deduplicate map channel | 0.67x | Map obstacle grid (channel 0 of `x`) is identical across all timesteps of a scenario. Store once per map (~30 maps, negligible). Remove from .pt, reconstruct in DataLoader. |
| Float16 + int32 dtypes | 0.5x | Node features (BD + agent channels): float32→float16. Edge attr: float32→float16. Edge index: int64→int32. Velocities: float32→float16. |
| Downsample timesteps (every 3rd) | 0.33x | Keep every 3rd timestep per trajectory. Flow matching doesn't need consecutive frames — it learns velocity fields from independent snapshots. |
| **Combined** | **~0.11x** | **600 GB × 0.67 × 0.5 × 0.33 ≈ 66 GB** |

### 1.1 Re-encoding Script (`compact_dataset.py`)

Process one map at a time to stay within disk budget:

```
For each map_name in training maps:
  1. List all .pt files for this map
  2. Downsample: keep every 3rd file (sorted by timestep)
  3. For each kept file:
     a. Load .pt
     b. Strip map channel (channel 0) from x → x becomes (N, 2, 9, 9)
     c. Convert x to float16
     d. Convert edge_index to int32
     e. Convert edge_attr, y, node_weights, bd_pred to float16
     f. Add discrete_action label: derive from y (velocity→cardinal direction)
     g. Save as compact .pt (pickle_protocol=4)
  4. Verify compact files load correctly (spot-check 5 random files)
  5. Delete original .pt files for this map
  6. Report: original size → compact size, files kept/deleted
```

**Safety**: Verify before deleting. Log everything. Can resume from any map if interrupted.

### 1.2 Updated DataLoader (`dataset_preprocessed.py`)

Modify the DataLoader to reconstruct full features at load time:

```python
def __getitem__(self, idx):
    data = torch.load(self.files[idx])
    # Reconstruct 3-channel features: [map_patch, bd_patch, agent_patch]
    map_name = data.map_name  # stored as metadata
    map_grid = self.map_cache[map_name]  # loaded once at init
    # Extract per-agent map patches from global grid using agent positions
    map_patches = extract_patches(map_grid, data.positions, k=4)  # (N, 1, 9, 9)
    data.x = torch.cat([map_patches, data.x.float()], dim=1)  # (N, 3, 9, 9)
    # Upcast float16 → float32 for training
    data.y = data.y.float()
    data.edge_attr = data.edge_attr.float()
    return data
```

This adds minimal CPU overhead (patch extraction is a simple index operation on a cached grid) while keeping the BD heuristic pre-computed (avoiding the BFS bottleneck).

### 1.3 Add Discrete Action Labels

For the grid-world discrete classifier study (Phase 2), we need expert action labels (0-4).

**During re-encoding**: derive from existing velocity targets:
```python
# Map velocity → discrete action: 0=wait, 1=right, 2=down, 3=up, 4=left
ACTION_VECS = [[0,0], [0,1], [1,0], [-1,0], [0,-1]]
scores = y @ torch.tensor(ACTION_VECS).T  # (N, 5)
discrete_action = scores.argmax(dim=1)  # (N,)
# For near-zero velocities, force wait
discrete_action[y.norm(dim=1) < 0.3] = 0
```

Store as `data.action_label` (int8) in the compact .pt files. This avoids re-running EECBS.

### 1.4 Clean Train/Eval Split Verification

- Verify no held-out map data in preprocessed directories
- Add map_name metadata to each .pt file during re-encoding
- DataLoader can filter by map_name to enforce strict splits

### 1.5 Continuous-Space Data Pipeline Prep (for Phase 3)

Create `generate_continuous_data.py` stub:
- ORCA expert planner for continuous trajectories
- Can run on CPU in background during Phase 2 GPU experiments
- Separate from grid-world data — stored in `data/continuous/`

### Files to create (in Flow-CS-PIBT repo)
- `compact_dataset.py` — in-place re-encoding script
- `generate_continuous_data.py` — ORCA pipeline stub (Phase 3 prep)

### Files to modify
- `main_pys/dataset_preprocessed.py` — load compact format, reconstruct map channel
- `main_pys/model_inputs.py` — `extract_patches()` utility for map reconstruction

### Verification
1. Compact 1 map, train for 1 epoch, compare loss to original format → should be identical (float16 precision loss is negligible for flow matching)
2. Check discrete action labels match original EECBS actions on a sample
3. Monitor disk usage throughout migration

---

## Phase 2: Grid-World Comparison Study (Days 4-10)

### 2.1 Discrete Classifier Baseline (Days 4-5)
- Add `--discrete-only` mode to `train_flow.py`: same CNN+GNN, output = 5-class softmax with CE loss (no flow)
- Train on clean base data (no heldout contamination) — 1 GPU
- Also retrain clean flow model on same data — 1 GPU (parallel)

### 2.2 Evaluation Sweep (Days 5-6)
Run `eval_rishi_paper.py` for:
1. Clean flow matching model
2. Clean discrete classifier (same arch)
3. Existing wave8_heldout model (reference)
4. Action head mode (`--useActionHead True`)
Compare all against Rishi paper's published numbers. 3 GPUs in parallel.

### 2.3 Ablation Experiments (Days 6-8)

| Ablation | Values | Purpose |
|----------|--------|---------|
| Euler steps | 1, 2, 3, 5, 10 | Quality vs speed |
| Consensus samples | 1, 3, 5, 10 | Variance reduction cost |
| Temperature (tau) | 0.1, 0.2, 0.3, 0.5, 1.0 | Action sharpness |
| Model size | hidden {256, 512, 1024}, layers {2, 4, 6} | Capacity |
| Wait threshold | 0.1, 0.2, 0.25, 0.3, 0.5 | Wait sensitivity |

### 2.4 Analysis & Figures (Days 9-10)
- Generalization: flow vs discrete on unseen maps
- Uncertainty: consensus sample variance vs difficulty
- Computational cost: FLOPs and wall-time breakdown
- Failure analysis: model vs shield attribution

### Files to modify
- `main_pys/train_flow.py` — `--discrete-only` mode
- `main_pys/generative_model.py` — discrete-only forward path
- `sweep_inference.py` — extended ablation configs

---

## Phase 3: Continuous-Space MAPF (Days 11-17)

### 3.1 Environment (Days 11-12)
- Continuous 2D MAPF: circular agents, continuous positions/velocities, speed-limited
- Reuse grid maps as obstacle boundaries
- Start with `empty-48-48` (no obstacles), then add complexity
- **File**: `main_pys/continuous_env.py`

### 3.2 Expert Data + Training (Days 12-14)
- Generate ORCA expert trajectories (from Phase 1.4 pipeline)
- Adapt FlowGNNModel: output = velocity applied directly (no discretization)
- Safety layer: ORCA post-processing or velocity clipping
- Train flow model + discrete baseline (8/16 direction bins) in parallel

### 3.3 Evaluation & Analysis (Days 15-17)
- Baselines: ORCA standalone, social force, discrete + interpolation
- Metrics: success rate, path quality, smoothness, runtime
- Key result: smooth trajectories in continuous space
- Scalability curves, trajectory visualizations

### Files to create
- `main_pys/continuous_env.py`
- `eval_continuous.py`

### Files to modify
- `main_pys/model_inputs.py` — continuous position handling
- `main_pys/simulator.py` — continuous stepping mode

---

## Phase 4: Paper Writing & Polish (Days 18-21)

### 4.1 Gap-filling experiments (Days 18-19)
- Multi-seed runs (3-5 seeds) for statistical significance
- Missing ablations identified during writing
- 3 GPUs in parallel

### 4.2 Writing (Days 19-21)
**Title**: "Flow Matching for Multi-Agent Path Finding: From Discrete Grids to Continuous Spaces"

**Structure**:
1. Introduction: MAPF + discrete classifier limitations + flow matching motivation
2. Background: MAPF, CS-PIBT/SSIL, Rectified Flow
3. Method: FlowGNNModel, flow-to-action pipeline, continuous extension
4. Grid-World Experiments: comparison, ablations, generalization
5. Continuous-Space Experiments: environment, baselines, results
6. Analysis: when flow helps, uncertainty, computational tradeoffs
7. Conclusion

---

## GPU Allocation (3x A6000)

| Phase | GPU 1 | GPU 2 | GPU 3 |
|-------|-------|-------|-------|
| Phase 1 | Data regeneration | Data regeneration | Continuous data (ORCA, CPU-bound) |
| Phase 2a | Train clean flow | Train discrete | Eval wave8 models |
| Phase 2b | Ablation (steps) | Ablation (tau/size) | Ablation (generalization) |
| Phase 3 | Train continuous flow | Train continuous discrete | Eval + visualize |
| Phase 4 | Multi-seed runs | Gap experiments | Final evals |

---

## Verification

1. Discrete baseline should match Rishi numbers within ~5%
2. Flow with 1 Euler step ≈ discrete classifier (sanity check)
3. ORCA solves simple scenarios perfectly before using as expert
4. Continuous model on straight-line paths → near-optimal cost
5. 3+ seeds on key experiments, report mean ± std

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Flow doesn't beat discrete in grids | Frame as "when/why" analysis — publishable either way |
| Continuous env too slow to build | Start obstacle-free, add complexity incrementally |
| Not enough time for both | Grid-world comparison is priority; continuous can be "preliminary" |
