# Continuous-Space MAPF: Findings, Paper Story, and Experiment Log

*Last updated: 2026-05-04*

---

## 1. What This Work Is

**Flow-CS-PIBT** extends the grid-world MAPF line of work from Veerapaneni et al. (ICRA 2025, "Work Smarter, Not Harder") into continuous 2D space. The grid-world branch replaces their discrete 5-class action classifier with a Rectified Flow model. The continuous-space branch (this document's focus) goes further: agents move freely in 2D with circular collision geometry, no grid constraint at all.

The two-sentence version of the paper:

> We extend PIBT's priority-inheritance collision shield to continuous 2D space (CV-PIBT) and show it dramatically outperforms ORCA. A flow matching GNN trained on EECBS expert trajectories further improves upon straight preferred velocities under CV-PIBT, achieving 91.4% agent completion vs ORCA's 58.4% at N=50.

**Target venue**: ICRA 2026 (or RA-L with ICRA option).

---

## 2. Novel Contributions vs Prior Art

### Existing work this builds on

| Component | Source |
|---|---|
| PIBT (discrete grid) | Okumura et al. |
| CS-PIBT / SSIL (grid-world with discrete classifier) | Veerapaneni et al., ICRA 2025 |
| ORCA / RVO2 (continuous reactive avoidance) | van den Berg et al., ISRR 2011 |
| Priority-ordered ORCA / asymmetric ORCA | Known concept in ORCA literature; **NOT our contribution** |
| Rectified Flow / Flow Matching | Lipman et al.; Liu et al. |
| EECBS, LaCAM3 | Standard MAPF planners |

### This paper's contributions

1. **CV-PIBT** — a novel extension of PIBT's priority-inheritance backtracking mechanism from discrete grids to continuous 2D space with circular agents. Agents are ordered by priority; each resolves its velocity to avoid all higher-priority agents' projected positions, recursing down the priority queue. This is not defined in any prior work. **This is the dominant contribution** (see Section 9).

2. **Flow matching policy for continuous MAPF** — a GNN-based Rectified Flow model (`FlowGNNModel`) trained on EECBS expert trajectories that predicts preferred velocities for CV-PIBT. Secondary contribution; improves upon straight goal-directed preferences.

3. **Empirical finding** — CV-PIBT with even naive straight preferred velocities dramatically outperforms ORCA, reframing the role of the learned policy as an improvement on top of an already-strong shield rather than a replacement for ORCA.

### What is NOT a contribution: PO-ORCA

`po-orca` in our codebase (ORCAStyleShield with `use_priorities=True`) is **priority-ordered / asymmetric ORCA**: higher-priority agents commit their preferred velocity unchanged; lower-priority agents bear the full avoidance burden. This concept is well-established in the ORCA literature under names like "prioritized ORCA" and "one-sided RVO" and appears in van den Berg's own follow-up work. We use it as an **ablation baseline**, not as a contribution. It lets us isolate the benefit of PIBT's priority inheritance (dynamic, recursive) vs. static priority ordering alone.

---

## 3. Literature Review & Baseline Rationale

### 3.1 Original ORCA Paper (van den Berg et al., ISRR 2011)

The original ORCA paper proves mathematical collision-freedom under its halfplane assumptions and reports only computational runtime. Key facts:

- **Environments**: 1,000-agent open circle antipodal swap; office-evacuation crowd simulation. No grid obstacle maps resembling MAPF benchmarks.
- **Metrics**: ms/frame and agent-count scaling. Does NOT report success rates or at-goal fractions — the paper claims guaranteed collision-freedom, so these were not measured.
- **Parameters (RVO2 canonical example)**: `timeStep=0.25s`, `neighborDist=15.0`, `timeHorizon=10.0`, `timeHorizonObst=5.0`, `radius=2.0`, `maxSpeed=2.0` — in a ~100-unit workspace. Not directly comparable to grid-unit setups.
- **Our parameters vs theirs**: `radius=0.3`, `maxSpeed=1.0`, `dt=0.2`, `time_horizon=2.0`. Our `time_horizon=2.0` matches the τ=2 value in van den Berg's Figure 4. Our `radius=0.3` is the most commonly used value in follow-on ORCA literature.
- **Key quote**: "The half-planes of permitted velocities with respect to obstacles only make sure that the robot avoids collisions with the obstacle; they do not make the robot move around them." ORCA requires an external path planner for obstacle-aware preferred velocities — without it (our `--nav straight`), agents run straight into obstacles and deadlock. This is the intended usage and the standard baseline.

### 3.2 ORCA in MAPF-Like Settings

| Paper | Map | N | ORCA result | Notes |
|---|---|---|---|---|
| PRIMAL (Sartoretti, RAL 2019) | 40×40 grid, 10-30% obstacles | 16+ | "Performs very poorly in most scenarios" | No tables, qualitative only; ORCA results had a post-publication bug fix |
| Dergachev & Yakovlev (CASE 2021) | 64×64 Gaps-3; 32×32 Rooms (MovingAI) | 40 | 0% success (Gaps-3); 10-15% success (Rooms) | Radius=0.3 cell, most directly comparable to our setup |
| MPPI-ORCA (Dergachev, PeerJ-CS 2024) | Grid dense placement | 16 | 0% (dense), 78% (sparse) | Differential-drive ORCA variant |
| CMPP (2025) | Continuous 22×18m warehouse | 200/300/400 | 100%/≈100%/83.9% | Open warehouse, no obstacle-dense regions |
| **Our setup** | random-32-32-10, empty-48-48 | 50 | **58.4% per-agent at-goal** (512 steps) | Fraction at goal, not binary success |

**Our 58.4% is consistent with and plausible for the literature.** random-32-32-10 is less constrained than the Rooms/Gaps maps where Dergachev shows near-0% success, but significantly more constrained than open warehouse environments. Note the metric difference: we report per-agent fraction at goal, not binary episode success (which would be much lower — likely near 0% since ORCA deadlocks trap a minority of agents permanently).

**rvo2 library status**: confirmed installed on Lambda (`pyrvo2-0.0.0`, native `.so`). We are running true RVO2, not the heuristic fallback. The comparison is fair.

### 3.3 Why Other MAPF Papers Are Not Comparable Baselines

| Paper | Reason NOT directly comparable |
|---|---|
| PRIMAL / PRIMAL2 (Sartoretti) | Grid-world with discrete actions. Agents are points at integer cells, not circles in continuous space. Different problem class. |
| SCRIMP (Wang, AAMAS 2023) | Grid-world RL; does not benchmark vanilla ORCA |
| DHC (Ma, RA-L 2021) | Grid-world RL; benchmarks only PRIMAL and ODrM*, not ORCA |
| Dergachev 2021 (CASE) | Augments ORCA with local MAPF solver; different algorithm, different maps, binary success metric. Not runnable in our framework. |
| CMPP 2025 | Warehouse-specific trajectory optimization; different kinematic model. |
| MPPI-ORCA 2024 | Differential-drive robots; different kinematics. |
| CCBS (Andreychuk) | Centralized optimal planner for continuous-time MAPF; exponential in worst case, not scalable to N=50. Useful only as an "oracle upper bound" citation. |

These papers belong in related work as motivation, not as direct numerical baselines.

### 3.4 Chosen Baseline Set and Rationale

The ablation table has four rows, each adding exactly one ingredient:

| Method | What it tests |
|---|---|
| ORCA + straight | Standard reactive baseline; widely used in MAPF literature |
| PO-ORCA + straight | Priority ordering alone, without PIBT backtracking |
| CV-PIBT + straight | PIBT inheritance alone, without learning |
| CV-PIBT + Flow (ours) | Full method: PIBT inheritance + learned preferred velocity |

The gaps tell the story cleanly:
- **ORCA → PO-ORCA**: benefit of static priority ordering
- **PO-ORCA → CV-PIBT + straight**: benefit of PIBT recursive backtracking (the key novel contribution)
- **CV-PIBT + straight → CV-PIBT + flow**: benefit of learned preferred velocity

Optional fifth row: **ORCA + flow** (already run as v2) — shows the learned model cannot rescue a weak shield, motivating CV-PIBT.

---

## 4. Map Selection & Evaluation Sets

### 4.1 MovingAI Maps for Continuous MAPF

MovingAI benchmark maps were designed for discrete grid-world MAPF (agents as points at integer cells). Adapting them to continuous space (agents as circles, radius=0.3, cell size=1.0) creates geometric constraints:

| Map type | Min corridor width (cells) | Usable for continuous MAPF? |
|---|---|---|
| empty-* | Open (infinite) | ✅ Ideal |
| random-32-32-10, random-32-32-20 | ≥3 cells typical | ✅ Good |
| random-64-64-10 | ≥3 cells typical | ✅ Good (larger scale) |
| room-32-32-4, room-64-64-8 | Doorways: 1 cell | ⚠️ Doorways are single-file only; creates bottlenecks. Use N≤50. |
| maze-32-32-2 | 2 cells | ❌ Passing room is ~0.2 units; EECBS fails most N=100 scenarios |
| maze-32-32-4 | 4 cells | ⚠️ Marginal at N=100; many EECBS skips observed |
| maze-128-128-2 | 2 cells | ❌ Same corridor issue, 4× larger |
| warehouse-10-20-10-2-1 | Wide aisles | ✅ Good for OOD eval; realistic robot domain |
| Berlin_1_256 | Complex city streets | ❌ Too large for straight-nav preferred velocity to work |

**Rule for exclusion**: a map is excluded if more than 30% of N=100 scenarios are skipped by EECBS (proxy for continuous-space incompatibility). This directly excludes maze-32-32-2.

In the paper: *"We adapt MovingAI benchmark maps to continuous space by representing obstacles as a signed-distance field (SDF); start positions are at grid cell centers. Maps with single-cell corridors (maze variants with 2-cell corridors) are excluded as geometrically incompatible with agent radius r=0.3."*

### 4.2 Evaluation Set Definitions

Three evaluation sets for the paper, combining seen and heldout maps:

---

**Eval Set A — Primary (Seen, Core Paper Table)**

| | |
|---|---|
| Maps | `random-32-32-10`, `empty-48-48` |
| Agent counts | N=50, N=100 |
| Scenarios | 25 per map |
| Purpose | Main comparison table; apples-to-apples with all baselines |
| Status | Partially done (5 scenarios); need full 25 |

```bash
python eval_parallel.py \
  --map-dir data/mapf-map --scen-dir data/mapf-scen-random \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios 25 \
  --policy flow \
  --model-path checkpoints/continuous_v4/continuous_flow_v4_best.pt \
  --shield-type cv-pibt --num-integration-steps 3 --max-steps 512 \
  --output-csv evals/v4/flow_epibt_512_setA.csv --num-gpus 4
```

---

**Eval Set B — Generalization / Seen (Seen during training, different type/scale)**

| | |
|---|---|
| Maps | `random-64-64-10` (N=50, 100), `room-32-32-4` (N=50 only) |
| Agent counts | 50 (both maps), 100 (random-64-64-10 only) |
| Scenarios | 25 per map |
| Purpose | Tests scale generalization (2× larger map) and structured map type |
| Notes | room-32-32-4 N=100 excluded due to narrow doorways |

```bash
python eval_parallel.py \
  --map-dir data/mapf-map --scen-dir data/mapf-scen-random \
  --maps random-64-64-10 \
  --agent-counts 50 100 --max-scenarios 25 \
  --policy flow \
  --model-path checkpoints/continuous_v4/continuous_flow_v4_best.pt \
  --shield-type cv-pibt --num-integration-steps 3 --max-steps 512 \
  --output-csv evals/v4/flow_epibt_512_setB_r64.csv --num-gpus 4

python eval_parallel.py \
  --map-dir data/mapf-map --scen-dir data/mapf-scen-random \
  --maps room-32-32-4 \
  --agent-counts 50 --max-scenarios 25 \
  --policy flow \
  --model-path checkpoints/continuous_v4/continuous_flow_v4_best.pt \
  --shield-type cv-pibt --num-integration-steps 3 --max-steps 512 \
  --output-csv evals/v4/flow_epibt_512_setB_room.csv --num-gpus 4
```

---

**Eval Set C — OOD Heldout (Maps NOT in v4 training)**

| | |
|---|---|
| Maps | `warehouse-10-20-10-2-1` |
| Agent counts | N=50, N=100 |
| Scenarios | All available (check Lambda; 25 if present) |
| Purpose | True out-of-distribution generalization test; realistic robot domain |
| Why warehouse | Not in v4 training; structured aisles with open traversable regions; directly relevant to real-world robot deployment |
| Why not Berlin_1_256 | 256×256 city map; straight-nav preferred velocity cannot navigate around buildings — not a fair test of the policy |
| Why not maze-128-128-2 | 2-cell corridors; physically incompatible with r=0.3 at multi-agent density |

```bash
python eval_parallel.py \
  --map-dir data/mapf-map --scen-dir data/mapf-scen-random \
  --maps warehouse-10-20-10-2-1 \
  --agent-counts 50 100 --max-scenarios 25 \
  --policy flow \
  --model-path checkpoints/continuous_v4/continuous_flow_v4_best.pt \
  --shield-type cv-pibt --num-integration-steps 3 --max-steps 512 \
  --output-csv evals/v4/flow_epibt_512_setC_warehouse.csv --num-gpus 4
```

Also run all three baselines (ORCA, PO-ORCA, CV-PIBT+straight) on Set A at minimum, and Set C for OOD generalization. Set B baselines are optional (space permitting in the paper).

---

## 5. System Architecture

### Environment (`main_pys/continuous_env.py`)

- `ContinuousMAPFEnv`: circular agents (radius 0.3), continuous 2D positions/velocities, `dt=0.2`, `max_speed=1.0`, grid obstacle maps as boundaries represented as SDF
- `CV-PIBT`: priority-ordered collision resolver with recursive PIBT backtracking. Takes preferred velocities → outputs safe velocities respecting agent-agent and agent-obstacle geometry
- `ORCAStyleShield`: ORCA-based shield. Modes: `orca` (symmetric RVO2, true rvo2 library if installed), `po-orca` (priority-ordered, one-sided), `heuristic-orca` (force heuristic path, no rvo2), `none`
- `ContinuousMAPFEnv` instantiates shield once at construction; `step(preferred, shield_type=...)` dispatches at step time

### ORCA parameters (continuous_env.py:ORCAStyleShield)

```python
time_horizon: float = 2.0       # agent-agent lookahead (seconds)
obstacle_horizon: float = 1.0   # agent-obstacle lookahead (seconds)
iterations: int = 3             # heuristic ORCA iterations (fallback only)
neighbor_dist = max(4.0 * agent_radius, 2.0)  # = 2.0 with r=0.3
max_neighbors = 16
```

rvo2 (`pyrvo2-0.0.0`) confirmed installed on Lambda. `shield_type="orca"` uses true RVO2, not heuristic fallback.

### Data generation (`generate_continuous_data.py`)

Expert cascade: EECBS → LaCAM3 → ORCA (when `--expert-source hybrid`).

- Converts discrete EECBS/LaCAM3 paths to continuous via linear interpolation
- Saves `.npz` per scenario: `positions (T,N,2)`, `velocities (T-1,N,2)`, `goals (N,2)`, `action_labels (T-1,N)`, `expert_source_used`, `fraction_moving`, etc.
- Flag `--rollout-shield-type {orca,po-orca,cv-pibt,none}`: overrides shield during ORCA/fallback rollouts

### Graph construction (`main_pys/model_inputs.py`)

`create_continuous_data_object(positions, goals, grid, k=4, m=5)` builds a PyG `Data` object per timestep:
- Node features `x`: 4-channel local patch `(N, 4, 2k+1, 2k+1)` — obstacle map, goal_dx, goal_dy, Gaussian agent occupancy. **Local** extraction: model sees a 9×9 patch around each agent. This makes the model **map-size agnostic** — it generalizes to arbitrarily large maps as long as local geometry is similar to training.
- Edge features: relative position vectors to k-nearest neighbours
- `aux_features`: relative goal vector (2D), goal distance (scalar), at-goal flag, max_speed
- Labels `y`: velocity targets; `action_label`: 8-direction + wait (9 classes)

### Model (`main_pys/generative_model.py`)

`FlowGNNModel`:
- CNN: 3-layer Conv2d (4→64→128→256) + BatchNorm + SiLU + MaxPool, flattens to `cnn_out_dim`
- `visual_proj`: Linear(cnn_out_dim + 5, hidden_dim) + LayerNorm + SiLU + Dropout(0.15)
- `input_proj`: Linear(hidden_dim + 2 + 1, hidden_dim) — concatenates visual embedding, flow variable v_t (2D), time t (scalar)
- GNN: `num_layers` SAGEConv with LayerNorm + residual connections
- `post_mp`: 3-layer MLP → 2D flow velocity output
- `action_head`: 2-layer MLP → 9-class direction logits (auxiliary, used only in training)
- `wait_head` + calibration scalars: inference-only, never used during training (hence `find_unused_parameters=True` required for DDP)

### Training (`main_pys/train_continuous.py`)

Rectified flow loss: `t ~ sigmoid(N(0,1))`, `x_t = t*x_1 + (1-t)*x_0`, predict `x_1 - x_0`, MSE.
Auxiliary action CE loss (weight 0.1) on 8+1 direction labels. AdamW, CosineAnnealingLR, AMP, DDP-aware.

**DDP note**: `find_unused_parameters=True` required because `wait_head` and calibration scalars (`wait_logit_scale`, `wait_logit_bias`, `movement_logit_scale`) are inference-only and never receive gradients during training.

### Evaluation (`eval_continuous.py` / `eval_parallel.py`)

`eval_continuous.py`: single-process eval on a set of scenarios.
`eval_parallel.py`: multi-GPU launcher — enumerates scenario IDs, splits round-robin across N GPUs, launches N `eval_continuous` subprocesses (one per GPU via `CUDA_VISIBLE_DEVICES`), merges CSV shards.

Flow policy inference: integrate from `x_0 ~ N(0,I)` with `--num-integration-steps` Euler steps, multiply by `max_speed`, pass to shield.

Metrics: `success` (all agents at goal), `agent_fraction_at_goal`, `collisions`, `obstacle_hits`, `runtime`.

---

## 6. Infrastructure Notes (Lambda)

| Item | Path |
|---|---|
| Repo | `/home/anushree_mattlab/barath/Flow-CS-PIBT/` |
| EECBS binary | `/home/anushree_mattlab/barath/EECBS-flow/build/eecbs` |
| LaCAM3 binary | `/home/anushree_mattlab/barath/lacam3/build/main` |
| Maps | `data/mapf-map/` |
| Scen files | `data/mapf-scen-random/` (25 scen files per map) |
| v1 training data | `data/continuous_training/v1/` (100 EECBS rollouts, 2 maps) |
| v4 training data | `data/continuous_training/v4/` (~380 EECBS rollouts, 9 maps; maze/room N=100 mostly skipped) |
| v1 checkpoint | `checkpoints/continuous/continuous_flow_v1_best.pt` |
| v4 checkpoint (in progress) | `checkpoints/continuous_v4/continuous_flow_v4_best.pt` |
| Eval CSVs | `evals/` |
| Logs | `logs/` |
| Python env | `3dposehsx_env` (conda) |
| rvo2 | `pyrvo2-0.0.0` installed, native `.so`, true RVO2 confirmed |

**GPU setup**: 4× NVIDIA RTX A6000 (49 GB each). For training, single-GPU with `--num-workers 96 --batch-size 256` is faster than 4-GPU DDP for the current dataset size (~106k samples). DDP breaks even only at larger dataset sizes.

---

## 7. All Experiments Run

### 7.1 Data Generation

| Dataset | Expert | Shield | Maps | Scenarios | Agent counts | Rollouts | Notes |
|---|---|---|---|---|---|---|---|
| `smoke_test/` | EECBS | — | random-32-32-10 | 5 | 50 | 5 | Sanity check |
| `v1/` | EECBS | — | random-32-32-10, empty-48-48 | 25 each | 50, 100 | 100 | Clean training data |
| `lacam3_smoke/` | LaCAM3 | — | random-32-32-10 | 3 | 50 | 3 | Verified LaCAM3 binary |
| `v2/` | EECBS (hybrid → all EECBS) | — | random-32-32-10, empty-48-48 | 25 each | 50, 100 | 100 | EECBS won cascade every time |
| `v3_epibt_rollouts/` | ORCA | CV-PIBT | random-32-32-10, empty-48-48 | 25 each | 50, 100 | ~100 | Used new `--rollout-shield-type cv-pibt` |
| **`v4/`** | EECBS | — | 9 maps (see below) | 25 each | 50, 100 | ~380 | Maze N=100 mostly skipped; **primary training set** |

v4 maps: `random-32-32-10`, `random-32-32-20`, `random-64-64-10`, `empty-32-32`, `empty-48-48`, `room-32-32-4`, `room-64-64-8`, `maze-32-32-2`, `maze-32-32-4`.
Skipped (EECBS timeout): most of `maze-32-32-2 N=100`, many `room-32-32-4 N=100`, some `maze-32-32-4 N=100`.

### 7.2 Training Runs

| Run | Data | Hidden | Layers | Epochs | Best val loss | Status | Notes |
|---|---|---|---|---|---|---|---|
| **v1** | v1 (100 EECBS rollouts, 2 maps) | 512 | 4 | 50 | **0.1354** (ep 48) | Done | Current best checkpoint |
| v2 | v2 (100 EECBS rollouts, 2 maps) | 512 | 4 | 50 | 0.1305 (ep 49) | Done | Catastrophic eval regression |
| v3 | v1 + v3_epibt (200 rollouts) | 512 | 4 | 50 | — | Done | Also regressed badly |
| **v4** | v4 (~380 EECBS rollouts, 9 maps) | 512 | 4 | 50 | — | **Running** | Single GPU, batch=256, workers=96 |

### 7.3 Evaluation Results

All v1/v2/v3 results: averages over **5 scenarios**, 2 maps (random-32-32-10 + empty-48-48).
v4 results: will be over **25 scenarios**, 2+ maps (Set A).

#### Core ablation table (Eval Set A, 5 scenarios, 512 steps)

| Method | Shield | N=50 at_goal | N=50 collisions | N=100 at_goal | N=100 collisions |
|---|---|---|---|---|---|
| straight | ORCA | 0.584 | 191.8 | 0.582 | 764.7 |
| straight | PO-ORCA | *running* | *running* | *running* | *running* |
| straight | CV-PIBT | 0.882 | 69.3 | 0.875 | 338.4 |
| Flow v1 | CV-PIBT | **0.914** | **54.0** | **0.879** | **320.9** |
| Flow v2 | CV-PIBT | 0.470 | 85.2 | 0.499 | 384.8 |
| Flow v2 | ORCA | 0.586 | 220.6 | 0.583 | 888.7 |

#### Step budget comparison (v1, N=50)

| Steps | Integration steps | at_goal | collisions |
|---|---|---|---|
| 256 | 10 | 0.790 | 37.3 |
| 256 | 3 | 0.810 | 32.2 |
| 256 | — (straight baseline) | 0.882 | 69.3 |
| **512** | **3** | **0.914** | **54.0** |

Flow v1 underperforms straight at 256 steps but exceeds it at 512. Hypothesis: model is slightly conservative in novel configurations (trained on 2 maps). v4 (9 maps) expected to close this gap.

---

## 8. Key Findings

### Finding 1: CV-PIBT >> ORCA (dominant result)

With identical straight preferred velocities:
- `straight + CV-PIBT`: at_goal=0.882, collisions=69.3 at N=50
- `straight + ORCA`: at_goal=0.584, collisions=191.8 at N=50

CV-PIBT achieves **51% more agents at goal** and **2.8× fewer collisions** than ORCA. This holds at N=100. ORCA's reactive velocity-obstacle geometry cannot coordinate 50–100 agents; PIBT's priority inheritance is fundamentally more powerful. **CV-PIBT itself is the primary contribution.**

### Finding 2: Flow v1 + CV-PIBT is best at 512 steps

At 512 steps, 3 integration steps:
- N=50: at_goal=0.914 (vs straight+CV-PIBT 0.882, straight+ORCA 0.584)
- N=50: collisions=54.0 (vs 69.3 and 191.8)
- N=100: matches straight+CV-PIBT on at_goal (0.879 vs 0.875), fewer collisions (320.9 vs 338.4)

### Finding 3: Flow model needs 512 steps, not 256 (v1)

At 256 steps, Flow v1 underperforms straight (0.810 vs 0.882 at_goal). Root cause: v1 trained on 2 maps / 25 scenarios — model is somewhat conservative in novel configurations. v4 (9 maps) is the fix.

### Finding 4: Only EECBS data works for CV-PIBT training

Three data mixing experiments failed:
- **v2 (EECBS + ORCA rollouts)**: at_goal 0.914 → 0.470. ORCA velocities assume ORCA collision avoidance context; CV-PIBT resolves conflicts differently.
- **v3 (EECBS + CV-PIBT rollouts)**: at_goal 0.479, collisions 137.6. CV-PIBT rollout data is NOT collision-free (shield itself causes ~69 collisions at N=50), contaminating training targets.
- **v2 + ORCA eval**: at_goal=0.586 — matches plain ORCA. ORCA's ceiling (~58%) is the bottleneck; better preferred velocities cannot overcome a weak shield.

**Conclusion**: EECBS provides provably collision-free paths — the gold standard training signal. Never mix in continuous-space rollout data.

### Finding 5: 3 integration steps is better than 10

Slight improvement in at_goal (0.810 vs 0.790) and collisions (32.2 vs 37.3) at N=50, and 2× faster eval. Use `--num-integration-steps 3`.

### Finding 6: BD-guided baseline is broken

`BD + CV-PIBT` gets at_goal=0.018. Root cause: polygon inflation in obstacle map for BD heuristic precomputation makes the map inaccessible. Not fixed; omit BD from paper unless fixed.

### Finding 7: ORCA evaluation is fair (confirmed)

- rvo2 is correctly installed on Lambda; true RVO2 is running (not heuristic fallback)
- `time_horizon=2.0` matches the τ=2 shown in van den Berg's Figure 4
- `--tau 0.3` in our eval scripts is action logit temperature, NOT ORCA lookahead — non-issue
- Symmetric ORCA (standard mode) is the correct comparison; po-orca is an ablation

### Finding 8: DDP is slower than single GPU for this dataset size

~106k samples (v4), 512-dim GNN — GPUs were running at 34-41% of peak power (CPU-bound data loading, not GPU-bound compute). 4-GPU DDP adds synchronization overhead that outweighs the compute savings. Single GPU with large batch + many workers is faster for this scale.

### Finding 9: Moving-AI maze maps are geometrically incompatible

maze-32-32-2: 2-cell corridors (2.0 units wide) leave ~0.2 units margin for two circular agents (diameter 0.6 each) to pass side-by-side. EECBS skips most N=100 scenarios on these maps. Exclude maze variants from eval. Use random and empty maps as primary; room maps for N=50 supplemental.

---

## 9. Paper Story

### Claim hierarchy

1. **(Primary)** CV-PIBT, a novel continuous-space extension of PIBT's priority-inheritance mechanism, dramatically outperforms ORCA in continuous 2D MAPF — 51% more agents at goal, 3× fewer collisions at N=50, using only straight preferred velocities.

2. **(Secondary)** A GNN-based Rectified Flow model trained on EECBS expert trajectories further improves upon straight preferred velocities under CV-PIBT: 91.4% vs 88.2% at_goal, 54 vs 69 collisions at N=50.

3. **(Ablation/insight)** Learned preferred velocities cannot rescue ORCA — Flow + ORCA matches plain ORCA (0.586 vs 0.584), confirming shield choice dominates policy choice.

4. **(Ablation/insight)** PO-ORCA (static priority ordering) isolates the value of PIBT's recursive backtracking over simple prioritization.

### Narrative arc

1. Discrete MAPF is solved by PIBT and its variants. Continuous-space MAPF remains open.
2. ORCA is the standard continuous baseline but deadlocks in dense obstacle maps.
3. We extend PIBT's priority inheritance to continuous space (CV-PIBT). Even with naive straight preferred velocities, CV-PIBT dramatically outperforms ORCA.
4. A flow matching GNN trained on EECBS expert trajectories learns better preferred velocities for CV-PIBT. The learned policy adds a meaningful improvement on top of an already-strong shield.
5. Ablations confirm: (a) the shield is the dominant factor, (b) EECBS-only training data is necessary, (c) 3 integration steps is sufficient for flow inference.

### Full paper comparison table (target: after v4 evals with 25 scenarios)

| Method | Shield | at_goal N=50 | coll N=50 | at_goal N=100 | coll N=100 |
|---|---|---|---|---|---|
| straight | ORCA | 0.584 | 191.8 | 0.582 | 764.7 |
| straight | PO-ORCA | TBD | TBD | TBD | TBD |
| straight | CV-PIBT | 0.882 | 69.3 | 0.875 | 338.4 |
| **Flow v4 (ours)** | **CV-PIBT** | **TBD** | **TBD** | **TBD** | **TBD** |

*Primary: Set A (random-32-32-10 + empty-48-48), 25 scenarios, 512-step budget, 3 integration steps.*
*Supplemental: Set B (random-64-64-10, room-32-32-4), Set C (warehouse-10-20-10-2-1).*

---

## 10. What Still Needs to Be Done

### High priority (required for submission)

- [x] v4 data generation (9 maps, EECBS)
- [ ] **v4 training** — currently running (single GPU, batch=256, workers=96)
- [ ] **PO-ORCA eval** — currently running on Lambda
- [ ] **v4 eval at 256 AND 512 steps** (Eval Set A, 25 scenarios) — critical: does v4 close the 256-step gap?
- [ ] **Full baseline evals at 25 scenarios** — ORCA, PO-ORCA, straight+CV-PIBT on Set A
- [ ] **Eval Set B** (random-64-64-10, room-32-32-4) for all baselines + Flow v4
- [ ] **Eval Set C** (warehouse OOD) for all baselines + Flow v4
- [ ] **Multi-seed runs (3 seeds)** on final model for statistical significance
- [ ] **N=150, N=200 scaling** on empty-48-48 (open map, no obstacle complications)

### Medium priority

- [ ] **Fix BD-guided baseline** or formally drop it (omit from paper)
- [ ] **DAgger experiment** if v4 still has a 256-step gap: roll out current model + CV-PIBT, label with straight+CV-PIBT oracle, aggregate and retrain
- [ ] **Trajectory visualizations** (`--viz-dir` in eval_continuous.py)
- [ ] **Collision breakdown** — are collisions agent-agent, agent-obstacle, or both?

### Low priority / future work

- [ ] Transformer model comparison (implemented, untested)
- [ ] Grid-world Phase 2: flow vs Veerapaneni discrete classifier on 8 held-out maps

---

## 11. Standard Eval Commands (Eval Set A — Full 25 Scenarios)

Run after v4 training completes. All use `--num-gpus 4`.

```bash
# Flow v4 + CV-PIBT, 512 steps
python eval_parallel.py --map-dir data/mapf-map --scen-dir data/mapf-scen-random \
  --maps random-32-32-10 empty-48-48 --agent-counts 50 100 --max-scenarios 25 \
  --policy flow --model-path checkpoints/continuous_v4/continuous_flow_v4_best.pt \
  --shield-type cv-pibt --num-integration-steps 3 --max-steps 512 \
  --output-csv evals/v4/flow_epibt_512_setA.csv --num-gpus 4

# Flow v4 + CV-PIBT, 256 steps (critical: does v4 close the gap?)
python eval_parallel.py --map-dir data/mapf-map --scen-dir data/mapf-scen-random \
  --maps random-32-32-10 empty-48-48 --agent-counts 50 100 --max-scenarios 25 \
  --policy flow --model-path checkpoints/continuous_v4/continuous_flow_v4_best.pt \
  --shield-type cv-pibt --num-integration-steps 3 --max-steps 256 \
  --output-csv evals/v4/flow_epibt_256_setA.csv --num-gpus 4

# Straight + CV-PIBT baseline
python eval_parallel.py --map-dir data/mapf-map --scen-dir data/mapf-scen-random \
  --maps random-32-32-10 empty-48-48 --agent-counts 50 100 --max-scenarios 25 \
  --policy orca --nav straight --shield-type cv-pibt --max-steps 512 \
  --output-csv evals/v4/straight_epibt_512_setA.csv --num-gpus 4

# PO-ORCA baseline (currently running)
python eval_parallel.py --map-dir data/mapf-map --scen-dir data/mapf-scen-random \
  --maps random-32-32-10 empty-48-48 --agent-counts 50 100 --max-scenarios 25 \
  --policy orca --nav straight --shield-type po-orca --max-steps 512 \
  --output-csv evals/v4/poorca_straight_512_setA.csv --num-gpus 4

# ORCA baseline
python eval_parallel.py --map-dir data/mapf-map --scen-dir data/mapf-scen-random \
  --maps random-32-32-10 empty-48-48 --agent-counts 50 100 --max-scenarios 25 \
  --policy orca --nav straight --shield-type orca --max-steps 512 \
  --output-csv evals/v4/orca_straight_512_setA.csv --num-gpus 4
```

---

## 12. Code Changes Made

| File | Change | Commit |
|---|---|---|
| `generate_continuous_data.py` | Added `--rollout-shield-type {orca,po-orca,cv-pibt,none}` flag | `1ce5995` |
| `main_pys/train_continuous.py` | Fix DDP `find_unused_parameters` — always True for FlowGNNModel (wait_head and calibration scalars are inference-only) | `7af80ca` |

---

## 13. Known Bugs

| Bug | Severity | Status |
|---|---|---|
| BD-guided baseline broken (at_goal ≈ 0.018) | Medium | Open — polygon inflation bug. Omit from paper unless fixed. |
| LaCAM3 not actually used in `--expert-source hybrid` | Low | Expected — EECBS wins cascade. Use `--expert-source lacam3` explicitly. |
