# Handoff: grid-world work → Phase 3 (continuous MAPF)

Paste this file (or its path) at the start of a new LLM session. It summarizes **what is done**, **what Phase 2 work is deferred**, and **what to do next** per `PAPER_PLAN.md`. Full roadmap detail lives in `PAPER_PLAN.md`; training/eval mechanics in `PROJECT_CONTEXT_HANDOFF.md` and `README.md`.

---

## 1. Project in one sentence

**Flow-CS-PIBT** replaces a discrete action classifier with **rectified flow** over 2D velocities for multi-agent path finding on **grids**, with CS-PIBT (or LaCAM variants) as the collision shield at inference. The paper plan adds a **continuous-space** extension (Phase 3).

---

## 2. Completed work (through end of Phase 1 + grid training/eval)

### 2.1 Data pipeline (Phase 1.1)

- Ran **`compact_dataset.py`** on `data/preprocessed`: removed downsampled timestep files, re-encoded kept `.pt` to compact format (float16 / int32, map channel stripped per compaction design), ~**547k** files, ~**92 GB** on disk (order-of-magnitude shrink from pre-compaction).
- **`main_pys/dataset_preprocessed.py`** loads compact tensors and upcasts for training (see repo; supports compact vs legacy float32).

### 2.2 Training

- Trained **`wave9_compact`** (run name) on compact preprocessed data via `python -m main_pys.train_flow --preprocessed-dir data/preprocessed ...` (exact flags on training machine).
- Best checkpoint convention: **`checkpoints/large_scale_flow_wave9_compact_best.pt`** (and epoch checkpoints as saved by `train_flow`).

### 2.3 Evaluation (Rishi held-out benchmark)

- **Quick smoke:** `evals/rishi_quick_wave9_best.csv` (`eval_rishi_paper.py --quick`).
- **Full paper-scale grid** (8 maps × up to 25 scenarios × agents 100…1000, same cell count as prior full CSVs): **`evals/rishi_full_wave9_compact_best.csv`**.

**Result vs stored baseline** `evals/wave8_best_full_eval.csv` (paired 1,850 cells):

| Metric | wave9_compact_best | wave8_base (repo CSV) |
|--------|-------------------|------------------------|
| Overall success (all agents at goal) | **64.59%** (1195/1850) | 60.76% (1124/1850) |

Per-map: strong gains on several maps (e.g. random-64, maze, random-32, empty, den312d vs wave8_base); **warehouse** remains very low for both; wave9 slightly lower than wave8_base on warehouse in this pairing.

**Artifacts in repo:** `evals/rishi_full_wave9_compact_best.csv`, `evals/wave8_best_full_eval.csv`, `evals/wave8_heldout_full_eval.csv`, `evals/rishi_quick_wave9_best.csv`.

### 2.4 Operational notes for future runs

- **`train_full.py`** re-extracts zips and wipes/rebuilds `data/` for extraction — **do not** use it when relying on in-place `data/preprocessed` compaction; prefer **`python -m main_pys.train_flow`** with **`--preprocessed-dir data/preprocessed`**.
- **`main_pys/train_flow.py`** defines **`PREPROCESSED_DIRS`** (external path first, then `data/preprocessed`). On machines where the external path exists, pass **`--preprocessed-dir`** explicitly so training uses the intended tree.
- After compaction, **`agent_counts.json`** may be deleted; first training run rebuilds agent-count cache (can be slow once).

---

## 3. Phase 2 status — **paused here**

Phase 2 in `PAPER_PLAN.md` is the **grid-world comparison study** (flow vs discrete classifier, full eval matrix, ablations, figures). **Not completed** before Phase 3 handoff:

| Planned item (PAPER_PLAN §2) | Status |
|------------------------------|--------|
| **`--discrete-only`** in `train_flow.py` + discrete forward path in `generative_model.py` | Not done (or not handed off as complete) |
| Train discrete baseline + clean flow on same data (parallel GPUs) | Not done |
| Full eval sweep vs Rishi paper numbers (flow, discrete, wave8 refs, action head) | Partially: wave9 vs wave8_base CSV compared; discrete/action-head matrix not done |
| Ablations: Euler steps, consensus, τ, model size, wait threshold (`sweep_inference.py`) | Not done |
| Analysis figures (generalization, uncertainty, cost, failure attribution) | Not done |

**Intention:** Stop Phase 2 implementation for now; **next coding focus is Phase 3** unless the user resumes Phase 2 later.

---

## 4. Remaining tasks — Phase 3 (continuous-space MAPF)

From `PAPER_PLAN.md` §3 (prioritize 3.1 → 3.2 → 3.3):

### 4.1 Environment

- **Create** `main_pys/continuous_env.py`: continuous 2D MAPF (e.g. circular agents, speed limits), **reuse grid maps as obstacles**, start with **`empty-48-48`** then add difficulty.

### 4.2 Data + training

- **Stub / pipeline:** `generate_continuous_data.py` (Phase 1.5 in plan) — ORCA (or similar) expert trajectories; store under e.g. `data/continuous/`.
- **Adapt model:** velocity used directly (less discretization); safety via ORCA post-processing or clipping.
- Train **continuous flow** and a **discrete baseline** (e.g. 8/16 direction bins) in parallel if resources allow.

### 4.3 Evaluation

- **Create** `eval_continuous.py`.
- Baselines: ORCA alone, social force, discrete + interpolation, etc.
- Metrics: success, path quality, smoothness, runtime; scalability and trajectory plots.

### 4.4 Code touch points (plan)

- **Modify** `main_pys/model_inputs.py` — continuous positions.
- **Modify** `main_pys/simulator.py` — continuous stepping mode (or parallel eval path).

### 4.5 Phase 1 stub still relevant

- **`generate_continuous_data.py`** may not exist yet; add when starting expert data for Phase 3.

---

## 5. Later: Phase 4 (after Phase 3)

- Multi-seed runs, gap experiments, writing (see `PAPER_PLAN.md` §4).

---

## 6. Science / interpretability warnings (carry forward)

- If training ever merges **held-out map** preprocessed dirs (`--preprocessed-dir a,b`), eval on those maps is **not** pure zero-shot for overlapping scenarios — see `PROJECT_CONTEXT_HANDOFF.md`.
- **Warehouse** and high-density **maze / den312d** remain hard; interpret improvements against shield deadlocks vs policy quality.

---

## 7. Quick command reference

```bash
# Training (preprocessed compact data)
python -m main_pys.train_flow --run-name <name> --preprocessed-dir data/preprocessed

# Full Rishi eval
python eval_rishi_paper.py -m checkpoints/large_scale_flow_wave9_compact_best.pt \
  -o evals/rishi_full_wave9_compact_best.csv

# Quick Rishi eval
python eval_rishi_paper.py -m <ckpt> --quick -o evals/quick.csv
```

---

## 8. Files to read first in a new session

1. `PAPER_PLAN.md` — full phased plan and verification checklist  
2. `PROJECT_CONTEXT_HANDOFF.md` — architecture, held-out map list, eval entry points  
3. This file — **LLM_HANDOFF_PHASE3.md** — status snapshot  
4. `README.md` — install, conda env name, basic simulator CLI  

---

*Snapshot for LLM handoff: Phase 2 paused after wave9 compact full eval; Phase 3 continuous MAPF is the intended next implementation block.*
