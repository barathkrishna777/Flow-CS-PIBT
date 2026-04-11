# Flow-CS-PIBT — context for a new coding session

Paste this file (or its path) at the start of a new Claude Code chat to align on the project, stack, and **current experimental situation** (wave8 evaluation, mixed training data).

---

## 1. What this repository is

**Flow-CS-PIBT** extends the CS-PIBT / SSIL-style MAPF line of work by replacing a **discrete action classifier** with a **Rectified Flow (flow matching)** model that predicts a **2D continuous velocity field**. At inference, that field is integrated (Euler steps), mapped to discrete action preferences (cardinal directions + wait via magnitude threshold), and passed to a **collision shield** (default **CS-PIBT**; alternatives include LaCAM / Real-Time LaCAM in `main_pys/simulator.py`).

**Problem domain:** Multi-Agent Path Finding (MAPF) on **2D grids** — many agents, individual goals, no vertex/edge collisions, evaluated on standard map/scenario formats with Backward Dijkstra (BD) heuristic features.

**Primary references** (see `README.md`): Veerapaneni et al., ICAPS 2024 / ICRA 2025; Real-Time LaCAM (SoCS 2025).

---

## 2. Technical framework (what to touch)

| Piece | Role |
|-------|------|
| `main_pys/generative_model.py` | `FlowGNNModel`: CNN local map patch + PyG **SAGEConv** stack; outputs flow velocity (+ optional auxiliary 5-way action head). |
| `main_pys/train_flow.py` | Rectified flow loss + auxiliary CE; AMP; checkpointing `large_scale_flow_{run_name}_epoch_*.pt` and `*_best.pt`; **preprocessed** or on-the-fly dataset. |
| `main_pys/dataset.py` | On-the-fly loading from `data/flow_training_data_multi/*.npz`. |
| `main_pys/dataset_preprocessed.py` | Loads pre-built `.pt` graphs; supports **multiple directories** (comma-separated) merged into one file list. |
| `main_pys/model_inputs.py` | Graph construction, BD features, k-hop / m-neighbor logic, normalization. |
| `main_pys/simulator.py` | Loads checkpoint, runs flow integration + shield, writes CSV metrics. |
| `preprocess_dataset.py` | One-time CPU-heavy pipeline → `.pt` per sample for fast training. |
| `train_full.py` | Extracts `data.zip` + trajectory zip into `data/`, then spawns `python -m main_pys.train_flow` (does **not** pass `--preprocessed-dir` by default — training uses whatever `train_flow` auto-detects). |

**Conda env:** `environment.yml` → env name `mlmapf` (per README).

---

## 3. Data: “base” vs held-out test maps (paper protocol)

**Held-out test set (8 maps)** — same names in `generate_flow_data_multi.py` and `eval_rishi_paper.py`:

- `Paris_1_256`, `empty-48-48`, `maze-128-128-2`, `random-64-64-10`, `random-32-32-10`, `warehouse-10-20-10-2-1`, `den312d`, `den520d`

**Canonical expert data generation** (`generate_flow_data_multi.py`):

- **BD heuristics** are still built for held-out maps (needed for eval).
- **EECBS trajectories** for flow training are **skipped** for maps in `HELD_OUT_TEST` (lines ~185–189): those maps are intended to stay **out of training trajectories** for strict generalization experiments.

**Omitted maps** (no processing): `brc202d`, `orz900`, `maze-128-128-1`, `maze-128-128-10` (`OMITTED_MAPS`).

---

## 4. Current situation (wave8 + mixed training data)

**Naming:** Training uses `--run-name` → checkpoints like `large_scale_flow_<run_name>_epoch_N.pt` and `large_scale_flow_<run_name>_best.pt`. The **wave8** line includes a checkpoint referred to in-repo as e.g. `large_scale_flow_wave8_base_best.pt` (see `eval_rishi_paper.py` docstring example).

**Training data mix (what you described):**

- **Base data:** Standard large-scale preprocessed pool (e.g. `data/preprocessed` and/or external path in `PREPROCESSED_DIRS` inside `train_flow.py`).
- **Additional held-out-map data:** Some **preprocessed samples from the 8 held-out maps** have been included in training (e.g. a second directory such as `data/preprocessed_heldout` merged via `--preprocessed-dir data/preprocessed,data/preprocessed_heldout` — this pattern is explicitly mentioned in `train_flow.py`’s `--preprocessed-dir` help text).

**Implication for science / eval interpretation:**  
`eval_rishi_paper.py` still runs the **same 8 maps** as the paper-style benchmark, but **performance on scenarios that overlap with what was added to training** is **not** a pure zero-shot test for those map/scenario/agent slices. Keep this distinction clear when comparing to prior waves trained only on non-held-out trajectories.

**Current focus:** **Evaluating the wave8 model** (e.g. best checkpoint) on the held-out benchmark and/or smaller smoke evals, and iterating on **next tasks** (training changes, data ablations, inference hyperparameters, analysis).

---

## 5. Evaluation entry points (useful commands)

**Full 8-map “Rishi paper” held-out sweep** (many runs: 8 maps × up to 25 scenarios × agent ladder 100…1000):

```bash
python eval_rishi_paper.py -m large_scale_flow_wave8_base_best.pt \
  --output checkpoints_and_evaluations/eval_rishi_wave8_base.csv
```

Flags: `--quick` (smoke), `--maps`, `--agents`, `--max-scenario`, integration/consensus/tau aligned with `eval_full` / paper-style defaults in the script.

**Shorter multi-map eval** (5 maps, sweeps Euler steps): `eval_full.py`  
**Legacy small batch** (3 maps, few agents): `run_experiments.py`  
**Wave4-style wrapper:** `eval_wave4.py` (pattern for epoch/best eval on a fixed small map set)

**Simulator directly** (single run): see `README.md` (`python -m main_pys.simulator ...`).

Typical inference knobs: `--numIntegrationSteps`, `--numConsensusSamples`, `--tau`, `--waitThreshold`, `--timeLimit`, `--maxSteps` (e.g. `3x`), `--hiddenDim`, `--numLayers` (must match checkpoint architecture).

---

## 6. Training entry points

```bash
# After zips are in place (see README)
python train_full.py --base-data /path/to/data.zip --trajectories /path/to/massive_flow_dataset_large_scale.zip

# Direct training with preprocessed + run name + merged dirs
python -m main_pys.train_flow --run-name wave8_base \
  --preprocessed-dir data/preprocessed,data/preprocessed_heldout
```

`train_flow` auto-picks first existing path in `PREPROCESSED_DIRS` if `--preprocessed-dir` is omitted. Default **val_split=0.05**, weighted sampling on preprocessed data unless `--no-weighted-sampling`.

---

## 7. Repo layout (quick)

```
main_pys/          # model, train, sim, datasets, model_inputs
analysis_scripts/  # overfit tests, diagnostics
train_full.py
eval_rishi_paper.py
eval_full.py
preprocess_dataset.py
generate_flow_data_multi.py   # EECBS + SG velocities; skips traj for HELD_OUT_TEST
data/                         # maps, scen-random, bd_npzs/large_scale, flow_training_data_multi, preprocessed*
```

---

## 8. Suggested next-task checklist for the assistant

1. Confirm **exact** checkpoint path and `--run-name` used for wave8.  
2. Confirm **which preprocessed directories** and **approximate sample counts** from held-out maps were merged.  
3. When reporting `eval_rishi_paper.py` results, separate or annotate any **train/eval overlap** if scenario-level leakage is possible.  
4. Match **hidden_dim / num_layers** at eval to the trained checkpoint.  
5. Use `README.md` for install and citation blocks.

---

*Generated to onboard a new Claude Code session; revise the “wave8” and paths section as your local artifacts evolve.*
