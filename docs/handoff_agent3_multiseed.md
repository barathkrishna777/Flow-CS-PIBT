# Agent 3 Handoff: Multi-Seed Training Infrastructure

**Branch:** `barath/continuous_space_epibt`  
**Constraint:** No code runs locally. Push all code changes, then provide Lambda shell commands for the user to execute and paste back.

---

## Context

The flow model (v4b) is trained with hardcoded seed `42` in two places inside `main_pys/train_flow.py`. There is no CLI argument to change the seed. For the paper, we need 3 training runs with different seeds to report mean±std across seeds and establish that results are not seed-sensitive.

The training launcher is `train_full.py`, which calls into `main_pys/train_flow.py`. The checkpoint naming uses a `run_name` suffix, producing files like `large_scale_flow_{run_name}_best.pt`.

---

## Task 1: Add `--seed` argument to `main_pys/train_flow.py`

**File:** `main_pys/train_flow.py`

**Current state (hardcoded seeds):**

Search for these two lines (around line 557 and 597):
```python
generator=torch.Generator().manual_seed(42)
```
```python
seed=42,
```

**Change needed:**

1. Add a `--seed` argument to the argument parser in `train_flow.py`. Find the `argparse` section near the bottom of the file (around line 1047). Add:
   ```python
   parser.add_argument("--seed", type=int, default=42,
                       help="Global random seed for reproducibility")
   ```

2. In the `train()` function signature (line 500), add `seed: int = 42` as a parameter.

3. At the top of the `train()` body, add global seed setting immediately after any imports/setup:
   ```python
   import random
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed_all(seed)
   ```

4. Replace the two hardcoded `42` values with the `seed` parameter:
   - `generator=torch.Generator().manual_seed(seed)` (the DataLoader generator)
   - `seed=seed,` (the WeightedDistributedSampler)

5. When parsing args and calling `train()`, pass `seed=args.seed`.

**Checkpoint naming:** The checkpoint prefix is already `large_scale_flow_{run_name}_` (line 836). By passing `--run-name v4b_seed42` (or seed123, seed456), the checkpoints will be named `large_scale_flow_v4b_seed42_best.pt` etc. No change needed here.

---

## Task 2: Thread `--seed` through `train_full.py`

**File:** `train_full.py`

The `train_full.py` script builds a subprocess command calling `main_pys/train_flow.py`. Find where it adds the `--run-name` flag (around line 115) and also forward `--seed`:

```python
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for training")
```

And in the command builder:
```python
if args.seed != 42:  # always pass it explicitly
    train_cmd.extend(["--seed", str(args.seed)])
```

Or simpler: always forward it unconditionally.

---

## Task 3: Add multi-seed entries to `analyze_evals.py` CATALOG

**File:** `analyze_evals.py`

Add a new eval set `"A_multiseed"` and entries for the 3 seeds. The CSVs don't exist yet — they'll be generated after training completes.

```python
# ── Multi-seed robustness (Set A, 512 steps) ─────────────────────────────
("evals/v4b_seed42/flow_epibt_512_setA.csv",  "Flow v4b seed=42",  "A_multiseed", 512),
("evals/v4b_seed123/flow_epibt_512_setA.csv", "Flow v4b seed=123", "A_multiseed", 512),
("evals/v4b_seed456/flow_epibt_512_setA.csv", "Flow v4b seed=456", "A_multiseed", 512),
```

Also add to `SET_NAMES`:
```python
"A_multiseed": "Set A — Multi-seed robustness (512 steps)",
```

---

## Task 4: Lambda commands

Create `scripts/run_agent3_training.sh` committed to the repo. The user will need to confirm the path to the training data zips on Lambda before running — add a comment noting this.

```bash
#!/usr/bin/env bash
# Agent 3 training commands — run on Lambda from repo root
# Prerequisites:
#   - BASE_DATA: path to base data zip (maps, BDs, etc.)
#   - TRAJ_DATA: path to trajectories zip (the v4b training dataset)
# Adjust these paths to match where your data lives on Lambda:
set -e

BASE_DATA=/path/to/base_data.zip          # EDIT THIS
TRAJ_DATA=/path/to/trajectories_v4b.zip   # EDIT THIS

# ── Seed 42 (replication of v4b baseline) ───────────────────────────────────
python train_full.py \
  --base-data $BASE_DATA \
  --trajectories $TRAJ_DATA \
  --run-name v4b_seed42 \
  --seed 42

# ── Seed 123 ─────────────────────────────────────────────────────────────────
python train_full.py \
  --base-data $BASE_DATA \
  --trajectories $TRAJ_DATA \
  --run-name v4b_seed123 \
  --seed 123

# ── Seed 456 ─────────────────────────────────────────────────────────────────
python train_full.py \
  --base-data $BASE_DATA \
  --trajectories $TRAJ_DATA \
  --run-name v4b_seed456 \
  --seed 456

echo "All 3 training runs complete."
echo "Checkpoints at:"
echo "  checkpoints/continuous_v4b/large_scale_flow_v4b_seed42_best.pt"
echo "  checkpoints/continuous_v4b/large_scale_flow_v4b_seed123_best.pt"
echo "  checkpoints/continuous_v4b/large_scale_flow_v4b_seed456_best.pt"
```

---

## Task 5: Post-training eval commands

After training, create `scripts/run_agent3_evals.sh`:

```bash
#!/usr/bin/env bash
# Evaluate all 3 seeds on Set A (512 steps)
set -e

MAPDIR=data/mapf-map
SCENDIR=data/mapf-scen-random
N=25

for SEED in 42 123 456; do
  python eval_parallel.py \
    --map-dir $MAPDIR --scen-dir $SCENDIR \
    --maps random-32-32-10 empty-48-48 \
    --agent-counts 50 100 --max-scenarios $N \
    --policy flow \
    --model-path checkpoints/continuous_v4b/large_scale_flow_v4b_seed${SEED}_best.pt \
    --shield-type cv-pibt \
    --max-steps 512 \
    --run-name v4b_seed${SEED} \
    --output-csv evals/v4b_seed${SEED}/flow_epibt_512_setA.csv
done

echo "Multi-seed evals complete."
```

---

## What the Results Will Tell Us

Once all 3 seeds are evaluated, report in the paper:

> Across 3 random seeds, Flow v4b + CV-PIBT achieves **mean ± std** at_goal on random-32-32-10 N=100 at 512 steps.

If std < 0.02 (2pp), training is stable and the result is reliable. If std > 0.05, there is meaningful training variance that should be acknowledged.

The `analyze_evals.py` output for `A_multiseed` will show all three rows side by side, making it easy to report the range.
