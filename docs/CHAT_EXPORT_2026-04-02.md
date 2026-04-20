# Chat Export

Date: 2026-04-02
Workspace: `/Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT`

This is an abridged Markdown export of the current chat, preserving the key requests, commands, errors, fixes, and outcomes.

---

## 1. Original plan and baseline training

User shared the earlier baseline training command:

```bash
python -m main_pys.train_continuous \
    --data-dir data/continuous_eecbs/raw \
    --map-dir data/mapf-map \
    --policy-type flow \
    --run-name eecbs_7maps \
    --output-dir checkpoints/continuous_eecbs \
    --epochs 50 \
    --batch-size 64 \
    --num-workers 8 \
    --hidden-dim 512 \
    --num-layers 4 \
    --k 4 \
    --m 5 \
    --lr 1e-4 \
    --train-scenario-start 1 --train-scenario-end 17 \
    --val-scenario-start 18 --val-scenario-end 21 \
    --flow-loss-weight 1.0 \
    --action-loss-weight 0.1 \
    --oversample-difficult \
    --seed 42
```

The archived transcript confirmed this was the intended first baseline:

- Train a `flow` model on `continuous_eecbs`
- No shield-aware loss on the first pass
- Train scenarios `1-17`, validation `18-21`
- Reserve `22-25` for held-out evaluation

---

## 2. Early training progress

User reported the first 9 epochs:

```text
Epoch 1: train=0.2551 val=0.2113
Epoch 2: train=0.2109 val=0.2059
Epoch 3: train=0.1989 val=0.1969
Epoch 4: train=0.1903 val=0.1929
Epoch 5: train=0.1830 val=0.1918
Epoch 6: train=0.1785 val=0.1861
Epoch 7: train=0.1738 val=0.1848
Epoch 8: train=0.1707 val=0.1841
Epoch 9: train=0.1665 val=0.1842
```

Assessment:

- Learning looked healthy
- Validation improvements were tapering
- Best visible validation at epoch 8

---

## 3. Why training was slow

User shared `nvidia-smi` and `htop` screenshots from the Lambda.

Key diagnosis:

- GPU utilization was near `0%`
- Several Python worker processes were saturating CPU
- The training loop was CPU/dataloader bound, not GPU bound

Root cause found in code:

- `main_pys/dataset_continuous.py` was constructing graphs on the fly in `__getitem__`
- `main_pys/model_inputs.py` was doing expensive per-sample CPU work:
  - patch extraction
  - all-pairs distance calculations
  - occupancy rasterization
  - neighbor graph construction

Conclusion:

- The GPU was being starved by CPU preprocessing

---

## 4. Storage planning before preprocessing

User provided disk and dataset stats:

```text
Main NVMe free: 507G
External drive free: 900G
data/preprocessed: 88G
data/continuous_eecbs/raw: 129M
Continuous rollout files: 874
Total timestep samples: 986,435
```

Important filesystem note:

- External drive was mounted as `vfat`

Recommendation:

- Do not use the external VFAT drive as the active training cache
- Keep the new continuous preprocessed cache on local NVMe
- Use a compact sharded format

---

## 5. Implemented code changes

Added compact sharded preprocessing support:

- `scripts/preprocess_continuous_shards.py`
- `main_pys/dataset_continuous_preprocessed.py`
- Updated `main_pys/train_continuous.py`

Design:

- Store compact shard files instead of one `.pt` per sample
- Use:
  - `float16` for feature tensors
  - `int32` for `edge_index`
  - `uint8` for `action_label`
- Add `--preprocessed-dir` to `train_continuous.py`

These changes were committed and pushed.

Commits:

- `d3f53af` — `Add compact continuous shard preprocessing`
- `f18e03e` — `Avoid fd exhaustion in continuous preprocessing`

Branch:

- `po-orca-sdf-shield`

---

## 6. First preprocessing attempt and fix

User ran:

```bash
python -m scripts.preprocess_continuous_shards \
    --data-dir data/continuous_eecbs/raw \
    --map-dir data/mapf-map \
    --out data/continuous_eecbs/preprocessed_shards \
    --workers 24 \
    --batch-size 64 \
    --shard-size 2048
```

This failed with:

```text
RuntimeError: unable to open shared memory object ... Too many open files
OSError: [Errno 24] Too many open files
```

Fix applied:

- Set PyTorch multiprocessing sharing strategy to `"file_system"` in:
  - `scripts/preprocess_continuous_shards.py`
  - `main_pys/train_continuous.py`

After pulling the fix, user reran preprocessing with fewer workers:

```bash
python -m scripts.preprocess_continuous_shards \
    --data-dir data/continuous_eecbs/raw \
    --map-dir data/mapf-map \
    --out data/continuous_eecbs/preprocessed_shards \
    --workers 12 \
    --batch-size 64 \
    --shard-size 2048
```

Successful result:

```text
Done. samples=986,435 shards=482 size_gb=49.07 output=data/continuous_eecbs/preprocessed_shards
```

---

## 7. Baseline training stopped

User stopped the original slow baseline run and listed checkpoints:

```text
checkpoints/continuous_eecbs/continuous_flow_eecbs_7maps_best.pt
checkpoints/continuous_eecbs/continuous_flow_eecbs_7maps_epoch_1.pt
...
checkpoints/continuous_eecbs/continuous_flow_eecbs_7maps_epoch_9.pt
```

Noted:

- `best.pt` timestamp matched epoch 8, consistent with best visible validation

---

## 8. First training attempt from preprocessed shards

User launched:

```bash
python -m main_pys.train_continuous \
    --preprocessed-dir data/continuous_eecbs/preprocessed_shards \
    --map-dir data/mapf-map \
    --policy-type flow \
    --run-name eecbs_7maps_preprocessed \
    --output-dir checkpoints/continuous_eecbs \
    --epochs 10 \
    --batch-size 128 \
    --num-workers 12 \
    --hidden-dim 512 \
    --num-layers 4 \
    --k 4 \
    --m 5 \
    --lr 1e-4 \
    --train-scenario-start 1 --train-scenario-end 17 \
    --val-scenario-start 18 --val-scenario-end 21 \
    --flow-loss-weight 1.0 \
    --action-loss-weight 0.1 \
    --oversample-difficult \
    --seed 42 \
    --data-dir data/continuous_eecbs/preprocessed_shards/
```

Observed early progress:

```text
Epoch 1/10: 36/5281 [03:50<2:38:46, 1.82s/it, loss=0.8382]
```

Screenshots still showed:

- GPU 0 near `0%` utilization
- 12 Python worker processes saturated on CPU

Diagnosis:

- Even though graph construction was precomputed, the shards were too large for random-access loading
- `49.07 GB / 482 shards` implied about `~102 MB` per shard
- For random sampling, workers were loading and deserializing large shards to fetch very small samples

Recommendation:

- Stop this run
- Rebuild with much smaller shards, e.g. `--shard-size 256`

Suggested command:

```bash
rm -rf data/continuous_eecbs/preprocessed_shards_256

python -m scripts.preprocess_continuous_shards \
    --data-dir data/continuous_eecbs/raw \
    --map-dir data/mapf-map \
    --out data/continuous_eecbs/preprocessed_shards_256 \
    --workers 12 \
    --batch-size 64 \
    --shard-size 256
```

Suggested retraining command:

```bash
python -m main_pys.train_continuous \
    --preprocessed-dir data/continuous_eecbs/preprocessed_shards_256 \
    --map-dir data/mapf-map \
    --policy-type flow \
    --run-name eecbs_7maps_preprocessed_256 \
    --output-dir checkpoints/continuous_eecbs \
    --epochs 10 \
    --batch-size 128 \
    --num-workers 12 \
    --hidden-dim 512 \
    --num-layers 4 \
    --k 4 \
    --m 5 \
    --lr 1e-4 \
    --train-scenario-start 1 --train-scenario-end 17 \
    --val-scenario-start 18 --val-scenario-end 21 \
    --flow-loss-weight 1.0 \
    --action-loss-weight 0.1 \
    --oversample-difficult \
    --seed 42
```

---

## 9. Git / remote actions

User asked to push the changes to remote.

Actions performed:

- Staged only the intended files
- Left unrelated untracked workspace files untouched
- Pushed to:

```text
origin/po-orca-sdf-shield
```

Latest pushed commits mentioned in chat:

```text
d3f53af  Add compact continuous shard preprocessing
f18e03e  Avoid fd exhaustion in continuous preprocessing
```

---

## 10. Current status at end of export

Current state:

- Compact continuous shard preprocessing exists and works
- Shard cache at size `2048` is still too coarse for efficient random-access training
- Best next step is to rebuild with smaller shards, likely `256`

---

## Files changed during this chat

- `/Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT/main_pys/train_continuous.py`
- `/Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT/main_pys/dataset_continuous_preprocessed.py`
- `/Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT/preprocess_continuous_shards.py`

