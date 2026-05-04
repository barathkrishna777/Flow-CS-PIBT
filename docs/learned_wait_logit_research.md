# Learned Wait-Logit Research Handoff

Date: 2026-05-02  
Branch: `barath/learned-wait-logit`  
Local repo: `/Users/barathkrishna/Documents/GitHub/Flow-CS-PIBT`  
Lambda repo: `~/barath/Flow-CS-PIBT`  
Lambda env: `conda activate 3dposehsx_env`

This document records the learned wait-logit work end-to-end: motivation, code changes, training/eval protocol, results, statistical test, and next steps. It is intended as a complete handoff for another LLM or researcher.

## Executive Summary

The original FLOMAP inference path converted predicted continuous velocity into five discrete action scores using dot products with action vectors, then applied a fixed velocity-magnitude wait threshold. A small threshold sweep showed that wait handling matters:

| Wait Threshold | Result |
|---:|---|
| `0.10` | Overall best, `69.47%` |
| `0.25` | Stress-slice best, `55.00%` |
| `0.40`, `0.60` | Much worse |

This motivated a learned wait-logit interface:

```text
logits = [wait_logit, v·right, v·down, v·up, v·left]
```

Movement directions still preserve velocity geometry; only the wait action is learned separately.

The current full result is a statistically significant improvement over the FLOMAP wave9 compact full baseline:

| Eval | Overall | Stress | Non-Stress | Overall Agents At Goal | Stress Agents At Goal | Non-Stress Agents At Goal |
|---|---:|---:|---:|---:|---:|---:|
| Baseline FLOMAP wave9 compact | `1195/1850 = 64.59%` | `146/350 = 41.71%` | `1049/1500 = 69.93%` | `91.93%` | `76.98%` | `95.41%` |
| Learned wait-logit DDP, `bias=-5` | `1222/1850 = 66.05%` | `161/350 = 46.00%` | `1061/1500 = 70.73%` | `93.83%` | `84.36%` | `96.04%` |
| Delta | `+27 wins`, `+1.46 pp` | `+15 wins`, `+4.29 pp` | `+12 wins`, `+0.80 pp` | `+1.90 pp` | `+7.38 pp` | `+0.63 pp` |

Paired comparison over the same 1850 rows:

```text
matched rows: 1850
both success: 1154
neither success: 587
baseline only: 41
learned only: 68
net learned gain: 27
exact McNemar/binomial p = 0.0124036
```

Known SSIL full primary8 from the report was approximately:

```text
SSIL overall: 66.43%
SSIL stress: 51.71%
```

Therefore, learned wait-logit is competitive with SSIL overall but does not yet clearly beat SSIL, and it still trails the reported SSIL stress slice.

## Key Artifacts

Baseline checkpoint:

```text
checkpoints/large_scale_flow_wave9_compact/large_scale_flow_wave9_compact_best.pt
```

Best learned wait checkpoint so far:

```text
large_scale_flow_learned_wait_head_ddp_best.pt
```

Best full learned wait eval:

```text
evals/waitlogit_full_ddp_learned_s1p0_bm5p0.csv
```

Historical baseline full eval:

```text
evals/rishi_full_wave9_compact_best.csv
```

Important commits on `barath/learned-wait-logit`:

```text
6663cf8 Add learned wait logit interface
24e3196 Stabilize learned wait probability sampling
d4aeeea Add learned wait logit calibration flags
64207bd Add DDP support for flow training
d742f37 Add parallel Rishi eval launcher
a6d0cf6 Support resumable parallel eval shards
```

## Code-Level Interface

Files touched for the learned wait-logit path:

```text
main_pys/generative_model.py
main_pys/train_flow.py
main_pys/simulator.py
eval_rishi_paper.py
test_wait_logit_interface.py
analysis_scripts/eval_direct.py
analysis_scripts/eval_direct_sim.py
analysis_scripts/eval_nofix_sim.py
analysis_scripts/diagnose_action_mapping.py
```

### Model

`FlowGNNModel` now includes:

```text
wait_head: hidden_dim -> action_head_dim -> 1
```

It supports:

```python
model(v_t, t, data, return_wait_logit=True)
model(v_t, t, data, return_action_logits=True, return_wait_logit=True)
```

The helper `hybrid_action_logits_from_velocity(predicted_velocity, wait_logit)` builds:

```text
[wait_logit, v·right, v·down, v·up, v·left]
```

where directions are ordered:

```text
right = [0, 1]
down  = [1, 0]
up    = [-1, 0]
left  = [0, -1]
```

The action label order remains:

```text
0 = wait
1 = right
2 = down
3 = up
4 = left
```

### Inference

`main_pys/simulator.py` adds:

```text
--waitMode / --wait-mode threshold|learned
--waitLogitBias / --wait-logit-bias
--waitLogitScale / --wait-logit-scale
```

Default behavior is unchanged:

```text
--wait-mode threshold
```

The learned mode integrates flow velocity as before, then queries the wait head at `t=0.99` and builds hybrid logits:

```text
wait_logit = wait_logit_scale * wait_logit + wait_logit_bias
scores = [wait_logit, v·right, v·down, v·up, v·left]
probs = softmax(scores / tau)
```

We added probability normalization/flooring before sampling preferences. This fixed a crash where learned wait could produce nearly one-hot probabilities and `torch.multinomial(..., replacement=False)` failed after the first sampled action:

```text
RuntimeError: invalid multinomial distribution (sum of probabilities <= 0)
```

### Training

`main_pys/train_flow.py` adds:

```text
--wait-head-loss-weight
--wait-head-only
--wait-head-lr
```

The current wait-head training objective is:

```text
target_wait = (action_y == 0)
loss += wait_head_loss_weight * BCEWithLogitsLoss(wait_logit, target_wait)
```

The best checkpoint used DDP and wait-head-only training. The trunk was frozen, and the new wait head was trained using existing action labels.

### Compatibility

Old checkpoints load with `strict=False`. Missing `wait_head.*` weights are initialized from scratch. Threshold inference and old eval paths remain available and unchanged.

## Smoke Test

Added:

```text
test_wait_logit_interface.py
```

On Lambda:

```text
[PASS] flow output shape -- (3, 2)
[PASS] action head shape -- (3, 5)
[PASS] wait logit shape -- (3,)
[PASS] hybrid logits shape -- (3, 5)
[PASS] hybrid movement logits preserve dot products

Summary: 5 passed, 0 failed, 0 skipped
```

On local Mac without Torch/PyG, it skips cleanly:

```text
[SKIP] wait-logit shape smoke -- missing optional dependency: torch

Summary: 0 passed, 0 failed, 1 skipped
```

## Initial Baseline Verification

Old checkpoint with learned wait disabled:

```bash
CKPT=checkpoints/large_scale_flow_wave9_compact/large_scale_flow_wave9_compact_best.pt

python eval_rishi_paper.py -m "$CKPT" \
  --quick --wait-mode threshold --wait-thresh 0.25 \
  -o evals/waitlogit_preflight_threshold025_quick.csv
```

This confirmed the threshold path still worked.

## First Wait-Head Training Run

Command:

```bash
python -m main_pys.train_flow \
  --run-name learned_wait_head_ft \
  --resume checkpoints/large_scale_flow_wave9_compact/large_scale_flow_wave9_compact_best.pt \
  --wait-head-only \
  --wait-head-loss-weight 1.0 \
  --action-loss-weight 0.0 \
  --discrete-forward-mode x1_t1 \
  --reset-best-val-loss \
  --epochs 1 \
  --no-wandb
```

Training notes:

```text
Device: cuda | AMP: True
Using PREPROCESSED dataset from data/preprocessed
Preprocessed dataset: 1,169,520 valid samples
Train/Val split: 1,111,044 / 58,476
Weighted sampling: ON
head_only: freezing trunk, training 262,657 params
Missing checkpoint keys initialized from scratch:
  wait_head.0.weight
  wait_head.0.bias
  wait_head.3.weight
  wait_head.3.bias
Epoch 11/11: 4341/4341 batches
Train Loss: 0.1343
Val Loss: 0.1341 (best)
Checkpoint: large_scale_flow_learned_wait_head_ft_best.pt
```

## First Quick A/B

Checkpoint:

```text
large_scale_flow_learned_wait_head_ft_best.pt
```

Fixed threshold quick:

```text
evals/waitlogit_quick_fixed025.csv
overall   : success 14/23 = 60.87% | agents_at_goal 94.39%
stress    : success  1/5  = 20.00% | agents_at_goal 86.57%
non_stress: success 13/18 = 72.22% | agents_at_goal 96.56%
```

Raw learned quick initially crashed due to degenerate probabilities. After the probability normalization fix, raw learned quick was:

```text
evals/waitlogit_quick_learned.csv
overall   : success  6/23 = 26.09% | agents_at_goal 95.08%
stress    : success  2/5  = 40.00% | agents_at_goal 91.72%
non_stress: success  4/18 = 22.22% | agents_at_goal 96.01%
```

Interpretation: the learned wait signal helped stress maps but was badly miscalibrated globally. This motivated inference-time calibration sweeps.

## First Bias Sweep

Checkpoint:

```text
large_scale_flow_learned_wait_head_ft_best.pt
```

Scale fixed at `1.0`; swept wait-logit bias:

| CSV | Overall | Stress | Non-Stress | Overall Agents At Goal | Stress Agents At Goal | Non-Stress Agents At Goal |
|---|---:|---:|---:|---:|---:|---:|
| `waitlogit_quick_fixed025.csv` | `14/23 = 60.87%` | `1/5 = 20.00%` | `13/18 = 72.22%` | `94.39%` | `86.57%` | `96.56%` |
| `waitlogit_quick_learned_bias_m0p25.csv` | `7/23 = 30.43%` | `2/5 = 40.00%` | `5/18 = 27.78%` | `95.48%` | `93.40%` | `96.06%` |
| `waitlogit_quick_learned_bias_m0p50.csv` | `5/23 = 21.74%` | `1/5 = 20.00%` | `4/18 = 22.22%` | `95.33%` | `93.12%` | `95.94%` |
| `waitlogit_quick_learned_bias_m0p75.csv` | `7/23 = 30.43%` | `2/5 = 40.00%` | `5/18 = 27.78%` | `95.76%` | `93.70%` | `96.33%` |
| `waitlogit_quick_learned_bias_m1p00.csv` | `7/23 = 30.43%` | `2/5 = 40.00%` | `5/18 = 27.78%` | `96.11%` | `93.60%` | `96.81%` |
| `waitlogit_quick_learned_bias_m1p50.csv` | `8/23 = 34.78%` | `2/5 = 40.00%` | `6/18 = 33.33%` | `96.20%` | `94.18%` | `96.76%` |
| `waitlogit_quick_learned_bias_m2p00.csv` | `9/23 = 39.13%` | `2/5 = 40.00%` | `7/18 = 38.89%` | `96.15%` | `94.67%` | `96.56%` |

More negative bias improved overall, so a wider sweep was run.

## Wider Calibration Sweep

Checkpoint:

```text
large_scale_flow_learned_wait_head_ft_best.pt
```

Results:

| CSV | Overall | Stress | Non-Stress | Overall Agents At Goal | Stress Agents At Goal | Non-Stress Agents At Goal |
|---|---:|---:|---:|---:|---:|---:|
| `waitlogit_quick_learned_s1p0_bm2p50.csv` | `10/23 = 43.48%` | `1/5 = 20.00%` | `9/18 = 50.00%` | `96.18%` | `93.77%` | `96.85%` |
| `waitlogit_quick_learned_s1p0_bm3p00.csv` | `11/23 = 47.83%` | `2/5 = 40.00%` | `9/18 = 50.00%` | `96.19%` | `94.38%` | `96.69%` |
| `waitlogit_quick_learned_s1p0_bm4p00.csv` | `13/23 = 56.52%` | `1/5 = 20.00%` | `12/18 = 66.67%` | `96.22%` | `94.30%` | `96.75%` |
| `waitlogit_quick_learned_s1p0_bm5p00.csv` | `15/23 = 65.22%` | `1/5 = 20.00%` | `14/18 = 77.78%` | `96.10%` | `93.47%` | `96.83%` |

The `scale=1.0`, `bias=-5.0` setting beat fixed threshold on quick:

```text
fixed quick: 14/23
learned b=-5 quick: 15/23
```

## First Mini Eval

Mini protocol:

```text
Rishi8 maps
max_scenario = 5
agents = 100 200 400 600 800
expected rows = 190
stress maps = random-32-32-10, maze-128-128-2
```

Checkpoint:

```text
large_scale_flow_learned_wait_head_ft_best.pt
```

Compared fixed `0.25`, learned `bias=-3.0`, and learned `bias=-5.0`:

| Eval | Overall | Stress | Non-Stress | Overall Agents At Goal | Stress Agents At Goal | Non-Stress Agents At Goal |
|---|---:|---:|---:|---:|---:|---:|
| `waitlogit_mini_fixed025.csv` | `130/190 = 68.42%` | `18/40 = 45.00%` | `112/150 = 74.67%` | `95.12%` | `87.94%` | `97.03%` |
| `waitlogit_mini_learned_s1p0_bm3p0.csv` | `96/190 = 50.53%` | `25/40 = 62.50%` | `71/150 = 47.33%` | `97.23%` | `96.53%` | `97.42%` |
| `waitlogit_mini_learned_s1p0_bm5p0.csv` | `128/190 = 67.37%` | `25/40 = 62.50%` | `103/150 = 68.67%` | `97.31%` | `96.77%` | `97.45%` |

Interpretation: the first learned head gave a large stress improvement but was not yet as good overall.

## 4-GPU DDP Training

DDP support was added later on the branch.

DDP training completed successfully:

```text
4-GPU DDP completed.
Two epochs ran in about 9 minutes total.
Val improved from 0.1345 to 0.1330.
Best checkpoint:
large_scale_flow_learned_wait_head_ddp_best.pt
```

The intended command shape was:

```bash
torchrun --standalone --nproc_per_node=4 -m main_pys.train_flow \
  --run-name learned_wait_head_ddp \
  --resume checkpoints/large_scale_flow_wave9_compact/large_scale_flow_wave9_compact_best.pt \
  --wait-head-only \
  --wait-head-loss-weight 1.0 \
  --action-loss-weight 0.0 \
  --discrete-forward-mode x1_t1 \
  --reset-best-val-loss \
  --epochs 2 \
  --no-wandb
```

## DDP Quick Calibration

Checkpoint:

```text
large_scale_flow_learned_wait_head_ddp_best.pt
```

Results:

| Eval | Overall | Stress | Non-Stress | Overall Agents At Goal | Stress Agents At Goal | Non-Stress Agents At Goal |
|---|---:|---:|---:|---:|---:|---:|
| `waitlogit_quick_fixed025.csv` | `14/23 = 60.87%` | `1/5 = 20.00%` | `13/18 = 72.22%` | `94.39%` | `86.58%` | `96.56%` |
| `waitlogit_quick_ddp_bias_m3p0.csv` | `14/23 = 60.87%` | `2/5 = 40.00%` | `12/18 = 66.67%` | `95.85%` | `92.83%` | `96.69%` |
| `waitlogit_quick_ddp_bias_m4p0.csv` | `16/23 = 69.57%` | `2/5 = 40.00%` | `14/18 = 77.78%` | `96.03%` | `92.75%` | `96.94%` |
| `waitlogit_quick_ddp_bias_m5p0.csv` | `16/23 = 69.57%` | `2/5 = 40.00%` | `14/18 = 77.78%` | `96.05%` | `92.60%` | `97.01%` |
| `waitlogit_quick_ddp_bias_m6p0.csv` | `16/23 = 69.57%` | `2/5 = 40.00%` | `14/18 = 77.78%` | `95.97%` | `92.00%` | `97.08%` |

The DDP head was clearly better on quick than the first 1-epoch head.

## DDP Mini Eval

Checkpoint:

```text
large_scale_flow_learned_wait_head_ddp_best.pt
```

Setting:

```text
--wait-mode learned
--wait-logit-scale 1.0
--wait-logit-bias -5.0
```

Results:

| Eval | Overall | Stress | Non-Stress | Overall Agents At Goal | Stress Agents At Goal | Non-Stress Agents At Goal |
|---|---:|---:|---:|---:|---:|---:|
| `waitlogit_mini_fixed025.csv` | `130/190 = 68.42%` | `18/40 = 45.00%` | `112/150 = 74.67%` | `95.12%` | `87.94%` | `97.03%` |
| `waitlogit_mini_ddp_learned_s1p0_bm5p0.csv` | `134/190 = 70.53%` | `23/40 = 57.50%` | `111/150 = 74.00%` | `97.00%` | `95.58%` | `97.37%` |

This was the first strong mini result:

```text
overall: +4 wins
stress: +5 wins
non-stress: -1 win
agents-at-goal improved, especially stress.
```

## Additional DDP Training

We trained further from the DDP checkpoint:

```bash
torchrun --standalone --nproc_per_node=4 -m main_pys.train_flow \
  --run-name learned_wait_head_ddp_more \
  --resume large_scale_flow_learned_wait_head_ddp_best.pt \
  --wait-head-only \
  --wait-head-loss-weight 1.0 \
  --action-loss-weight 0.0 \
  --discrete-forward-mode x1_t1 \
  --reset-best-val-loss \
  --epochs 3 \
  --no-wandb
```

Checkpoint:

```text
large_scale_flow_learned_wait_head_ddp_more_best.pt
```

Quick calibration:

| Eval | Overall | Stress | Non-Stress | Overall Agents At Goal | Stress Agents At Goal | Non-Stress Agents At Goal |
|---|---:|---:|---:|---:|---:|---:|
| `waitlogit_quick_ddp_more_bias_m4p0.csv` | `14/23 = 60.87%` | `2/5 = 40.00%` | `12/18 = 66.67%` | `95.96%` | `93.23%` | `96.72%` |
| `waitlogit_quick_ddp_more_bias_m5p0.csv` | `14/23 = 60.87%` | `2/5 = 40.00%` | `12/18 = 66.67%` | `95.92%` | `93.05%` | `96.72%` |
| `waitlogit_quick_ddp_more_bias_m6p0.csv` | `14/23 = 60.87%` | `2/5 = 40.00%` | `12/18 = 66.67%` | `96.08%` | `92.83%` | `96.99%` |
| `waitlogit_quick_ddp_more_bias_m7p0.csv` | `16/23 = 69.57%` | `2/5 = 40.00%` | `14/18 = 77.78%` | `95.85%` | `92.00%` | `96.92%` |

The useful bias shifted to `-7.0`, but the quick best did not improve beyond the older DDP checkpoint.

Mini at `bias=-7.0`:

| Eval | Overall | Stress | Non-Stress | Overall Agents At Goal | Stress Agents At Goal | Non-Stress Agents At Goal |
|---|---:|---:|---:|---:|---:|---:|
| `waitlogit_mini_fixed025.csv` | `130/190 = 68.42%` | `18/40 = 45.00%` | `112/150 = 74.67%` | `95.12%` | `87.94%` | `97.03%` |
| `waitlogit_mini_ddp_learned_s1p0_bm5p0.csv` | `134/190 = 70.53%` | `23/40 = 57.50%` | `111/150 = 74.00%` | `97.00%` | `95.58%` | `97.37%` |
| `waitlogit_mini_ddp_more_learned_s1p0_bm7p0.csv` | `128/190 = 67.37%` | `24/40 = 60.00%` | `104/150 = 69.33%` | `97.02%` | `95.37%` | `97.45%` |

Interpretation:

```text
More training improved stress by one row but hurt overall and non-stress too much.
Best checkpoint remains large_scale_flow_learned_wait_head_ddp_best.pt with bias -5.0.
```

## Full Eval

Best candidate:

```text
checkpoint: large_scale_flow_learned_wait_head_ddp_best.pt
scale: 1.0
bias: -5.0
```

Full eval command:

```bash
WAIT_CKPT=large_scale_flow_learned_wait_head_ddp_best.pt

python eval_rishi_paper.py -m "$WAIT_CKPT" \
  --wait-mode learned \
  --wait-logit-scale 1.0 \
  --wait-logit-bias -5.0 \
  -o evals/waitlogit_full_ddp_learned_s1p0_bm5p0.csv
```

The actual full run was executed with a parallel/resumable launcher and completed:

```text
1850/1850 done
Done. Wrote 1850 rows to:
/home/anushree_mattlab/barath/Flow-CS-PIBT/evals/waitlogit_full_ddp_learned_s1p0_bm5p0.csv
```

Full result:

| Eval | Overall | Stress | Non-Stress | Overall Agents At Goal | Stress Agents At Goal | Non-Stress Agents At Goal |
|---|---:|---:|---:|---:|---:|---:|
| `evals/rishi_full_wave9_compact_best.csv` | `1195/1850 = 64.59%` | `146/350 = 41.71%` | `1049/1500 = 69.93%` | `91.93%` | `76.98%` | `95.41%` |
| `evals/waitlogit_full_ddp_learned_s1p0_bm5p0.csv` | `1222/1850 = 66.05%` | `161/350 = 46.00%` | `1061/1500 = 70.73%` | `93.83%` | `84.36%` | `96.04%` |

## Statistical Test

Script used:

```python
import csv
a_path = "evals/rishi_full_wave9_compact_best.csv"
b_path = "evals/waitlogit_full_ddp_learned_s1p0_bm5p0.csv"

def key(r):
    return (r["mapName"], r["scenFile"], int(r["agentNum"]), int(r["seed"]))

def ok(r):
    return r["success"].lower() == "true"

a = {key(r): ok(r) for r in csv.DictReader(open(a_path))}
b = {key(r): ok(r) for r in csv.DictReader(open(b_path))}
ks = sorted(set(a) & set(b))

base_only = sum(a[k] and not b[k] for k in ks)
learned_only = sum(b[k] and not a[k] for k in ks)
both = sum(a[k] and b[k] for k in ks)
neither = sum((not a[k]) and (not b[k]) for k in ks)

print(f"matched rows: {len(ks)}")
print(f"both success: {both}")
print(f"neither success: {neither}")
print(f"baseline only: {base_only}")
print(f"learned only: {learned_only}")
print(f"net learned gain: {learned_only - base_only}")

from scipy.stats import binomtest
p = binomtest(min(base_only, learned_only), base_only + learned_only, 0.5).pvalue
print(f"exact McNemar/binomial p={p:.6g}")
```

Output:

```text
matched rows: 1850
both success: 1154
neither success: 587
baseline only: 41
learned only: 68
net learned gain: 27
exact McNemar/binomial p=0.0124036
```

## Interpretation

The learned wait-logit idea works, but the current training objective is not the final form.

What worked:

- The learned wait head improves FLOMAP full held-out success.
- The gain is statistically significant by paired exact McNemar/binomial test.
- The largest gains are on the intended coordination-stress maps.
- Agents-at-goal improves strongly, especially on stress maps.

What remains limited:

- Overall learned wait-logit is still slightly below the known SSIL full primary8 overall number (`66.05%` vs approximately `66.43%`).
- Stress still trails the known SSIL stress number (`46.00%` vs approximately `51.71%`).
- Manual wait-logit bias `-5.0` was required, indicating calibration mismatch.

Core diagnosis:

```text
Training objective:
  BCE(wait_head, action_y == 0)

Inference objective:
  arg/softmax over [wait_logit, v·right, v·down, v·up, v·left]
```

BCE wait/not-wait logits are not naturally calibrated against velocity dot-product movement logits. The manual bias sweep is effectively compensating for a training/inference mismatch.

## Recommended Next Improvements

The next implementation should train the exact inference interface.

### 1. Hybrid Action CE Loss

Add a hybrid action CE loss:

```text
logits = [calibrated_wait_logit, movement_logits]
movement_logits = velocity_for_logits dot cardinal directions
target = action_y
loss = CE(logits / tau_train, target)
```

For the first version, use teacher velocity `x_1`:

```text
velocity_for_logits = x_1
```

This trains `wait_head` to compete directly against the movement scores used by inference, without changing the pretrained velocity model.

Suggested flags:

```text
--hybrid-action-loss-weight float default 0.0
--hybrid-velocity-source teacher_x1,predicted_x1 default teacher_x1
```

### 2. Learn Calibration Instead of Hand Bias

Add trainable scalar calibration parameters:

```text
wait_logit_scale
wait_logit_bias
movement_logit_scale
```

Initialize:

```text
wait_logit_scale = 1
wait_logit_bias = 0
movement_logit_scale = 1
```

Use:

```text
calibrated_wait = wait_logit_scale * wait_logit + wait_logit_bias
calibrated_movement = movement_logit_scale * movement_logits
hybrid_logits = [calibrated_wait, calibrated_movement]
```

In inference, use model calibration by default, while keeping CLI overrides for ablation:

```text
--wait-logit-scale
--wait-logit-bias
```

### 3. Fine-Tune Trunk Lightly

After head-only hybrid CE works:

- unfreeze last GNN layer/block, or
- unfreeze the whole trunk with a small LR,
- use hybrid CE with low weight to avoid destroying velocity geometry.

### 4. Stress-Weighted Hybrid Loss

Since the gap to SSIL is mostly stress, consider weighting hybrid loss higher for:

```text
random-32-32-10
maze-128-128-2
high-agent-count samples
existing node_weights / congestion proxies
```

### 5. Threshold Prior + Learned Correction

A future interface could combine the useful fixed threshold prior with the learned signal:

```text
final_wait_logit = learned_wait_logit + alpha * (threshold - ||v||)
```

This may preserve non-stress behavior while learning stress exceptions.

## Suggested Prompt For Next Coding Session

Use this with a new Codex chat:

```text
We may be starting in the local Mac repo, not on Lambda.

Local repo path:
~/Documents/GitHub/Flow-CS-PIBT

Lambda repo path:
~/barath/Flow-CS-PIBT

Branch:
barath/learned-wait-logit

Important environment rules:
- If you are in the local Mac repo, implement/edit/test lightweight code only.
- Local Mac may not have Torch/CUDA/data.
- Do not run heavy MAPF eval locally.
- Heavy training/eval commands must be prepared for Lambda only:
  cd ~/barath/Flow-CS-PIBT
  conda activate 3dposehsx_env
- Preserve dirty worktree files; do not revert unrelated changes.

Context:
We implemented a learned wait-logit interface for FLOMAP / Flow-CS-PIBT.

Current architecture/inference:
- FlowGNNModel predicts velocity as before.
- Added wait_head producing one wait logit.
- Movement logits preserve velocity geometry:
  right, down, up, left = v dot cardinal action vectors.
- Learned inference logits are:
  [wait_logit, v·right, v·down, v·up, v·left]
- Simulator/eval flags:
  --wait-mode threshold|learned
  --wait-logit-scale
  --wait-logit-bias
- DDP training support may already exist on the branch.

Current best learned full eval on Lambda:
- checkpoint: large_scale_flow_learned_wait_head_ddp_best.pt
- eval: evals/waitlogit_full_ddp_learned_s1p0_bm5p0.csv
- setting: --wait-mode learned --wait-logit-scale 1.0 --wait-logit-bias -5.0

Full results:
- baseline FLOMAP wave9 compact:
  evals/rishi_full_wave9_compact_best.csv
  overall 1195/1850 = 64.59%
  stress 146/350 = 41.71%
  non-stress 1049/1500 = 69.93%
- learned wait-logit:
  overall 1222/1850 = 66.05%
  stress 161/350 = 46.00%
  non-stress 1061/1500 = 70.73%
- paired test:
  matched rows 1850
  baseline only 41
  learned only 68
  net +27
  exact McNemar/binomial p = 0.0124

Known SSIL full primary8 from report:
- overall about 66.43%
- stress about 51.71%
So learned wait-logit is competitive with SSIL overall but not clearly above it yet, and stress still trails SSIL.

Problem:
Current training only uses BCE:
  wait_head -> action_y == 0
But inference compares that BCE logit against movement dot-product logits:
  [wait_logit, v·right, v·down, v·up, v·left]
Manual calibration bias -5.0 was needed, which indicates the training objective is miscalibrated.

Task:
Implement training improvements to extract more from the learned wait-logit interface.

Primary improvement:
1. Add hybrid action CE loss that trains exactly the inference interface:
   logits = [calibrated_wait_logit, movement_logits]
   movement_logits = velocity_for_logits dot cardinal directions
   target = action_y
2. For wait-head-only training, support teacher movement logits using expert velocity x_1:
   velocity_for_logits = x_1
   This trains wait_head/calibration to compete against exact movement scores without changing the pretrained velocity model.
3. Also support predicted_x1 if clean, but teacher_x1 is first priority.

Calibration improvement:
4. Add trainable scalar calibration parameters to FlowGNNModel:
   wait_logit_scale
   wait_logit_bias
   movement_logit_scale
   Initial behavior:
   wait scale=1, wait bias=0, movement scale=1
5. Hybrid CE should use:
   calibrated_wait = wait_logit_scale * wait_logit + wait_logit_bias
   calibrated_movement = movement_logit_scale * movement_logits
   hybrid_logits = [calibrated_wait, calibrated_movement]
6. In learned inference, use model calibration parameters by default.
   Keep CLI overrides --wait-logit-scale and --wait-logit-bias.
   Add clear semantics:
   - if CLI override is provided, use it
   - otherwise use model calibration params

Training CLI:
Add flags to main_pys/train_flow.py:
- --hybrid-action-loss-weight float default 0.0
- --hybrid-velocity-source choices teacher_x1,predicted_x1 default teacher_x1
- --calibration-only optional
- preserve existing:
  --wait-head-loss-weight
  --wait-head-only
  --action-loss-weight
  DDP support if present

Compatibility:
- Old checkpoints must load with strict=False.
- Missing calibration params from old checkpoints initialize to defaults.
- Existing threshold mode unchanged.
- Existing learned mode with explicit --wait-logit-bias -5.0 should still work.

Suggested Lambda experiments after implementation:

Option A: train wait_head + calibration with hybrid CE from wave9:
cd ~/barath/Flow-CS-PIBT
conda activate 3dposehsx_env
git pull --ff-only

torchrun --standalone --nproc_per_node=4 -m main_pys.train_flow \
  --run-name learned_wait_hybrid_teacher \
  --resume checkpoints/large_scale_flow_wave9_compact/large_scale_flow_wave9_compact_best.pt \
  --wait-head-only \
  --hybrid-action-loss-weight 1.0 \
  --wait-head-loss-weight 0.0 \
  --action-loss-weight 0.0 \
  --hybrid-velocity-source teacher_x1 \
  --discrete-forward-mode x1_t1 \
  --reset-best-val-loss \
  --epochs 2 \
  --no-wandb

Option B: refine from current best:
torchrun --standalone --nproc_per_node=4 -m main_pys.train_flow \
  --run-name learned_wait_hybrid_refine \
  --resume large_scale_flow_learned_wait_head_ddp_best.pt \
  --wait-head-only \
  --hybrid-action-loss-weight 1.0 \
  --wait-head-loss-weight 0.0 \
  --action-loss-weight 0.0 \
  --hybrid-velocity-source teacher_x1 \
  --discrete-forward-mode x1_t1 \
  --reset-best-val-loss \
  --epochs 2 \
  --no-wandb

Validation:
- In local Mac: run syntax checks and smoke tests only.
- If Torch/PyG missing locally, tests may skip; that is okay.
- Do not run heavy local eval.
- Extend test_wait_logit_interface.py to cover hybrid logits and calibration defaults.
- Provide exact Lambda commands for quick calibration and mini evaluation after training.
- Commit and push changes to branch barath/learned-wait-logit.
```

## Reusable Summary Snippets

Slice summary:

```bash
python - <<'PY'
import csv
stress = {"random-32-32-10", "maze-128-128-2"}
files = [
    "evals/rishi_full_wave9_compact_best.csv",
    "evals/waitlogit_full_ddp_learned_s1p0_bm5p0.csv",
]
for path in files:
    rows = list(csv.DictReader(open(path)))
    print("\n" + path)
    for label, pred in [
        ("overall", lambda r: True),
        ("stress", lambda r: r["mapName"] in stress),
        ("non_stress", lambda r: r["mapName"] not in stress),
    ]:
        xs = [r for r in rows if pred(r)]
        ok = sum(r["success"].lower() == "true" for r in xs)
        goal = sum(float(r["num_agents_at_goal"]) / float(r["agentNum"]) for r in xs) / max(len(xs), 1)
        print(f"{label:10s}: success {ok}/{len(xs)} = {100*ok/max(len(xs),1):5.2f}% | agents_at_goal {100*goal:5.2f}%")
PY
```

Paired exact McNemar/binomial:

```bash
python - <<'PY'
import csv
from scipy.stats import binomtest

a_path = "evals/rishi_full_wave9_compact_best.csv"
b_path = "evals/waitlogit_full_ddp_learned_s1p0_bm5p0.csv"

def key(r):
    return (r["mapName"], r["scenFile"], int(r["agentNum"]), int(r["seed"]))

def ok(r):
    return r["success"].lower() == "true"

a = {key(r): ok(r) for r in csv.DictReader(open(a_path))}
b = {key(r): ok(r) for r in csv.DictReader(open(b_path))}
ks = sorted(set(a) & set(b))

base_only = sum(a[k] and not b[k] for k in ks)
learned_only = sum(b[k] and not a[k] for k in ks)
both = sum(a[k] and b[k] for k in ks)
neither = sum((not a[k]) and (not b[k]) for k in ks)
p = binomtest(min(base_only, learned_only), base_only + learned_only, 0.5).pvalue

print(f"matched rows: {len(ks)}")
print(f"both success: {both}")
print(f"neither success: {neither}")
print(f"baseline only: {base_only}")
print(f"learned only: {learned_only}")
print(f"net learned gain: {learned_only - base_only}")
print(f"exact McNemar/binomial p={p:.6g}")
PY
```

