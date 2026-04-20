# Flow vs SSIL Ablation Plan

This document is a handoff plan for improving and evaluating the Flow-CS-PIBT
model against Rishi Veerapaneni's SSIL classifier on the Rishi-paper held-out
grid-world benchmark.

The goal is to determine whether the flow model is intrinsically worse, or
whether the current training objective and inference interface are misaligned
with CS-PIBT's discrete action-selection loop.

## Current Result

Full Rishi-paper eval, 1850 rows:

| Model | Rows | All-agents success | Mean agents at goal | Avg runtime |
| --- | ---: | ---: | ---: | ---: |
| Flow best | 1850 | 64.05% | 92.32% | 31.34s |
| SSIL classifier | 1850 | 66.43% | 95.69% | 18.96s |

Per-map success:

| Map | Flow best | SSIL classifier | Gap |
| --- | ---: | ---: | ---: |
| Paris_1_256 | 100.00% | 100.00% | 0.00 pp |
| den312d | 32.80% | 32.00% | Flow +0.80 pp |
| den520d | 99.60% | 99.60% | 0.00 pp |
| empty-48-48 | 97.60% | 100.00% | SSIL +2.40 pp |
| maze-128-128-2 | 28.00% | 40.00% | SSIL +12.00 pp |
| random-32-32-10 | 68.00% | 81.00% | SSIL +13.00 pp |
| random-64-64-10 | 86.40% | 85.20% | Flow +1.20 pp |
| warehouse-10-20-10-2-1 | 2.40% | 2.40% | 0.00 pp |

Main diagnosis:

- Flow is competitive on some maps, especially `den312d` and
  `random-64-64-10`.
- The gap mostly comes from `maze-128-128-2`, `random-32-32-10`, and a smaller
  gap on `empty-48-48`.
- Rishi's classifier is directly trained to output 5 discrete action
  probabilities, while the current flow model learns continuous velocities and
  then converts them into discrete action probabilities.
- The most likely bottleneck is the flow-vector-to-action interface and weak or
  noisy discrete action supervision.

## Important Files

- `main_pys/simulator.py`
- `main_pys/generative_model.py`
- `main_pys/train_flow.py`
- `scripts/preprocess_dataset.py`
- `scripts/eval_rishi_paper.py`
- `analysis_scripts/summarize_grid_eval.py`
- `analysis_scripts/compare_grid_1v1.py`

Important checkpoints:

```text
checkpoints/large_scale_flow_wave8_heldout_best.pt
data/model/ssil_model.pt
```

Important CSVs:

```text
evals/rishi_full_flow_best.csv
evals/rishi_full_ssil_classifier.csv
```

## Fair Evaluation Rules

For the main comparison to Rishi, keep the benchmark protocol fixed:

```text
same 8 held-out maps
same random scenarios
same agent counts
same CS-PIBT shield
same max steps
same time limit
```

Do not change these for the headline table:

```bash
--max-steps-multiplier 3x
--time-limit 120
--shieldType CS-PIBT
--max-scenario 25
--agents 100 200 300 400 500 600 700 800 900 1000
```

It is fair to tune model inference settings, as long as the tuned result is
reported separately and the tuning procedure is disclosed.

Fair knobs to tune:

| Setting | Reason |
| --- | --- |
| `--wait-thresh` | Converts low-magnitude flow velocity into wait. |
| `--tau` | Controls action softmax temperature. |
| `--consensus` | Multiple flow samples/votes. |
| `--num-integration-steps` | Flow ODE integration accuracy vs runtime. |
| `flow_action_head` vs `flow` | Tests direct action logits from the same model. |

Do not use per-map tuned settings for the main number unless clearly labeled as
diagnostic or oracle tuning.

Recommended final table structure:

| Method | Description |
| --- | --- |
| SSIL classifier | Rishi checkpoint, discrete classifier |
| Flow default | Current flow-vector inference |
| Flow action head | Same flow checkpoint, direct action logits |
| Flow tuned-global | One globally selected inference setting |

## Rishi Paper Takeaways

From the paper's training data section:

- The test set is:
  - `Paris_1_256`
  - `empty-48-48`
  - `maze-128-128-2`
  - `random-64-64-10`
  - `random-32-32-10`
  - `warehouse-10-20-10-2-1`
  - `den312d`
  - `den520d`
- All other Moving AI maps are used for training.
- Instances where EECBS times out are excluded.
- Each MAPF solution timestep is treated as an independent training example.
- The label for each agent is the next discrete action it took.
- About 50% of labels are wait actions.
- They did not throw away wait labels or down-weight wait labels.
- They tried DAgger, but did not find major gains from it.

From the paper's architecture section:

- Model is a SageConv GNN.
- Nodes are agents.
- Edges connect agents within field of view.
- No edge features are used.
- Node inputs:
  - 3 local `D x D` images centered at the agent:
    - obstacle map
    - normalized BD heuristic
    - agent occupancy
  - 5-dimensional binary vector indicating which actions minimize the BD
    heuristic
- Image shape is `(3, D=9, D=9)`.
- CNN encoder uses `(3, 3, 3)` kernel, stride 1, no padding.
- Flatten image embedding, concatenate 5-action BD vector.
- Linear projection to 128-dimensional node embedding.
- Separate linear projection to 128-dimensional message vector.
- 3 SageConv layers.
- Layer norms after SageConv layers.
- LeakyReLU activations.
- Final linear layer, dropout 0.25, output 5 action probabilities.
- They found SageConv was the most critical component; changing number of
  layers, sizes, or activations did not significantly change success rate.

Implication for us:

- Rishi's training and architecture are tightly aligned with CS-PIBT's final
  discrete action ranking problem.
- Our flow model should not treat the action head as merely auxiliary.
- The likely best path is a hybrid model:

```text
shared SageConv encoder
  -> flow head for continuous velocity
  -> action head for CS-PIBT action probabilities
```

Training:

```text
loss = flow_loss + lambda_action * exact_discrete_action_CE
```

Evaluation:

```text
use action head for CS-PIBT
optionally use flow vector as an extra prior or diagnostic baseline
```

## Phase 0: Baseline Checks

Run on Lambda:

```bash
git branch --show-current
git status --short
ls -lh evals/rishi_full_flow_best.csv evals/rishi_full_ssil_classifier.csv
```

Regenerate the current comparison:

```bash
python -m analysis_scripts.compare_grid_1v1 \
  evals/rishi_full_flow_best.csv \
  evals/rishi_full_ssil_classifier.csv \
  --labels "Flow best" "SSIL classifier" \
  --out-dir evals/rishi_1v1
```

Expected output:

```text
evals/rishi_1v1/flow_best_vs_ssil_classifier.md
evals/rishi_1v1/flow_best_vs_ssil_classifier.png
```

## Phase 1: No-Retrain Ablations

Do these before any retraining.

### 1A. Add Or Verify Action-Head Inference

Check whether `main_pys/simulator.py` already supports direct action-head
inference for flow models. Search for:

```bash
rg -n "useActionHead|return_action_logits|policyType|runNNOnState" main_pys/simulator.py scripts/eval_rishi_paper.py main_pys/generative_model.py
```

Target interface:

```bash
--policy-type flow_action_head
```

Desired behavior:

- `--policy-type flow` uses the current flow-vector-to-action path.
- `--policy-type classifier` uses Rishi's SSIL classifier.
- `--policy-type flow_action_head` loads the flow model but uses its direct
  action logits/probabilities for CS-PIBT.

Implementation recommendation:

- Add `flow_action_head` to `scripts/eval_rishi_paper.py --policy-type` choices.
- When `policy_type == flow_action_head`, pass simulator args:

```text
--policyType=flow
--useActionHead=True
```

- In `main_pys/simulator.py`, ensure `args.useActionHead` calls the flow model
  with `return_action_logits=True`, softmaxes the action logits, and passes the
  resulting 5-action probabilities to CS-PIBT.

Run quick evals:

```bash
mkdir -p evals/ablations logs

python -m scripts.eval_rishi_paper \
  -m checkpoints/large_scale_flow_wave8_heldout_best.pt \
  -o evals/ablations/quick_flow_vector.csv \
  --quick \
  --policy-type flow

python -m scripts.eval_rishi_paper \
  -m checkpoints/large_scale_flow_wave8_heldout_best.pt \
  -o evals/ablations/quick_flow_action_head.csv \
  --quick \
  --policy-type flow_action_head

python -m scripts.eval_rishi_paper \
  -m data/model/ssil_model.pt \
  -o evals/ablations/quick_ssil_classifier.csv \
  --quick \
  --policy-type classifier
```

Summarize:

```bash
python -m analysis_scripts.summarize_grid_eval \
  evals/ablations/quick_flow_vector.csv \
  evals/ablations/quick_flow_action_head.csv \
  evals/ablations/quick_ssil_classifier.csv \
  --labels flow_vector flow_action_head ssil_classifier
```

Decision:

- If `flow_action_head` beats flow-vector inference, run a medium/full eval.
- If it is worse, action supervision is probably too weak or incorrectly
  labeled.

Full action-head eval:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m scripts.eval_rishi_paper \
  -m checkpoints/large_scale_flow_wave8_heldout_best.pt \
  -o evals/rishi_full_flow_action_head.csv \
  --policy-type flow_action_head \
  > logs/rishi_full_flow_action_head.log 2>&1 &
```

### 1B. Inference Parameter Sweep

Do a staged sweep, not a full cartesian product.

Weak maps:

```text
maze-128-128-2
random-32-32-10
empty-48-48
```

Use quick eval first.

#### Sweep wait threshold

```bash
mkdir -p evals/ablations logs

for wt in 0.10 0.15 0.20 0.25 0.30 0.35; do
  python -m scripts.eval_rishi_paper \
    -m checkpoints/large_scale_flow_wave8_heldout_best.pt \
    -o evals/ablations/quick_flow_wait_${wt}.csv \
    --quick \
    --maps maze-128-128-2 random-32-32-10 empty-48-48 \
    --policy-type flow \
    --wait-thresh ${wt}
done
```

Summarize:

```bash
python -m analysis_scripts.summarize_grid_eval \
  evals/ablations/quick_flow_wait_0.10.csv \
  evals/ablations/quick_flow_wait_0.15.csv \
  evals/ablations/quick_flow_wait_0.20.csv \
  evals/ablations/quick_flow_wait_0.25.csv \
  evals/ablations/quick_flow_wait_0.30.csv \
  evals/ablations/quick_flow_wait_0.35.csv \
  --labels wait_0.10 wait_0.15 wait_0.20 wait_0.25 wait_0.30 wait_0.35
```

Pick the best wait threshold by:

1. All-agents success.
2. Mean agents at goal.
3. Runtime as tie-breaker.

#### Sweep tau

Replace `<BEST_WAIT>`:

```bash
for tau in 0.10 0.20 0.30 0.50 0.75; do
  python -m scripts.eval_rishi_paper \
    -m checkpoints/large_scale_flow_wave8_heldout_best.pt \
    -o evals/ablations/quick_flow_wait_<BEST_WAIT>_tau_${tau}.csv \
    --quick \
    --maps maze-128-128-2 random-32-32-10 empty-48-48 \
    --policy-type flow \
    --wait-thresh <BEST_WAIT> \
    --tau ${tau}
done
```

#### Sweep consensus

Replace `<BEST_WAIT>` and `<BEST_TAU>`:

```bash
for c in 1 3 5 7; do
  python -m scripts.eval_rishi_paper \
    -m checkpoints/large_scale_flow_wave8_heldout_best.pt \
    -o evals/ablations/quick_flow_wait_<BEST_WAIT>_tau_<BEST_TAU>_consensus_${c}.csv \
    --quick \
    --maps maze-128-128-2 random-32-32-10 empty-48-48 \
    --policy-type flow \
    --wait-thresh <BEST_WAIT> \
    --tau <BEST_TAU> \
    --consensus ${c}
done
```

#### Sweep integration steps

Replace `<BEST_WAIT>`, `<BEST_TAU>`, and `<BEST_C>`:

```bash
for s in 1 2 3 5 8; do
  python -m scripts.eval_rishi_paper \
    -m checkpoints/large_scale_flow_wave8_heldout_best.pt \
    -o evals/ablations/quick_flow_wait_<BEST_WAIT>_tau_<BEST_TAU>_consensus_<BEST_C>_steps_${s}.csv \
    --quick \
    --maps maze-128-128-2 random-32-32-10 empty-48-48 \
    --policy-type flow \
    --wait-thresh <BEST_WAIT> \
    --tau <BEST_TAU> \
    --consensus <BEST_C> \
    --num-integration-steps ${s}
done
```

### 1C. Medium Eval For Best Inference Setting

Once a candidate global setting is selected, run medium eval on all 8 maps with
5 scenarios:

```bash
python -m scripts.eval_rishi_paper \
  -m checkpoints/large_scale_flow_wave8_heldout_best.pt \
  -o evals/ablations/medium_flow_tuned_inference.csv \
  --max-scenario 5 \
  --policy-type flow \
  --wait-thresh <BEST_WAIT> \
  --tau <BEST_TAU> \
  --consensus <BEST_C> \
  --num-integration-steps <BEST_STEPS>
```

Matching SSIL medium eval:

```bash
python -m scripts.eval_rishi_paper \
  -m data/model/ssil_model.pt \
  -o evals/ablations/medium_ssil_classifier.csv \
  --max-scenario 5 \
  --policy-type classifier
```

Summarize:

```bash
python -m analysis_scripts.summarize_grid_eval \
  evals/ablations/medium_flow_tuned_inference.csv \
  evals/ablations/medium_ssil_classifier.csv \
  --labels flow_tuned ssil_classifier
```

If promising, run full tuned eval:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -m scripts.eval_rishi_paper \
  -m checkpoints/large_scale_flow_wave8_heldout_best.pt \
  -o evals/rishi_full_flow_tuned_global.csv \
  --policy-type flow \
  --wait-thresh <BEST_WAIT> \
  --tau <BEST_TAU> \
  --consensus <BEST_C> \
  --num-integration-steps <BEST_STEPS> \
  > logs/rishi_full_flow_tuned_global.log 2>&1 &
```

## Phase 2: Light Retraining Ablations

Only do this if no-retrain inference changes do not close the gap.

### 2A. Exact Discrete Action Labels

Rishi trains on exact next discrete action labels. Our current action labels may
be derived from smoothed expert velocities, which can blur turns and waits.

Modify `scripts/preprocess_dataset.py`:

- For each sample at timestep `t`, compute:

```text
delta = discrete_positions[:, t + 1] - discrete_positions[:, t]
```

- For the final timestep, use wait.
- Store:

```python
graph_data.action_y = torch.tensor(labels, dtype=torch.long)
```

Label mapping:

| Delta `[row, col]` | Action |
| --- | --- |
| `[0, 0]` | `0 = wait` |
| `[0, +1]` | `1 = right` |
| `[+1, 0]` | `2 = down` |
| `[-1, 0]` | `3 = up` |
| `[0, -1]` | `4 = left` |

Modify `main_pys/train_flow.py`:

```python
if hasattr(batch, "action_y"):
    expert_actions = batch.action_y.to(device)
else:
    expert_actions = velocity_to_action_labels(x_1, device)
```

### 2B. Separate Weighting For Flow Loss And Action Loss

Rishi did not down-weight wait labels. Our `node_weights` may be useful for flow
MSE but harmful for action CE.

Add an option:

```bash
--unweighted-action-loss
```

Loss variants:

| Variant | Flow loss | Action loss |
| --- | --- | --- |
| current | weighted | weighted |
| action_unweighted | weighted | unweighted |
| all_unweighted | unweighted | unweighted |

Start with:

```text
flow loss weighted
action loss unweighted
```

### 2C. Make Action Loss Weight Configurable

In `main_pys/train_flow.py`, add:

```python
parser.add_argument("--action-loss-weight", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=10)
```

Pass both into `train(...)`.

Keep `--quick` overriding epochs to `1`.

Recommended short retrains:

```bash
python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed_exact_actions \
  --run-name exact_action_w1_e3 \
  --action-loss-weight 1.0 \
  --unweighted-action-loss \
  --epochs 3

python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed_exact_actions \
  --run-name exact_action_w2_e3 \
  --action-loss-weight 2.0 \
  --unweighted-action-loss \
  --epochs 3

python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed_exact_actions \
  --run-name exact_action_w4_e3 \
  --action-loss-weight 4.0 \
  --unweighted-action-loss \
  --epochs 3
```

Evaluate each with both vector inference and action-head inference:

```bash
python -m scripts.eval_rishi_paper \
  -m large_scale_flow_exact_action_w2_e3_best.pt \
  -o evals/ablations/quick_exact_action_w2_e3_vector.csv \
  --quick \
  --policy-type flow

python -m scripts.eval_rishi_paper \
  -m large_scale_flow_exact_action_w2_e3_best.pt \
  -o evals/ablations/quick_exact_action_w2_e3_head.csv \
  --quick \
  --policy-type flow_action_head
```

Summarize all quick retrains:

```bash
python -m analysis_scripts.summarize_grid_eval \
  evals/ablations/quick_exact_action_w1_e3_vector.csv \
  evals/ablations/quick_exact_action_w2_e3_vector.csv \
  evals/ablations/quick_exact_action_w4_e3_vector.csv \
  evals/ablations/quick_exact_action_w1_e3_head.csv \
  evals/ablations/quick_exact_action_w2_e3_head.csv \
  evals/ablations/quick_exact_action_w4_e3_head.csv \
  evals/ablations/quick_ssil_classifier.csv \
  --labels w1_vec w2_vec w4_vec w1_head w2_head w4_head ssil
```

Promote the best one to medium eval, then full eval.

## Phase 3: Rishi-Like Classifier Sanity Check

Before trying more complex flow ideas, reproduce a Rishi-like classifier in our
pipeline.

Purpose:

- If our local Rishi-like classifier approaches SSIL, our dataset/input/eval
  pipeline is good.
- If it underperforms badly, our graph construction, data split, or labels
  differ from Rishi's implementation.

Architecture target:

| Component | Setting |
| --- | --- |
| Input image | `3 x 9 x 9` |
| BD-best-action vector | 5 |
| GNN | 3 SageConv layers |
| Hidden size | 128 |
| Dropout | 0.25 |
| Output | 5 action logits |
| Loss | unweighted cross entropy |
| Labels | exact next discrete action |

Implementation options:

- Add `--policy-head-only` mode to `main_pys/train_flow.py`.
- Or create `analysis_scripts/train_rishi_like_classifier.py`.
- Add eval policy type `local_classifier` if needed.

Suggested command:

```bash
python -m analysis_scripts.train_rishi_like_classifier \
  --preprocessed-dir data/preprocessed_exact_actions \
  --run-name rishi_like_classifier \
  --hidden-dim 128 \
  --num-layers 3 \
  --dropout 0.25 \
  --epochs 10
```

Evaluate:

```bash
python -m scripts.eval_rishi_paper \
  -m checkpoints/rishi_like_classifier_best.pt \
  -o evals/ablations/quick_rishi_like_classifier.csv \
  --quick \
  --policy-type local_classifier
```

## Phase 4: Smaller Hybrid Flow Models

Rishi found SageConv mattered more than size. Our 6-layer, 1024-hidden model may
be slower and not better for discrete action quality.

Try smaller hybrid flow models with exact action labels:

```bash
python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed_exact_actions \
  --run-name flow_small_exact_w2_e3 \
  --hidden-dim 128 \
  --num-layers 3 \
  --action-loss-weight 2.0 \
  --unweighted-action-loss \
  --epochs 3

python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed_exact_actions \
  --run-name flow_mid_exact_w2_e3 \
  --hidden-dim 256 \
  --num-layers 3 \
  --action-loss-weight 2.0 \
  --unweighted-action-loss \
  --epochs 3
```

Evaluate both with vector and action-head inference.

## Phase 5: SSIL Distillation

Only attempt this after exact-action retraining.

Goal:

- Use Rishi's classifier as a teacher for action probabilities.
- Preserve flow training, but make action logits sharper and more SSIL-like.

Training loss:

```text
loss = flow_loss
     + action_weight * exact_action_CE
     + distill_weight * KL(student_action_logits, teacher_action_logits)
```

Suggested CLI:

```bash
--teacher-model data/model/ssil_model.pt
--distill-weight 0.5
--distill-temperature 2.0
```

Suggested command:

```bash
python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed_exact_actions \
  --run-name distill_ssil_w05_e3 \
  --action-loss-weight 1.0 \
  --unweighted-action-loss \
  --teacher-model data/model/ssil_model.pt \
  --distill-weight 0.5 \
  --distill-temperature 2.0 \
  --epochs 3
```

Only do this if the teacher can consume the same PyG batch cleanly. Do not
spend too much time here before validating exact-action retraining.

## Phase 6: DAgger, Last Resort

Rishi tried DAgger and did not see major gains, so do this last.

If attempting it:

- Focus only on weak maps:
  - `maze-128-128-2`
  - `random-32-32-10`
- Run current flow policy inside CS-PIBT.
- For each rollout, sample 10 timesteps.
- Feed those current locations as starts into EECBS.
- Add resulting exact next-action labels to training data.
- Retrain the hybrid model.

Do not make this the first improvement path.

## Reporting

For each candidate that reaches medium or full eval:

```bash
python -m analysis_scripts.summarize_grid_eval \
  <flow_candidate_csv> \
  evals/rishi_full_ssil_classifier.csv \
  --labels <flow_label> ssil_classifier
```

For final candidates:

```bash
python -m analysis_scripts.compare_grid_1v1 \
  <flow_candidate_csv> \
  evals/rishi_full_ssil_classifier.csv \
  --labels "<Flow label>" "SSIL classifier" \
  --out-dir evals/<flow_label>_vs_ssil
```

Final report should include:

1. Best inference-only result.
2. Whether action-head inference helped.
3. Best retrained checkpoint, if any.
4. Overall success, mean agents-at-goal, runtime.
5. Per-map deltas against SSIL.
6. Recommendation: which checkpoint/settings to use for the paper-style
   comparison.

## Preferred Order Of Work

1. Implement `flow_action_head` eval mode.
2. Run quick vector vs action-head vs SSIL.
3. Run targeted inference sweeps on weak maps.
4. Run medium eval for best no-retrain inference setting.
5. If promising, run full tuned inference eval.
6. If still below SSIL, implement exact discrete action labels.
7. Make action loss weight and action-loss weighting configurable.
8. Train action-loss variants for 3 epochs.
9. Evaluate quick, then medium, then full only for the best candidate.
10. Train local Rishi-like classifier as a sanity check.
11. Try smaller hybrid flow models.
12. Only then attempt SSIL distillation or DAgger.

## Success Criteria

Useful improvement:

- Full Rishi eval success above `64.05%`.
- Mean agents-at-goal closer to `95.69%`.
- No large runtime regression.
- Better results on `maze-128-128-2` and `random-32-32-10`.

Strong result:

- Flow or hybrid flow reaches or exceeds SSIL's `66.43%` overall success.
- Flow keeps its relative strengths on `den312d` and `random-64-64-10`.

Immediate map-level targets:

| Map | Current flow | SSIL | Target |
| --- | ---: | ---: | ---: |
| `maze-128-128-2` | 28% | 40% | 35-40% |
| `random-32-32-10` | 68% | 81% | 76-82% |
| `empty-48-48` | 97.6% | 100% | 99-100% |

## Bottom Line

Do not change the Rishi benchmark itself for the headline claim. Instead:

1. Add fair inference modes for the flow model.
2. Tune global inference settings.
3. Align training labels with Rishi's exact next-action labels.
4. Treat the action head as central, not auxiliary.

The most plausible winning model is not pure flow-vector inference. It is:

```text
Flow model + strong exact-action head + globally tuned inference
```
