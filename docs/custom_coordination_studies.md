# Custom Coordination Map Studies

Use this workflow on Lambda to add training-only dense coordination data without
training on the exact Rishi held-out maps.

## 1. Generate Custom Maps And Scenarios

```sh
python generate_custom_coordination_maps.py \
  --output-root data/custom_coordination \
  --scenarios 128 \
  --seed 0
```

This creates:

```text
data/custom_coordination/maps/*.map
data/custom_coordination/scens/*.scen
data/custom_coordination/agent_counts.json
data/custom_coordination/manifest.json
```

Default generated map families:

- `random-32-32-10-custom-*`: dense small random maps, 100-400 agents
- `random-64-64-10-custom-*`: larger random maps, 100-800 agents
- `corridor-30-30-custom-*`: bottleneck maps, 50-200 agents
- `dense-15-15-custom-*`: very small high-density maps, 20-50 agents

The names intentionally do not collide with the Rishi held-out map names.

## 2. Generate Expert Trajectories

```sh
CUDA_VISIBLE_DEVICES=0 python generate_flow_data_multi.py \
  --map-dir data/custom_coordination/maps \
  --scen-dir data/custom_coordination/scens \
  --output-npz-dir data/flow_training_data_custom_coordination \
  --bd-dir data/bd_npzs/custom_coordination \
  --agent-counts-json data/custom_coordination/agent_counts.json \
  --allow-large-32x32 \
  --training-scenario-limit 128 \
  --workers 64
```

Preview the job count first:

```sh
python generate_flow_data_multi.py \
  --map-dir data/custom_coordination/maps \
  --scen-dir data/custom_coordination/scens \
  --output-npz-dir data/flow_training_data_custom_coordination \
  --bd-dir data/bd_npzs/custom_coordination \
  --agent-counts-json data/custom_coordination/agent_counts.json \
  --allow-large-32x32 \
  --dry-run
```

## 3. Preprocess Custom Data

```sh
python preprocess_dataset.py \
  --data-dir data/flow_training_data_custom_coordination \
  --map-dir data/custom_coordination/maps \
  --bd-dir data/bd_npzs/custom_coordination \
  --out data/preprocessed_custom_coordination \
  --output-prefix custom \
  --workers 32
```

The `--output-prefix custom` avoids filename overlap with the base
`data/preprocessed` directory.

## 4. Train Hybrid With Base Plus Custom Data

Use the existing balanced discrete-primary setup, but point `--preprocessed-dir`
at both directories:

```sh
CUDA_VISIBLE_DEVICES=0 python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed,data/preprocessed_custom_coordination \
  --run-name flow_hybrid_discrete_balanced_w1_5_h512_l3_custom_coord \
  --hidden-dim 512 \
  --num-layers 3 \
  --action-loss-weight 1.5 \
  --unweighted-action-loss \
  --discrete-forward-mode x1_t1 \
  --epochs 20 \
  --no-wandb
```

Then evaluate on the unchanged Rishi held-out protocol. Keep passing the matching
model size:

```sh
python eval_rishi_paper.py \
  -m large_scale_flow_flow_hybrid_discrete_balanced_w1_5_h512_l3_custom_coord_best.pt \
  -o evals/ablations/flow_hybrid_discrete_balanced_w1_5_h512_l3_custom_coord_medium.csv \
  --max-scenario 5 \
  --policy-type flow_action_head \
  --action-head-conditioning integrated \
  --hidden-dim 512 \
  --num-layers 3
```

## 5. Train Flow-Primary Baselines With Base Plus Custom Data

These runs isolate whether the custom coordination data itself improves normal
flow-vector inference. They should be evaluated with `--policy-type flow`, not
the action head.

`main_pys.train_flow` currently defaults to `--action-loss-weight 0.3`, so
omitting the flag keeps the repo's normal auxiliary action-loss behavior. For a
strict flow-only control, add `--action-loss-weight 0.0`.

Strong flow-capacity baseline:

```sh
CUDA_VISIBLE_DEVICES=0 python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed,data/preprocessed_custom_coordination \
  --run-name flow_customcoord_h1024_l6 \
  --hidden-dim 1024 \
  --num-layers 6 \
  --epochs 20 \
  --no-wandb
```

Hybrid-capacity matched baseline:

```sh
CUDA_VISIBLE_DEVICES=1 python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed,data/preprocessed_custom_coordination \
  --run-name flow_customcoord_h512_l3 \
  --hidden-dim 512 \
  --num-layers 3 \
  --epochs 20 \
  --no-wandb
```

Optional smaller comparison:

```sh
CUDA_VISIBLE_DEVICES=1 python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed,data/preprocessed_custom_coordination \
  --run-name flow_customcoord_h256_l3 \
  --hidden-dim 256 \
  --num-layers 3 \
  --epochs 20 \
  --no-wandb
```

Quick checkpoint sweep for the 512x3 flow baseline:

```sh
for ep in 5 10 12 15 20; do
  ckpt="large_scale_flow_flow_customcoord_h512_l3_epoch_${ep}.pt"
  [ -f "$ckpt" ] || { echo "missing $ckpt"; continue; }

  python eval_rishi_paper.py \
    -m "$ckpt" \
    -o "evals/ablations/flow_customcoord_h512_l3_epoch${ep}_quick_flow.csv" \
    --quick \
    --policy-type flow \
    --hidden-dim 512 \
    --num-layers 3
done
```

Summarize the sweep:

```sh
python -m analysis_scripts.summarize_grid_eval \
  evals/ablations/flow_customcoord_h512_l3_epoch5_quick_flow.csv \
  evals/ablations/flow_customcoord_h512_l3_epoch10_quick_flow.csv \
  evals/ablations/flow_customcoord_h512_l3_epoch12_quick_flow.csv \
  evals/ablations/flow_customcoord_h512_l3_epoch15_quick_flow.csv \
  evals/ablations/flow_customcoord_h512_l3_epoch20_quick_flow.csv \
  --labels flow_cc_e5 flow_cc_e10 flow_cc_e12 flow_cc_e15 flow_cc_e20
```

Medium eval for a promising 512x3 checkpoint:

```sh
python eval_rishi_paper.py \
  -m large_scale_flow_flow_customcoord_h512_l3_epoch_5.pt \
  -o evals/ablations/flow_customcoord_h512_l3_epoch5_medium_flow.csv \
  --max-scenario 5 \
  --policy-type flow \
  --hidden-dim 512 \
  --num-layers 3
```

For any non-default model size, keep passing the matching `--hidden-dim` and
`--num-layers` during evaluation.

## Decision Signal

This study is worth continuing if medium overall is near or above the flow-vector
baseline while `random-32-32-10` improves materially. If custom data only boosts
random-32 while hurting Paris, empty, or random-64, reduce the custom ratio by
generating fewer custom scenarios or training with fewer custom maps.

The key comparison is:

- custom-data flow model evaluated with normal flow-vector inference
- custom-data hybrid/discrete-primary model evaluated with action-head inference
