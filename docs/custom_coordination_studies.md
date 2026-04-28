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

## 4. Train With Base Plus Custom Data

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

## Decision Signal

This study is worth continuing if medium overall is near or above the flow-vector
baseline while `random-32-32-10` improves materially. If custom data only boosts
random-32 while hurting Paris, empty, or random-64, reduce the custom ratio by
generating fewer custom scenarios or training with fewer custom maps.
