#!/usr/bin/env bash
# Agent 2 eval commands - run on Lambda from repo root.
set -e

# TODO: Confirm this checkpoint exists on Lambda. It is not present in this local checkout.
MODEL=checkpoints/continuous_v4b/large_scale_flow_v4b_best.pt
MAPDIR=data/mapf-map
SCENDIR=data/mapf-scen-random
N=25

# Integration step ablation.
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type epibt \
  --max-steps 512 --num-integration-steps 5 \
  --output-csv evals/ablations/flow_epibt_512_setA_steps5.csv

python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type epibt \
  --max-steps 512 --num-integration-steps 10 \
  --output-csv evals/ablations/flow_epibt_512_setA_steps10.csv

python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type epibt \
  --max-steps 512 --num-integration-steps 20 \
  --output-csv evals/ablations/flow_epibt_512_setA_steps20.csv

echo "Agent 2 evals complete."
