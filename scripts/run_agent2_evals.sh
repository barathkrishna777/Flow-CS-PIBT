#!/usr/bin/env bash
# Agent 2 eval commands - run on Lambda from repo root.
set -e

MODEL="${MODEL:-checkpoints/continuous_v4b/continuous_flow_v4b_best.pt}"
MAPDIR="${MAPDIR:-data/mapf-map}"
SCENDIR="${SCENDIR:-data/mapf-scen-random}"
N="${N:-25}"
NUM_GPUS="${NUM_GPUS:-4}"

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model checkpoint not found: $MODEL" >&2
  exit 1
fi

# Integration step ablation.
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type cv-pibt \
  --max-steps 512 --num-integration-steps 5 \
  --output-csv evals/ablations/flow_epibt_512_setA_steps5.csv \
  --num-gpus "$NUM_GPUS"

python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type cv-pibt \
  --max-steps 512 --num-integration-steps 10 \
  --output-csv evals/ablations/flow_epibt_512_setA_steps10.csv \
  --num-gpus "$NUM_GPUS"

python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type cv-pibt \
  --max-steps 512 --num-integration-steps 20 \
  --output-csv evals/ablations/flow_epibt_512_setA_steps20.csv \
  --num-gpus "$NUM_GPUS"

echo "Agent 2 evals complete."
