#!/usr/bin/env bash
# Evaluate all 3 seeds on Set A (512 steps) from repo root.
set -euo pipefail

MAPDIR="${MAPDIR:-data/mapf-map}"
SCENDIR="${SCENDIR:-data/mapf-scen-random}"
N="${N:-25}"
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_DIR="${MODEL_DIR:-checkpoints/continuous_v4b}"
MODEL_42="${MODEL_42:-${MODEL_DIR}/continuous_flow_v4b_best.pt}"
MODEL_TEMPLATE="${MODEL_TEMPLATE:-${MODEL_DIR}/continuous_flow_v4b_seed__SEED___best.pt}"

model_path_for_seed() {
  local seed="$1"
  if [[ "$seed" == "42" ]]; then
    echo "$MODEL_42"
  else
    echo "${MODEL_TEMPLATE/__SEED__/$seed}"
  fi
}

missing=0
for SEED in 42 123 456; do
  MODEL="$(model_path_for_seed "$SEED")"
  if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: missing checkpoint for seed ${SEED}: $MODEL" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Wait for continuous training to finish, or override MODEL_42 / MODEL_TEMPLATE / MODEL_DIR." >&2
  exit 1
fi

for SEED in 42 123 456; do
  RUN_NAME="v4b_seed${SEED}"
  MODEL="$(model_path_for_seed "$SEED")"

  python eval_parallel.py \
    --map-dir "$MAPDIR" \
    --scen-dir "$SCENDIR" \
    --maps random-32-32-10 empty-48-48 \
    --agent-counts 50 100 \
    --max-scenarios "$N" \
    --policy flow \
    --model-path "$MODEL" \
    --shield-type cv-pibt \
    --max-steps 512 \
    --run-name "$RUN_NAME" \
    --train-seed "$SEED" \
    --output-csv "evals/${RUN_NAME}/flow_epibt_512_setA.csv" \
    --num-gpus "$NUM_GPUS"
done

echo "Multi-seed evals complete."
