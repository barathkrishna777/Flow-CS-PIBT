#!/usr/bin/env bash
# Evaluate all 3 seeds on Set A (512 steps) from repo root.
set -euo pipefail

MAPDIR="${MAPDIR:-data/mapf-map}"
SCENDIR="${SCENDIR:-data/mapf-scen-random}"
N="${N:-25}"
NUM_GPUS="${NUM_GPUS:-4}"

for SEED in 42 123 456; do
  RUN_NAME="v4b_seed${SEED}"

  python eval_parallel.py \
    --map-dir "$MAPDIR" \
    --scen-dir "$SCENDIR" \
    --maps random-32-32-10 empty-48-48 \
    --agent-counts 50 100 \
    --max-scenarios "$N" \
    --policy flow \
    --model-path "checkpoints/continuous_v4b/large_scale_flow_${RUN_NAME}_best.pt" \
    --shield-type epibt \
    --max-steps 512 \
    --run-name "$RUN_NAME" \
    --train-seed "$SEED" \
    --output-csv "evals/${RUN_NAME}/flow_epibt_512_setA.csv" \
    --num-gpus "$NUM_GPUS"
done

echo "Multi-seed evals complete."
