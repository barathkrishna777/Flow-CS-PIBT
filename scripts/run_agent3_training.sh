#!/usr/bin/env bash
# Multi-seed continuous training — run on Lambda from repo root.
#
# IMPORTANT: use main_pys.train_continuous for continuous-space v4b.
# main_pys.train_flow produces older 3-channel checkpoints that cannot be
# evaluated with eval_continuous.py's 4-channel continuous inputs.
set -euo pipefail

NGPUS="${NGPUS:-4}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-data/continuous_eecbs/preprocessed_shards_256}"
DATA_DIR="${DATA_DIR:-data/continuous_eecbs/raw}"
MAPDIR="${MAPDIR:-data/mapf-map}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/continuous_v4b}"
BATCH_SIZE="${BATCH_SIZE:-32}"  # per GPU; 32 x 4 GPUs = effective batch 128
NUM_WORKERS="${NUM_WORKERS:-12}"
HIDDEN_DIM="${HIDDEN_DIM:-1024}"
NUM_LAYERS="${NUM_LAYERS:-6}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-4}"
SEEDS="${SEEDS:-123 456}"  # seed=42 is the existing continuous_flow_v4b_best.pt

if [[ ! -d "$PREPROCESSED_DIR" ]]; then
  echo "ERROR: PREPROCESSED_DIR not found: $PREPROCESSED_DIR" >&2
  echo "Set PREPROCESSED_DIR to the v4b continuous shard directory used for the main model." >&2
  exit 1
fi
if [[ ! -d "$MAPDIR" ]]; then
  echo "ERROR: MAPDIR not found: $MAPDIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

for SEED in $SEEDS; do
  RUN_NAME="v4b_seed${SEED}"
  echo "======================================================================"
  echo "Training ${RUN_NAME} with main_pys.train_continuous"
  echo "GPUs=${NGPUS} per-GPU batch=${BATCH_SIZE} hidden=${HIDDEN_DIM} layers=${NUM_LAYERS}"
  echo "======================================================================"

  torchrun --standalone --nproc_per_node="$NGPUS" -- main_pys/train_continuous.py \
    --preprocessed-dir "$PREPROCESSED_DIR" \
    --data-dir "$DATA_DIR" \
    --map-dir "$MAPDIR" \
    --policy-type flow \
    --run-name "$RUN_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-layers "$NUM_LAYERS" \
    --k 4 \
    --num-neighbors 5 \
    --lr "$LR" \
    --flow-loss-weight 1.0 \
    --action-loss-weight 0.1 \
    --oversample-difficult \
    --seed "$SEED"

  echo "${RUN_NAME} complete — checkpoint: ${OUTPUT_DIR}/continuous_flow_${RUN_NAME}_best.pt"
done

echo "All requested continuous seed trainings complete."
