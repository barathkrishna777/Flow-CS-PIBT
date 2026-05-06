#!/usr/bin/env bash
# Multi-seed training — run on Lambda from repo root.
# Data must already be extracted to data/ before running.
# Uses torchrun for 4-GPU DDP; --batch-size 64 keeps effective batch=256 (matching single-GPU seed42).
# Checkpoints are written to the repo root as large_scale_flow_v4b_seed{N}_best.pt
set -euo pipefail

NGPUS="${NGPUS:-4}"
# Per-GPU batch: 64 → effective batch = 64×4 = 256, same as single-GPU seed42.
# Set BATCH_SIZE=256 to maximise throughput at the cost of slightly different training dynamics.
BATCH_SIZE="${BATCH_SIZE:-64}"

SEEDS=(123 456)   # seed=42 already trained single-GPU; start from 123

for SEED in "${SEEDS[@]}"; do
  RUN_NAME="v4b_seed${SEED}"
  echo "======================================================================"
  echo "Training ${RUN_NAME}  (${NGPUS} GPUs, per-GPU batch=${BATCH_SIZE})"
  echo "======================================================================"

  torchrun --nproc_per_node="$NGPUS" main_pys/train_flow.py \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --hidden-dim 1024 \
    --num-layers 6 \
    --batch-size "$BATCH_SIZE"

  echo "${RUN_NAME} complete — checkpoint: large_scale_flow_${RUN_NAME}_best.pt"
done

echo "All remaining seeds complete."
