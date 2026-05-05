#!/usr/bin/env bash
# Agent 3 training commands - run on Lambda from repo root.
#
# TODO: Fill in the two zip paths below before running this script.
#   BASE_DATA: path to the base data zip (maps, BDs, scen files, etc.)
#   TRAJ_DATA: path to the v4b trajectory training dataset zip
set -euo pipefail

BASE_DATA="/path/to/base_data.zip"             # TODO: EDIT THIS
TRAJ_DATA="/path/to/trajectories_v4b.zip"      # TODO: EDIT THIS

CHECKPOINT_DIR="checkpoints/continuous_v4b"
SEEDS=(42 123 456)

if [[ "$BASE_DATA" == "/path/to/base_data.zip" || "$TRAJ_DATA" == "/path/to/trajectories_v4b.zip" ]]; then
  echo "ERROR: edit BASE_DATA and TRAJ_DATA in scripts/run_agent3_training.sh before running." >&2
  exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

for SEED in "${SEEDS[@]}"; do
  RUN_NAME="v4b_seed${SEED}"

  echo "======================================================================"
  echo "Training ${RUN_NAME}"
  echo "======================================================================"

  python train_full.py \
    --base-data "$BASE_DATA" \
    --trajectories "$TRAJ_DATA" \
    --run-name "$RUN_NAME" \
    --seed "$SEED"

  shopt -s nullglob
  CHECKPOINTS=(large_scale_flow_${RUN_NAME}_*.pt)
  shopt -u nullglob

  if (( ${#CHECKPOINTS[@]} == 0 )); then
    echo "ERROR: no checkpoints found for ${RUN_NAME}" >&2
    exit 1
  fi

  mv "${CHECKPOINTS[@]}" "$CHECKPOINT_DIR"/
  echo "Moved ${RUN_NAME} checkpoints to ${CHECKPOINT_DIR}/"
done

echo "All 3 training runs complete."
echo "Checkpoints at:"
echo "  checkpoints/continuous_v4b/large_scale_flow_v4b_seed42_best.pt"
echo "  checkpoints/continuous_v4b/large_scale_flow_v4b_seed123_best.pt"
echo "  checkpoints/continuous_v4b/large_scale_flow_v4b_seed456_best.pt"
