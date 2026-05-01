#!/usr/bin/env bash
set -euo pipefail

# Dormant scaffold for staged-mixture FLOMAP training.
# This is intentionally not part of the current wait-threshold/interface
# analysis. Use later when we are ready to run training experiments on Lambda.
#
# Intended curriculum:
#   Stage A: short coordination-heavy warm start.
#   Stage B: resume on full + coordination-heavy data for a mixed curriculum.
#   Stage C: brief full-distribution polish.
#
# Run on Lambda only:
#   cd ~/barath/Flow-CS-PIBT
#   conda activate 3dposehsx_env
#   bash scripts/run_lambda_staged_mixture_training.sh

if [[ "${CONDA_DEFAULT_ENV:-}" != "3dposehsx_env" ]]; then
  echo "WARNING: CONDA_DEFAULT_ENV is '${CONDA_DEFAULT_ENV:-unset}', expected '3dposehsx_env'." >&2
  echo "Activate it first with: conda activate 3dposehsx_env" >&2
fi

RUN_PREFIX="${RUN_PREFIX:-staged_mix_v1}"
FULL_PREPROCESSED_DIR="${FULL_PREPROCESSED_DIR:-data/preprocessed}"
COORD_PREPROCESSED_DIR="${COORD_PREPROCESSED_DIR:-data/preprocessed_coordination_stress}"
COORD_MAPS=(${COORD_MAPS:-maze-128-128-2 random-32-32-10 random-64-64-10})

STAGE_A_EPOCHS="${STAGE_A_EPOCHS:-2}"
STAGE_B_EPOCHS="${STAGE_B_EPOCHS:-6}"
STAGE_C_EPOCHS="${STAGE_C_EPOCHS:-8}"

COMMON_ARGS=(
  --hidden-dim 1024
  --num-layers 6
  --action-loss-weight 0.3
  --val-split 0.05
  --patience 0
  --wandb-project flow-mapf
)

if [[ ! -d "$FULL_PREPROCESSED_DIR" ]]; then
  echo "ERROR: full preprocessed dataset missing: $FULL_PREPROCESSED_DIR" >&2
  exit 1
fi

if [[ ! -d "$COORD_PREPROCESSED_DIR" || -z "$(find "$COORD_PREPROCESSED_DIR" -maxdepth 1 -name '*.pt' -print -quit)" ]]; then
  echo "Building coordination-heavy preprocessed subset at $COORD_PREPROCESSED_DIR"
  python preprocess_dataset.py \
    --data-dir data/flow_training_data_multi \
    --map-dir data/mapf-map \
    --bd-dir data/bd_npzs/large_scale \
    --out "$COORD_PREPROCESSED_DIR" \
    --output-prefix coord \
    --include-maps "${COORD_MAPS[@]}"
else
  echo "Using existing coordination-heavy preprocessed subset: $COORD_PREPROCESSED_DIR"
fi

echo "Stage A: coordination-heavy warm start"
python -m main_pys.train_flow \
  --run-name "${RUN_PREFIX}_stageA_coord" \
  --preprocessed-dir "$COORD_PREPROCESSED_DIR" \
  --epochs "$STAGE_A_EPOCHS" \
  "${COMMON_ARGS[@]}"

STAGE_A_CKPT="large_scale_flow_${RUN_PREFIX}_stageA_coord_best.pt"
if [[ ! -f "$STAGE_A_CKPT" ]]; then
  echo "ERROR: expected Stage A checkpoint missing: $STAGE_A_CKPT" >&2
  exit 1
fi

echo "Stage B: mixed curriculum, full data plus coordination oversampling"
python -m main_pys.train_flow \
  --run-name "${RUN_PREFIX}_stageB_mixed" \
  --preprocessed-dir "${COORD_PREPROCESSED_DIR},${FULL_PREPROCESSED_DIR}" \
  --resume "$STAGE_A_CKPT" \
  --start-epoch "$STAGE_A_EPOCHS" \
  --reset-best-val-loss \
  --epochs "$STAGE_B_EPOCHS" \
  "${COMMON_ARGS[@]}"

STAGE_B_CKPT="large_scale_flow_${RUN_PREFIX}_stageB_mixed_best.pt"
if [[ ! -f "$STAGE_B_CKPT" ]]; then
  echo "ERROR: expected Stage B checkpoint missing: $STAGE_B_CKPT" >&2
  exit 1
fi

echo "Stage C: full-distribution polish"
python -m main_pys.train_flow \
  --run-name "${RUN_PREFIX}_stageC_full_polish" \
  --preprocessed-dir "$FULL_PREPROCESSED_DIR" \
  --resume "$STAGE_B_CKPT" \
  --start-epoch "$STAGE_B_EPOCHS" \
  --reset-best-val-loss \
  --epochs "$STAGE_C_EPOCHS" \
  "${COMMON_ARGS[@]}"

FINAL_CKPT="large_scale_flow_${RUN_PREFIX}_stageC_full_polish_best.pt"
echo "Staged training complete."
echo "Final checkpoint candidate: $FINAL_CKPT"
echo "Recommended evaluation:"
echo "  python eval_rishi_paper.py -m \"$FINAL_CKPT\" --policy-type flow -o evals/${RUN_PREFIX}_rishi8.csv"
