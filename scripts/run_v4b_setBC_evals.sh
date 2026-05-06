#!/usr/bin/env bash
# Re-run v4b flow evals on Sets B and C to populate arrived_path_length_ratio.
# These CSVs were originally generated before the metric was added.
# Run from repo root on Lambda.
set -euo pipefail

MODEL="checkpoints/continuous_v4b/continuous_flow_v4b_best.pt"
MAPDIR="${MAPDIR:-data/mapf-map}"
SCENDIR="${SCENDIR:-data/mapf-scen-random}"
N="${N:-25}"
NUM_GPUS="${NUM_GPUS:-4}"

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model checkpoint not found: $MODEL" >&2
  exit 1
fi

# ── Set B: random-64-64-10 + room-32-32-4 ────────────────────────────────────
python eval_parallel.py \
  --map-dir "$MAPDIR" --scen-dir "$SCENDIR" \
  --maps random-64-64-10 room-32-32-4 \
  --agent-counts 50 --max-scenarios "$N" \
  --policy flow --model-path "$MODEL" \
  --shield-type epibt --max-steps 512 \
  --output-csv evals/v4b/flow_epibt_512_setB.csv \
  --num-gpus "$NUM_GPUS"

# ── Set C: warehouse ──────────────────────────────────────────────────────────
python eval_parallel.py \
  --map-dir "$MAPDIR" --scen-dir "$SCENDIR" \
  --maps warehouse-10-20-10-2-1 \
  --agent-counts 50 100 --max-scenarios "$N" \
  --policy flow --model-path "$MODEL" \
  --shield-type epibt --max-steps 512 \
  --output-csv evals/v4b/flow_epibt_512_setC.csv \
  --num-gpus "$NUM_GPUS"

echo "Set B/C v4b re-evals complete."
