#!/usr/bin/env bash
set -euo pipefail

# Run from the Lambda checkout:
#   cd ~/barath/Flow-CS-PIBT
#   conda activate 3dposehsx_env
#   bash analysis_scripts/run_wait_threshold_sweep_lambda.sh

resolve_ckpt() {
  local name="$1"
  local candidate
  for candidate in "$name" "checkpoints/$name" "checkpoints_and_evaluations/$name"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  local found
  found="$(find . -path './.git' -prune -o -type f -name "$name" -print -quit)"
  if [[ -n "$found" ]]; then
    printf '%s\n' "${found#./}"
    return 0
  fi

  echo "ERROR: checkpoint not found: $name" >&2
  echo "Looked in repo root, checkpoints/, checkpoints_and_evaluations/, and find ." >&2
  return 1
}

CKPT=${CKPT:-$(resolve_ckpt large_scale_flow_wave9_compact_best.pt)}
echo "Using checkpoint: $CKPT"

OUT_ROOT=${OUT_ROOT:-evals/wait_threshold_sweep_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT_ROOT/logs"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for WT in 0.00 0.10 0.25 0.40 0.60; do
  TAG=$(printf "wt_%0.2f" "$WT" | tr '.' 'p')
  mkdir -p "$OUT_ROOT/$TAG/shards"

  CUDA_VISIBLE_DEVICES=0 python eval_rishi_paper.py \
    -m "$CKPT" \
    --policy-type flow \
    --wait-thresh "$WT" \
    --maps Paris_1_256 empty-48-48 \
    -o "$OUT_ROOT/$TAG/shards/${TAG}_gpu0.csv" \
    > "$OUT_ROOT/logs/${TAG}_gpu0.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=1 python eval_rishi_paper.py \
    -m "$CKPT" \
    --policy-type flow \
    --wait-thresh "$WT" \
    --maps maze-128-128-2 random-64-64-10 \
    -o "$OUT_ROOT/$TAG/shards/${TAG}_gpu1.csv" \
    > "$OUT_ROOT/logs/${TAG}_gpu1.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=2 python eval_rishi_paper.py \
    -m "$CKPT" \
    --policy-type flow \
    --wait-thresh "$WT" \
    --maps random-32-32-10 warehouse-10-20-10-2-1 \
    -o "$OUT_ROOT/$TAG/shards/${TAG}_gpu2.csv" \
    > "$OUT_ROOT/logs/${TAG}_gpu2.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=3 python eval_rishi_paper.py \
    -m "$CKPT" \
    --policy-type flow \
    --wait-thresh "$WT" \
    --maps den312d den520d \
    -o "$OUT_ROOT/$TAG/shards/${TAG}_gpu3.csv" \
    > "$OUT_ROOT/logs/${TAG}_gpu3.log" 2>&1 &

  wait

  python analysis_scripts/combine_simulator_csvs.py \
    "$OUT_ROOT/$TAG"/shards/*.csv \
    -o "$OUT_ROOT/$TAG/${TAG}_combined.csv" \
    --expect-rows 1850
done

python research_report/scripts/summarize_wait_threshold_sweep.py \
  "$OUT_ROOT"/wt_*/wt_*_combined.csv

echo
echo "Sweep complete."
echo "OUT_ROOT=$OUT_ROOT"
echo "Paste the printed summary plus research_report/tables/wait_threshold_sweep.csv back into Codex."
