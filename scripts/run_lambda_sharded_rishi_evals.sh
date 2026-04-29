#!/usr/bin/env bash
set -euo pipefail

# Launch the two requested paper-scale evals on a 4-GPU Lambda machine.
# Each GPU runs one Rishi-12 shard followed by one Rishi-8 shard, avoiding GPU
# oversubscription while keeping all four GPUs busy.

REPO_DIR="${REPO_DIR:-$HOME/barath/Flow-CS-PIBT}"
cd "$REPO_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" != "3dposehsx_env" ]]; then
  echo "WARNING: CONDA_DEFAULT_ENV is '${CONDA_DEFAULT_ENV:-unset}', expected '3dposehsx_env'." >&2
  echo "Activate it first with: conda activate 3dposehsx_env" >&2
fi

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

MAIN_CKPT="${MAIN_CKPT:-$(resolve_ckpt large_scale_flow_wave9_compact_best.pt)}"
HYBRID_CKPT="${HYBRID_CKPT:-$(resolve_ckpt large_scale_flow_hybrid_w1_5_focus_tiny_h512_l3_epoch_10.pt)}"

for path in "$MAIN_CKPT" "$HYBRID_CKPT" data/all_maps.npz data/bd_npzs/large_scale data/scen-random eval_rishi_paper.py; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path missing: $path" >&2
    exit 1
  fi
done

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-evals/lambda_sharded_${RUN_TAG}}"

mkdir -p \
  "$OUT_ROOT/logs" \
  "$OUT_ROOT/rishi12_wave9/shards" \
  "$OUT_ROOT/rishi8_hybrid/shards"

R12_SHARDS=(
  "Berlin_1_256 empty-32-32 maze-32-32-4"
  "random-64-64-20 warehouse-20-40-10-2-1 room-64-64-16"
  "Paris_1_256 empty-48-48 maze-128-128-2"
  "random-64-64-10 warehouse-10-20-10-2-1 den312d"
)

R8_SHARDS=(
  "Paris_1_256 empty-48-48"
  "maze-128-128-2 random-64-64-10"
  "random-32-32-10 warehouse-10-20-10-2-1"
  "den312d den520d"
)

PID_FILE="$OUT_ROOT/worker_pids.txt"
: > "$PID_FILE"

{
  echo "started_at=$(date -Is)"
  echo "repo=$PWD"
  echo "main_ckpt=$MAIN_CKPT"
  echo "hybrid_ckpt=$HYBRID_CKPT"
  echo "out_root=$OUT_ROOT"
  echo "strategy=4 workers; each GPU runs Rishi-12 shard then Rishi-8 shard"
} > "$OUT_ROOT/run_info.txt"

cat > "$OUT_ROOT/combine_and_summarize.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$HOME/barath/Flow-CS-PIBT}"
cd "$REPO_DIR"

R12_COMBINED="$ROOT/rishi12_wave9/rishi12_wave9_flow_combined.csv"
R8_COMBINED="$ROOT/rishi8_hybrid/rishi8_hybrid_action_head_combined.csv"

python -m analysis_scripts.combine_simulator_csvs \
  --output "$R12_COMBINED" \
  --expect-rows 2700 \
  "$ROOT"/rishi12_wave9/shards/rishi12_wave9_gpu0.csv \
  "$ROOT"/rishi12_wave9/shards/rishi12_wave9_gpu1.csv \
  "$ROOT"/rishi12_wave9/shards/rishi12_wave9_gpu2.csv \
  "$ROOT"/rishi12_wave9/shards/rishi12_wave9_gpu3.csv

python -m analysis_scripts.combine_simulator_csvs \
  --output "$R8_COMBINED" \
  --expect-rows 1850 \
  "$ROOT"/rishi8_hybrid/shards/rishi8_hybrid_gpu0.csv \
  "$ROOT"/rishi8_hybrid/shards/rishi8_hybrid_gpu1.csv \
  "$ROOT"/rishi8_hybrid/shards/rishi8_hybrid_gpu2.csv \
  "$ROOT"/rishi8_hybrid/shards/rishi8_hybrid_gpu3.csv

{
  echo "# Rishi-12 wave9 flow"
  python -m analysis_scripts.summarize_grid_eval \
    "$R12_COMBINED" \
    --labels rishi12_wave9_flow
  echo
  echo "# Rishi-8 hybrid action head"
  python -m analysis_scripts.summarize_grid_eval \
    "$R8_COMBINED" \
    --labels rishi8_hybrid_action_head
} > "$ROOT/summary.md"

cat "$ROOT/summary.md"
EOF
chmod +x "$OUT_ROOT/combine_and_summarize.sh"

cat > "$OUT_ROOT/wait_then_combine.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/worker_pids.txt"

while true; do
  alive=0
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    if kill -0 "$pid" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done < "$PID_FILE"

  if (( alive == 0 )); then
    break
  fi

  echo "$(date -Is): $alive worker(s) still running"
  sleep 60
done

bash "$ROOT/combine_and_summarize.sh"
EOF
chmod +x "$OUT_ROOT/wait_then_combine.sh"

for gpu in 0 1 2 3; do
  log="$OUT_ROOT/logs/gpu${gpu}_worker.log"
  nohup bash -c '
    set -euo pipefail
    repo="$1"
    gpu="$2"
    r12_maps="$3"
    r8_maps="$4"
    main_ckpt="$5"
    hybrid_ckpt="$6"
    out_root="$7"
    cd "$repo"

    echo "[$(date -Is)] GPU ${gpu} worker start"
    echo "Rishi-12 maps: ${r12_maps}"
    CUDA_VISIBLE_DEVICES="$gpu" python eval_rishi_paper.py \
      -m "$main_ckpt" \
      --output "$out_root/rishi12_wave9/shards/rishi12_wave9_gpu${gpu}.csv" \
      --map-set rishi12 \
      --maps ${r12_maps} \
      --policy-type flow

    echo "[$(date -Is)] GPU ${gpu} finished Rishi-12; starting Rishi-8"
    echo "Rishi-8 maps: ${r8_maps}"
    CUDA_VISIBLE_DEVICES="$gpu" python eval_rishi_paper.py \
      -m "$hybrid_ckpt" \
      --output "$out_root/rishi8_hybrid/shards/rishi8_hybrid_gpu${gpu}.csv" \
      --map-set rishi8 \
      --maps ${r8_maps} \
      --policy-type flow_action_head \
      --action-head-conditioning integrated \
      --hidden-dim 512 \
      --num-layers 3

    echo "[$(date -Is)] GPU ${gpu} worker done"
  ' worker "$PWD" "$gpu" "${R12_SHARDS[$gpu]}" "${R8_SHARDS[$gpu]}" "$MAIN_CKPT" "$HYBRID_CKPT" "$OUT_ROOT" > "$log" 2>&1 &
  pid="$!"
  echo "$pid" >> "$PID_FILE"
  echo "Launched GPU $gpu worker pid=$pid log=$log"
done

echo
echo "Output root: $OUT_ROOT"
echo "Run info:    $OUT_ROOT/run_info.txt"
echo "PID file:    $PID_FILE"
echo
echo "Monitor:"
echo "  nvidia-smi -l 5"
echo "  tail -f $OUT_ROOT/logs/gpu0_worker.log $OUT_ROOT/logs/gpu1_worker.log $OUT_ROOT/logs/gpu2_worker.log $OUT_ROOT/logs/gpu3_worker.log"
echo "  ps -fp \$(paste -sd, $PID_FILE)"
echo
echo "When done, combine and summarize:"
echo "  bash $OUT_ROOT/combine_and_summarize.sh"
echo
echo "Or wait in the foreground and combine automatically:"
echo "  bash $OUT_ROOT/wait_then_combine.sh"
