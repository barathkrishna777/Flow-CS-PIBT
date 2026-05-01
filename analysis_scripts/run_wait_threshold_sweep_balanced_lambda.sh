#!/usr/bin/env bash
set -euo pipefail

# Balanced/resumable wait-threshold sweep for Lambda.
#
# This runner splits slow maps by scenario windows instead of assigning whole
# topologies to a single GPU. It supports MODE=mini or MODE=full and skips shard
# CSVs that already have the expected row count, making it safe to resume after
# stopping an earlier run at a threshold boundary.
#
# Example:
#   cd ~/barath/Flow-CS-PIBT
#   conda activate 3dposehsx_env
#   MODE=mini OUT_ROOT=evals/wait_threshold_mini_sweep_20260501_164538 \
#     bash analysis_scripts/run_wait_threshold_sweep_balanced_lambda.sh

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
  return 1
}

csv_rows() {
  local path="$1"
  if [[ -f "$path" ]]; then
    local lines
    lines=$(wc -l < "$path")
    echo $((lines > 0 ? lines - 1 : 0))
  else
    echo 0
  fi
}

run_or_skip() {
  local gpu="$1"
  local expected="$2"
  local output="$3"
  shift 3

  local current
  current=$(csv_rows "$output")
  if [[ "$current" -eq "$expected" ]]; then
    echo "skip $(basename "$output"): $current/$expected rows"
    return 0
  fi

  rm -f "$output"
  echo "run  gpu=$gpu expected=$expected output=$output"
  CUDA_VISIBLE_DEVICES="$gpu" python eval_rishi_paper.py "$@" -o "$output"
}

combine_threshold() {
  local tag="$1"
  local expected="$2"
  python analysis_scripts/combine_simulator_csvs.py \
    "$OUT_ROOT/$tag"/shards/*.csv \
    -o "$OUT_ROOT/$tag/${tag}_combined.csv" \
    --expect-rows "$expected"
}

CKPT=${CKPT:-$(resolve_ckpt large_scale_flow_wave9_compact_best.pt)}
MODE=${MODE:-full}
OUT_ROOT=${OUT_ROOT:-evals/wait_threshold_${MODE}_balanced_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT_ROOT/logs"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "$MODE" in
  mini)
    AGENTS=(100 200 400 600 800)
    SCEN_A=(--scenario-start 1 --max-scenario 3)
    SCEN_B=(--scenario-start 4 --max-scenario 2)
    # Per threshold:
    # non-random32 maps: 7 maps * 5 scens * 5 agents = 175
    # random32: 5 scens * 3 supported agents = 15
    EXPECT_TOTAL=190
    EXPECT_3SCEN_FULL=15
    EXPECT_2SCEN_FULL=10
    EXPECT_RANDOM32_A=9
    EXPECT_RANDOM32_B=6
    ;;
  full)
    AGENTS=(100 200 300 400 500 600 700 800 900 1000)
    SCEN_A=(--scenario-start 1 --max-scenario 13)
    SCEN_B=(--scenario-start 14 --max-scenario 12)
    EXPECT_TOTAL=1850
    EXPECT_3SCEN_FULL=130
    EXPECT_2SCEN_FULL=120
    # random-32 has fewer high-agent scenarios in this benchmark, so these are
    # not used in the full balanced task list; it stays as a whole-map shard.
    ;;
  *)
    echo "ERROR: MODE must be mini or full, got: $MODE" >&2
    exit 1
    ;;
esac

echo "Using checkpoint: $CKPT"
echo "Mode: $MODE"
echo "OUT_ROOT=$OUT_ROOT"

for WT in 0.00 0.10 0.25 0.40 0.60; do
  TAG=$(printf "wt_%0.2f" "$WT" | tr '.' 'p')
  mkdir -p "$OUT_ROOT/$TAG/shards"
  echo "== $TAG =="

  combined="$OUT_ROOT/$TAG/${TAG}_combined.csv"
  combined_rows=$(csv_rows "$combined")
  if [[ "$combined_rows" -eq "$EXPECT_TOTAL" ]]; then
    echo "skip $TAG: combined CSV already has $combined_rows/$EXPECT_TOTAL rows"
    continue
  fi

  common=(-m "$CKPT" --policy-type flow --wait-thresh "$WT" --agents "${AGENTS[@]}")

  if [[ "$MODE" == "mini" ]]; then
    run_or_skip 0 "$EXPECT_3SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_paris_a.csv" "${common[@]}" "${SCEN_A[@]}" --maps Paris_1_256 > "$OUT_ROOT/logs/${TAG}_paris_a.log" 2>&1 &
    run_or_skip 1 "$EXPECT_3SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_empty_a.csv" "${common[@]}" "${SCEN_A[@]}" --maps empty-48-48 > "$OUT_ROOT/logs/${TAG}_empty_a.log" 2>&1 &
    run_or_skip 2 "$EXPECT_3SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_maze_a.csv" "${common[@]}" "${SCEN_A[@]}" --maps maze-128-128-2 > "$OUT_ROOT/logs/${TAG}_maze_a.log" 2>&1 &
    run_or_skip 3 "$EXPECT_3SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_random64_a.csv" "${common[@]}" "${SCEN_A[@]}" --maps random-64-64-10 > "$OUT_ROOT/logs/${TAG}_random64_a.log" 2>&1 &
    wait

    run_or_skip 0 "$EXPECT_RANDOM32_A" "$OUT_ROOT/$TAG/shards/${TAG}_random32_a.csv" "${common[@]}" "${SCEN_A[@]}" --maps random-32-32-10 > "$OUT_ROOT/logs/${TAG}_random32_a.log" 2>&1 &
    run_or_skip 1 "$EXPECT_3SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_warehouse_a.csv" "${common[@]}" "${SCEN_A[@]}" --maps warehouse-10-20-10-2-1 > "$OUT_ROOT/logs/${TAG}_warehouse_a.log" 2>&1 &
    run_or_skip 2 "$EXPECT_3SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_den312d_a.csv" "${common[@]}" "${SCEN_A[@]}" --maps den312d > "$OUT_ROOT/logs/${TAG}_den312d_a.log" 2>&1 &
    run_or_skip 3 "$EXPECT_3SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_den520d_a.csv" "${common[@]}" "${SCEN_A[@]}" --maps den520d > "$OUT_ROOT/logs/${TAG}_den520d_a.log" 2>&1 &
    wait

    run_or_skip 0 "$EXPECT_2SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_paris_b.csv" "${common[@]}" "${SCEN_B[@]}" --maps Paris_1_256 > "$OUT_ROOT/logs/${TAG}_paris_b.log" 2>&1 &
    run_or_skip 1 "$EXPECT_2SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_empty_b.csv" "${common[@]}" "${SCEN_B[@]}" --maps empty-48-48 > "$OUT_ROOT/logs/${TAG}_empty_b.log" 2>&1 &
    run_or_skip 2 "$EXPECT_2SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_maze_b.csv" "${common[@]}" "${SCEN_B[@]}" --maps maze-128-128-2 > "$OUT_ROOT/logs/${TAG}_maze_b.log" 2>&1 &
    run_or_skip 3 "$EXPECT_2SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_random64_b.csv" "${common[@]}" "${SCEN_B[@]}" --maps random-64-64-10 > "$OUT_ROOT/logs/${TAG}_random64_b.log" 2>&1 &
    wait

    run_or_skip 0 "$EXPECT_RANDOM32_B" "$OUT_ROOT/$TAG/shards/${TAG}_random32_b.csv" "${common[@]}" "${SCEN_B[@]}" --maps random-32-32-10 > "$OUT_ROOT/logs/${TAG}_random32_b.log" 2>&1 &
    run_or_skip 1 "$EXPECT_2SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_warehouse_b.csv" "${common[@]}" "${SCEN_B[@]}" --maps warehouse-10-20-10-2-1 > "$OUT_ROOT/logs/${TAG}_warehouse_b.log" 2>&1 &
    run_or_skip 2 "$EXPECT_2SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_den312d_b.csv" "${common[@]}" "${SCEN_B[@]}" --maps den312d > "$OUT_ROOT/logs/${TAG}_den312d_b.log" 2>&1 &
    run_or_skip 3 "$EXPECT_2SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_den520d_b.csv" "${common[@]}" "${SCEN_B[@]}" --maps den520d > "$OUT_ROOT/logs/${TAG}_den520d_b.log" 2>&1 &
    wait
  else
    # Full mode keeps random-32 as its original shard because scenario capacity
    # differs by file; the other slow maps are split by scenario window.
    run_or_skip 0 250 "$OUT_ROOT/$TAG/shards/${TAG}_paris.csv" "${common[@]}" --maps Paris_1_256 > "$OUT_ROOT/logs/${TAG}_paris.log" 2>&1 &
    run_or_skip 1 "$EXPECT_3SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_maze_a.csv" "${common[@]}" "${SCEN_A[@]}" --maps maze-128-128-2 > "$OUT_ROOT/logs/${TAG}_maze_a.log" 2>&1 &
    run_or_skip 2 "$EXPECT_3SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_den312d_a.csv" "${common[@]}" "${SCEN_A[@]}" --maps den312d > "$OUT_ROOT/logs/${TAG}_den312d_a.log" 2>&1 &
    run_or_skip 3 250 "$OUT_ROOT/$TAG/shards/${TAG}_empty.csv" "${common[@]}" --maps empty-48-48 > "$OUT_ROOT/logs/${TAG}_empty.log" 2>&1 &
    wait

    run_or_skip 0 "$EXPECT_2SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_maze_b.csv" "${common[@]}" "${SCEN_B[@]}" --maps maze-128-128-2 > "$OUT_ROOT/logs/${TAG}_maze_b.log" 2>&1 &
    run_or_skip 1 250 "$OUT_ROOT/$TAG/shards/${TAG}_random64.csv" "${common[@]}" --maps random-64-64-10 > "$OUT_ROOT/logs/${TAG}_random64.log" 2>&1 &
    run_or_skip 2 "$EXPECT_2SCEN_FULL" "$OUT_ROOT/$TAG/shards/${TAG}_den312d_b.csv" "${common[@]}" "${SCEN_B[@]}" --maps den312d > "$OUT_ROOT/logs/${TAG}_den312d_b.log" 2>&1 &
    run_or_skip 3 250 "$OUT_ROOT/$TAG/shards/${TAG}_den520d.csv" "${common[@]}" --maps den520d > "$OUT_ROOT/logs/${TAG}_den520d.log" 2>&1 &
    wait

    run_or_skip 0 100 "$OUT_ROOT/$TAG/shards/${TAG}_random32.csv" "${common[@]}" --maps random-32-32-10 > "$OUT_ROOT/logs/${TAG}_random32.log" 2>&1 &
    run_or_skip 1 250 "$OUT_ROOT/$TAG/shards/${TAG}_warehouse.csv" "${common[@]}" --maps warehouse-10-20-10-2-1 > "$OUT_ROOT/logs/${TAG}_warehouse.log" 2>&1 &
    wait
  fi

  combine_threshold "$TAG" "$EXPECT_TOTAL"
done

python research_report/scripts/summarize_wait_threshold_sweep.py \
  "$OUT_ROOT"/wt_*/wt_*_combined.csv \
  --expect-rows "$EXPECT_TOTAL"
