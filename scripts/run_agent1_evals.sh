#!/usr/bin/env bash
# Agent 1 eval commands — run on Lambda from repo root
# Assumes 4 GPUs, conda env activated, data at data/mapf-map and data/mapf-scen-random
set -e

# TODO(agent1): This local checkout has no checkpoints/ directory, so confirm
# the v4b checkpoint path on Lambda before running this script.
# Expected path from the handoff:
#   checkpoints/continuous_v4b/large_scale_flow_v4b_best.pt
MODEL=TODO_FILL_IN_V4B_CHECKPOINT_PATH
MAPDIR=data/mapf-map
SCENDIR=data/mapf-scen-random
N=25  # scenarios per map

if [[ "$MODEL" == "TODO_FILL_IN_V4B_CHECKPOINT_PATH" ]]; then
  echo "ERROR: set MODEL to the confirmed v4b checkpoint path before running."
  echo "Expected path: checkpoints/continuous_v4b/large_scale_flow_v4b_best.pt"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "ERROR: model checkpoint not found: $MODEL"
  exit 1
fi

# ── 1. ORCA vel + EPIBTShield  (new ablation, Set A) ────────────────────────
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy orca --nav straight --shield-type epibt \
  --max-steps 512 \
  --output-csv evals/ablations/orca_epibt_512_setA.csv

# ── 2. Flow v4b + ORCA shield  (new ablation, Set A, 256 steps) ─────────────
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type orca \
  --max-steps 256 \
  --output-csv evals/ablations/flow_orca_256_setA.csv

# ── 3. Flow v4b + ORCA shield  (new ablation, Set A, 512 steps) ─────────────
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-32-32-10 empty-48-48 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy flow --model-path $MODEL --shield-type orca \
  --max-steps 512 \
  --output-csv evals/ablations/flow_orca_512_setA.csv

# ── 4. ORCA baseline on Set B (random-64-64-10 + room-32-32-4) ───────────────
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-64-64-10 room-32-32-4 \
  --agent-counts 50 --max-scenarios $N \
  --policy orca --nav straight --shield-type orca \
  --max-steps 512 \
  --output-csv evals/baselines/straight_orca_512_setB.csv

# ── 5. PO-ORCA baseline on Set B ─────────────────────────────────────────────
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-64-64-10 room-32-32-4 \
  --agent-counts 50 --max-scenarios $N \
  --policy orca --nav straight --shield-type po-orca \
  --max-steps 512 \
  --output-csv evals/baselines/straight_po-orca_512_setB.csv

# ── 6. ORCA vel + EPIBTShield on Set B ───────────────────────────────────────
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps random-64-64-10 room-32-32-4 \
  --agent-counts 50 --max-scenarios $N \
  --policy orca --nav straight --shield-type epibt \
  --max-steps 512 \
  --output-csv evals/ablations/orca_epibt_512_setB.csv

# ── 7. ORCA baseline on Set C (warehouse) ────────────────────────────────────
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps warehouse-10-20-10-2-1 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy orca --nav straight --shield-type orca \
  --max-steps 512 \
  --output-csv evals/baselines/straight_orca_512_setC.csv

# ── 8. PO-ORCA baseline on Set C ─────────────────────────────────────────────
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps warehouse-10-20-10-2-1 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy orca --nav straight --shield-type po-orca \
  --max-steps 512 \
  --output-csv evals/baselines/straight_po-orca_512_setC.csv

# ── 9. ORCA vel + EPIBTShield on Set C ───────────────────────────────────────
python eval_parallel.py \
  --map-dir $MAPDIR --scen-dir $SCENDIR \
  --maps warehouse-10-20-10-2-1 \
  --agent-counts 50 100 --max-scenarios $N \
  --policy orca --nav straight --shield-type epibt \
  --max-steps 512 \
  --output-csv evals/ablations/orca_epibt_512_setC.csv

echo "All Agent 1 evals complete."
