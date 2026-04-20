#!/bin/bash
#SBATCH --job-name=flow-preprocess
#SBATCH --partition=RM
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH -A cis250252p
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Bridges-2 RM full node: 2x AMD EPYC 7742 (128 physical cores), 256 GB RAM.
# The RM partition gives the whole node exclusively; memory is not requestable
# separately (you get all of it). Max walltime on RM is 48h.

set -eo pipefail

REPO="${FLOW_CSPIBT_ROOT:-/ocean/projects/cis250252p/psingh13/Flow-CS-PIBT}"
cd "$REPO"

mkdir -p logs/slurm

module load anaconda3/2022.10
conda activate /ocean/projects/cis250252p/psingh13/conda_envs/mlmapf

# --- Thread pinning ---------------------------------------------------------
# Each worker process does numpy / torch / scipy work. By default those spawn
# OpenMP / MKL / OpenBLAS thread pools sized to the full machine, which leads
# to NxN oversubscription once we also use multiprocessing. Force each worker
# to a single math thread so the Pool workers truly parallelize, not fight.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Force fork start method (Linux default; keeps maps dict COW-shared instead of
# pickled into every worker).
export PYTHONHASHSEED=0

# --- File descriptor / process limits --------------------------------------
# 120+ workers each opening several npz/pt files concurrently can blow past
# the default ulimit -n (often 1024 on login shells). Raise if allowed.
ulimit -n 65536 || true
ulimit -u 65536 || true

# --- Worker count -----------------------------------------------------------
# RM has 128 physical cores. We leave ~8 cores of headroom for:
#   - the main process (index build, tqdm aggregator)
#   - OS / Lustre client / filesystem daemons
#   - slurmstepd + logging
# 120 workers * ~1.5-2 GB peak per worker ~= 180-240 GB, fits in 256 GB with
# margin. If you see OOMs in `sacct`, drop to 96 or 64.
WORKERS="${FLOW_PREPROCESS_WORKERS:-120}"

# --- Inputs / outputs -------------------------------------------------------
DATA_DIR="${FLOW_PREPROCESS_DATA_DIR:-data/flow_training_data_multi}"
MAP_DIR="${FLOW_PREPROCESS_MAP_DIR:-data/mapf-map}"
OUT_DIR="${FLOW_PREPROCESS_OUT:-data/preprocessed}"

mkdir -p "$OUT_DIR"

echo "=============================================================="
echo "Flow-CS-PIBT preprocess"
echo "  host        : $(hostname)"
echo "  partition   : ${SLURM_JOB_PARTITION:-unknown}"
echo "  job id      : ${SLURM_JOB_ID:-unknown}"
echo "  cpus        : ${SLURM_CPUS_ON_NODE:-unknown} (requested ${SLURM_CPUS_PER_TASK:-?})"
echo "  workers     : $WORKERS"
echo "  data dir    : $DATA_DIR"
echo "  map dir     : $MAP_DIR"
echo "  output dir  : $OUT_DIR"
echo "  OMP/MKL     : $OMP_NUM_THREADS / $MKL_NUM_THREADS"
echo "  ulimit -n   : $(ulimit -n)"
echo "=============================================================="

# `stdbuf -oL -eL` keeps tqdm updates flushing line-by-line instead of being
# held in a big stdio buffer. Makes log monitoring usable.
stdbuf -oL -eL python -u preprocess_dataset.py \
  --data-dir "$DATA_DIR" \
  --map-dir "$MAP_DIR" \
  --out "$OUT_DIR" \
  --workers "$WORKERS"
