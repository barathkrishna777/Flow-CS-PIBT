#!/bin/bash
#SBATCH --job-name=flow-rishi-eval-2g
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -A cis250252p
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -eo pipefail

REPO="${FLOW_CSPIBT_ROOT:-/ocean/projects/cis250252p/psingh13/Flow-CS-PIBT}"
cd "$REPO"

mkdir -p logs/slurm evals

module load anaconda3/2022.10
conda activate /ocean/projects/cis250252p/psingh13/conda_envs/mlmapf

CKPT="${FLOW_EVAL_CKPT:-large_scale_flow_parth_wave2_ft_v2_best.pt}"
OUT="${FLOW_EVAL_CSV:-evals/rishi8_flow_run.csv}"

# 2 workers × 2 GPUs: each subprocess sees one GPU via CUDA_VISIBLE_DEVICES
export EVAL_NUM_GPUS=2
if [ -z "$FLOW_EVAL_EXTRA" ]; then
  FLOW_EVAL_EXTRA="--parallel 2 --num-gpus 2 --resume"
fi

python eval_rishi_paper.py \
  -m "$CKPT" \
  -o "$OUT" \
  --map-set rishi8 \
  --policy-type flow \
  $FLOW_EVAL_EXTRA
