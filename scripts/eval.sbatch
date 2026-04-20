#!/bin/bash
#SBATCH --job-name=flow-rishi-eval
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
# Full rishi8 (~2000 sims) often needs >8h; raise if you hit time limit
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

CKPT="${FLOW_EVAL_CKPT:-large_scale_flow_parth_wave2_ft_best.pt}"
OUT="${FLOW_EVAL_CSV:-evals/rishi8_flow_run.csv}"

# Optional: FLOW_EVAL_EXTRA='--resume --parallel 2 --num-gpus 2' (use scripts/eval_2gpu.sbatch for 2 GPUs)
python eval_rishi_paper.py \
  -m "$CKPT" \
  -o "$OUT" \
  --map-set rishi8 \
  --policy-type flow \
  ${FLOW_EVAL_EXTRA:-}
