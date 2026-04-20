#!/bin/bash
#SBATCH --job-name=flow-eval-quick
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
OUT="${FLOW_EVAL_CSV:-evals/quick_flow_run.csv}"

# --quick: 8 maps x 1 scenario x 3 agent counts (100/400/800) = 24 runs, ~10-20 min
# --parallel: run 4 sims simultaneously to use all 5 CPUs
python eval_rishi_paper.py \
  -m "$CKPT" \
  -o "$OUT" \
  --map-set rishi8 \
  --policy-type flow \
  --quick \
  --parallel 4
