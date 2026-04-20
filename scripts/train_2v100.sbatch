#!/bin/bash
#SBATCH --job-name=flow-train
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
mkdir -p logs/slurm

module load anaconda3/2022.10
conda activate /ocean/projects/cis250252p/psingh13/conda_envs/mlmapf

MODEL_CKPT="${FLOW_RESUME_CKPT:-checkpoints/large_scale_flow_wave8_heldout_best.pt}"
RUN_NAME="${FLOW_RUN_NAME:-parth_wave2_ft_2v100}"

python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed \
  --run-name "$RUN_NAME" \
  --hidden-dim 1024 \
  --num-layers 6 \
  --epochs 15 \
  --patience 5 \
  --action-loss-weight 0.5 \
  --no-weighted-sampling \
  --resume "$MODEL_CKPT" \
  --wandb-project flow-mapf \
  --wandb-entity flow-cspibt
