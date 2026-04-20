#!/bin/bash
#SBATCH --job-name=flow-train
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
# V100-32 on GPU-shared (H100: --gpus=h100-80:1, cpus 12; L40s: --gpus=l40s-48:1)
#SBATCH --gpus=v100-32:1
# GPU-shared caps CPUs at 5 per GPU on Bridges-2
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
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

# Edit --run-name and paths as needed. Example: fine-tune from friend's checkpoint.
MODEL_CKPT="${FLOW_RESUME_CKPT:-checkpoints/large_scale_flow_wave8_heldout_best.pt}"
RUN_NAME="${FLOW_RUN_NAME:-parth_wave2_ft_v2}"

# Batch 512 is stable on V100-32 for this model; override with FLOW_TRAIN_EXTRA if needed
if [ -z "$FLOW_TRAIN_EXTRA" ]; then
  FLOW_TRAIN_EXTRA="--batch-size 512"
fi
python -m main_pys.train_flow \
  --preprocessed-dir data/preprocessed \
  --run-name "$RUN_NAME" \
  --hidden-dim 1024 \
  --num-layers 6 \
  --epochs 20 \
  --patience 5 \
  --action-loss-weight 0.5 \
  --no-weighted-sampling \
  --resume "$MODEL_CKPT" \
  --wandb-project flow-mapf \
  --wandb-entity flow-cspibt \
  ${FLOW_TRAIN_EXTRA:-}
