#!/usr/bin/env bash
#SBATCH --job-name=millst
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-59%1
#SBATCH --output=runs/slurm-%A-%a.log
set -euo pipefail
# Run from repository root; create matrix + shared warm starts first.
source .venv/bin/activate
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
millst --config configs/main.yaml run-index --index "$SLURM_ARRAY_TASK_ID" --resume
