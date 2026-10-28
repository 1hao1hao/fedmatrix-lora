#!/bin/bash
#SBATCH -J p2_train
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -o logs/p2_train_%A_%a.out
#SBATCH -e logs/p2_train_%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
module load cuda/11.8
source ~/.bashrc
source activate bjh_lora || conda activate bjh_lora
set -u
export HF_HOME="$SLURM_SUBMIT_DIR/hf_cache"
export HF_DATASETS_CACHE="$SLURM_SUBMIT_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

distributions=(iid dirichlet:1.0 dirichlet:0.1)
seeds=(2024 2025 2026)
task="${SLURM_ARRAY_TASK_ID:?P2 train requires an array task}"
group=$((task / 10))
subtask=$((task % 10))
distribution="${distributions[$((group % 3))]}"
seed="${seeds[$((group / 3))]}"
client=$((subtask % 5))
if ((subtask < 5)); then checkpoint=early; else checkpoint=middle; fi
python -u scripts/run_matrix_granularity_p2.py train --distribution "$distribution" \
  --seed "$seed" --client "$client" --checkpoint "$checkpoint"
