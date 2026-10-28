#!/bin/bash
#SBATCH -J p2_masks
#SBATCH -p cpuXeon6458
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o logs/p2_masks_%A_%a.out
#SBATCH -e logs/p2_masks_%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
source activate bjh_lora || conda activate bjh_lora
set -u
export HF_HOME="$SLURM_SUBMIT_DIR/hf_cache"
export HF_DATASETS_CACHE="$SLURM_SUBMIT_DIR/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

distributions=(iid dirichlet:1.0 dirichlet:0.1)
seeds=(2024 2025 2026)
task="${SLURM_ARRAY_TASK_ID:?P2 prepare requires an array task}"
distribution="${distributions[$((task % 3))]}"
seed="${seeds[$((task / 3))]}"
python -u scripts/run_matrix_granularity_p2.py prepare --distribution "$distribution" --seed "$seed"
