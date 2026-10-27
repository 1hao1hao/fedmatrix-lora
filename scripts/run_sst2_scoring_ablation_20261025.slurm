#!/bin/bash
#SBATCH -J sst2_score_abl_1025
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=96:00:00
#SBATCH -o logs/sst2_score_abl_1025_%A_%a.out
#SBATCH -e logs/sst2_score_abl_1025_%A_%a.err

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

variants=(Full Server-Proxy Layer-Level No-Depth)
seeds=(2024 2025 2026)
task="${SLURM_ARRAY_TASK_ID:?array task required}"
variant="${variants[$((task % 4))]}"
seed="${seeds[$((task / 4))]}"
extra=()
if [[ "${1:-full}" == pilot ]]; then extra+=(--pilot); fi
echo "variant=$variant seed=$seed mode=${1:-full}"
python -u scripts/run_sst2_scoring_ablation_20261025.py \
  --variant "$variant" --seed "$seed" "${extra[@]}"
