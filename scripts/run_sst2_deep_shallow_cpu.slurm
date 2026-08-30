#!/bin/bash
#SBATCH -J sst2_depth_cpu
#SBATCH -p cpuXeon6458
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-5%6
#SBATCH -o logs/sst2_depth_cpu_%A_%a.out
#SBATCH -e logs/sst2_depth_cpu_%A_%a.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

module purge
source ~/.bashrc
source activate bjh_lora || conda activate bjh_lora

export HF_HOME="$SLURM_SUBMIT_DIR/hf_cache"
export HF_DATASETS_CACHE="$SLURM_SUBMIT_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

strategies=(shallow_qkv deep_qkv)
seeds=(2024 2025 2026)
strategy_index=$((SLURM_ARRAY_TASK_ID % ${#strategies[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID / ${#strategies[@]}))
strategy="${strategies[$strategy_index]}"
seed="${seeds[$seed_index]}"
output_root="paper_assets/02_experiments/sst2_deep_shallow_cpu_${SLURM_ARRAY_JOB_ID}"

mkdir -p "$output_root"
echo "job_id=$SLURM_ARRAY_JOB_ID task_id=$SLURM_ARRAY_TASK_ID strategy=$strategy seed=$seed"
lscpu

python -u scripts/run_sst2_matrix_position_motivation.py \
  --strategy "$strategy" \
  --seed "$seed" \
  --max-steps 600 \
  --eval-interval 50 \
  --train-batch-size 16 \
  --eval-batch-size 128 \
  --learning-rate 5e-4 \
  --max-length 128 \
  --warmup-ratio 0.06 \
  --output-root "$output_root"
