#!/bin/bash
#SBATCH -J matrix_score_1025
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/matrix_score_1025_%A_%a.out
#SBATCH -e logs/matrix_score_1025_%A_%a.err

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

stage="${1:?pass prepare or evaluate}"
mode="${2:-full}"
distributions=(iid dirichlet:1.0 dirichlet:0.1)
seeds=(2024 2025 2026)
task="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$mode" == pilot ]]; then
  distribution=iid
  seed=2024
  client=0
  checkpoint=early
  extra=(--pilot)
else
  if [[ "$stage" == prepare ]]; then
    group="$task"
  else
    group=$((task / 10))
    subtask=$((task % 10))
    client=$((subtask % 5))
    if ((subtask < 5)); then checkpoint=early; else checkpoint=middle; fi
  fi
  distribution="${distributions[$((group % 3))]}"
  seed="${seeds[$((group / 3))]}"
  extra=()
fi

if [[ "$stage" == evaluate ]]; then
  extra+=(--client "$client" --checkpoint "$checkpoint")
fi
echo "stage=$stage mode=$mode task=$task distribution=$distribution seed=$seed extra=${extra[*]}"
python -u scripts/run_matrix_score_validation_20261025.py "$stage" \
  --distribution "$distribution" --seed "$seed" "${extra[@]}"
