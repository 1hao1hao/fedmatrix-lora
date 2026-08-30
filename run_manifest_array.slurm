#!/bin/bash
#SBATCH -J FL_LORA_GRID
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH -o logs/manifest_%A_%a.out
#SBATCH -e logs/manifest_%A_%a.err
#
# Submit example:
#   sbatch --array=1-16 run_manifest_array.slurm
#   sbatch --array=1-16 run_manifest_array.slurm experiments/critical_gap_manifest.csv
#   sbatch --array=1-35 run_manifest_array.slurm experiments/main_grid_manifest.csv
#   sbatch --array=1-6 run_manifest_array.slurm experiments/ablation_representative_manifest.csv

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs save evaluation_results monitor_logs

module purge
module load cuda/11.8

source ~/.bashrc
source activate bjh_lora || conda activate bjh_lora

export HF_HOME=$SLURM_SUBMIT_DIR/hf_cache
export HF_DATASETS_CACHE=$SLURM_SUBMIT_DIR/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

MANIFEST=${1:-experiments/critical_gap_manifest.csv}

echo "=== Manifest array job ==="
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "Manifest: $MANIFEST"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

python -u scripts/run_from_manifest.py \
  --manifest "$MANIFEST" \
  --row "$SLURM_ARRAY_TASK_ID"
