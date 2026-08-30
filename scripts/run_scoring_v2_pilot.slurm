#!/bin/bash
#SBATCH -J scoring_v2_pilot
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH -o logs/scoring_v2_pilot_%j.out
#SBATCH -e logs/scoring_v2_pilot_%j.err

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs save evaluation_results monitor_logs paper_assets/02_experiments/scoring_v2_pilot_traces

module purge
module load cuda/11.8

source ~/.bashrc
source activate bjh_lora || conda activate bjh_lora

export HF_HOME=$SLURM_SUBMIT_DIR/hf_cache
export HF_DATASETS_CACHE=$SLURM_SUBMIT_DIR/hf_cache
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== scoring_v2_pilot ==="
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PILOT_MAX_ROUNDS: ${PILOT_MAX_ROUNDS:-20}"
echo "PILOT_TOTAL_BUDGET: ${PILOT_TOTAL_BUDGET:-2400}"
nvidia-smi

bash scripts/run_scoring_v2_pilot.sh
