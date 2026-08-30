#!/bin/bash
#SBATCH -J SAMSUM_FL_OPT
#SBATCH -p gpuA800
#SBATCH --qos=normal
#SBATCH -A a_xmwang
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH -o logs/samsum_%j.out
#SBATCH -e logs/samsum_%j.err

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs
mkdir -p save
mkdir -p evaluation_results
mkdir -p monitor_logs

module purge
module load cuda/11.8

source ~/.bashrc
source activate bjh_lora || conda activate bjh_lora

export HF_HOME="$SLURM_SUBMIT_DIR/hf_cache"
export HF_DATASETS_CACHE="$SLURM_SUBMIT_DIR/hf_cache"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== 运行环境信息 ==="
echo "节点: $(hostname)"
echo "目录: $(pwd)"
echo "显卡: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "===================="

# echo "-----------------------------------------------------------"
# echo "[1/3] SAMSum + OPT-1.3B + IID"
# echo "-----------------------------------------------------------"
# python -u main_drl.py \
#   --dataset samsum \
#   --distribution iid \
#   --local_bs 1 \
#   --lr 2e-5 \
#   --local_ep 1 \
#   --total_budget 10000 \
#   --max_rounds 40 \
#   --gpu 0 \
#   --num_parallel_envs 1

# echo "-----------------------------------------------------------"
# echo "[2/3] SAMSum + OPT-1.3B + Length Skew Mild"
# echo "-----------------------------------------------------------"
# python -u main_drl.py \
#   --dataset samsum \
#   --distribution length_skew:mild \
#   --local_bs 1 \
#   --lr 2e-5 \
#   --local_ep 1 \
#   --total_budget 10000 \
#   --max_rounds 40 \
#   --gpu 0 \
#   --num_parallel_envs 1

echo "-----------------------------------------------------------"
echo "[3/3] SAMSum + OPT-1.3B + Length Skew Severe"
echo "-----------------------------------------------------------"
python -u main_drl.py \
  --dataset samsum \
  --distribution length_skew:severe \
  --local_bs 1 \
  --lr 2e-5 \
  --local_ep 1 \
  --total_budget 10000 \
  --max_rounds 40 \
  --gpu 0 \
  --num_parallel_envs 1
