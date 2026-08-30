#!/bin/bash
#SBATCH -J GLUE_FL_SOTA         # 作业名
#SBATCH -p gpuA800              # 分区名 (强制要求)
#SBATCH --qos=normal            # 服务质量 (强制要求)
#SBATCH -A a_xmwang             # 账户名 (强制要求)
#SBATCH -N 1                    # 节点数
#SBATCH -n 1                    # 任务数
#SBATCH --gres=gpu:1            # 申请 1 块 GPU
#SBATCH --cpus-per-task=8       # [防扣费关键] 申请 8 核，绝对不要超过 9！
#SBATCH --mem=128G              # 内存给足，防止 DeBERTa 多线程加载 OOM
#SBATCH --time=96:00:00         # 申请 48 小时
#SBATCH -o logs/glue_%j.out     # 标准输出日志
#SBATCH -e logs/glue_%j.err     # 错误日志

# 1. 确保当前目录是提交脚本的目录 (项目根目录)
cd $SLURM_SUBMIT_DIR

# 2. 准备必需的文件夹 (防止由于路径不存在报错)
mkdir -p logs
mkdir -p save
mkdir -p evaluation_results
mkdir -p monitor_logs

# 3. 加载集群环境模块
module purge                   # 清除默认模块
module load cuda/11.8          # 加载 CUDA

source ~/.bashrc

# 4. 激活虚拟环境
source activate bjh_lora || conda activate bjh_lora

# =========================================================
# [核心配置] 离线模式与缓存路径
# 强制 HuggingFace 直接去本地寻找，严禁联网报错
# =========================================================
export HF_HOME=$SLURM_SUBMIT_DIR/hf_cache
export HF_DATASETS_CACHE=$SLURM_SUBMIT_DIR/hf_cache
export HF_HUB_OFFLINE=1
# 限制 PyTorch 矩阵运算的底层 CPU 线程数，防止 10 个客户端并发时把 CPU 彻底卡死
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# 5. 打印调试信息
echo "=== 运行环境信息 ==="
echo "节点: $(hostname)"
echo "目录: $(pwd)"
echo "显卡: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "===================="

echo "-----------------------------------------------------------"

python -u main_drl.py \
    --dataset qqp \
    --distribution iid \
    --local_bs 16 \
    --lr 5e-5 \
    --local_ep 1 \
    --total_budget 2500 \
    --max_rounds 2 \
    --gpu 0 \
    --num_parallel_envs 5
# 注意：只跑 2 轮！