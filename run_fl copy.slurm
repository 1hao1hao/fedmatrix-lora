#!/bin/bash
#SBATCH -J DRL_FL_Job           # 作业名
#SBATCH -p gpuA800              # [截图确认] 分区名，必须填这个
#SBATCH --qos=normal            # [截图确认] 服务质量，保持默认
#SBATCH -A a_xmwang             # [截图确认] 账户名 (非常重要！缺少这个可能导致权限拒绝)
#SBATCH -N 1                    # 节点数
#SBATCH -n 1                    # 任务数
#SBATCH --gres=gpu:1            # [截图确认] 申请显卡数 (截图中是1，你想跑快点可以试着改2)
#SBATCH --cpus-per-task=8     # [核心修改] 拉高到 32 核，为数据加载提供充足算力
#SBATCH --mem=128G             # [建议] 内存也稍微给大点，防止多线程数据加载 OOM
#SBATCH --time=48:00:00         # [修改] 截图默认30分钟太短，这里改为48小时
#SBATCH -o logs/job_%j.out      # 标准输出日志
#SBATCH -e logs/job_%j.err      # 错误日志

# 1. 确保当前目录是提交脚本的目录 (项目根目录)
cd $SLURM_SUBMIT_DIR
# 2. 准备日志文件夹 (防止报错)
mkdir -p logs
mkdir -p save
mkdir -p ppo_tensorboard_cifar100
mkdir -p evaluation_results

# 3. 加载环境
module purge                   # 清除默认模块
module load cuda/11.8          # 加载 CUDA (版本需匹配你的 PyTorch)

source ~/.bashrc

# 4. 激活虚拟环境
# 有些集群需要先 source conda.sh，如果 source activate 报错，尝试用 conda activate
source activate bjh_lora || conda activate bjh_lora
# =========================================================
# [核心修复] 配置离线模式与缓存路径
# 告诉 HuggingFace 直接去项目目录下的 hf_cache 找文件，不要联网
# =========================================================
export HF_HOME=$SLURM_SUBMIT_DIR/hf_cache
export HF_DATASETS_CACHE=$SLURM_SUBMIT_DIR/hf_cache
export HF_HUB_OFFLINE=1  # 强制离线模式，双重保险
# 5. 调试信息
echo "=== 运行环境信息 ==="
echo "节点: $(hostname)"
echo "目录: $(pwd)"
echo "显卡: $CUDA_VISIBLE_DEVICES"
nvidia-smi
echo "===================="


# 4. 运行代码
python -u main_drl.py \
    --dataset cifar100 \
    --distribution dirichlet:0.1 \
    --local_bs 32 \
    --lr 1e-4 \
    --local_ep 1 \
    --total_budget 6000 \
    --max_rounds 100 \
    --gpu 0  \
    --num_parallel_envs 8  # 如果申请了多核CPU，可以适当增加并行环境数



echo "Job End Time: $(date)"