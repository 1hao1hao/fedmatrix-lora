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


# =================================================================================
#
# random_agent = RandomStrategy(low=0.15, high=0.5, seed=12345)
#
# =================================================================================
# =========================================================
# 6. 依次执行 4 个 GLUE 任务对比实验
# =========================================================
# =================================================================================
# 【第一组】 IID (理想同分布) 实验
# 目的: 作为性能上限，展示系统在理想状态下的基准表现
# 策略: 正常学习率 (5e-5)，常规轮次 (40轮)
# =================================================================================

# echo "-----------------------------------------------------------"
# echo "[1/6] 开始运行 SST-2 (IID) + RoBERTa-base"
# echo "-----------------------------------------------------------"
# python -u main_drl.py \
#     --dataset sst2 \
#     --distribution iid \
#     --local_bs 32 \
#     --lr 5e-5 \
#     --local_ep 1 \
#     --total_budget 1800\
#     --max_rounds 20\
#     --gpu 0 \
#     --num_parallel_envs 8

# echo "-----------------------------------------------------------"
# echo "[2/6] 开始运行 QNLI (IID) + RoBERTa-base"
# echo "-----------------------------------------------------------"
# python -u main_drl.py \
#     --dataset qnli \
#     --distribution iid \
#     --local_bs 32 \
#     --lr 5e-5 \
#     --local_ep 1 \
#     --total_budget 3500 \
#     --max_rounds 40 \
#     --gpu 0 \
#     --num_parallel_envs 8

# # =================================================================================
# # 【第二组】 中度 Non-IID (dirichlet: 1)
# # 目的: 探究算法在数据中度倾斜下的抗压能力
# # =================================================================================
# echo "==========================================================="
# echo "[3/6] 开始运行 SST-2 (情感分析) + RoBERTa-base"
# echo "==========================================================="
# python -u main_drl.py \
#     --dataset sst2 \
#     --distribution dirichlet:1.0 \
#     --local_bs 32 \
#     --lr 5e-5 \
#     --local_ep 1 \
#     --total_budget 1800\
#     --max_rounds 20\
#     --gpu 0 \
#     --num_parallel_envs 8

# echo "==========================================================="
# echo "[4/6] 开始运行 QNLI (自然语言推理) + RoBERTa-base"
# echo "==========================================================="
# python -u main_drl.py \
#     --dataset qnli \
#     --distribution dirichlet:1.0 \
#     --local_bs 32 \
#     --lr 5e-5 \
#     --local_ep 1 \
#     --total_budget 3500 \
#     --max_rounds 40 \
#     --gpu 0 \
#     --num_parallel_envs 8

# # =================================================================================
# # 【第三组】 极端 Non-IID (dirichlet:0.1) 重跑实验
# # 目的: 探究算法在数据极度倾斜下的抗压能力
# # 策略: 极低学习率 (2e-5) 防止梯度爆炸，适当延长训练轮次 (50轮) 和总预算 (4000)
# # =================================================================================

# echo "-----------------------------------------------------------"
# echo "[5/6] 重跑 SST-2 (Dirichlet: 0.1) + RoBERTa-base"
# echo "-----------------------------------------------------------"
# python -u main_drl.py \
#     --dataset sst2 \
#     --distribution dirichlet:0.1 \
#     --local_bs 32 \
#     --lr 2e-5 \
#     --local_ep 1 \
#     --total_budget 2200 \
#     --max_rounds 25 \
#     --gpu 0 \
#     --num_parallel_envs 8

# echo "-----------------------------------------------------------"
# echo "[6/6] 重跑 QNLI (Dirichlet: 0.1) + RoBERTa-base"
# echo "-----------------------------------------------------------"
# python -u main_drl.py \
#     --dataset qnli \
#     --distribution dirichlet:0.1 \
#     --local_bs 32 \
#     --lr 2e-5 \
#     --local_ep 1 \
#     --total_budget 4400 \
#     --max_rounds 50 \
#     --gpu 0 \
#     --num_parallel_envs 8

# "---------------------------------------------------------------------------------------------------------"
# =================================================================================
# 【第一组】 IID (理想同分布)
# 目的: 作为性能上限，展示系统在理想状态下的基准表现
# 任务: QQP / MNLI (DeBERTa-Large, 24层, 72个LoRA矩阵)
# 策略: 降低 local_bs 防 OOM，预算翻倍以匹配 72 个矩阵的规模，轮次设为 50 轮
# =================================================================================
echo "-----------------------------------------------------------"
echo "[1/6] 开始运行 QQP (问题等价性 - IID) + DeBERTa-Large"
echo "-----------------------------------------------------------"
python -u main_drl.py \
    --dataset qqp \
    --distribution iid \
    --local_bs 16 \
    --lr 5e-5 \
    --local_ep 1 \
    --total_budget 9500 \
    --max_rounds 50 \
    --gpu 0 \
    --num_parallel_envs 8

echo "-----------------------------------------------------------"
echo "[2/6] 开始运行 MNLI (多体裁推断 - IID) + DeBERTa-Large"
echo "-----------------------------------------------------------"
python -u main_drl.py \
    --dataset mnli \
    --distribution iid \
    --local_bs 16 \
    --lr 5e-5 \
    --local_ep 1 \
    --total_budget 9500 \
    --max_rounds 50 \
    --gpu 0 \
    --num_parallel_envs 8

=================================================================================
【第二组】 中度 Non-IID (dirichlet: 1.0)
目的: 探究算法在数据中度倾斜下的抗压能力
=================================================================================
echo "==========================================================="
echo "[3/6] 开始运行 QQP (问题等价性 - Dir:1.0) + DeBERTa-Large"
echo "==========================================================="
python -u main_drl.py \
    --dataset qqp \
    --distribution dirichlet:1.0 \
    --local_bs 16 \
    --lr 5e-5 \
    --local_ep 1 \
    --total_budget 9500 \
    --max_rounds 50 \
    --gpu 0 \
    --num_parallel_envs 8

echo "==========================================================="
echo "[4/6] 开始运行 MNLI (多体裁推断 - Dir:1.0) + DeBERTa-Large"
echo "==========================================================="
python -u main_drl.py \
    --dataset mnli \
    --distribution dirichlet:1.0 \
    --local_bs 16 \
    --lr 5e-5 \
    --local_ep 1 \
    --total_budget 9500 \
    --max_rounds 50 \
    --gpu 0 \
    --num_parallel_envs 8

=================================================================================
【第三组】 极端 Non-IID (dirichlet:0.1)
目的: 探究算法在数据极度倾斜下的抗压能力
策略: 极低学习率 (2e-5) 防止梯度爆炸，延长训练轮次 (60轮) 和总预算 (10500)
=================================================================================
echo "-----------------------------------------------------------"
echo "[5/6] 重跑 QQP (问题等价性 - Dir:0.1) + DeBERTa-Large"
echo "-----------------------------------------------------------"
python -u main_drl.py \
    --dataset qqp \
    --distribution dirichlet:0.1 \
    --local_bs 16 \
    --lr 2e-5 \
    --local_ep 1 \
    --total_budget 10500 \
    --max_rounds 60 \
    --gpu 0 \
    --num_parallel_envs 8

echo "-----------------------------------------------------------"
echo "[6/6] 重跑 MNLI (多体裁推断 - Dir:0.1) + DeBERTa-Large"
echo "-----------------------------------------------------------"
python -u main_drl.py \
    --dataset mnli \
    --distribution dirichlet:0.1 \
    --local_bs 16 \
    --lr 2e-5 \
    --local_ep 1 \
    --total_budget 10500 \
    --max_rounds 60 \
    --gpu 0 \
    --num_parallel_envs 8

echo "🎉 所有 GLUE 实验运行完毕！"