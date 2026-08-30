#!/bin/bash
#SBATCH -J Layer_Test           # 作业名
#SBATCH -p gpuA800              # [必须] 分区名
#SBATCH --qos=normal            # [必须] 服务质量
#SBATCH -A a_xmwang             # [必须] 账户名
#SBATCH -N 1                    # 节点数
#SBATCH -n 1                    # 任务数
#SBATCH --gres=gpu:1            # 申请 1 张显卡
#SBATCH --cpus-per-task=8       # 申请 8 个 CPU 核心 (给 DataLoader 用)
#SBATCH --mem=32G               # 申请 32GB 内存 (单脚本测试足够了)
#SBATCH --time=02:00:00         # 运行时间限制 (2小时足够跑完这200步)
#SBATCH -o logs/test_%j.out      # 标准输出日志
#SBATCH -e logs/test_%j.err      # 错误日志

# 1. 确保当前目录是提交脚本的目录
cd $SLURM_SUBMIT_DIR

# 2. 加载必要的模块
module purge
module load cuda/11.8

# 3. 激活虚拟环境
source ~/.bashrc
source activate bjh_lora || conda activate bjh_lora

# =========================================================
# 4. [极其重要] 离线模式与缓存路径配置
# 保证断网环境下能读到之前下载的 ViT 模型和数据集
# =========================================================
export HF_HOME=$SLURM_SUBMIT_DIR/hf_cache
export HF_DATASETS_CACHE=$SLURM_SUBMIT_DIR/hf_cache
export HF_HUB_OFFLINE=1

# 5. 打印调试信息
echo "================================================="
echo "Job Start Time: $(date)"
echo "节点: $(hostname)"
echo "显卡: $CUDA_VISIBLE_DEVICES"
echo "================================================="

# 6. 运行测试脚本
python -u test_layer_importance.py

echo "================================================="
echo "Job End Time: $(date)"
echo "================================================="