# 2026-10-25 矩阵评分与消融实验提交记录

依据 `plan/experiment_10.25.md` 执行。以下均为新实验，不覆盖历史原始结果。截至 2026-10-27，三部分正式实验和分析均已完成；表中的作业 ID 是最初提交链，失败后的补跑使用了新的 ID，实际运行 ID 以结果账本为准。

| 实验 | 数据集与划分 | 方法 | 种子 | 指标 | 作业 ID | 配置/脚本 | 原始结果路径 | 状态 |
|---|---|---|---|---|---|---|---|---|
| 评分来源与效用有效性，试运行 | CIFAR100 / IID | Local、Local-Mean、Server-Proxy；原始/深度评分 | 2024 | Top-12 Jaccard；验证损失下降；Spearman | 准备 1764157；评估 1764158 | `scripts/run_matrix_score_validation_20261025.py`、`scripts/run_matrix_score_validation_20261025.slurm` | `paper_assets/02_experiments/matrix_score_20261025/pilot/cifar100_iid_seed2024/` | 已完成；仅用于冒烟检查 |
| 评分来源与效用有效性，正式实验 | CIFAR100 / IID、Dirichlet:1.0、Dirichlet:0.1 | Local、Local-Mean、Server-Proxy；原始/深度评分 | 2024、2025、2026 | Top-12 Jaccard 距离；20-step 验证损失下降；Spearman | 准备 1764159；评估 1764160；分析 1764161 | `scripts/run_matrix_score_validation_20261025.py`、`scripts/analyze_experiment_20261025.py` | `paper_assets/02_experiments/matrix_score_20261025/full/cifar100_{iid,dirichlet_1.0,dirichlet_0.1}_seed{2024,2025,2026}/` | 已完成：9 组、90 个客户端×检查点、1080 个配置 |
| 联邦消融，试运行 | SST-2 / Dirichlet:1.0 | Full、Server-Proxy、Layer-Level、No-Depth | 2024 | 验证准确率；实际矩阵预算；选择轨迹 | 1764162 | `scripts/run_sst2_scoring_ablation_20261025.py`、`scripts/run_sst2_scoring_ablation_20261025.slurm` | `evaluation_results/sst2/SCORE_ABL_20261025_SST2_D1_*_seed2024_PILOT_*/traces/` 及同目录 `eval_*.csv` | 已完成；仅用于冒烟检查 |
| 联邦消融，正式实验 | SST-2 / Dirichlet:1.0 | Full、Server-Proxy、Layer-Level、No-Depth | 2024、2025、2026 | 最终/最佳验证准确率；相同累计预算曲线 | 1764163；分析 1764164 | `scripts/run_sst2_scoring_ablation_20261025.py`、`scripts/analyze_experiment_20261025.py` | `evaluation_results/sst2/SCORE_ABL_20261025_SST2_D1_{Full,ServerProxy,LayerLevel,NoDepth}_seed{2024,2025,2026}_*/` | 已完成：12 次正式运行，每次 80 轮 |

说明：CIFAR100 的 run_id 为结果目录名；SST-2 的 run_id 包含实际 Slurm job ID，具体路径以各运行目录内 `metadata.json` 为准。CIFAR100 的代理集、独立 warmup 集、客户端训练/验证集均从训练划分拆出；SST-2 的代理样本从训练划分预留，所有四组均使用剩余的同一客户端数据。两项实验均不使用测试集做评分。

图与汇总已由分析脚本生成至 `paper_assets/05_figures/experiment_20261025/`。CIFAR100 的九组运行索引见 `matrix_score/run_ledger.csv`，逐配置原始路径见 `matrix_score/config_results.csv`；SST-2 的十二次运行索引见 `sst2_ablation/run_summary.csv`。分析按完整集合严格检查，没有把缺失结果补零。客观统计和方法检查见 `paper_assets/02_experiments/experiment_20261025_completed_review.md`。

## 2026-10-26 启动故障与重提

原试运行 1761615 和 1761620 均失败，日志分别位于 `logs/matrix_score_1025_1761615_0.err` 与 `logs/sst2_score_abl_1025_1761620_0.err`，报错同为 `/etc/bashrc: line 12: BASHRCSOURCED: unbound variable`。这是 Slurm 脚本在加载 shell 环境之前启用 `set -u` 引起的启动故障，尚未进入训练。已将三个 20261025 Slurm 脚本中的 `set -u` 移到环境激活之后，`bash -n` 检查通过；取消失效的旧依赖作业并提交新作业链。正式结果现已齐全，失败的启动尝试未计入统计。
