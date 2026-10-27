# 实验 P / A / B 集中审阅包

本目录把 `plan/experiment_10.25.md` 对应的三组已完成实验集中展示。图来自已完成运行的 CSV；未重跑训练、未修改原始结果。图中误差条为**三个种子所得均值的样本标准差**，不是置信区间。代码和完整方法核对见 [`experiment_20261025_completed_review.md`](../../../02_experiments/experiment_20261025_completed_review.md)。

| 实验 | 要回答的问题 | 主图 | 数据 | 实际观察 |
|---|---|---|---|---|
| **P：评分来源** | 本地评分、客户端均值评分、服务器代理评分是否选出不同矩阵；短期训练收益是否不同？ | [PDF](experiment_P.pdf) / [PNG](experiment_P.png) | [选择距离](P_selection_distance.csv)、[配对收益差](P_paired_gain.csv)、[三方法绝对收益](P_method_gain.csv)、[画像开销](P_profiling_overhead.csv)、[运行索引](P_run_ledger.csv) | 选择有差异，但 Top-12 仍高度重合；Local 的短期收益未明显优于两种共享评分。P(b) 的零线表示两方法收益相同，负值表示 Local 更低。 |
| **A：评分有效性** | 配置预测效用与 20 步验证损失下降是否同向排序？ | [PDF](experiment_A.pdf) / [PNG](experiment_A.png) | [客户端×检查点相关系数](A_spearman_cells.csv)、[三种子汇总](A_spearman_summary.csv) | 校准评分在 90/90 个客户端×检查点单元的 Spearman 系数为正；但原始评分在一些设置中更强。 |
| **B：完整联邦消融** | 本地评分、矩阵粒度、深度校准对 SST-2 有多少实际贡献？ | [PDF](experiment_B.pdf) / [PNG](experiment_B.png) | [12 次运行索引](B_run_summary.csv)、[三种子端点](B_three_seed_summary.csv)、[逐轮预算曲线](B_round_curves.csv) | Full 最终准确率均值最高（0.9060），但优势较小；No-Depth 的最佳准确率均值略高于 Full（0.9093 对 0.9087）。 |

## 阅读口径

- P(a) 为 `1 - Jaccard overlap`，只比较 **Local 与 Server-Proxy** 的 Top-12 选择；P(b) 为同一客户端、种子和检查点的 **Local 减对照**验证损失下降量。早期与中期都展示，IID 与两种 Dirichlet 划分都展示。
- A 的 Spearman 在每个客户端、每个检查点的 12 个资源可行配置上计算；汇总先平均客户端，再计算三个种子的均值和标准差。正相关仅说明本协议中的短期排序关系，不等于最终准确率保证。
- B(a) 是相同累计矩阵等价预算上的三种子**均值**，不画逐种子细线；B(b) 才显示种子间标准差。所有四组均完成 80 轮，实际累计预算均为 7200。两图不是硬件实测时间曲线。
- CIFAR100 的实验 P/A 原始 `scores.json` 和 `short_train/*.csv` 仍保留在本地 `paper_assets/02_experiments/matrix_score_20261025/full/`；SST-2 原始 `eval_*.csv` 和选择轨迹仍在本地 `evaluation_results/sst2/<run_id>/`。本审阅包提供可复核的逐单元/逐轮派生数据及精确 `run_id`、配置和原始路径，未将大型原始目录纳入 Git。

重建本目录：`python scripts/analyze_experiment_20261025.py all`，然后 `python scripts/build_experiment_20261025_review.py`。这两步仅分析已有结果，不启动训练。
