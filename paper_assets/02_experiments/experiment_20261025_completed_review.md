# 2026-10-25 三组实验完成核对

## 范围与结果索引

`plan/experiment_10.25.md` 的评分来源对比（P）、矩阵评分有效性（A）和完整联邦消融（B）均已运行及分析。新结果没有覆盖既有主实验或原始 CSV。试运行仅用于检查流程，以下数字只使用正式运行。

| 部分 | 数据集与方法 | 种子、指标和完整运行索引 | 原始结果 |
|---|---|---|---|
| P/A | CIFAR100/ViT-base；Local、Local-Mean、Server-Proxy；原始及深度校准评分 | IID、Dirichlet 1.0/0.1 各使用 2024/2025/2026；Top-12 Jaccard 距离、独立验证损失下降、Spearman；9 个 `run_id` 及配置路径见 [`run_ledger.csv`](../05_figures/experiment_20261025/matrix_score/run_ledger.csv) | 每个运行的 `scores.json` 和 `short_train/*.csv`；逐配置精确路径见 [`config_results.csv`](../05_figures/experiment_20261025/matrix_score/config_results.csv) |
| B | SST-2/RoBERTa-base，Dirichlet 1.0；Full、Server-Proxy、Layer-Level、No-Depth | 每组使用 2024/2025/2026；最终/最佳验证 Accuracy、实际累计矩阵成本；12 个 `run_id`、配置路径及原始 `eval_*.csv` 见 [`run_summary.csv`](../05_figures/experiment_20261025/sst2_ablation/run_summary.csv) | `evaluation_results/sst2/<run_id>/eval_*.csv`、`metadata.json` 和 `traces/*.csv`；完整路径在运行索引内 |

## 协议与完整性

- **代码事实：** CIFAR100 的代理集、warmup 集与客户端训练/验证集从训练划分中互不相交地拆出，并检查不重叠；验证集不参与画像和训练（[`run_matrix_score_validation_20261025.py`](../../scripts/run_matrix_score_validation_20261025.py)，第 100--133 行）。画像每个本地客户端用 16 个样本，服务器代理评分由 5 个互斥的 16 样本批次平均而成，即共 80 个代理样本；这与 5 个本地客户端的总画像样本数相同，但不等于单个客户端的样本数（同脚本第 204--246 行）。
- **代码与结果事实：** P/A 在早期和独立 warmup 后的中期检查点，各评估 5 个客户端、12 个可行且互异的配置；所有配置使用 12 个矩阵、rank 8、20 步及相同起始适配器。共有 9 个划分/种子运行、90 个客户端×检查点单元、1080 条配置记录；每单元的初始验证损失完全一致，收益均为有限值。配置生成、可行性检查及冻结方式见同脚本第 281--312、343--382 行。
- **代码与审计事实：** 未选矩阵在主比较中保留于前向计算但被冻结，而不是从模型中删除。90 个构建审计中有 45 个的“删除未选矩阵”初始损失不同，最大绝对差为 0.1357；因此不能把两种构建方式的训练收益混作同一对比。审计原件位于各运行 `short_train/*_construction_audit.json`。
- **代码事实：** B 的代理样本从 SST-2 训练划分预留；Server-Proxy 在刷新时使用独立代理数据，Layer-Level 将同块 Q/K/V 绑定，No-Depth 经 `Random_FIM_NoDepth` 关闭深度缩放；相关入口见 [`run_sst2_scoring_ablation_20261025.py`](../../scripts/run_sst2_scoring_ablation_20261025.py) 第 60--95、108--145、178--233 行及 [`server.py`](../../models/server.py) 第 925--934 行。
- **日志与元数据事实：** B 的 12 次正式运行均有 80 条评估记录，最终实际累计矩阵成本均为 7200。同一种子下四组的私有训练集与代理集索引哈希分别一致；`score_trace` 中只有 Server-Proxy 标为 `independent_train_server_proxy`，其他三组为 `local_profile`。Layer-Level 的选中配置成本均为 3 的倍数。所有 12 个原始结果文件存在。

## P：评分来源

对应论文风格双子图：[PDF](../05_figures/experiment_20261025/matrix_score/scoring_source_p_comparison.pdf)；数值见 [`top12_jaccard_distance_summary.csv`](../05_figures/experiment_20261025/matrix_score/top12_jaccard_distance_summary.csv)、[`method_loss_reduction_summary.csv`](../05_figures/experiment_20261025/matrix_score/method_loss_reduction_summary.csv) 和 [`paired_loss_reduction_summary.csv`](../05_figures/experiment_20261025/matrix_score/paired_loss_reduction_summary.csv)。距离定义为 `1 - Jaccard overlap`；原始重合度和距离分别保留，不混用。

| 划分 | Local--Server-Proxy 距离：早期 / 中期 | Local 减 Server-Proxy 的验证损失下降量：早期 / 中期 |
|---|---:|---:|
| IID | 0.103 / 0.111 | -0.00014 / -0.00201 |
| Dirichlet 1.0 | 0.128 / 0.139 | -0.00046 / -0.00100 |
| Dirichlet 0.1 | 0.062 / 0.138 | -0.00556 / +0.00025 |

每个单元格的均值先在同一种子内跨 5 客户端计算，再在 3 个种子间汇总；标准差见 CSV。Top-12 选择存在差异，但交集仍很大，且 Local 的短期损失下降在六个设置中仅一个略高于 Server-Proxy、没有一个高于 Local-Mean。**本实验不支持“本地评分必然带来更大短期收益”的结论。**

画像开销记录在 [`profiling_overhead.csv`](../05_figures/experiment_20261025/matrix_score/profiling_overhead.csv)。18 个运行×检查点的本地客户端最大画像时间均值为 0.084 秒，本地 5 客户端画像时间之和均值为 0.419 秒，5 批代理评分时间之和均值为 0.553 秒。这些是插桩阶段时间，不能直接当作完整轮次时延或十台设备的实测并行时间。

## A：矩阵评分有效性

按“客户端×检查点”而不是跨不同客户端混算的 Spearman 结果见 [`client_checkpoint_spearman.csv`](../05_figures/experiment_20261025/matrix_score/client_checkpoint_spearman.csv)、[`spearman_summary.csv`](../05_figures/experiment_20261025/matrix_score/spearman_summary.csv) 和[散点图 PDF](../05_figures/experiment_20261025/matrix_score/predicted_utility_vs_gain.pdf)。深度校准评分的 90/90 个相关系数为正；六个划分×检查点的三种子均值介于 0.653 与 0.878。原始评分也应结合表内数值阅读，不能说深度校准在每个设置都更好，例如 Dirichlet 0.1 的早期原始评分均值 0.885 高于校准评分 0.770。

结论限于本协议中生成的 12 个可行配置和 20 步短期验证损失：预测效用具有正的组内排序相关性。这不是对长期训练准确率或所有可行配置的保证。

## B：完整联邦消融

三种子统计见 [`three_seed_mean_std.csv`](../05_figures/experiment_20261025/sst2_ablation/three_seed_mean_std.csv)；[等累计预算曲线 PDF](../05_figures/experiment_20261025/sst2_ablation/accuracy_vs_equal_budget.pdf) 展示各次轨迹与均值。

| 方法 | 最终 Accuracy 均值±样本标准差 | 最佳 Accuracy 均值±样本标准差 | 实际累计成本 |
|---|---:|---:|---:|
| Full | 0.9060 ± 0.0072 | 0.9087 ± 0.0050 | 7200 |
| Server-Proxy | 0.9047 ± 0.0081 | 0.9080 ± 0.0087 | 7200 |
| Layer-Level | 0.9007 ± 0.0133 | 0.9073 ± 0.0042 | 7200 |
| No-Depth | 0.9047 ± 0.0061 | 0.9093 ± 0.0042 | 7200 |

Full 的最终均值分别高于 Server-Proxy、Layer-Level、No-Depth 约 0.13、0.53、0.13 个百分点；但幅度小且只有三个种子。No-Depth 的最佳均值比 Full 高约 0.07 个百分点，故不能声称深度校准在最终与最佳准确率上都取得优势。三种子结果支持“在本设置下 Full 最终均值最高”，不支持更强的普遍优势结论。

## 状态

计划要求的可复现入口、原始 CSV、运行账本、均值/标准差、配对比较、向量图及客观结论均已交付。没有启动新的训练任务；没有将本批结果自动写入论文的 P/A/B 占位，因为 P 的局部优势及 B 的深度校准优势未得到明确验证，论文表述需要单独审定。Slurm 数据库查询在当前受限环境中返回 `Operation not permitted`，完成状态依据完整结果文件、严格分析脚本成功退出和运行日志判定。
