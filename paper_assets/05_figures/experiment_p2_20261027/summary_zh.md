# P2 固定协议结果

共保留 360 条方法记录；不可行记录 0 条。
复用原 P 非首配置的方法记录 57 条；本次新训练去重后 244 个配置。

正值表示矩阵级选择的 20 步验证损失下降大于完整层绑定；误差为先在每个种子内跨 5 客户端求均值、再跨 3 种子计算的样本标准差。

| 划分 | 检查点 | 来源 | Matrix-Layer 均值±标准差 |
|---|---|---|---:|
| iid | early | Local | +0.00219 ± 0.00140 |
| iid | early | Proxy | +0.00191 ± 0.00125 |
| iid | middle | Local | +0.00669 ± 0.00319 |
| iid | middle | Proxy | +0.00582 ± 0.00591 |
| dirichlet:1.0 | early | Local | +0.01328 ± 0.00211 |
| dirichlet:1.0 | early | Proxy | +0.01335 ± 0.00335 |
| dirichlet:1.0 | middle | Local | +0.01411 ± 0.00179 |
| dirichlet:1.0 | middle | Proxy | +0.01130 ± 0.00330 |
| dirichlet:0.1 | early | Local | +0.13752 ± 0.01708 |
| dirichlet:0.1 | early | Proxy | +0.13907 ± 0.01669 |
| dirichlet:0.1 | middle | Local | +0.03412 ± 0.00757 |
| dirichlet:0.1 | middle | Proxy | +0.02726 ± 0.01002 |

逐种子配对均值：35/36 为正；逐客户端配对：161/180 为正。负值也完整保存在 paired_seed_differences.csv 和 paired_client_differences.csv。
图中各划分采用独立纵轴刻度，以便同时看清 IID 与 Dirichlet 0.1 的差值。

本实验保持 rank=8、12 个矩阵、相同评分来源/深度校准/归一化、原 P 的训练及验证划分；没有调参或删选负结果。
跨划分比较不能单独归因于数据异构性，因为原 P 深度系数也随划分变化。
