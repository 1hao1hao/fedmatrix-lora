#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# 文件: utils/options.py
import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # ===== 提速与并发参数 (新增) =====
    parser.add_argument('--test_method', type=str, default='all',
                        help="指定要评估的方法，如 Ours_FIM、Ours_Taylor、FedAvg_LoRA_homogeneous、HETLoRA、FAH_QLoRA。设为 all 则串行跑主实验方法。")
    parser.add_argument('--experiment_id', type=str, default='',
                        help="可选实验 ID；用于 metadata.json 和 manifest/registry 对齐。")
    parser.add_argument('--data_subsample_ratio', type=float, default=0.1,
                        help="安全缩减训练集比例 (例如 0.1 表示只使用 10% 的训练数据)")

    # ===== 并行化参数 =====
    parser.add_argument('--num_parallel_envs', type=int, default=1,
                        help="保留的兼容参数；当前非强化学习实验入口每个方法使用单个环境。")
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help="DataLoader worker 数；CIFAR/ViT 多客户端并发不稳定时可设为 0。")
    parser.add_argument('--max_client_train_workers', type=int, default=10,
                        help="单轮客户端本地训练线程池最大 worker 数；默认保持原并行度。")

    # ===== 数据分布异质性参数 =====
    parser.add_argument('--distribution', type=str, default='iid',
                        help="数据分布设置。格式: 'iid', 'patho:X', 'dirichlet:Y', 'X/Y'。"
                             "例如: 'patho:10' 或 'dirichlet:0.5' 或 '20/1.0'")

    # ===== 资源约束与目标参数 =====
    parser.add_argument('--total_budget', type=int, default=7000,
                        help="一次完整联邦微调流程的总预算 (例如, 可分配的总层数)")
    parser.add_argument('--max_rounds', type=int, default=80,
                        help="一次完整联邦微调流程的最大轮次数上限")
    # [核心修改] 将损失目标改为通用的性能目标
    # 注意: 您需要为不同的任务设置一个合理的 performance_target。例如，CIFAR-100 的目标可能是 0.75，而 IMDB 的目标可能是 0.90。
    parser.add_argument('--performance_target', type=float, default=0.3,
                        help="模型微tuning的最终性能目标 (例如, 准确率达到 0.75)")

    parser.add_argument('--fixed_t_max', type=float, default=200.0,
                        help="固定的最大容忍延迟")
    parser.add_argument('--enable_trace', action='store_true',
                        help="写出 selection/score/memory/scheduler trace CSV，用于 heatmap 和 feasibility 分析。")
    parser.add_argument('--trace_dir', type=str, default='',
                        help="可选 trace 输出目录；默认写入 evaluation_results/{dataset}/{run_id}/traces。")

    # ===== 联邦学习核心参数 =====
    parser.add_argument('--epochs', type=int, default=5, help="联邦学习的通信轮次")
    parser.add_argument('--num_clients', type=int, default=10, help="客户端总数")
    # parser.add_argument('--frac', type=float, default=0.1, help="每轮选择的客户端比例")

    # ===== 模型与数据参数 =====
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Hugging Face 模型名称')
    # [核心修改] 增加数据集选择功能
    parser.add_argument('--dataset', type=str, default='sst2',
                        choices=['imdb', 'cifar100', 'domainnet', 'ledgar', 'natural_instructions', 'dolly15k', 'samsum',
                                 'sst2', 'qnli', 'qqp', 'mnli'], # 新增了 4 个 GLUE 任务
                        help='选择要使用的数据集')
    parser.add_argument('--samsum_train_size', type=int, default=10000,
                        help='SAMSum 训练集抽样大小；-1 表示使用完整 train split')
    parser.add_argument('--samsum_val_size', type=int, default=500,
                        help='SAMSum 验证集大小；-1 表示使用完整 validation split')
    parser.add_argument('--samsum_test_size', type=int, default=-1,
                        help='SAMSum 测试集大小；-1 表示使用完整 test split')

    # [核心修改] 将 num_classes 设为 None，由数据加载时动态确定
    parser.add_argument('--num_classes', type=int, default=None, help='分类任务的类别数 (由数据集自动确定)')
    parser.add_argument('--max_length', type=int, default=256, help='文本最大长度 (用于语言任务)')


    # ===== 本地训练参数 =====
    parser.add_argument('--local_ep', type=int, default=1, help="客户端本地训练的 epoch 数")
    parser.add_argument('--local_bs', type=int, default=1, help="客户端本地训练的 batch size")
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--optimizer', type=str, default='adamw', help='优化器类型')

    # ===== LoRA 异构微调算法参数 =====
    # parser.add_argument('--total_lora_layers', type=int, default=48, help="模型中可微调的 LoRA 总层数 (BERT-base 有 24 个)")
    parser.add_argument('--lambda_prime', type=float, default=0.1, help="集成调度中的延迟惩罚系数")
    parser.add_argument('--taylor_fim_blend', type=float, default=0.05,
                        help="Taylor importance 的 FIM 稳定项系数，用于避免 LoRA 近零初始化时 Taylor 分数退化。")
    parser.add_argument('--depth_non_iid_extra', type=float, default=1.0,
                        help="普通 non-IID 设置的深度权重额外指数。")
    parser.add_argument('--depth_severe_extra', type=float, default=2.0,
                        help="Dirichlet alpha<=0.1 等强异构设置的深度权重额外指数。")
    parser.add_argument('--use_score_ema', action='store_true',
                        help="启用客户端本地 FIM/Taylor 分数 EMA 平滑。默认关闭以保持旧行为。")
    parser.add_argument('--score_ema_beta', type=float, default=0.8,
                        help="客户端分数 EMA 的 beta。")
    parser.add_argument('--score_stability_eps', type=float, default=1e-8,
                        help="EMA stable score 的数值稳定项。")
    parser.add_argument('--score_utility_mode', type=str, default='raw',
                        choices=['raw', 'ema', 'depth_mix', 'agreement', 'full'],
                        help="分数效用模式。raw 保持旧行为；其他模式启用 utility-aware scoring。")
    parser.add_argument('--cifar100_prior_mode', type=str, default='legacy_prior',
                        choices=['legacy_prior', 'generic'],
                        help="CIFAR100 的评分路径；generic 仅使用通用 FIM、深度缩放和资源调度。")
    parser.add_argument('--baseline_granularity', type=str, default='matrix',
                        choices=['matrix', 'physical'],
                        help="FedAvg/Uniform baseline 的资源分配粒度；matrix 保持旧结果，physical 用物理层整组分配。")
    parser.add_argument('--fedavg_physical_policy', type=str, default='uniform',
                        choices=['uniform', 'prefix'],
                        help="FedAvg 在 physical 粒度下的同构物理层选择策略；默认 uniform 保持旧行为。")
    parser.add_argument('--fedhello_anchor_layers', type=int, default=0,
                        help="Fed-HeLLo 在原有 FIM 层选择前固定的最浅物理层数量；0 保持原行为。")
    parser.add_argument('--depth_prior_weight', type=float, default=0.0,
                        help="非自适应 depth prior mixture 权重；0 表示完全使用归一化局部分数。")
    parser.add_argument('--adaptive_depth_mix', action='store_true',
                        help="启用基于本地 FIM 浅/深层对比的双向位置先验混合。")
    parser.add_argument('--depth_mix_min', type=float, default=0.4,
                        help="旧版置信度混合的兼容参数；双向位置校准不使用。")
    parser.add_argument('--depth_mix_max', type=float, default=0.8,
                        help="旧版置信度混合的兼容参数；双向位置校准不使用。")
    parser.add_argument('--position_prior_max_weight', type=float, default=0.45,
                        help="双向自适应位置先验的最大混合权重。")
    parser.add_argument('--position_signal_temperature', type=float, default=0.15,
                        help="浅/深层 FIM 对比映射为位置先验权重时的温度。")
    parser.add_argument('--position_band_fraction', type=float, default=1.0 / 3.0,
                        help="估计浅层与深层效用时两端各自覆盖的 block 比例。")
    parser.add_argument(
        '--position_calibration_mode',
        type=str,
        default='legacy',
        choices=['legacy', 'none', 'fixed_deep', 'adaptive'],
        help=(
            "位置校准模式。legacy 保持历史参数组合；none 仅使用原始 FIM；"
            "fixed_deep 使用固定深层增强；adaptive 根据本地 FIM 自动选择浅层或深层方向。"
        ),
    )
    parser.add_argument('--agreement_weight', type=float, default=0.0,
                        help="跨客户端 agreement prior 的加性权重。")
    parser.add_argument('--agreement_eps', type=float, default=1e-8,
                        help="跨客户端 agreement 计算的数值稳定项。")
    parser.add_argument('--marginal_memory_penalty', type=float, default=0.0,
                        help="客户端 DP 转移中的边际显存惩罚系数；0 保持旧行为。")
    parser.add_argument('--marginal_latency_penalty', type=float, default=0.0,
                        help="客户端 DP 转移中的边际延迟惩罚系数；0 保持旧行为。")
    parser.add_argument('--use_block_coherence', action='store_true',
                        help="客户端 DP 中启用同一 block/group 的软 coherence bonus。")
    parser.add_argument('--block_coherence_weight', type=float, default=0.0,
                        help="block coherence bonus 权重。")
    parser.add_argument('--qkv_completeness_weight', type=float, default=0.0,
                        help="q/k/v 完整性 bonus 权重；0 保持旧行为。")
    parser.add_argument('--log_selection_analysis', action='store_true',
                        help="写出 selection_analysis_{strategy}.csv，记录分数分解和选择结果。")
    parser.add_argument('--log_update_cosine', action='store_true',
                        help="写出 update_cosine_{strategy}.csv，记录客户端更新方向一致性。")
    parser.add_argument('--memory_model', type=str, default='position', choices=['position', 'additive'],
                        help="客户端 DP 内存模型：position 为当前位置相关模型，additive 为消融用加性近似。")
    parser.add_argument('--client_l_min', type=int, default=1, help="客户端最少贡献的 LoRA 层数")
    parser.add_argument('--client_l_max', type=int, default=48, help="客户端最多贡献的 LoRA 层数")
    """
        如果您使用的是 peft 库的默认配置 (LoraConfig)，注入的模块数量如下表：
        模型代号 (args.dataset)	    模型名称	Transformer层数	默认 LoRA 注入位置	    LoRA 模块总数 (Adapters)
        cifar100 / domainnet	    ViT-base	12	            Query, Value	        24
        ledgar / default	        BERT-base	12	            Query, Value	        24
        natural_instructions	    GPT-2	    12	            c_attn (QKV合体)	     12
        dolly15k	                T-1.3B	    24	            q_proj, v_proj	        48
    """
    # ===== 客户端异构性模拟参数 =====
    # parser.add_argument('--mem_min', type=float, default=4.0, help="客户端最小内存 (GB)")
    # parser.add_argument('--mem_max', type=float, default=30.0, help="客户端最大内存 (GB)")

        # [新增] 可控的内存分层配置
    # # roberta-base 36层lora矩阵
    # parser.add_argument('--mem_levels_gb', type=str, default="25:15:10",
    #                     help="定义高、中、低三种内存水平 (GB)，以冒号分隔")
    # deberta-base 72层lora矩阵
    parser.add_argument('--mem_levels_gb', type=str, default="40:25:15",
                        help="定义高、中、低三种内存水平 (GB)，以冒号分隔")
    parser.add_argument('--mem_dist_ratio', type=str, default="6:3:1",
                        help="定义高、中、低内存客户端的比例，以冒号分隔 (例如 '6:3:1' 或 '1:1:1')")

    parser.add_argument('--delay_min', type=float, default=0.8, help="客户端训练单层延迟最小值(s)")
    parser.add_argument('--delay_max', type=float, default=2.0, help="客户端训练单层延迟最大值(s)")

    # ===== LAIWA 聚合算法参数 =====
    parser.add_argument('--eta_global', type=float, default=1.0,
                        help="服务器端的全局学习率 eta_global (用于控制全局模型更新步长)")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="LAIWA 中控制数据量影响权重的超参数 gamma (默认: 0.5)")
    parser.add_argument('--delta', type=float, default=0.1,
                        help="LAIWA 中平衡历史稀疏性和当前层重要性的超参数 delta (默认: 0.1)")

    # ===== 其他参数 =====
    parser.add_argument('--gpu', type=int, default=0, help="使用哪个 GPU (-1 表示 CPU)")
    parser.add_argument('--seed', type=int, default=1, help='随机种子')

    args = parser.parse_args()
    return args
