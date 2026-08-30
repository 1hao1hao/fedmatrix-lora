# 文件: main_drl.py
import torch
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
import numpy as np

import os
import csv
import json
from transformers import ViTImageProcessor as ViTFeatureExtractor

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForImageClassification, AutoModelForCausalLM
)
from models.lora_utils import find_lora_candidates # 导入新函数
from FedEnv import FedEnv
from utils.options import args_parser
from utils.plot_results import generate_all_plots
from datetime import datetime
from baselines.budget_schedule_strategy import BudgetScheduleStrategy
from baselines.random_strategy import RandomStrategy  # [新增] 导入随机策略
from utils.evaluation_runner import EvaluationRunner  # [新增] 导入评估运行器
# main_drl.py 顶部
from utils.data_utils import load_and_preprocess_data, distribute_data, custom_collate_fn_for_generation # <--- 加上这个


def load_method_name_mapping():
    mapping_path = os.path.join("experiments", "method_name_mapping.csv")
    if not os.path.exists(mapping_path):
        return {}
    with open(mapping_path, newline="", encoding="utf-8") as handle:
        return {
            row["raw_method"]: row["paper_method"]
            for row in csv.DictReader(handle)
            if row.get("raw_method") and row.get("paper_method")
        }


def write_run_metadata(args, run_id, baselines, method_mapping):
    save_dir = os.path.join("evaluation_results", args.dataset, run_id)
    os.makedirs(save_dir, exist_ok=True)
    metadata = {
        "experiment_id": args.experiment_id or "[TODO-METADATA]",
        "run_id": run_id,
        "dataset": args.dataset,
        "distribution": args.distribution,
        "total_budget": args.total_budget,
        "max_rounds": args.max_rounds,
        "local_bs": args.local_bs,
        "local_ep": args.local_ep,
        "lr": args.lr,
        "memory_model": getattr(args, "memory_model", "position"),
        "lambda_prime": getattr(args, "lambda_prime", "[TODO-METADATA]"),
        "enable_trace": getattr(args, "enable_trace", False),
        "trace_dir": getattr(args, "trace_dir", ""),
        "dataloader_num_workers": getattr(args, "dataloader_num_workers", "[TODO-METADATA]"),
        "max_client_train_workers": getattr(args, "max_client_train_workers", "[TODO-METADATA]"),
        "score_utility_mode": getattr(args, "score_utility_mode", "raw"),
        "cifar100_prior_mode": getattr(args, "cifar100_prior_mode", "legacy_prior"),
        "baseline_granularity": getattr(args, "baseline_granularity", "matrix"),
        "fedavg_physical_policy": getattr(args, "fedavg_physical_policy", "uniform"),
        "fedhello_anchor_layers": getattr(args, "fedhello_anchor_layers", 0),
        "use_score_ema": getattr(args, "use_score_ema", False),
        "score_ema_beta": getattr(args, "score_ema_beta", "[TODO-METADATA]"),
        "depth_prior_weight": getattr(args, "depth_prior_weight", "[TODO-METADATA]"),
        "adaptive_depth_mix": getattr(args, "adaptive_depth_mix", False),
        "position_prior_max_weight": getattr(args, "position_prior_max_weight", "[TODO-METADATA]"),
        "position_signal_temperature": getattr(args, "position_signal_temperature", "[TODO-METADATA]"),
        "position_band_fraction": getattr(args, "position_band_fraction", "[TODO-METADATA]"),
        "position_calibration_mode": getattr(args, "position_calibration_mode", "legacy"),
        "agreement_weight": getattr(args, "agreement_weight", "[TODO-METADATA]"),
        "marginal_memory_penalty": getattr(args, "marginal_memory_penalty", "[TODO-METADATA]"),
        "marginal_latency_penalty": getattr(args, "marginal_latency_penalty", "[TODO-METADATA]"),
        "use_block_coherence": getattr(args, "use_block_coherence", False),
        "block_coherence_weight": getattr(args, "block_coherence_weight", "[TODO-METADATA]"),
        "qkv_completeness_weight": getattr(args, "qkv_completeness_weight", "[TODO-METADATA]"),
        "log_selection_analysis": getattr(args, "log_selection_analysis", False),
        "log_update_cosine": getattr(args, "log_update_cosine", False),
        "seed": args.seed,
        "test_seed": 2024,
        "test_method": args.test_method,
        "raw_methods": baselines,
        "paper_methods": {method: method_mapping.get(method, method) for method in baselines},
        "raw_result_paths": {
            method: os.path.join("evaluation_results", args.dataset, run_id, f"eval_{method}.csv")
            for method in baselines
        },
        "overhead_result_paths": {
            method: os.path.join("evaluation_results", args.dataset, run_id, f"overhead_{method}.csv")
            for method in baselines
        },
    }
    path = os.path.join(save_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"--- 已写出运行 metadata: {path} ---")


def get_model_and_tokenizer(args, share_memory=False):
    """
    [最终修正版] 根据 args.dataset 加载指定的模型和 tokenizer。
    并为不同类型的模型进行恰当的配置。
    """
    tokenizer, global_model = None, None
    is_causal_lm = False  # 标志位，用于后续处理
    # ==========================================================
    # 1. 根据数据集名称，选择并加载模型 (离线支持版)
    # ==========================================================
    # --- GLUE (SST-2, QNLI) -> RoBERTa-base ---
    if args.dataset in ['sst2', 'qnli']:
        print("任务类型: 文本理解。加载 RoBERTa-base...")
        args.num_classes = 2
        local_path = './models_zoo/roberta-base'
        is_local = os.path.exists(local_path)
        model_name = local_path if is_local else 'roberta-base'

        # [核心修复] 增加 local_files_only=is_local，禁止它去网上查漏补缺
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=is_local)
        global_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=args.num_classes, local_files_only=is_local
        )

    # --- GLUE (QQP, MNLI) -> DeBERTa-V3-Large ---
    elif args.dataset in ['qqp', 'mnli']:
        print("任务类型: 文本理解。加载 DeBERTa-V3-Large...")
        args.num_classes = 3 if args.dataset == 'mnli' else 2
        # 修改为新的离线路径
        local_path = './models_zoo/deberta-v3-large'
        is_local = os.path.exists(local_path)
        model_name = local_path if is_local else 'microsoft/deberta-v3-large'

        # 增加 local_files_only=is_local，禁止联网
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=is_local, use_fast=False)
        global_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=args.num_classes, local_files_only=is_local
        )

    # --- 视觉任务 ---
    elif args.dataset in ['cifar100', 'domainnet']:
        print("任务类型: 视觉分类。加载 ViT-base...")
        model_name = 'google/vit-base-patch16-224-in21k'
        from transformers import ViTFeatureExtractor
        # 对于视觉模型，tokenizer 实际上是 feature_extractor
        tokenizer = ViTFeatureExtractor.from_pretrained(model_name)
        # 加载模型时，先忽略分类头的尺寸不匹配问题
        global_model = AutoModelForImageClassification.from_pretrained(
            model_name, num_labels=1000, ignore_mismatched_sizes=True
        )

    # --- 文本分类任务 ---
    elif args.dataset == 'ledgar':
        print("任务类型: 文本分类。加载 BERT-base...")
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 直接加载，类别数在 data_utils 中已经确定
        global_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=args.num_classes
        )

    # --- 指令遵循/生成式任务 ---
    elif args.dataset == 'natural_instructions':
        is_causal_lm = True
        print("任务类型: 指令遵循。加载 DataJuicer-1B (使用 gpt2 作为占位符)...")
        # 提示: DataJuicer-1B 可能需要从特定地方下载。这里用 gpt2 作为功能占位符。
        model_name = 'gpt2'
        # 为生成式模型设置 padding_side='left'
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        global_model = AutoModelForCausalLM.from_pretrained(model_name)

    elif args.dataset in ['dolly15k', 'samsum']:
        is_causal_lm = True
        print("任务类型: 生成式任务。加载 OPT-1.3B...")
        local_model_path = './models_zoo/opt-1.3b'
        model_name = local_model_path if os.path.isdir(local_model_path) else 'facebook/opt-1.3b'
        # model_name = 'facebook/opt-125m' # 如果资源不足，请切换到这个小模型

        # 为生成式模型设置 padding_side='left'
        # tokenizer_performance = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        # 之前是为了 .generate() 才用的 left，现在我们只算 loss，改回 right
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right', local_files_only=os.path.isdir(local_model_path))

        global_model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=os.path.isdir(local_model_path))

    # --- 默认任务 (例如 imdb) ---
    else:
        print(f"任务类型: 默认文本分类 ({args.dataset})。加载 BERT-base...")
        model_name = 'bert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        global_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=args.num_classes or 2 # 如果 num_classes 未设置，默认为2
        )

    # ==========================================================
    # 2. 对加载后的模型和 tokenizer 进行通用后处理
    # ==========================================================

    # 对于所有生成式模型 (Causal LM)，确保 pad_token 被正确设置
    if is_causal_lm or args.dataset in ['dolly15k', 'samsum']:
        if tokenizer.pad_token is None:
            print("检测到生成式模型缺少 pad_token，将其设置为 eos_token。")
            tokenizer.pad_token = tokenizer.eos_token
            global_model.config.pad_token_id = global_model.config.eos_token_id

    # 如果是为多进程准备，则将模型参数移动到共享内存
    if share_memory:
        print("--- [主进程] 将模型移动到 CPU 共享内存 ---")
        global_model.share_memory()

    return global_model, tokenizer

# 文件: main_drl.py (放在主函数 if __name__ == '__main__': 之前)

def build_loader_kwargs(num_workers):
    kwargs = {
        "num_workers": max(0, int(num_workers)),
        "pin_memory": True,
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
    return kwargs

# --------------------------1.入口程序启动 (main_drl.py)--------------------------
if __name__ == '__main__':
        # 1. 设置环境变量和多进程启动方式
    # [核心修正-1] 在程序最开始就设置环境变量，消除 Tokenizer 警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


    # 1. [新] 强制设置多进程启动方式为 'spawn'
    # 这对于在多进程中使用 CUDA 是至关重要的，可以避免很多奇怪的 CUDA 错误
    if torch.cuda.is_available() and mp.get_start_method(allow_none=True) != 'spawn':
        print("警告: 检测到非 'spawn' 启动方式，强制设置为 'spawn' 以确保 CUDA 安全。")
        mp.set_start_method('spawn', force=True)
    start_method = mp.get_start_method()
    args = args_parser()
    loader_kwargs = build_loader_kwargs(args.dataloader_num_workers)

    # ==========================================================
    # [核心修复] 必须在所有操作之前固定全局种子！
    # ==========================================================
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"--- [全局设置] 已固定全局随机种子: {args.seed} ---")

    # ==========================================================
    # [核心优化] 生成带有丰富实验配置信息的文件夹名称 (Run ID)
    # 将非法路径字符（冒号、斜杠）替换为下划线
    # ==========================================================
    safe_dist = args.distribution.replace(":", "_").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Include experiment_id when available so concurrent Slurm array tasks do
    # not write different metadata/results into the same timestamp directory.
    safe_experiment = (args.experiment_id or "").replace(":", "_").replace("/", "_")
    safe_experiment = safe_experiment.replace(" ", "_").replace(",", "_")
    prefix = f"{safe_experiment}__" if safe_experiment else ""
    run_id = f"{prefix}Dist-{safe_dist}_Bgt-{args.total_budget}_{timestamp}"
    args.run_id = run_id
    print(f"--- 本次实验 Run ID: {run_id} ---")

    # [核心修正-1] 在程序最开始就检查并确定设备
    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"--- [主进程] 检测到 CUDA，将强制在 GPU:{args.gpu} 上运行 ---")
    else:
        device = torch.device("cpu")
        print("--- [主进程] CUDA 不可用或被禁用，将在 CPU 上运行 ---")
        # 打印 CUDA 不可用的详细原因
        if args.gpu != -1 and not torch.cuda.is_available():
             print("  >> 警告: 您指定了 GPU，但 torch.cuda.is_available() 返回 False。")
             print("  >> 请检查您的 NVIDIA 驱动、CUDA 工具包和 PyTorch 安装是否匹配。")

    # 2. 在主进程中准备所有需要共享的资源
    print("--- [主进程] 准备全局共享资源 ---")

    global_model, tokenizer = get_model_and_tokenizer(args,share_memory=True)

    # [核心修改] 动态获取 LoRA 总层数
    lora_candidates = find_lora_candidates(global_model)
    args.total_lora_layers = len(lora_candidates)
    args.client_l_max = len(lora_candidates)
    print(f"--- [模型分析] 动态检测到 {args.total_lora_layers} 个可应用的 LoRA 层 ---")

    train_dataset, test_dataset, test_text_labels = load_and_preprocess_data(args, tokenizer)

    # # 动态调整分类模型的分类头
    if args.num_classes and hasattr(global_model, 'classifier'):
        if isinstance(global_model.classifier, torch.nn.Linear):
            global_model.classifier = torch.nn.Linear(global_model.config.hidden_size, args.num_classes)
            global_model.config.num_labels = args.num_classes
            # 3. [核心修复] 必须同时更新实例变量 num_labels
            # 否则 forward 函数计算 Loss 时仍会使用旧值 (1000) 进行 reshape
            global_model.num_labels = args.num_classes
    print(f"数据集 '{args.dataset}' 加载完成。任务类别数: {args.num_classes}")

    # [新] 将模型参数移动到 CPU 共享内存
    # 这一步是多进程并行化的关键！
    print("--- [主进程] 将模型移动到 CPU 共享内存 ---")
    global_model.share_memory()

    # 从测试集中划分一个验证集给服务器，但不超过测试集本身的大小
    # [核心修正] 使用 .select() 方法创建验证集，而不是 torch.utils.data.Subset
    print(f"原始测试集大小: {len(test_dataset)}")
    # 完整验证集 (用于最终评估)
    full_val_indices = list(range(min(500, len(test_dataset))))
    full_val_dataset = test_dataset.select(full_val_indices)

    full_val_loader = DataLoader(
        full_val_dataset,
        batch_size=args.local_bs,
        shuffle=False,
        collate_fn=custom_collate_fn_for_generation, # [关键] 必须用这个通用版函数

        **loader_kwargs  # [HPC 优化]
    )

    print(f"已创建包含 {len(full_val_dataset)} 样本的完整验证集。")

    # [确认] 创建小验证集,用于层重要性计算
    small_val_indices = list(range(min(32, len(test_dataset)))) # 取32个样本够了
    small_val_dataset = test_dataset.select(small_val_indices)
    # 注意：视觉任务不需要 collate_fn (传 None)，文本任务需要
    # 根据你的 dataset 类型决定是否传 collate_fn
    collate_fn = custom_collate_fn_for_generation if args.dataset in ['dolly15k', 'natural_instructions', 'samsum'] else None
    small_val_loader = DataLoader(small_val_dataset, batch_size=args.local_bs, shuffle=False, collate_fn=collate_fn,**loader_kwargs)


    # [核心修正] 为验证集创建对应的文本标签列表
    val_text_labels = None
    if test_text_labels:
        val_text_labels = [test_text_labels[i] for i in full_val_indices]

    env_kwargs = {
        "args": args,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "val_loader": full_val_loader,
        "small_val_loader": small_val_loader,  # <--- 必须有这一行
        "val_text_labels": val_text_labels, # <-- 新增
        "global_model": global_model
    }

    # ==========================================
    # 非强化学习实验阶段：按 baseline/method 逐个运行
    # ==========================================
    print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>> 进入联邦 LoRA 对比实验阶段 <<<")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    test_seed = 2024
    per_round_cap = args.num_clients * args.client_l_max
    budget_agent = BudgetScheduleStrategy(
        total_budget=args.total_budget,
        max_rounds=args.max_rounds,
        per_round_cap=per_round_cap,
        min_ratio=0.1,
        max_ratio=1.0,
    )
    random_agent = budget_agent
    random_agent_hello = budget_agent
    eval_run_id = run_id

    if args.test_method == 'all':
        baselines = [
            'FedAvg_LoRA_homogeneous',
            'Uniform_allocation',
            'Fed-HeLLo',
            'LEGEND',
            'HETLoRA',
            'FAH_QLoRA',
            'MS',
            'Random_FIM',
            'Random_Taylor',
        ]
    elif args.test_method == 'ablations':
        baselines = [
            'Physical_FIM',
            'Random_Scores',
            'Random_FIM_NoDepth',
            'Greedy_Value',
            'Greedy_ValueCost',
        ]
    else:
        baselines = [item.strip() for item in args.test_method.split(',') if item.strip()]

    method_mapping = load_method_name_mapping()
    write_run_metadata(args, run_id, baselines, method_mapping)

    for strategy_name in baselines:
        print(f"\n{'#'*50}")
        print(f"### 准备运行方法: {strategy_name} ###")
        print(f"{'#'*50}")

        current_env_kwargs = env_kwargs.copy()
        current_env_kwargs['allocation_strategy'] = strategy_name
        baseline_env = FedEnv(**current_env_kwargs)
        runner = EvaluationRunner(baseline_env, args, run_id=eval_run_id)

        if strategy_name == 'Fed-HeLLo':
            runner.run(strategy_model=random_agent_hello, strategy_name=strategy_name, seed=test_seed)
        else:
            runner.run(strategy_model=random_agent, strategy_name=strategy_name, seed=test_seed)
        if hasattr(baseline_env, "close"):
            baseline_env.close()

    print("\n所有对比实验已顺利完成！您可以查看 evaluation_results 下的 CSV 文件。")

    # ==========================================
    # 阶段四：自动生成数据对比图表
    # ==========================================
    print("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(">>> 正在自动生成学术图表 (Plotting) <<<")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    try:
        # [核心修正] 把分布参数传进去，让画图脚本标明标题
        generate_all_plots(args.dataset, distribution=args.distribution)
        print("\n🎉 实验全部结束！完美收官！")
    except Exception as e:
        print(f"\n⚠️ 图表生成出现小错误: {e}")
        print("不用担心，数据 CSV 已安全保存，稍后可以手动运行 utils/plot_results.py")
