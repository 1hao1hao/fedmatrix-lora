# 文件: utils/data_utils.py
# 新增：用于数据预处理和分配
import numpy as np
import torch
from datasets import load_dataset, DatasetDict, load_from_disk, concatenate_datasets  # [关键] 引入 load_from_disk
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import collections
import os


def _sample_split(split, size, seed):
    """Sample a Hugging Face split with a deterministic shuffle."""
    if size is None or size < 0 or size >= len(split):
        return split
    return split.shuffle(seed=seed).select(range(size))


def _get_effective_lengths(dataset):
    """Return effective sequence lengths using attention_mask when available."""
    if 'attention_mask' in dataset.column_names:
        lengths = []
        for mask in dataset['attention_mask']:
            if hasattr(mask, 'sum'):
                lengths.append(int(mask.sum()))
            else:
                lengths.append(int(sum(mask)))
        return np.array(lengths)

    # Fallback for datasets without attention masks.
    return np.array([len(x) for x in dataset['input_ids']])


def _load_samsum_from_local_dir(local_path):
    """Load SAMSum from either a save_to_disk directory or raw split files."""
    if os.path.exists(os.path.join(local_path, "dataset_dict.json")):
        return load_from_disk(local_path)

    split_aliases = {
        "train": ["train"],
        "validation": ["validation", "val", "dev"],
        "test": ["test"],
    }
    data_files_by_ext = {}

    for split_name, aliases in split_aliases.items():
        for alias in aliases:
            for ext in ["jsonl", "json", "csv"]:
                candidate = os.path.join(local_path, f"{alias}.{ext}")
                if os.path.exists(candidate):
                    data_files_by_ext.setdefault(ext, {})[split_name] = candidate
                    break
            if any(split_name in files for files in data_files_by_ext.values()):
                break

    if not data_files_by_ext:
        raise FileNotFoundError(
            "未找到 SAMSum 本地 split 文件。支持 HuggingFace save_to_disk 目录，"
            "或 train/validation/test 的 json、jsonl、csv 文件。"
        )

    if len(data_files_by_ext) > 1:
        raise ValueError(
            f"SAMSum 本地 split 文件格式不一致: {sorted(data_files_by_ext.keys())}。"
            "请统一使用 json/jsonl/csv 中的一种。"
        )

    ext, data_files = next(iter(data_files_by_ext.items()))
    if "train" not in data_files:
        raise ValueError(f"SAMSum 本地目录 {local_path} 缺少 train split 文件。")

    loader_name = "json" if ext in ["json", "jsonl"] else "csv"
    print(f"--- [Data] 使用本地 {loader_name} 文件加载 SAMSum: {data_files} ---")
    return load_dataset(loader_name, data_files=data_files)

# =================================================================================
# 图像任务处理 (CIFAR-100, DomainNet)
# =================================================================================
def get_vision_datasets(dataset_name, args):
    """加载和预处理视觉数据集 (最终优化版)"""
    print(f"正在加载视觉数据集: {dataset_name}")

    # 1. 保留你原来的 ImageNet 标准归一化参数 (这对 ViT 更好，且能利用你旧的缓存)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if dataset_name == 'cifar100':
        # # Hugging Face Hub 支持 CIFAR-100
        # dataset = load_dataset("cifar100")
                # [核心修改] 优先加载本地解压的 cifar100_local 文件夹
        # 假设该文件夹在项目根目录下
        local_path = os.path.join(os.getcwd(), "cifar100_local")

        if os.path.exists(local_path):
            print(f"--- [Data] 发现本地数据集: {local_path}，正在加载... ---")
            dataset = load_from_disk(local_path)
        else:
            print("--- [Data] 未发现本地数据集，尝试在线加载 (可能会失败)... ---")
            dataset = load_dataset("cifar100")

        # 处理列名差异
        if 'img' in dataset['train'].column_names:
            dataset = dataset.rename_column("img", "image")
        if 'fine_label' in dataset['train'].column_names:
            dataset = dataset.rename_column("fine_label", "labels")

        args.num_classes = 100

    elif dataset_name == 'domainnet':
        print("警告: DomainNet 需要手动下载。此处使用 CIFAR-100 作为占位符。")
        dataset = load_dataset("cifar100")
        dataset = dataset.rename_column("img", "image")
        dataset = dataset.rename_column("fine_label", "labels")
        args.num_classes = 126
    else:
        raise ValueError(f"不支持的视觉数据集: {dataset_name}")

    # 定义 map 函数
    def apply_transform(examples):
        examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['image']]
        return examples

    # 2. 应用处理
    # 注意：如果你的变换逻辑(transform)和之前完全一样，这一步会直接命中缓存，瞬间完成
    transformed_dataset = dataset.map(
        apply_transform,
        batched=True,
        remove_columns=['image'] # 移除原始图像，减少内存占用
    )

    # 3. [关键修复] 清理多余列 + 设置 Torch 格式
    # 这一步是为了解决 'list' object has no attribute 'to'

    # 获取我们需要保留的列名
    cols_to_keep = ['pixel_values', 'labels']

    # 移除数据集里可能存在的其他列 (如 'coarse_label', 'idx' 等)
    # 这些列如果是 list 格式，会干扰 DataLoader
    columns_to_remove = [c for c in transformed_dataset['train'].column_names if c not in cols_to_keep]
    if columns_to_remove:
        transformed_dataset = transformed_dataset.remove_columns(columns_to_remove)

    # 显式设置为 torch 格式 (这一步很快，不涉及数据处理，只是元数据标记)
    transformed_dataset.set_format(type='torch', columns=cols_to_keep)

    train_dataset = transformed_dataset['train']
    test_dataset = transformed_dataset['test']

    # 4. 返回 3 个值 (适配 main_drl.py 的解包逻辑)
    return train_dataset, test_dataset, None

# =================================================================================
# 语言任务处理 (IMDB, LEDGAR, Natural Instructions, Dolly-15k)（samsum）
# =================================================================================
def get_language_datasets(dataset_name, tokenizer, args):
    """
    [最终修正版] 加载和预处理语言数据集。
    修复了分类任务(IMDB/LEDGAR)中 labels 列意外被删除的 Bug。
    """
    print(f"正在加载语言数据集: {dataset_name}")

    is_causal_lm = False
    text_cols = ['text'] # 默认单句列名

    # --- 1. 加载和初步格式化 ---
    if dataset_name in ['sst2', 'qnli', 'qqp', 'mnli']:
        args.num_classes = 3 if dataset_name == 'mnli' else 2
        # 本地离线路径设定
        local_path = os.path.join(os.getcwd(), "data_local", dataset_name)
        if os.path.exists(local_path):
            print(f"--- [Data] 发现本地数据集: {local_path}，正在加载... ---")
            dataset = load_from_disk(local_path)
        else:
            print(f"--- [Data] 未发现本地数据集，尝试联网加载 glue-{dataset_name}... ---")
            dataset = load_dataset("glue", dataset_name)

        # 统一标签列名并划分测试集
        if 'label' in dataset['train'].column_names:
            dataset = dataset.rename_column("label", "labels")
        # GLUE 的测试集通常没有标签，我们用 validation 作为 test
        if 'validation' in dataset.keys():
            dataset['test'] = dataset['validation']

        # 确定需要 tokenize 的列
        if dataset_name == 'sst2':
            text_cols = ['sentence']
        elif dataset_name == 'qnli':
            text_cols = ['question', 'sentence']
        elif dataset_name == 'qqp':
            text_cols = ['question1', 'question2']
        elif dataset_name == 'mnli':
            text_cols = ['premise', 'hypothesis']
            if 'validation_matched' in dataset.keys():
                dataset['test'] = dataset['validation_matched'] # MNLI 特殊处理

    elif dataset_name in ['natural_instructions', 'dolly15k']:
        is_causal_lm = True
        args.num_classes = None

        # 加载原始数据
        # 如果网络不好，建议本地下载后改为本地路径
        # raw_dataset = load_dataset("databricks/databricks-dolly-15k")
        # 新代码: 从本地 jsonl 文件加载
        # 假设你把文件放在项目根目录下的 data 文件夹中
        # 也可以在 options.py 里增加一个 --data_path 参数来灵活指定
        local_data_path = os.path.join("data", "databricks-dolly-15k.jsonl")

        if os.path.exists(local_data_path):
            print(f"正在加载本地数据集: {local_data_path}")
            raw_dataset = load_dataset("json", data_files=local_data_path)
        else:
            #以此作为回退，万一你是在有网环境跑
            print(f"本地文件不存在，尝试联网加载: databricks/databricks-dolly-15k")
            raw_dataset = load_dataset("databricks/databricks-dolly-15k")

        # 定义格式化函数
        def format_dolly(example):
            # [优化] 使用标准的 Alpaca 模版，增加 System Prompt
            # 这能让模型更清楚它正在做一个“遵循指令”的任务

            bos = tokenizer.bos_token if tokenizer.bos_token else ""

            instruction = example['instruction']
            context = example['context']
            response = example['response']

            if context:
                prompt = (
                    f"{bos}Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Input:\n{context}\n\n"
                    f"### Response:\n"
                )
            else:
                prompt = (
                    f"{bos}Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{instruction}\n\n"
                    f"### Response:\n"
                )

            return {
                "text": prompt + response + tokenizer.eos_token,
                "text_labels": response,
                "prompts": prompt
            }

        # 应用格式化
        dataset = raw_dataset.map(format_dolly)

        # 手动划分
        train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=args.seed)
        dataset = DatasetDict({
            'train': train_test_split['train'],
            'test': train_test_split['test']
        })

    elif dataset_name == 'samsum':
        is_causal_lm = True
        args.num_classes = None

        local_candidates = [
            os.path.join(os.getcwd(), "data_local", "samsum"),
            os.path.join(os.getcwd(), "data", "samsum"),
        ]

        dataset = None
        for local_path in local_candidates:
            if os.path.exists(local_path):
                print(f"--- [Data] 发现本地 SAMSum 数据集: {local_path}，正在加载... ---")
                try:
                    dataset = _load_samsum_from_local_dir(local_path)
                    break
                except Exception as e:
                    print(f"--- [Data] 本地 SAMSum 加载失败: {e}，继续尝试其他路径 ---")

        if dataset is None:
            if os.environ.get("HF_HUB_OFFLINE", "0").lower() in ["1", "true", "yes"]:
                raise FileNotFoundError(
                    "当前启用了 HF_HUB_OFFLINE=1，但未找到本地 SAMSum。"
                    "请将数据放到 ./data_local/samsum 或 ./data/samsum。"
                    "支持 HuggingFace save_to_disk 格式，或 train/validation/test.jsonl(json/csv)。"
                )
            print("--- [Data] 未发现本地 SAMSum，尝试在线加载 samsum ---")
            dataset = load_dataset("samsum")

        if 'train' not in dataset:
            raise ValueError("SAMSum 数据集缺少 train split。")
        if 'validation' not in dataset and 'test' not in dataset:
            split = dataset['train'].train_test_split(test_size=0.1, seed=args.seed)
            dataset = DatasetDict({
                'train': split['train'],
                'validation': split['test'],
                'test': split['test']
            })
        elif 'validation' not in dataset and 'test' in dataset:
            dataset = DatasetDict({
                'train': dataset['train'],
                'validation': dataset['test'],
                'test': dataset['test']
            })
        elif 'validation' in dataset and 'test' not in dataset:
            dataset = DatasetDict({
                'train': dataset['train'],
                'validation': dataset['validation'],
                'test': dataset['validation']
            })

        # 根据参数抽样，不改动输出目录结构。
        dataset['train'] = _sample_split(dataset['train'], args.samsum_train_size, args.seed)
        dataset['validation'] = _sample_split(dataset['validation'], args.samsum_val_size, args.seed)
        dataset['test'] = _sample_split(dataset['test'], args.samsum_test_size, args.seed)

        print(
            f"--- [SAMSum] 实际样本数 train={len(dataset['train'])}, "
            f"val={len(dataset['validation'])}, test={len(dataset['test'])} ---"
        )

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos = tokenizer.eos_token or ""

        def preprocess_samsum(examples):
            prompts, text_labels = [], []
            input_ids_list, attention_masks, labels_list = [], [], []
            prompt_lens, summary_label_counts, non_ignore_counts = [], [], []

            dialogues = examples['dialogue']
            summaries = examples['summary']

            for dialogue, summary in zip(dialogues, summaries):
                dialogue = "" if dialogue is None else str(dialogue)
                summary = "" if summary is None else str(summary).strip()
                prompt = f"Dialogue:\n{dialogue}\n\nSummarize the dialogue:\n"
                full_text = prompt + summary + eos

                tokenized = tokenizer(
                    full_text,
                    padding='max_length',
                    truncation=True,
                    max_length=args.max_length
                )
                prompt_ids = tokenizer(
                    prompt,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.max_length
                )['input_ids']

                labels = []
                summary_token_count = 0
                non_ignore = 0
                for idx, tok in enumerate(tokenized['input_ids']):
                    if tok == pad_id:
                        labels.append(-100)
                    elif idx < len(prompt_ids):
                        labels.append(-100)
                    else:
                        labels.append(tok)
                        summary_token_count += 1
                        non_ignore += 1

                prompts.append(prompt)
                text_labels.append(summary)
                input_ids_list.append(tokenized['input_ids'])
                attention_masks.append(tokenized['attention_mask'])
                labels_list.append(labels)
                prompt_lens.append(len(prompt_ids))
                summary_label_counts.append(summary_token_count)
                non_ignore_counts.append(non_ignore)

            return {
                'prompts': prompts,
                'text_labels': text_labels,
                'input_ids': input_ids_list,
                'attention_mask': attention_masks,
                'labels': labels_list,
                'prompt_len_tokens': prompt_lens,
                'summary_label_tokens': summary_label_counts,
                'non_ignore_label_tokens': non_ignore_counts,
            }

        processed = dataset.map(
            preprocess_samsum,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        # Sanity check: 训练集不能全部为 -100
        non_ignore_total = 0
        for labels in processed['train']['labels']:
            non_ignore_total += sum(1 for t in labels if t != -100)
        if non_ignore_total == 0:
            raise ValueError("SAMSum 训练集 labels 全为 -100，请检查 prompt/summary 切分逻辑。")

        # 打印一个样本统计，便于快速确认 summary-only loss 行为
        if len(processed['train']) > 0:
            s0 = processed['train'][0]
            print(
                f"--- [SAMSum Sanity] prompt_len={s0['prompt_len_tokens']}, "
                f"summary_label_tokens={s0['summary_label_tokens']}, "
                f"non_ignore_labels={s0['non_ignore_label_tokens']} ---"
            )

        train_dataset = processed['train']
        val_dataset = processed['validation']
        test_dataset = processed['test']

        # 验证/测试必须保留 prompts 与 text_labels
        for split_name, split in [("validation", val_dataset), ("test", test_dataset)]:
            missing = [c for c in ['prompts', 'text_labels'] if c not in split.column_names]
            if missing:
                raise ValueError(f"SAMSum {split_name} 缺少字段: {missing}")

        # 主流程使用 test_dataset 来切 500 条验证集：将 validation 放在前面，再拼接 test。
        eval_dataset = concatenate_datasets([val_dataset, test_dataset])

        train_remove = [c for c in train_dataset.column_names if c not in ['input_ids', 'attention_mask', 'labels']]
        if train_remove:
            train_dataset = train_dataset.remove_columns(train_remove)
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        test_keep = ['input_ids', 'attention_mask', 'labels', 'prompts', 'text_labels']
        test_remove = [c for c in eval_dataset.column_names if c not in test_keep]
        if test_remove:
            eval_dataset = eval_dataset.remove_columns(test_remove)
        eval_dataset.set_format(type=None)

        if 'prompts' not in eval_dataset.column_names or 'text_labels' not in eval_dataset.column_names:
            raise ValueError("SAMSum 生成式测试集缺少 prompts/text_labels。")

        test_text_labels = eval_dataset['text_labels']
        return train_dataset, eval_dataset, test_text_labels

    elif dataset_name == 'imdb':
        # IMDB: text, label -> labels
        dataset = load_dataset("imdb").rename_column("label", "labels")
        args.num_classes = 2

    elif dataset_name == 'ledgar':
        # LEDGAR: label 是 list -> 取第一个
        dataset = load_dataset("lex_glue", "ledgar").map(lambda x: {'labels': x['label'][0]})
        args.num_classes = 100

    else:
        raise ValueError(f"不支持的语言数据集: {dataset_name}")

# --- [新增] 安全缩减训练集规模 (Subsampling) ---
    # 增加条件：只有当数据集非常大（>15万条）时，才执行缩减，保护小数据集
    if args.data_subsample_ratio < 1.0 and len(dataset['train']) > 150000:
        print(f"\n--- [加速机制] 检测到超大数据集，正在安全缩减训练集至 {args.data_subsample_ratio*100}% ---")
        train_size = int(len(dataset['train']) * args.data_subsample_ratio)
        dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(range(train_size))
        print(f"--- 缩减后训练集大小: {len(dataset['train'])} 条 ---\n")
    elif args.data_subsample_ratio < 1.0:
        print(f"\n--- [加速机制跳过] 数据集总数不足 15 万条，为保证训练稳定性，跳过缩减。 ---")


    # --- 2. Tokenization ---
    def tokenize_function(examples):
        # 根据 text_cols 的长度，动态决定是单句还是双句输入
        if len(text_cols) == 1:
            return tokenizer(examples[text_cols[0]], padding='max_length', truncation=True, max_length=args.max_length)
        else:
            return tokenizer(examples[text_cols[0]], examples[text_cols[1]], padding='max_length', truncation=True, max_length=args.max_length)

    # [修改] 移除列的逻辑
    all_cols = dataset['train'].column_names
    # 只要不是我们需要保留的，都删掉。保留 prompts!
    if is_causal_lm:
        cols_to_keep = ['prompts', 'text_labels'] # 注意：text 列会被 tokenize 消耗掉变成 input_ids
    else:
        cols_to_keep = ['labels']

    cols_to_remove =[c for c in all_cols if c not in cols_to_keep and c not in text_cols]

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=cols_to_remove)

    # --- 3. 为 Causal LM 设置数值标签 ---
    if is_causal_lm:
        def set_causal_lm_labels_list(examples):
            input_ids = examples['input_ids']
            new_labels = []

            # 防御性编程：确保 pad_token_id 存在
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

            for inp in input_ids:
                # 将等于 pad_id 的位置设为 -100，这样计算 Loss 时会忽略 padding
                label = [(tok if tok != pad_id else -100) for tok in inp]
                new_labels.append(label)
            examples['labels'] = new_labels
            return examples

        tokenized_datasets = tokenized_datasets.map(set_causal_lm_labels_list, batched=True)

    # --- 4. 清理和格式化 ---
    train_dataset = tokenized_datasets['train']
    test_dataset = tokenized_datasets['test']

    # 提取测试集的文本标签 (仅 LLM 需要)
    test_text_labels = None
    if is_causal_lm and 'text_labels' in test_dataset.column_names:
        test_text_labels = test_dataset['text_labels']

    # 定义 Tensor 列
    tensor_cols = ['input_ids', 'attention_mask', 'labels']

    # 1. 训练集: 只需要 Tensor，高效
    # 移除 text_labels, prompts 等杂项，防止 default_collate 报错
    train_remove = [c for c in train_dataset.column_names if c not in tensor_cols]
    if train_remove:
        train_dataset = train_dataset.remove_columns(train_remove)
    train_dataset.set_format('torch', columns=tensor_cols)

    # 2. 测试集: 需要 Tensor + Strings
    # [绝对关键] 不要 remove prompts! 不要 set_format('torch')!
    # 我们只移除那些明显没用的列 (比如 text, 如果还没删的话)
    desired_test_cols = tensor_cols + ['prompts', 'text_labels']
    test_remove = [c for c in test_dataset.column_names if c not in desired_test_cols]
    if test_remove:
        test_dataset = test_dataset.remove_columns(test_remove)

        # [核心修复] 分情况设置输出格式
    if is_causal_lm:
        # 生成式任务(如 Dolly)需要保留字符串，输出原始 Python List
        test_dataset.set_format(type=None)
    else:
        # 分类任务(如 GLUE)直接输出 Torch Tensor，完美规避 collate 的矩阵转置 Bug！
        test_dataset.set_format('torch', columns=tensor_cols)

    return train_dataset, test_dataset, test_text_labels

# =================================================================================
# 主调度函数 (统一接口)
# =================================================================================
def load_and_preprocess_data(args, tokenizer=None):
    """
    根据 args.dataset 的名称，加载并预处理相应的数据集。
    """
    vision_datasets = ['cifar100', 'domainnet']
    #[核心修复] 把 GLUE 的四个数据集加入语言任务名单！
    language_datasets =['imdb', 'ledgar', 'natural_instructions', 'dolly15k', 'samsum', 'sst2', 'qnli', 'qqp', 'mnli']

    if args.dataset in vision_datasets:
        return get_vision_datasets(args.dataset, args)
    elif args.dataset in language_datasets:
        if tokenizer is None:
            raise ValueError("语言任务需要提供 tokenizer。")
        return get_language_datasets(args.dataset, tokenizer, args)
    else:
        raise ValueError(f"未知或不支持的数据集: {args.dataset}")


# =================================================================================
# 数据异质性分布函数集合
# =================================================================================
def _distribute_iid(dataset, num_clients):
    """
    将数据独立同分布 (I.I.D.) 地分配给客户端。
    """
    print("数据分布策略: I.I.D.")
    num_items = len(dataset)
    all_idxs = np.arange(num_items)
    np.random.shuffle(all_idxs)
    dict_users = {i: set() for i in range(num_clients)}

    # 使用 np.array_split 来均匀（或近似均匀）地分割索引
    split_idxs = np.array_split(all_idxs, num_clients)
    for i in range(num_clients):
        dict_users[i] = set(split_idxs[i])

    return dict_users

def _distribute_pathological_non_iid(dataset, num_clients, classes_per_client):
    """
    病态非独立同分布 (Pathological Non-I.I.D.)。
    每个客户端只拥有 `classes_per_client` 个类别的数据。
    """
    print(f"数据分布策略: Pathological Non-I.I.D. (每个客户端 {int(classes_per_client)} 个类)")
    num_classes = len(set(np.array(dataset['labels'])))

    # 1. 按标签对所有数据索引进行分组
    label_to_indices = collections.defaultdict(list)
    for i in range(len(dataset)):
        label = dataset[i]['labels']
        if hasattr(label, 'item'): label = label.item()
        label_to_indices[label].append(i)

    # 2. 创建数据分片 (shards)
    # 首先将所有索引打乱，以保证每个分片内部是随机的
    for label in label_to_indices:
        np.random.shuffle(label_to_indices[label])

    # 将每个类别的数据分成 num_clients 个分片
    shards_by_label = {label: np.array_split(indices, num_clients) for label, indices in label_to_indices.items()}

    # 3. 为每个客户端分配分片
    dict_users = collections.defaultdict(list)
    # 创建一个 [0, 1, ..., num_classes-1, 0, 1, ...] 的序列，用于循环分配类别
    class_ids_iter = np.random.permutation(np.repeat(np.arange(num_classes), num_clients))

    # 每个客户端分配 `classes_per_client` 个分片
    shards_per_client = classes_per_client
    shard_idx_counter = collections.defaultdict(int)

    for client_id in range(num_clients):
        # 为当前客户端随机选择 shards_per_client 个类别
        chosen_classes = np.random.choice(num_classes, int(shards_per_client), replace=False)
        for class_id in chosen_classes:
            # 从该类别的分片中取出一个分配给客户端
            shard_idx = shard_idx_counter[class_id]
            dict_users[client_id].extend(shards_by_label[class_id][shard_idx])
            shard_idx_counter[class_id] = (shard_idx + 1) % num_clients  # 循环使用分片

    # 将 list 转换为 set
    for client_id in dict_users:
        dict_users[client_id] = set(dict_users[client_id])

    return dict_users

def _distribute_dirichlet_label_skew(dataset, num_clients, alpha, classes_per_client=None):
    """
    [修复版] 使用狄利克雷分布模拟标签数量倾斜。
    """
    if classes_per_client:
        print(f"数据分布策略: Dirichlet Label Skew (alpha={alpha}) + Pathological (每个客户端最多 {int(classes_per_client)} 个类)")
    else:
        print(f"数据分布策略: Dirichlet Label Skew (alpha={alpha})")

    # --- 1. 极速获取所有标签 ---
    try:
        # 针对 Hugging Face Dataset，直接按列读取
        labels_array = np.array(dataset['labels'])
    except (KeyError, TypeError):
        try:
            labels_array = np.array(dataset.targets)
        except:
            from tqdm import tqdm
            print("正在逐个读取标签...")
            labels_array = np.array([dataset[i]['labels'] for i in tqdm(range(len(dataset)))])

    num_classes = len(set(labels_array))

    # --- 2. 为每个客户端分配其拥有的类别 ---
    if classes_per_client:
        times = [0] * num_classes
        contain = []
        for i in range(num_clients):
            current = [i % num_classes]
            times[i % num_classes] += 1
            j = 1
            while len(current) < classes_per_client:
                idx = np.random.randint(0, num_classes)
                if idx not in current:
                    current.append(idx)
                    times[idx] += 1
            contain.append(current)
        client_classes = {i: contain[i] for i in range(num_clients)}
    else:
        client_classes = {i: list(range(num_classes)) for i in range(num_clients)}

    # --- 3. 按标签向量化分组 ---
    label_to_indices = {}
    for c in range(num_classes):
        label_to_indices[c] = np.where(labels_array == c)[0]

    dict_users = {i: [] for i in range(num_clients)}

    print(f"正在进行 Dirichlet 分布划分 (Total: {len(dataset)} samples)...")

    # --- 4. 核心分配逻辑 ---
    for class_id in range(num_classes):
        indices = label_to_indices[class_id]
        np.random.shuffle(indices)
        num_indices = len(indices)

        # 找出拥有该类别的客户端
        clients_with_class = [cid for cid, classes in client_classes.items() if class_id in classes]
        if not clients_with_class:
            continue

        # 生成比例
        try:
            proportions = np.random.dirichlet(np.repeat(alpha, len(clients_with_class)))
        except ValueError:
            proportions = np.random.dirichlet(np.repeat(alpha + 1e-5, len(clients_with_class)))

        # [修复] 简单的归一化和切分
        proportions = proportions / proportions.sum()
        # 计算切分点：cumsum 得到累计比例，乘以总数得到切分索引
        split_points = (np.cumsum(proportions) * num_indices).astype(int)[:-1]

        # 执行切分
        split_indices = np.split(indices, split_points)

        # 分配给客户端
        for i, client_id in enumerate(clients_with_class):
            if len(split_indices[i]) > 0:
                dict_users[client_id].extend(split_indices[i])

    # --- 5. 防止极端 Dirichlet 小 alpha 产生空客户端 ---
    # 分类任务下如果某个 client 没有任何样本，后续 DataLoader 会报
    # "num_samples should be a positive integer"。这里从样本最多的客户端搬一个样本
    # 到空客户端，保证每个 client 至少能参与训练；搬移保持总样本集合不变。
    empty_clients = [cid for cid, values in dict_users.items() if len(values) == 0]
    if empty_clients:
        print(f"[Dirichlet rebalance] 发现 {len(empty_clients)} 个空客户端，正在从样本最多客户端补齐。")
    for empty_cid in empty_clients:
        donor_cid = max(dict_users, key=lambda cid: len(dict_users[cid]))
        if len(dict_users[donor_cid]) <= 1:
            break
        moved_index = dict_users[donor_cid].pop()
        dict_users[empty_cid].append(moved_index)

    # 转换为 set
    for i in dict_users:
        dict_users[i] = set(dict_users[i])

    print("数据分布完成。")
    return dict_users


def _distribute_domain_skew(dataset, num_clients, classes_per_client, domains_per_client):
    """
    模拟领域倾斜。
    """
    print(f"数据分布策略: Domain Skew (每个客户端 {classes_per_client} 个类, {domains_per_client} 个领域)")
    # 假设 'domain' 列存在且为整数编码
    if 'domain' not in dataset.column_names:
        print("警告: 数据集没有 'domain' 列。将退化为 Pathological Non-IID 分布。")
        return _distribute_pathological_non_iid(dataset, num_clients, classes_per_client)

    # 按领域和类别对索引进行分组
    domain_label_to_indices = collections.defaultdict(lambda: collections.defaultdict(list))
    for i in range(len(dataset)):
        domain = dataset[i]['domain']
        label = dataset[i]['labels']
        if hasattr(domain, 'item'): domain = domain.item()
        if hasattr(label, 'item'): label = label.item()
        domain_label_to_indices[domain][label].append(i)

    # 为每个客户端分配领域
    all_domains = list(domain_label_to_indices.keys())
    client_domains = {}
    for i in range(num_clients):
        client_domains[i] = np.random.choice(all_domains, int(domains_per_client), replace=False)

    # 分配数据
    dict_users = {i: [] for i in range(num_clients)}
    for client_id, domains in client_domains.items():
        client_indices = []
        for domain_id in domains:
            # 在这个领域内，为客户端随机选择 classes_per_client 个类别
            available_labels = list(domain_label_to_indices[domain_id].keys())
            chosen_labels = np.random.choice(available_labels, min(len(available_labels), int(classes_per_client)),
                                             replace=False)
            for label_id in chosen_labels:
                client_indices.extend(domain_label_to_indices[domain_id][label_id])
        dict_users[client_id] = set(client_indices)

    return dict_users


def _distribute_by_length_skew(dataset, num_clients, severity='severe'):
    """
    为生成式任务设计：按文本长度进行排序和切分，模拟 Non-IID。
    """
    if severity not in ['mild', 'severe']:
        raise ValueError(f"未知的 length_skew 类型: {severity}")
    print(f"数据分布策略: Length Skew ({severity}, 用于生成式任务)")

    # 1. 计算每个样本的有效长度 (优先用 attention_mask，避免 max_length padding 干扰)
    lengths = _get_effective_lengths(dataset)

    # 2. 获取按长度排序后的数据索引
    sorted_indices = np.argsort(lengths)

    dict_users = {i: set() for i in range(num_clients)}
    if severity == 'severe':
        split_indices = np.array_split(sorted_indices, num_clients)
        for i in range(num_clients):
            dict_users[i] = set(split_indices[i].tolist())
        return dict_users

    # mild: 先切更多 shard，再随机分配到客户端，减轻长度偏斜
    num_shards = max(num_clients * 4, num_clients)
    shards = [shard.tolist() for shard in np.array_split(sorted_indices, num_shards) if len(shard) > 0]
    np.random.shuffle(shards)
    for idx, shard in enumerate(shards):
        client_id = idx % num_clients
        dict_users[client_id].update(shard)

    return dict_users

# =================================================================================
# 主调度函数 (统一接口)
# =================================================================================
def distribute_data(dataset, num_clients, args):
    """
    根据 args.distribution 的策略，将数据分配给客户端。
    """
    # [核心修正-1] 使用更健壮的方式判断是否是生成式任务
    is_generative_task = False

    # 检查数据集中是否存在 'labels' 列
    if 'labels' in dataset.column_names:
        # 取出第一个样本的标签
        first_label = dataset[0]['labels']

        # 判断第一个标签是否是 "list-like" (列表、NumPy数组或PyTorch张量)
        # 并且检查其维度。标量（数字）没有 .ndim 属性或 .ndim == 0
        if hasattr(first_label, 'ndim') and first_label.ndim > 0:
            is_generative_task = True
        elif isinstance(first_label, list):
            is_generative_task = True
    else:
        # 如果连 'labels' 列都没有，那肯定是生成式任务（或需要特殊处理）
        is_generative_task = True

    if is_generative_task:
        dist_str = args.distribution.lower().strip()
        if dist_str == 'iid':
            print("检测到生成式任务，按 IID 划分。")
            return _distribute_iid(dataset, num_clients)
        if dist_str in ['length_skew', 'length_skew:severe']:
            return _distribute_by_length_skew(dataset, num_clients, severity='severe')
        if dist_str == 'length_skew:mild':
            return _distribute_by_length_skew(dataset, num_clients, severity='mild')
        if dist_str.startswith('length_skew:'):
            skew_type = dist_str.split(':', 1)[1]
            raise ValueError(f"未知的 length_skew 类型: {skew_type}")

        # 兼容历史生成式任务：旧命令中可能传了分类分布参数，默认回退到 severe length skew。
        if getattr(args, 'dataset', '').lower() in ['dolly15k', 'natural_instructions']:
            print(
                f"检测到生成式任务且 distribution='{args.distribution}' 不适配，"
                "回退到 length_skew:severe。"
            )
            return _distribute_by_length_skew(dataset, num_clients, severity='severe')
        raise ValueError(f"生成式任务不支持的数据分布策略: {args.distribution}")

    # --- 以下是为分类任务设计的分布策略 ---
    print("检测到分类任务 (标签是标量)，将根据 --distribution 参数进行分配。")
    dist_str = args.distribution.lower()

    if dist_str == 'iid':
        return _distribute_iid(dataset, num_clients)

    elif 'patho:' in dist_str:
        parts = dist_str.split(':')
        classes_per_client = int(parts[1])
        return _distribute_pathological_non_iid(dataset, num_clients, classes_per_client)

    elif 'dirichlet:' in dist_str:
        parts = dist_str.split(':')
        alpha = float(parts[1])
        return _distribute_dirichlet_label_skew(dataset, num_clients, alpha)

    elif '/' in dist_str:
        parts = dist_str.split('/')
        if 'd' in parts[1].lower():
            # 这是领域倾斜，我们在 Part 2 中实现
            classes_per_client = int(parts[0])
            domains_per_client = int(parts[1][1:])
            print("领域倾斜暂未实现，将使用标签倾斜作为替代。")
            return _distribute_domain_skew(dataset, num_clients, classes_per_client, domains_per_client)
        else:
            classes_per_client = int(parts[0])
            alpha = float(parts[1])
            return _distribute_dirichlet_label_skew(dataset, num_clients, alpha, classes_per_client)
    else:
        raise ValueError(f"未知的数据分布策略: {args.distribution}")

# 文件: utils/data_utils.py

def custom_collate_fn_for_generation(batch):
    """
    [最终修正版] 强健的 Collate 函数。
    1. 提取 'prompts', 'text_labels' 等字符串字段，保留为 List。
    2. 将 'input_ids', 'attention_mask', 'labels' 等数值字段转换为 Tensor。
    """
    from torch.utils.data.dataloader import default_collate

    # 1. 准备容器
    prompts = []
    text_labels = []
    clean_batch = [] # 用于存放只包含数值的 dict，交给 default_collate 处理

    for item in batch:
        # 提取字符串字段 (pop 会从 item 中移除它们)
        if 'prompts' in item:
            prompts.append(item.pop('prompts'))
        if 'text_labels' in item:
            text_labels.append(item.pop('text_labels'))

        # 剩下的 item 应该只包含 input_ids, labels 等数值
        clean_batch.append(item)

    # 2. 将数值字段转换为 Tensor (自动堆叠)
    # 这一步非常关键，它保证了 evaluate_model_loss 拿到的是 Tensor
    tensor_batch = default_collate(clean_batch)

    # 3. 把字符串字段塞回去
    if prompts:
        tensor_batch['prompts'] = prompts
    if text_labels:
        tensor_batch['text_labels'] = text_labels

    return tensor_batch

# old
# def distribute_data(dataset, num_clients):
#     """
#     将数据分成 non-iid 的分片。
#     """
#     num_items = len(dataset)
#     dict_users, all_idxs = {}, [i for i in range(num_items)]
#
#     # 模拟 Non-IID: 按标签排序后分发
#     print("正在模拟 Non-IID 数据分布...")
#     # [核心修正] 增加类型检查，使其对 Tensor 和普通 int 都兼容
#     labels = []
#     print(len(all_idxs))
#     for i in all_idxs:
#         if i % 500 == 0:
#             print(i)
#         label = dataset[i]['labels']
#         # 判断 label 是否是张量或 NumPy 类型
#         if hasattr(label, 'item'):
#             labels.append(label.item())
#         else: # 如果它就是普通的 int 或 float
#             labels.append(label)
#
#     labels = np.array(labels)
#
#     idxs_sorted = np.argsort(labels)
#
#     # 每个客户端分到的数据索引
#     items_per_client = num_items // num_clients
#
#     for i in range(num_clients):
#         start = i * items_per_client
#         end = start + items_per_client
#         dict_users[i] = set(idxs_sorted[start:end])
#
#     print("数据分配完成。")
#     return dict_users
