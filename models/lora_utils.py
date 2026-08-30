# 文件: models/lora_utils.py
import torch
from collections import OrderedDict
from peft import get_peft_model, LoraConfig, TaskType
import re
import torch.nn.functional as F

def natural_keys(text):
    """
    用于列表排序的辅助函数，将字符串中的数字提取出来作为排序依据。
    例如: 'layer.2' 会排在 'layer.10' 前面
    """
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]

def find_lora_candidates(model):
    """
    [增强版 Pro] 查找模型中所有可应用 LoRA 的层名。
    兼容性策略：
    1. 对 LLM (OPT/Llama)：匹配所有线性层 (q,k,v,o,fc) 以提升拟合能力。
    2. 对 ViT/BERT：保持原有逻辑，确保图像任务正常。
    3. 安全性：显式排除 LM Head 和 Classifier。
    """
    target_modules_candidates = []

    for name, module in model.named_modules():
        # 必须是线性层
        if not isinstance(module, torch.nn.Linear):
            continue

        # [新增] 安全过滤：排除最后的分类头或语言模型头
        # 这些层通常需要全量训练(modules_to_save)，不应挂载 LoRA
        if any(k in name for k in ['lm_head', 'score', 'classifier', 'head', 'embed_out']):
            continue

        # 1. LLM 增强逻辑 (OPT, LLaMA, GPT-Neo, Dolly任务)
        # OPT 结构: q_proj, k_proj, v_proj, out_proj, fc1, fc2
        # Llama 结构: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        # 这里的关键词涵盖了绝大多数 LLM 的线性层
        if any(k in name for k in ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']):
            target_modules_candidates.append(name)

        # ==========================================
        # [新增] DeBERTa V3 专属匹配逻辑
        # ==========================================
        elif 'attention' in name and any(k in name for k in['query_proj', 'key_proj', 'value_proj']):
            target_modules_candidates.append(name)
        # 2. BERT / RoBERTa 常见命名 (保持不变)
        elif 'attention.self' in name and any(k in name for k in ['query', 'value', 'key']):
             target_modules_candidates.append(name)

        # 3. ViT (Vision Transformer) 常见命名 (保持不变)
        # 仅针对 CIFAR-100 等视觉任务
        elif 'attention.attention' in name and any(k in name for k in ['query', 'value', 'key']):
            target_modules_candidates.append(name)

        # 4. Swin Transformer 等其他变体 (保持不变)
        elif ('attn.q' in name or 'attn.v' in name) and 'bias' not in name:
            target_modules_candidates.append(name)
    # [核心修复] 使用自然排序，而不是默认的字符串排序
    unique_candidates = sorted(list(set(target_modules_candidates)), key=natural_keys)
    # 去重并排序，确保顺序稳定
    return unique_candidates

def find_lora_target_modules(model):
    # """查找模型中所有适合应用LoRA的线性层"""
    # lora_module_names = set()
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Linear):
    #         names = name.split('.')
    #         lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    # if 'query' in lora_module_names and 'value' in lora_module_names:
    #     return ['query', 'value']
    # return list(lora_module_names)
    return find_lora_candidates(model)


def infer_lora_candidate_metadata(module_name: str) -> dict:
    """Best-effort metadata for a LoRA candidate module name."""
    block_match = re.search(r'(?:layers?|blocks?|h)\.(\d+)', module_name)
    if not block_match:
        block_match = re.search(r'\.(\d+)\.', module_name)
    block_id = int(block_match.group(1)) if block_match else -1

    lowered = module_name.lower()
    module_type = "unknown"
    group_type = "unknown"
    if any(k in lowered for k in ["q_proj", "query"]):
        module_type = "q"
        group_type = "attention"
    elif any(k in lowered for k in ["k_proj", "key"]):
        module_type = "k"
        group_type = "attention"
    elif any(k in lowered for k in ["v_proj", "value"]):
        module_type = "v"
        group_type = "attention"
    elif any(k in lowered for k in ["out_proj", "o_proj", "output.dense", "attention.output"]):
        module_type = "out"
        group_type = "attention"
    elif any(k in lowered for k in ["fc1", "gate_proj", "up_proj", "intermediate.dense"]):
        module_type = "ffn1"
        group_type = "mlp"
    elif any(k in lowered for k in ["fc2", "down_proj", "output_projection"]):
        module_type = "ffn2"
        group_type = "mlp"

    return {
        "module_name": module_name,
        "block_id": block_id,
        "module_type": module_type,
        "group_type": group_type,
    }


def apply_lora_to_model(model, assignment, r=8, lora_alpha=16, lora_dropout=0.1):
    """
    [重构版] 支持 LEGEND 的异构秩 (Heterogeneous Rank) 注入。
    如果 assignment 中的值大于 1，则将其作为该层特定的秩 (Rank)。
    """
    target_modules_candidates = find_lora_candidates(model)
    if not target_modules_candidates:
        return model

    actual_assignment = assignment[:len(target_modules_candidates)]
    target_modules =[]
    rank_pattern = {} # 存放异构秩的字典

    for i, val in enumerate(actual_assignment):
        if val > 0:
            mod_name = target_modules_candidates[i]
            target_modules.append(mod_name)
            # 如果 val 是 1，使用默认秩 r；如果 val > 1，使用 val 作为特定秩
            rank_pattern[mod_name] = int(val) if val > 1 else r

    if not target_modules:
        return model

    modules_to_save =['classifier','head', 'score', 'lm_head', 'output_projection']

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        rank_pattern=rank_pattern,  # [核心新增] PEFT 原生支持异构秩
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=modules_to_save
    )

    try:
        lora_model = get_peft_model(model, peft_config)
    except Exception as e:
        peft_config.task_type = None
        lora_model = get_peft_model(model, peft_config)

    return lora_model

def get_lora_state_dict(model):
    """
    [修复版] 从模型中提取所有可训练参数的 state_dict。
    包括 LoRA 参数 (lora_A, lora_B) 和 modules_to_save (classifier, score 等)。
    """
    my_state_dict = model.state_dict()
    trainable_keys = [n for n, p in model.named_parameters() if p.requires_grad]

    return OrderedDict((k, my_state_dict[k]) for k in trainable_keys if k in my_state_dict)

def average_lora_weights(weight_list: list) -> OrderedDict:
    """
    [重构版] 使用 FedAvg 聚合异构的 LoRA 权重。
    支持 LEGEND 等算法带来的形状不匹配 (不同秩) 问题，使用补零 (Zero-padding) 对齐。
    """
    if not weight_list:
        return OrderedDict()

    all_keys = set()
    for state_dict in weight_list:
        all_keys.update(state_dict.keys())

    avg_weights = OrderedDict()

    for k in sorted(list(all_keys)):
        tensors_for_key = [w[k].float() for w in weight_list if k in w]

        if tensors_for_key:
            # 获取该参数在所有客户端中的最大形状维度
            # 例如: lora_A 有的客户端是 (8, 768)，有的是 (16, 768)，则 max_shape 为 (16, 768)
            max_shape = [max(t.shape[i] for t in tensors_for_key) for i in range(len(tensors_for_key[0].shape))]

            padded_tensors =[]
            for t in tensors_for_key:
                if list(t.shape) != max_shape:
                    # 动态构建 pad 序列，从最后一位开始补齐
                    pad_tuple =[]
                    for i in reversed(range(len(max_shape))):
                        pad_tuple.extend([0, max_shape[i] - t.shape[i]])
                    padded_t = F.pad(t, pad_tuple)
                    padded_tensors.append(padded_t)
                else:
                    padded_tensors.append(t)

            stacked_tensors = torch.stack(padded_tensors)
            avg_tensor = stacked_tensors.sum(0) / len(padded_tensors)
            avg_weights[k] = avg_tensor

    return avg_weights
