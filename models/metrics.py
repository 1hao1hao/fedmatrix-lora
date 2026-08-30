# 文件: models/metrics.py
"""Evaluation helpers for federated LoRA experiments."""

import os
import string
import uuid

import torch
import evaluate
from sklearn.metrics import f1_score


def evaluate_model_loss(model, dataloader, device):
    """
    [最终稳健版]
    1. 兼容 List[Tensor] -> 使用 torch.stack
    2. 兼容 List[int/float] -> 使用 torch.tensor
    3. 自动跳过 prompts 等字符串
    """
    model.eval()
    model.to(device)
    total_loss = 0.0
    total_steps = 0

    # print(f"\n[DEBUG] 开始评估 Loss... 数据加载器长度: {len(dataloader)}")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_batch = {}

            # 1. 组装 input_batch
            for k, v in batch.items():
                # --- A. 跳过字符串辅助列 ---
                if k in ['prompts', 'text_labels', 'text']:
                    continue

                # --- B. 处理已经是 Tensor 的情况 ---
                if isinstance(v, torch.Tensor):
                    input_batch[k] = v.to(device)

                # --- C. 处理 List 的情况 (关键修复) ---
                elif isinstance(v, list):
                    if len(v) == 0: continue

                    # C1. 如果列表里装的是 Tensor (例如 [tensor(..), tensor(..)])
                    if isinstance(v[0], torch.Tensor):
                        try:
                            # [修复点] 使用 stack 进行堆叠
                            input_batch[k] = torch.stack(v, dim=1).to(device)
                        except Exception as e:
                            print(f"[警告] 无法 stack Key '{k}': {e}")
                            continue

                    # C2. 如果列表里装的是数字 (例如 [1, 2, 3])
                    elif isinstance(v[0], (int, float)):
                        try:
                            input_batch[k] = torch.tensor(v).to(device)
                        except Exception as e:
                            print(f"[警告] 无法 tensor Key '{k}': {e}")
                            continue

                    # C3. 其他情况 (如字符串列表)，直接跳过
                    else:
                        continue

            # 2. [核心修复] 检查有效性
            # 只要有 input_ids (文本) 或者 pixel_values (图像)，就是有效的
            if 'input_ids' not in input_batch and 'pixel_values' not in input_batch:
                # print(f"[警告] Batch {i} 缺少输入特征 (input_ids/pixel_values)，跳过。")
                continue

            # 生成式任务中如果 summary 被截断，labels 可能全是 -100。
            # 这种 batch 没有有效监督 token，直接计算 CrossEntropy 会得到 nan。
            if 'labels' in input_batch and torch.all(input_batch['labels'] == -100):
                continue

            # 3. 前向传播
            try:
                outputs = model(**input_batch)
                loss = outputs.loss

                if loss is not None and torch.isfinite(loss):
                    total_loss += loss.item()
                    total_steps += 1

            except Exception as e:
                print(f"[错误] Forward 阶段崩溃: {e}")
                break

    avg_loss = total_loss / total_steps if total_steps > 0 else 0.0
    # print(f"[DEBUG] 评估结束。Avg Loss: {avg_loss}")
    return avg_loss

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_batch = {}
            # 1. 安全的数据类型转换
            for k, v in batch.items():
                if k in ['prompts', 'text_labels', 'text']:
                    continue
                if isinstance(v, torch.Tensor):
                    input_batch[k] = v.to(device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                    # 防止 list 导致维度翻转，沿 dim=1 堆叠
                    input_batch[k] = torch.stack(v, dim=1).to(device)

            # 如果既没有图像特征，也没有文本特征，则跳过
            if 'pixel_values' not in input_batch and 'input_ids' not in input_batch:
                continue
            outputs = model(**input_batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # ==========================================
            # [核心修复] 彻底展平，防止 (32,) 和 (32,1) 错位对比
            # ==========================================
            labels = input_batch['labels'].view(-1)
            predictions = predictions.view(-1)

            # 过滤掉 -100 的 padding 标签（如果有的话）
            valid_mask = labels != -100

            correct += (predictions[valid_mask] == labels[valid_mask]).sum().item()
            total += valid_mask.sum().item()

    return correct / total if total > 0 else 0.0

def evaluate_macro_f1(model, dataloader, device):
    """计算 Macro F1 分数"""
    model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop('labels').to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return f1_score(all_labels, all_preds, average='macro')

def normalize_text(text):
    # 转小写，去除首尾空格
    text = text.lower().strip()
    # (可选) 去除标点: 视具体任务而定，Dolly通常可以不去
    # text = text.translate(str.maketrans('', '', string.punctuation))
    return text
# 文件: models/algorithm.py
def evaluate_rouge_l(model, tokenizer, dataloader, device):
    """
    [修复版] 计算 Rouge-L 分数。
    1. 自动处理 Padding Side。
    2. 限制评估样本数，防止卡死。
    3. 修复 nltk 依赖问题。
    """
    model.to(device)
    model.eval()
    # --- [修复 1] 确保 NLTK 数据存在 ---
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        print("punkt加载成功")
    except (LookupError, OSError):
        # 尝试下载（如果服务器有部分网络权限）
        print("  [提示] 正在尝试下载 NLTK punkt 分词数据...")
        try:
            nltk.download('punkt', quiet=True)
        except:
            print("  [警告] NLTK punkt 下载失败，Rouge 计算可能会报错。请手动下载 punkt 并放入 ~/nltk_data/tokenizers/")

    # --- [修复 2] 加载 Rouge ---
    import evaluate
    print("  [Debug] 正在尝试加载 Rouge 指标...")
    try:
        rouge_path = os.path.join(os.getcwd(), 'metrics', 'rouge', 'rouge.py')

        experiment_id = f"rouge_{os.getpid()}_{uuid.uuid4().hex}"
        if os.path.exists(rouge_path):
             rouge = evaluate.load(rouge_path, experiment_id=experiment_id)
        else:
             rouge = evaluate.load('rouge', experiment_id=experiment_id)
        print("  [Debug] Rouge 加载成功!")

    except Exception as e:
        print(f"  [错误] 无法加载 'rouge' 指标: {e}")
        return 0.0

    all_preds = []
    all_labels = []

    # 临时切换 Tokenizer 到 Left Padding 用于生成
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  [Rouge评估] 开始采样生成 (仅评估前 15 个 Batch)...")

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            # --- [关键优化] 限制评估数量 ---
            # 这里的 8 是 Batch 数。如果你 local_bs=4，那么就是评估 32 条数据。

            # print(step)
            if step >= 8:
                break

            prompts = batch.get('prompts')
            refs = batch.get('text_labels')

            if prompts is None:
                print("  [警告] Batch 中没有 prompts，跳过 Rouge 计算。")
                break

            # 重新 Tokenize Prompts (Left Padding)
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

            # 生成
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=64,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1
            )

            # 解码
            input_len = inputs.input_ids.shape[1]
            generated_tokens = generated_ids[:, input_len:]

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            # [新增] 归一化处理
            decoded_preds = [normalize_text(pred) for pred in decoded_preds]
            # 注意：refs 也要归一化
            batch_refs = [normalize_text(ref) for ref in refs]

            all_preds.extend(decoded_preds)
            all_labels.extend(batch_refs)
            # # [新增] 强制打印前 3 个生成的样本进行检查
            # if step == 0:
            #     print("\n[Debug Generation Content]")
            #     for idx in range(min(3, len(decoded_preds))):
            #         print(f"--- Sample {idx} ---")
            #         print(f"Label: {refs[idx].strip()[:100]}...") # 只打印前100字
            #         print(f"Pred : {decoded_preds[idx].strip()[:100]}...")
            #         print("--------------------")

    # 恢复 Tokenizer 设置
    tokenizer.padding_side = original_padding_side

    if len(all_preds) == 0:
        return 0.0

    # 计算分数
    try:
        results = rouge.compute(
            predictions=all_preds,
            references=all_labels,
        )
        return results['rougeL']
    except Exception as e:
        print(f"  [错误] Rouge 计算过程中出错 (可能是 NLTK 问题): {e}")
        return 0.0
