import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg') # 保证在无图形界面的服务器上画图不会报错崩溃
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime
from transformers import ViTForImageClassification
from peft import get_peft_model, LoraConfig
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

# ==========================================
# 1. 超参数与实验配置
# ==========================================
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
NUM_CLASSES = 100
BATCH_SIZE = 16
# 建议: 如果 5e-3 导致 loss 爆炸，请改回 1e-3
LEARNING_RATE = 5e-3
MAX_STEPS = 300
LOG_STEPS =[1,2,3,51, 52, 53,151, 152,153]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# ==========================================
# 2. 数据准备 (使用离线缓存 + 强数据增强)
# ==========================================
print("正在加载 CIFAR-100 数据集...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

local_path = os.path.join(os.getcwd(), "cifar100_local")
if os.path.exists(local_path):
    print(f"--- [Data] 发现本地数据集: {local_path}，正在加载... ---")
    dataset = load_from_disk(local_path)
else:
    print("--- [Data] 未发现本地数据集，尝试在线加载... ---")
    dataset = load_dataset("cifar100")

img_col = 'image' if 'image' in dataset['train'].column_names else 'img'
label_col = 'labels' if 'labels' in dataset['train'].column_names else 'fine_label'

class HFCIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform
        self.img_col = img_col
        self.label_col = label_col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.img_col]
        if not hasattr(image, 'convert'):
            from PIL import Image
            import numpy as np
            image = Image.fromarray(np.array(image))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transform(image)
        label = item[self.label_col]
        return image, label

trainset = HFCIFAR100Dataset(dataset['train'], transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# ==========================================
# 3. 模型准备 (ViT + QKV 联合 LoRA)
# ==========================================
print("正在加载预训练模型并插入 LoRA...")
base_model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["query", "key", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"]
)

model = get_peft_model(base_model, lora_config)
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. 核心统计函数 (★ 同时提取 12层 和 36矩阵 维度的三个指标 ★)
# ==========================================
def extract_all_metrics(model):
    """
    同时评估:
    1. 维度：12 层 (Block) vs 36 矩阵 (Module)
    2. 指标：梯度范数 (GN), 泰勒展开 (TA), FIM-HLA (FI)
    """
    # 数据结构初始化
    stats = {
        "block": {},   # 存放 12 层的统计 (0-11)
        "module": {}   # 存放 36 矩阵的统计 (L0_q, L0_k ...)
    }

    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad and param.grad is not None:
            match = re.search(r'\.layer\.(\d+)\..*?(query|key|value)', name)
            if match:
                layer_idx = int(match.group(1))
                m_type = match.group(2)[0]  # 取首字母 q/k/v
                module_id = f"L{layer_idx}_{m_type}"

                # 若字典不存在则初始化
                if layer_idx not in stats["block"]:
                    stats["block"][layer_idx] = {"gn": 0.0, "ta": 0.0, "fi": 0.0}
                if module_id not in stats["module"]:
                    stats["module"][module_id] = {"gn": 0.0, "ta": 0.0, "fi": 0.0}

                # 提取梯度与权重数值
                g = param.grad.detach()
                w = param.detach()

                # 计算三个指标项
                gn_val = g.norm(2).item() ** 2        # L2范数的平方 (待汇总后开根号)
                ta_val = (w * g).abs().sum().item()   # 泰勒一阶近似
                fi_val = g.pow(2).mean().item()       # FIM (基于平方梯度的平均值)

                # --- 累加到 Block 维度 (12层) ---
                stats["block"][layer_idx]["gn"] += gn_val
                stats["block"][layer_idx]["ta"] += ta_val
                stats["block"][layer_idx]["fi"] += fi_val

                # --- 累加到 Module 维度 (36矩阵) ---
                stats["module"][module_id]["gn"] += gn_val
                stats["module"][module_id]["ta"] += ta_val
                stats["module"][module_id]["fi"] += fi_val

    # 后处理：对梯度平方和进行开根号操作
    for idx in stats["block"]:
        stats["block"][idx]["gn"] = stats["block"][idx]["gn"] ** 0.5
    for mid in stats["module"]:
        stats["module"][mid]["gn"] = stats["module"][mid]["gn"] ** 0.5

    # 为了图表横轴顺序不错乱，进行统一排序
    # 1. Block 排序 (按 0,1,2...11)
    sorted_block = {k: stats["block"][k] for k in sorted(stats["block"].keys())}

    # 2. Module 排序 (按 层号 -> q -> k -> v)
    # 修复后的排序逻辑
    def sort_mod_key(mid):
        # 使用正则表达式提取出所有的数字部分
        # 例如 "L10_v" -> ["10"], "L2_q" -> ["2"]
        numbers = re.findall(r'\d+', mid)
        layer_num = int(numbers[0]) if numbers else 999

        # 提取模块类型 (q, k, v)
        m_type = mid.split('_')[1]
        order = {'q': 0, 'k': 1, 'v': 2}

        # 返回一个元组: (层号, 模块顺序)
        # 这样 Python 就会先比较层号数字大小，再比较模块顺序
        return (layer_num, order.get(m_type, 3))

    sorted_module = {k: stats["module"][k] for k in sorted(stats["module"].keys(), key=sort_mod_key)}

    return {"block": sorted_block, "module": sorted_module}

# ==========================================
# 5. 极简训练与记录循环
# ==========================================
print("开始训练并提取指标...")
history_stats = {}  # 存放所有被 LOG 的指标历史
step_losses =[]    # 存放每一步的 Loss

step = 0
pbar = tqdm(trainloader, desc="Training Steps", total=MAX_STEPS)

for inputs, labels in pbar:
    step += 1
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = model(pixel_values=inputs, labels=labels)
    loss = outputs.loss
    step_losses.append(loss.item())

    loss.backward()

    # 提取三大指标
    if step in LOG_STEPS:
        current_stats = extract_all_metrics(model)
        history_stats[step] = current_stats
        pbar.write(f"[*] Step {step} 指标提取完毕. Loss = {loss.item():.4f}")

    # 梯度裁剪防爆
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    optimizer.zero_grad()

    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    if step >= MAX_STEPS:
        break

pbar.close()
print("训练与提取结束，准备画图...")

# ==========================================
# 6. 【关键修改】拼接绘图函数 (生成 2 张图，每张包含 3 个子图)
# ==========================================
def save_combined_plots(history_stats, step_losses, timestamp):
    save_dir = "test_picture"
    os.makedirs(save_dir, exist_ok=True)

    for granularity in ["block", "module"]:
        fig, axes = plt.subplots(1, 3, figsize=(24, 6)) # 1行3列
        metrics = {"gn": "Gradient Norm", "ta": "Taylor Score", "fi": "FIM Score"}

        # 确定X轴顺序
        if history_stats:
            last_step = list(history_stats.keys())[-1]
            x_labels = sorted(list(history_stats[last_step][granularity].keys()))
            x_indices = range(len(x_labels))

        for i, (m_key, m_title) in enumerate(metrics.items()):
            ax = axes[i]
            for step in LOG_STEPS:
                if step not in history_stats: continue
                y_vals = [history_stats[step][granularity][key][m_key] for key in x_labels]
                ax.plot(x_indices, y_vals, marker='o', label=f'Step {step}')

            ax.set_title(f"{granularity.upper()} - {m_title}")
            ax.set_xticks(x_indices)
            if granularity == "module": ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)
            if i == 2: ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"combined_{granularity}_{timestamp}.png"), dpi=300)
        plt.close()

    # 最后单独存一张 Loss 图
    plt.figure(figsize=(8, 5))
    plt.plot(step_losses, color='red')
    plt.title("Training Loss Curve")
    plt.savefig(os.path.join(save_dir, f"loss_curve_{timestamp}.png"), dpi=300)
    plt.close()

# 运行绘图
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_combined_plots(history_stats, step_losses, timestamp)
print(f"🎉 实验 1 & 2 已全部完成！")