import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import os
import gc
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
LEARNING_RATE = 1e-3
MAX_STEPS = 450          # 450步，让损失曲线充分拉开差距

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的设备: {device}")

# ==========================================
# 2. 数据准备 (强增强逻辑)
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
    dataset = load_from_disk(local_path)
else:
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
# 3. 动态获取目标层逻辑 (6种实验策略)
# ==========================================
def get_target_modules(model, strategy):
    target_list =[]

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'attention' in name:
            match = re.search(r'\.layer\.(\d+)\.(.*?)(query|key|value)', name)
            if match:
                layer_idx = int(match.group(1))
                matrix_type = match.group(3)

                if strategy == "deep_1_3" and layer_idx >= 8:
                    target_list.append(name)
                elif strategy == "shallow_1_3" and layer_idx < 4:
                    target_list.append(name)
                elif strategy == "only_q" and matrix_type == "query":
                    target_list.append(name)
                elif strategy == "only_k" and matrix_type == "key":
                    target_list.append(name)
                elif strategy == "only_v" and matrix_type == "value":
                    target_list.append(name)
                elif strategy == "only_q_and_v":
                    target_list.append(name)
                elif strategy == "all_lora":  # 终极对比组：全部36个矩阵
                    target_list.append(name)


    return target_list

# ==========================================
# 4. 核心训练函数
# ==========================================
def run_experiment(strategy):
    print(f"\n" + "="*50)
    print(f"🚀 开始实验: {strategy}")
    print("="*50)

    base_model = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    )

    target_list = get_target_modules(base_model, strategy)
    expected_len = 36 if strategy == "all_lora" else 12
    print(f"  [*] 选中挂载 LoRA 的矩阵数量: {len(target_list)} 个 (期望为 {expected_len})")

    if len(target_list) != expected_len:
        print(f"  [警告] 提取的矩阵数量不等于 {expected_len}，请检查逻辑！")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=target_list,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"]
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    losses =[]

    step = 0
    pbar = tqdm(trainloader, desc=f"Training {strategy}", total=MAX_STEPS)

    for inputs, labels in pbar:
        step += 1
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(pixel_values=inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        if step >= MAX_STEPS:
            break

    pbar.close()

    del model, base_model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    return losses

# ==========================================
# 5. 执行 6 组实验
# ==========================================
experiment_strategies =["shallow_1_3", "deep_1_3", "only_q", "only_k", "only_v", "all_lora"]
all_results = {}

for strat in experiment_strategies:
    all_results[strat] = run_experiment(strat)

# ==========================================
# 6. 数据平滑算法
# ==========================================
def smooth_curve(scalars, weight=0.85):
    smoothed = []
    last = scalars[0]
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

# ==========================================
# 7. 绘制多组对比图的生成器
# ==========================================
print("\n正在生成 3 张对比曲线图...")
save_dir = "test_picture/compare"
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
steps_range = list(range(1, MAX_STEPS + 1))

# 定义图例标签和颜色
plot_config = {
    "shallow_1_3": {"label": "Shallow 1/3 (Layers 0-3, QKV, 12 Mat)", "color": "royalblue"},
    "deep_1_3":    {"label": "Deep 1/3 (Layers 8-11, QKV, 12 Mat)", "color": "crimson"},
    "only_q":      {"label": "Only Q (All 12 Layers, 12 Mat)", "color": "forestgreen"},
    "only_k":      {"label": "Only K (All 12 Layers, 12 Mat)", "color": "darkorange"},
    "only_v":      {"label": "Only V (All 12 Layers, 12 Mat)", "color": "purple"},
    "all_lora":    {"label": "Upper Bound: All LoRA (36 Mat)", "color": "black"} # 黑色作为绝对天花板基准
}

def plot_subset(strategies_to_plot, title, filename_suffix):
    """
    画图通用函数：传入想要在一张图里对比的策略列表，自动生成对比图。
    """
    plt.figure(figsize=(12, 8))

    for strat in strategies_to_plot:
        raw_losses = all_results[strat]
        smoothed_losses = smooth_curve(raw_losses)
        color = plot_config[strat]["color"]
        label = plot_config[strat]["label"]

        # 画带透明度的原始锯齿线
        plt.plot(steps_range, raw_losses, color=color, alpha=0.15)
        # 画平滑线
        plt.plot(steps_range, smoothed_losses, color=color, linewidth=2.5, label=label)

    plt.title(title, fontsize=16)
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Training Loss", fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{filename_suffix}_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 图表已保存: {save_path}")

# --- 生成三张图 ---

# 图 1：包含所有 6 组实验 (展示全景概貌)
plot_subset(["shallow_1_3", "deep_1_3", "only_q", "only_k", "only_v", "all_lora"],
    "Ablation: All 6 Strategies Comparison",
    "ablation_all_6_strategies"
)

# 图 2：只包含 2 组 (对比浅层和深层)
plot_subset(["shallow_1_3", "deep_1_3"],
    "Ablation: Shallow vs. Deep (Same Parameter Budget)",
    "ablation_shallow_vs_deep"
)

# 图 3：只包含 3 组 (对比 Q, K, V 的单项能力)
plot_subset(
    ["only_q", "only_k", "only_v"],
    "Ablation: Q vs. K vs. V Matrices Importance",
    "ablation_q_k_v"
)

print(f"🎉 所有实验完成！前往 {save_dir} 查看结果。")