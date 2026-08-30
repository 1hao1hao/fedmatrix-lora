import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
from peft import get_peft_model, LoraConfig
from utils.data_utils import get_vision_datasets
import argparse
from tqdm import tqdm

# 模拟参数
class Args:
    dataset = 'cifar100'
    num_classes = 100
    local_bs = 64
    gpu = 0
args = Args()
device = torch.device("cuda:0")

# 1. 加载数据
print("Loading Data...")
train_data, test_data, _ = get_vision_datasets('cifar100', args)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# 2. 加载模型
print("Loading Model...")
model_name = 'google/vit-base-patch16-224-in21k'
model = AutoModelForImageClassification.from_pretrained(model_name, num_labels=100)

# 3. 加 LoRA
print("Applying LoRA...")
peft_config = LoraConfig(
    r=8, target_modules=["query", "value"],
    modules_to_save=["classifier"] # 必须训练分类头
)
model = get_peft_model(model, peft_config)
model.to(device)

# 4. 简单训练循环
print("Start Training...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step, batch in enumerate(tqdm(train_loader)):
    # 手动处理 batch
    batch = {k: v.to(device) for k, v in batch.items() if k in ['pixel_values', 'labels']}

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
        if step > 100: break # 跑100步看看能不能降